import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Module, Sequential, ModuleList, Linear, Conv1d
from torch_geometric.nn import radius_graph, knn_graph
from torch_scatter import scatter_sum, scatter_softmax
import numpy as np
from math import pi as PI

from ..common import GaussianSmearing, ShiftedSoftplus
from ..protein_features import ProteinFeatures


class AttentionInteractionBlock(Module):

    def __init__(self, hidden_channels, edge_channels, key_channels, num_heads=1):
        super().__init__()

        assert hidden_channels % num_heads == 0 
        assert key_channels % num_heads == 0

        self.hidden_channels = hidden_channels
        self.key_channels = key_channels
        self.num_heads = num_heads

        self.k_lin = Conv1d(hidden_channels, key_channels, 1, groups=num_heads, bias=False)
        self.q_lin = Conv1d(hidden_channels, key_channels, 1, groups=num_heads, bias=False)
        self.v_lin = Conv1d(hidden_channels, hidden_channels, 1, groups=num_heads, bias=False)

        self.weight_k_net = Sequential(
            Linear(edge_channels, key_channels//num_heads),
            ShiftedSoftplus(),
            Linear(key_channels//num_heads, key_channels//num_heads),
        )
        self.weight_k_lin = Linear(key_channels//num_heads, key_channels//num_heads)

        self.weight_v_net = Sequential(
            Linear(edge_channels, hidden_channels//num_heads),
            ShiftedSoftplus(),
            Linear(hidden_channels//num_heads, hidden_channels//num_heads),
        )
        self.weight_v_lin = Linear(hidden_channels//num_heads, hidden_channels//num_heads)

        self.centroid_lin = Linear(hidden_channels, hidden_channels)
        self.act = ShiftedSoftplus()
        self.out_transform = Linear(hidden_channels, hidden_channels)
        self.layernorm_attention = nn.LayerNorm(hidden_channels)
        self.layernorm_ffn = nn.LayerNorm(hidden_channels)

    def forward(self, x, edge_index, edge_attr):
        """
        Args:
            x:  Node features, (N, H).
            edge_index: (2, E).
            edge_attr:  (E, H)
        """
        N = x.size(0)
        row, col = edge_index   # (E,) , (E,)

        # self-attention layer_norm
        y = self.layernorm_attention(x)

        # Project to multiple key, query and value spaces
        h_keys = self.k_lin(y.unsqueeze(-1)).view(N, self.num_heads, -1)    # (N, heads, K_per_head)
        h_queries = self.q_lin(y.unsqueeze(-1)).view(N, self.num_heads, -1) # (N, heads, K_per_head)
        h_values = self.v_lin(y.unsqueeze(-1)).view(N, self.num_heads, -1)  # (N, heads, H_per_head)

        # Compute keys and queries
        W_k = self.weight_k_net(edge_attr)  # (E, K_per_head)
        keys_j = self.weight_k_lin(W_k.unsqueeze(1) * h_keys[col])  # (E, heads, K_per_head)
        queries_i = h_queries[row]    # (E, heads, K_per_head)

        # Compute attention weights (alphas)
        qk_ij = (queries_i * keys_j).sum(-1)  # (E, heads)
        alpha = scatter_softmax(qk_ij, row, dim=0)

        # Compose messages
        W_v = self.weight_v_net(edge_attr)  # (E, H_per_head)
        msg_j = self.weight_v_lin(W_v.unsqueeze(1) * h_values[col])  # (E, heads, H_per_head)
        msg_j = alpha.unsqueeze(-1) * msg_j   # (E, heads, H_per_head)

        # Aggregate messages
        aggr_msg = scatter_sum(msg_j, row, dim=0, dim_size=N).view(N, -1) # (N, heads*H_per_head)
        x = aggr_msg + x
        y = self.layernorm_ffn(x)
        out = self.out_transform(self.act(y)) + x
        return out


class CFTransformerEncoder(Module):
    
    def __init__(self, hidden_channels=256, edge_channels=64, key_channels=128, num_heads=4, num_interactions=6, k=32, cutoff=10.0):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.edge_channels = edge_channels
        self.key_channels = key_channels
        self.num_heads = num_heads
        self.num_interactions = num_interactions
        self.k = k
        self.cutoff = cutoff

        self.distance_expansion = GaussianSmearing(stop=cutoff, num_gaussians=edge_channels)
        self.interactions = ModuleList()
        for _ in range(num_interactions):
            block = AttentionInteractionBlock(
                hidden_channels=hidden_channels,
                edge_channels=edge_channels,
                key_channels=key_channels,
                num_heads=num_heads,
            )
            self.interactions.append(block)

    @property
    def out_channels(self):
        return self.hidden_channels

    def forward(self, node_attr, pos, batch):
        # edge_index = radius_graph(pos, self.cutoff, batch=batch, loop=False)
        edge_index = knn_graph(pos, k=self.k, batch=batch, flow='target_to_source')
        edge_length = torch.norm(pos[edge_index[0]] - pos[edge_index[1]], dim=1)
        edge_attr = self.distance_expansion(edge_length)

        h = node_attr
        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_attr)
        return h


# residue level graph transformer
class AAEmbedding(nn.Module):

    def __init__(self):
        super(AAEmbedding, self).__init__()
        
        self.hydropathy = {'#': 0, "I":4.5, "V":4.2, "L":3.8, "F":2.8, "C":2.5, "M":1.9, "A":1.8, "W":-0.9, "G":-0.4, "T":-0.7, "S":-0.8, "Y":-1.3, "P":-1.6, "H":-3.2, "N":-3.5, "D":-3.5, "Q":-3.5, "E":-3.5, "K":-3.9, "R":-4.5}
        self.volume = {'#': 0, "G":60.1, "A":88.6, "S":89.0, "C":108.5, "D":111.1, "P":112.7, "N":114.1, "T":116.1, "E":138.4, "V":140.0, "Q":143.8, "H":153.2, "M":162.9, "I":166.7, "L":166.7, "K":168.6, "R":173.4, "F":189.9, "Y":193.6, "W":227.8}
        self.charge = {**{'R':1, 'K':1, 'D':-1, 'E':-1, 'H':0.1}, **{x:0 for x in 'ABCFGIJLMNOPQSTUVWXYZ#'}}
        self.polarity = {**{x:1 for x in 'RNDQEHKSTY'}, **{x:0 for x in "ACGILMFPWV#"}}
        self.acceptor = {**{x:1 for x in 'DENQHSTY'}, **{x:0 for x in "RKWACGILMFPV#"}}
        self.donor = {**{x:1 for x in 'RKWNQHSTY'}, **{x:0 for x in "DEACGILMFPV#"}}
        ALPHABET = ['#', 'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y','V']
        self.embedding = torch.tensor([
             [self.hydropathy[aa], self.volume[aa] / 100, self.charge[aa], self.polarity[aa], self.acceptor[aa], self.donor[aa]]
             for aa in ALPHABET])
#        self.embedding = torch.tensor([
#            [self.hydropathy[aa], self.volume[aa] / 100, self.charge[aa], self.polarity[aa], self.acceptor[aa], self.donor[aa]]
#            for aa in ALPHABET]).cuda()

    def to_rbf(self, D, D_min, D_max, stride):
        D_count = int((D_max - D_min) / stride)
        D_mu = torch.linspace(D_min, D_max, D_count).to(D.device)
        D_mu = D_mu.view(1,-1)  # [1, K]
        D_expand = torch.unsqueeze(D, -1)  # [N, 1]
        return torch.exp(-((D_expand - D_mu) / stride) ** 2)

    def transform(self, aa_vecs):
        return torch.cat([
            self.to_rbf(aa_vecs[:, 0], -4.5, 4.5, 0.1),
            self.to_rbf(aa_vecs[:, 1], 0, 2.2, 0.1),
            self.to_rbf(aa_vecs[:, 2], -1.0, 1.0, 0.25),
            torch.sigmoid(aa_vecs[:, 3:] * 6 - 3),
        ], dim=-1)

    def dim(self):
        return 90 + 22 + 8 + 3

    def forward(self, x, raw=False):
        #B, N = x.size(0), x.size(1)
        #aa_vecs = self.embedding[x.view(-1)].view(B, N, -1)
        self.embedding = self.embedding.to(x.device) 
        aa_vecs = self.embedding[x.view(-1)]
        rbf_vecs = self.transform(aa_vecs)
        return aa_vecs if raw else rbf_vecs


class TransformerLayer(nn.Module):
    def __init__(self, num_hidden, num_heads=4, dropout=0.1):
        super(TransformerLayer, self).__init__()
        self.num_heads = num_heads
        self.num_hidden = num_hidden
        self.dropout_attention = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)
        self.self_attention_norm = nn.LayerNorm(num_hidden)
        self.ffn_norm = nn.LayerNorm(num_hidden)

        self.attention = ResidueAttention(num_hidden, num_heads)
        self.ffn = PositionWiseFeedForward(num_hidden, num_hidden)

    def forward(self, h_V, h_E, E_idx):
        """ Parallel computation of full transformer layer """
        # Self-attention
        y = self.self_attention_norm(h_V)
        y = self.attention(y, h_E, E_idx)
        h_V = h_V + self.dropout_attention(y)

        # Position-wise feedforward
        y = self.ffn_norm(h_V)
        y = self.ffn(y)
        h_V = h_V + self.dropout_ffn(y)
        return h_V


class PositionWiseFeedForward(nn.Module):
    def __init__(self, num_hidden, num_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.W_in = nn.Linear(num_hidden, num_ff, bias=True)
        self.W_out = nn.Linear(num_ff, num_hidden, bias=True)

    def forward(self, h_V):
        h = F.relu(self.W_in(h_V))
        h = self.W_out(h)
        return h


class ResidueAttention(nn.Module):
    def __init__(self, num_hidden, num_heads=4):
        super(ResidueAttention, self).__init__()
        self.num_heads = num_heads
        self.num_hidden = num_hidden

        # Self-attention layers: {queries, keys, values, output}
        self.W_Q = nn.Linear(num_hidden, num_hidden, bias=False)
        self.W_K = nn.Linear(num_hidden*2, num_hidden, bias=False)
        self.W_V = nn.Linear(num_hidden*2, num_hidden, bias=False)
        self.W_O = nn.Linear(num_hidden, num_hidden, bias=False)
        self.act = ShiftedSoftplus()
        self.layernorm = nn.LayerNorm(num_hidden)

    def forward(self, h_V, h_E, edge_index):
        """ Self-attention, graph-structured O(Nk)
        Args:
            h_V:            Node features           [N_batch, N_nodes, N_hidden]
            h_E:            Neighbor features       [N_batch, N_nodes, K, N_hidden]
            mask_attend:    Mask for attention      [N_batch, N_nodes, K]
        Returns:
            h_V:            Node update
        """

        # Queries, Keys, Values
        n_edges = h_E.shape[0]
        n_nodes = h_V.shape[0]
        n_heads = self.num_heads
        row, col = edge_index  # (E,) , (E,)

        d = int(self.num_hidden / n_heads)
        Q = self.W_Q(h_V).view([n_nodes, n_heads, 1, d])
        K = self.W_K(torch.cat([h_E, h_V[col]], dim=-1)).view([n_edges, n_heads, d, 1])
        V = self.W_V(torch.cat([h_E, h_V[col]], dim=-1)).view([n_edges, n_heads, d])
        # Attention with scaled inner product
        attend_logits = torch.matmul(Q[row], K).view([n_edges, n_heads]) # (E, heads)
        alpha = scatter_softmax(attend_logits, row, dim=0) / np.sqrt(d)
        # Compose messages
        msg_j = alpha.unsqueeze(-1) * V   # (E, heads, H_per_head)

        # Aggregate messages
        aggr_msg = scatter_sum(msg_j, row, dim=0, dim_size=n_nodes).view(n_nodes, -1) # (N, heads*H_per_head)
        h_V_update = self.W_O(self.act(aggr_msg))
        return h_V_update


class ResidueTF(nn.Module):
    def __init__(self, node_features, edge_features,
        hidden_dim, num_encoder_layers=2, k_neighbors=8):
        super(ResidueTF, self).__init__()

        # Hyperparameters
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim

        # Featurization layers
        self.residue_feat = AAEmbedding()
        self.features = ProteinFeatures(top_k=k_neighbors)

        # Embedding layers
        self.W_v = nn.Linear(node_features, hidden_dim, bias=True)
        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)

        # Encoder layers
        self.encoder_layers = nn.ModuleList([TransformerLayer(hidden_dim, dropout=0.1) for _ in range(num_encoder_layers)])

        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, X, S_id, R, residue_batch):
        # Prepare node and edge embeddings
        V, E, edge_index = self.features(X, S_id, residue_batch)
        V= torch.cat([V, self.residue_feat(R)], dim=-1)
        h_V = self.W_v(V)
        h_E = self.W_e(E)

        for layer in self.encoder_layers:
            h_V = layer(h_V, h_E, edge_index)
        return h_V


# hierachical graph transformer encoder
class HierEncoder(Module):
    def __init__(self, hidden_channels=256, edge_channels=64, key_channels=128, num_heads=4, num_interactions=6, k=32,
                 cutoff=10.0):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.edge_channels = edge_channels
        self.key_channels = key_channels
        self.num_heads = num_heads
        self.num_interactions = num_interactions
        self.k = k
        self.cutoff = cutoff

        self.distance_expansion = GaussianSmearing(stop=cutoff, num_gaussians=edge_channels)
        self.interactions = ModuleList()
        for _ in range(num_interactions):
            block = AttentionInteractionBlock(
                hidden_channels=hidden_channels,
                edge_channels=edge_channels,
                key_channels=key_channels,
                num_heads=num_heads,
            )
            self.interactions.append(block)

        # Residue level settings
        self.residue_feat = AAEmbedding()
        self.features = ProteinFeatures(top_k=8)
        self.W_v = nn.Linear(hidden_channels+self.residue_feat.dim(), hidden_channels, bias=True)
        self.W_e = nn.Linear(self.features.feature_dimensions, hidden_channels, bias=True)
        self.residue_encoder_layers = nn.ModuleList([TransformerLayer(hidden_channels, dropout=0.1) for _ in range(3)])


    @property
    def out_channels(self):
        return self.hidden_channels

    def forward(self, node_attr, pos, batch, X=None, S_id=None, R=None, residue_batch=None, atom2residue=None, mask=None, node_level=False):
#         print("node_attr", node_attr.cpu().shape)
#         print("pos", pos.cpu().shape)
#         print("batch", batch.cpu().shape)
        edge_index = knn_graph(pos, k=self.k, batch=batch, flow='target_to_source')
        edge_length = torch.norm(pos[edge_index[0]] - pos[edge_index[1]], dim=1)
        edge_attr = self.distance_expansion(edge_length)
#         print("edge_index", edge_index.cpu().shape)
#         print("edge_attr", edge_attr.cpu().shape)
        h = node_attr
        for interaction in self.interactions:
            h = interaction(h, edge_index, edge_attr)

        if node_level:
#             print("h", h.cpu().shape)
            return h

        E, residue_edge_index = self.features(X, S_id, residue_batch)
#         print("E", E.cpu().shape)
#         print("residue_edge_index", residue_edge_index.cpu().shape)
        h_protein = h[mask]
        
#         print('tf', R.device, h_protein.device, atom2residue.device)
#         print('tf', self.residue_feat(R).device, scatter_sum(h_protein, atom2residue, dim=0).device)
        V = torch.cat([self.residue_feat(R), scatter_sum(h_protein, atom2residue, dim=0)], dim=-1)
#         print("self.residue_feat(R)", self.residue_feat(R).cpu().shape)
#         print("V", V.cpu().shape)
        h_V = self.W_v(V)
        h_E = self.W_e(E)

        for layer in self.residue_encoder_layers:
            h_V = layer(h_V, h_E, residue_edge_index)
#         print("h", h.cpu().shape)
#         print("h_V", h_V.cpu().shape)
        return h, h_V



class E_GCL(nn.Module):
    """
    E(n) Equivariant Convolutional Layer
    re
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, act_fn=nn.SiLU(), residual=True, attention=False, normalize=False, coords_agg='mean', tanh=False):
        super(E_GCL, self).__init__()
        input_edge = input_nf * 2
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8
        edge_coords_nf = 1

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp)

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

    def edge_model(self, source, target, radial, edge_attr):
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        if self.residual:
            out = x + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        if self.coords_agg == 'sum':
            agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        elif self.coords_agg == 'mean':
            agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        else:
            raise Exception('Wrong coords_agg parameter' % self.coords_agg)
        coord += agg
        return coord

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff**2, 1).unsqueeze(1)

        if self.normalize:
            norm = torch.sqrt(radial).detach() + self.epsilon
            coord_diff = coord_diff / norm

        return radial, coord_diff

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)

        return h, coord, edge_attr




class EGNN(nn.Module):
    def __init__(self, in_node_nf, hidden_nf, out_node_nf, in_edge_nf=0, device='cpu', act_fn=nn.SiLU(), n_layers=4, residual=True, attention=False, normalize=False, tanh=False):
        '''

        :param in_node_nf: Number of features for 'h' at the input
        :param hidden_nf: Number of hidden features
        :param out_node_nf: Number of features for 'h' at the output
        :param in_edge_nf: Number of features for the edge features
        :param device: Device (e.g. 'cpu', 'cuda:0',...)
        :param act_fn: Non-linearity
        :param n_layers: Number of layer for the EGNN
        :param residual: Use residual connections, we recommend not changing this one
        :param attention: Whether using attention or not
        :param normalize: Normalizes the coordinates messages such that:
                    instead of: x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)
                    we get:     x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)/||x_i - x_j||
                    We noticed it may help in the stability or generalization in some future works.
                    We didn't use it in our paper.
        :param tanh: Sets a tanh activation function at the output of phi_x(m_ij). I.e. it bounds the output of
                        phi_x(m_ij) which definitely improves in stability but it may decrease in accuracy.
                        We didn't use it in our paper.
        '''

        super(EGNN, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.embedding_in = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, E_GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf,
                                                act_fn=act_fn, residual=residual, attention=attention,
                                                normalize=normalize, tanh=tanh))
        self.to(self.device)


    def forward(self, h, x, edges, edge_attr):
        h = self.embedding_in(h)
        for i in range(0, self.n_layers):
            h, x, _ = self._modules["gcl_%d" % i](h, edges, x, edge_attr=edge_attr)
        h = self.embedding_out(h)
        return h, x




class EGNN_encoder(nn.Module):
    def __init__(self, hidden_channels=256, edge_channels=64, key_channels=128, num_heads=4, num_interactions=6, k=32,
                 cutoff=10.0, atom_hiddens=128, atom_layers=7, residue_in_node=379, residue_in_edge=39, residue_hiddens=64, residue_layers=4):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.edge_channels = edge_channels
        self.key_channels = key_channels
        self.num_heads = num_heads
        self.num_interactions = num_interactions
        self.k = k
        self.cutoff = cutoff

        self.distance_expansion = GaussianSmearing(stop=cutoff, num_gaussians=edge_channels)
        self.atom_encoder_layers = EGNN(
            in_node_nf=hidden_channels,  
            hidden_nf=atom_hiddens,  # 128 
            out_node_nf=hidden_channels, 
            in_edge_nf=edge_channels, 
            device='cuda:0',
            n_layers=atom_layers,  # 7
            attention=True,
            normalize=True,
            tanh=True,
        )
        # Residue level settings
        self.residue_feat = AAEmbedding()
        self.features = ProteinFeatures(top_k=8)
        self.residue_encoder_layers = EGNN(
            in_node_nf=residue_in_node,   
            hidden_nf=residue_hiddens,  # 64 
            out_node_nf=hidden_channels, 
            in_edge_nf=residue_in_edge,  
            device='cuda:0',
            n_layers=residue_layers,  # 4
            attention=True,
            normalize=True,
            tanh=True,
        )


    @property
    def out_channels(self):
        return self.hidden_channels

    
    def forward(self, node_attr, pos, batch, X=None, S_id=None, R=None, residue_batch=None, atom2residue=None, mask=None, node_level=False):
        print("node_attr", node_attr.cpu().shape)
        print("pos", pos.cpu().shape)
        print("batch", batch.cpu().shape)
        edge_index = knn_graph(pos, k=self.k, batch=batch, flow='target_to_source')
        edge_length = torch.norm(pos[edge_index[0]] - pos[edge_index[1]], dim=1)
        edge_attr = self.distance_expansion(edge_length)
        print("edge_index", edge_index.cpu().shape)
        print("edge_attr", edge_attr.cpu().shape)
        h = node_attr
        h, atom_x = self.atom_encoder_layers(h, pos, edge_index, edge_attr)

        if node_level:
            print("h", h.cpu().shape)
            print("atom_x", atom_x.cpu().shape)
            return h

        E, residue_edge_index = self.features(X, S_id, residue_batch)
        print("E", E.cpu().shape)
        print("residue_edge_index", residue_edge_index.cpu().shape)
        h_protein = h[mask]
        
        V = torch.cat([self.residue_feat(R), scatter_sum(h_protein, atom2residue, dim=0)], dim=-1)
        print("self.residue_feat(R)", self.residue_feat(R).cpu().shape)
        print("V", V.cpu().shape)
        # h_V = self.W_v(V)
        # h_E = self.W_e(E)

        # for layer in self.residue_encoder_layers:
        #     h_V = layer(h_V, h_E, residue_edge_index)

        # 

        h_V, residue_x = self.residue_encoder_layers(V, X[:,1,:], residue_edge_index, E)

        print("h", h.cpu().shape)
        print("h_V", h_V.cpu().shape)
        print("residue_x", residue_x.cpu().shape)
        return h, h_V



def unsorted_segment_sum(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)


def get_edges(n_nodes):
    rows, cols = [], []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                rows.append(i)
                cols.append(j)

    edges = [rows, cols]
    return edges


def get_edges_batch(n_nodes, batch_size):
    edges = get_edges(n_nodes)
    edge_attr = torch.ones(len(edges[0]) * batch_size, 1)
    edges = [torch.LongTensor(edges[0]), torch.LongTensor(edges[1])]
    if batch_size == 1:
        return edges, edge_attr
    elif batch_size > 1:
        rows, cols = [], []
        for i in range(batch_size):
            rows.append(edges[0] + n_nodes * i)
            cols.append(edges[1] + n_nodes * i)
        edges = [torch.cat(rows), torch.cat(cols)]
    return edges, edge_attr



    

if __name__ == '__main__':
    from torch_geometric.data import Data, Batch

    hidden_channels = 64
    edge_channels = 48
    key_channels = 32
    num_heads = 4

    data_list = []
    for num_nodes in [11, 13, 15]:
        data_list.append(Data(
            x = torch.randn([num_nodes, hidden_channels]),
            pos = torch.randn([num_nodes, 3]) * 2
        ))
    batch = Batch.from_data_list(data_list)

    model = CFTransformerEncoder(
        hidden_channels = hidden_channels,
        edge_channels = edge_channels,
        key_channels = key_channels,
        num_heads = num_heads,
    )
    out = model(batch.x, batch.pos, batch.batch)

    print(out)
    print(out.size())
