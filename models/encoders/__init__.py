from .schnet import SchNetEncoder
from .tf import CFTransformerEncoder, HierEncoder, EGNN_encoder
from .gnn import GNN_graphpred, MLP, WeightGNN


def get_encoder(config):
    if config.name == 'schnet':
        return SchNetEncoder(
            hidden_channels = config.hidden_channels,
            num_filters = config.num_filters,
            num_interactions = config.num_interactions,
            edge_channels = config.edge_channels,
            cutoff = config.cutoff,
        )
    elif config.name == 'tf':
        return CFTransformerEncoder(
            hidden_channels = config.hidden_channels,
            edge_channels = config.edge_channels,
            key_channels = config.key_channels,
            num_heads = config.num_heads,
            num_interactions = config.num_interactions,
            k = config.knn,
            cutoff = config.cutoff,
        )
    elif config.name == 'hierGT':
        return HierEncoder(
            hidden_channels = config.hidden_channels,
            edge_channels = config.edge_channels,
            key_channels = config.key_channels,
            num_heads = config.num_heads,
            num_interactions = config.num_interactions,
            k = config.knn,
            cutoff = config.cutoff,
        )
    elif config.name == 'EGNN':
        return EGNN_encoder(
            hidden_channels = config.hidden_channels,
            edge_channels = config.edge_channels,
            key_channels = config.key_channels,
            num_heads = config.num_heads,
            num_interactions = config.num_interactions,
            k = config.knn,
            cutoff = config.cutoff,     
            atom_hiddens = config.atom_hiddens,
            atom_layers = config.atom_layers,
            residue_in_node=config.residue_in_node, 
            residue_in_edge=config.residue_in_edge,
            residue_hiddens=config.residue_hiddens,
            residue_layers=config.residue_layers,
        )
    else:
        raise NotImplementedError('Unknown encoder: %s' % config.name)
