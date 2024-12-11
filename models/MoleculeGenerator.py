import sys

sys.path.append("..")
import numpy as np
from rdkit import RDConfig
import os
import torch
import torch.nn as nn
from torch.nn import Module, Linear, Embedding
from torch.nn import functional as F
from torch_scatter import scatter_add, scatter_mean
from torch_geometric.data import Data, Batch
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem.rdForceFieldHelpers import UFFOptimizeMolecule

from .encoders import get_encoder, GNN_graphpred, MLP, CFTransformerEncoder
from .common import *
from .vq import VQ
from utils import dihedral_utils, chemutils


ATOM_FAMILIES = ['Acceptor', 'Donor', 'Aromatic', 'Hydrophobe', 'LumpedHydrophobe', 'NegIonizable', 'PosIonizable', 'ZnBinder']
ATOM_FAMILIES_ID = {s: i for i, s in enumerate(ATOM_FAMILIES)}


class multilabel_categorical_crossentropy(nn.Module):
    def __init__(self):
        super(multilabel_categorical_crossentropy, self).__init__()

    def forward(self, y_pred, y_true):
        y_pred = (1 - 2 * y_true) * y_pred  
        y_pred_neg = y_pred - y_true * 1e12  # mask the pred outputs of pos classes，
        y_pred_pos = (y_pred - (1 - y_true) * 1e12)  # mask the pred outputs of neg classes
        zeros = torch.zeros_like(y_pred[..., :1])  # e^0=1
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
        return (neg_loss + pos_loss).mean()



class MoleculeGenerator(Module):

    def __init__(self, config, protein_atom_feature_dim, ligand_atom_feature_dim, vocab, device):
        super().__init__()
        self.config = config
        self.vocab = vocab
        self.device = device
        self.next_motif_loss = self.config.next_motif_loss
        self.protein_atom_emb = Linear(protein_atom_feature_dim, config.hidden_channels)
        self.ligand_atom_emb = Linear(ligand_atom_feature_dim, config.hidden_channels)
        self.embedding = nn.Embedding(vocab.size() + 1, config.hidden_channels)
        self.encoder = get_encoder(config.encoder)
        self.comb_head = GNN_graphpred(num_layer=3, emb_dim=config.hidden_channels, JK='last',
                                       drop_ratio=0.5, graph_pooling='mean', gnn_type='gin')

        self.motif_mlp = MLP(in_dim=config.hidden_channels * 3, out_dim=config.hidden_channels, num_layers=2)
        self.focal_mlp = MLP(in_dim=config.hidden_channels, out_dim=1, num_layers=1)
        self.attach_mlp = MLP(in_dim=config.hidden_channels * 1, out_dim=1, num_layers=1)

        if self.next_motif_loss == "KL":
            self.pred_loss = nn.KLDivLoss(reduction="batchmean")
        elif self.next_motif_loss == "BCE":
            self.pred_loss = nn.BCEWithLogitsLoss()
        else:
            self.pred_loss = multilabel_categorical_crossentropy()
        # self.pred_loss = nn.CrossEntropyLoss()
        self.comb_loss = nn.BCEWithLogitsLoss()
        self.focal_loss = nn.BCEWithLogitsLoss()


    def forward(self, protein_pos, protein_atom_feature, ligand_pos, ligand_atom_feature, batch_protein, batch_ligand, batch):
#         print("protein_atom_feature.device, ligand_atom_feature.device", protein_atom_feature.device, ligand_atom_feature.device)
        h_protein = self.protein_atom_emb(protein_atom_feature)
        h_ligand = self.ligand_atom_emb(ligand_atom_feature)

        h_ctx, pos_ctx, batch_ctx, protein_mask = compose_context_stable(h_protein=h_protein, h_ligand=h_ligand,
                                                                         pos_protein=protein_pos, pos_ligand=ligand_pos,
                                                                         batch_protein=batch_protein,
                                                                         batch_ligand=batch_ligand)
        # print(h_ctx.is_cuda, pos_ctx.is_cuda, batch_ctx.is_cuda, protein_mask.is_cuda)
        h_ctx, h_residue = self.encoder(node_attr=h_ctx, pos=pos_ctx, batch=batch_ctx, X=batch['residue_pos'],
                                        S_id=batch['res_idx'], R=batch['amino_acid'], residue_batch=batch['amino_acid_batch'], atom2residue=batch['atom2residue'], mask=protein_mask)
        focal_pred = self.focal_mlp(h_ctx)

        return focal_pred, protein_mask, h_ctx, pos_ctx, h_residue


    def forward_motif(self, h_ctx_focal, pos_ctx_focal, current_wid, current_atoms_batch, h_residue, residue_pos, amino_acid_batch):
        node_hiddens = scatter_add(h_ctx_focal, dim=0, index=current_atoms_batch)
        motif_hiddens = self.embedding(current_wid)

        center_pos = scatter_mean(pos_ctx_focal, dim=0, index=current_atoms_batch)
        residue_pos = residue_pos[:, 1, :]
        residue_index = torch.where(torch.norm(residue_pos - center_pos[amino_acid_batch], dim=1) < 6)
        residue_emb = torch.zeros_like(node_hiddens)
        added = scatter_add(h_residue[residue_index], dim=0, index=amino_acid_batch[residue_index])
        residue_emb[:added.shape[0]] = added

        pred_vecs = torch.cat([node_hiddens, motif_hiddens, residue_emb], dim=-1)
        pred_scores = torch.matmul(self.motif_mlp(pred_vecs), self.embedding.weight.transpose(1, 0))
        #_, preds = torch.max(pred_scores, dim=1)

        # random select in topk
        k = 5
        select_pool = torch.topk(pred_scores, k, dim=1)[1]
        index = torch.randint(k, (select_pool.shape[0],))
        preds = torch.cat([select_pool[i][index[i]].unsqueeze(0) for i in range(len(index))])
        return preds


    def forward_attach(self, mol_list, next_motif_smiles):
        decoupling_mol_list = []
        for mol in mol_list:
            decoupling_mol_list.append(chemutils.copy_edit_mol(mol))
            
        cand_mols, cand_batch, new_atoms, one_atom_attach, intersection, attach_fail = chemutils.assemble(decoupling_mol_list, next_motif_smiles)

#         print("type(cand_mols), type(cand_batch), type(new_atoms), type(one_atom_attach), type(intersection), type(attach_fail)", type(cand_mols), type(cand_batch), type(new_atoms), type(one_atom_attach), type(intersection), type(attach_fail))
        cand_batch = cand_batch.to(self.device)
        one_atom_attach = one_atom_attach.to(self.device)
        attach_fail = attach_fail.to(self.device)
#         print("cand_batch.device", cand_batch.device)
        graph_data = Batch.from_data_list([chemutils.mol_to_graph_data_obj_simple(mol) for mol in cand_mols])
#         print("type(graph_data)", type(graph_data))
#         print("graph_data.x.device, graph_data.edge_index.device, graph_data.edge_attr.device, graph_data.batch.device",graph_data.x.device, graph_data.edge_index.device, graph_data.edge_attr.device, graph_data.batch.device)
        cand_emb = self.comb_head(graph_data.x.to(self.device), graph_data.edge_index.to(self.device), graph_data.edge_attr.to(self.device), graph_data.batch.to(self.device))
        attach_pred = self.attach_mlp(cand_emb)  
        slice_idx = torch.cat([torch.tensor([0]).to(self.device), torch.cumsum(cand_batch.bincount(), dim=0)], dim=0)
        
        # random select
        select = []
        for k in range(len(slice_idx) - 1):
            id = torch.multinomial(torch.nn.functional.softmax(attach_pred[slice_idx[k]:slice_idx[k + 1]], dim=0).reshape(-1).float(), 1)
            select.append((id+slice_idx[k]).item())
            
        select_mols = [cand_mols[i] for i in select]
        new_atoms = [new_atoms[i] for i in select]
        one_atom_attach = [one_atom_attach[i] for i in select]
        intersection = [intersection[i] for i in select]
        return select_mols, new_atoms, one_atom_attach, intersection, attach_fail  




    def get_loss(self, protein_pos, protein_atom_feature, ligand_pos, ligand_atom_feature, 
                batch_protein, batch_ligand, batch):
        self.device = protein_pos.device
        h_protein = self.protein_atom_emb(protein_atom_feature)
        h_ligand = self.ligand_atom_emb(ligand_atom_feature)

        loss_list = [0, 0, 0]

        # Encode for motif prediction
        h_ctx, pos_ctx, batch_ctx, mask_protein = compose_context_stable(h_protein=h_protein, h_ligand=h_ligand,
                                                                         pos_protein=protein_pos, pos_ligand=ligand_pos,
                                                                         batch_protein=batch_protein,
                                                                         batch_ligand=batch_ligand)

        h_ctx, h_residue = self.encoder(node_attr=h_ctx, pos=pos_ctx, batch=batch_ctx, X=batch['residue_pos'],
                                        S_id=batch['res_idx'], R=batch['amino_acid'], residue_batch=batch['amino_acid_batch'], atom2residue=batch['atom2residue'], mask=mask_protein)  # (N_p+N_l, H)
        h_ctx_ligand = h_ctx[~mask_protein]
        h_ctx_protein = h_ctx[mask_protein]
        h_ctx_focal = h_ctx[batch['current_atoms']]
        pos_ctx_focal = pos_ctx[batch['current_atoms']]


        # next motif prediction
        node_hiddens = scatter_add(h_ctx_focal, dim=0, index=batch['current_atoms_batch'])
        center_pos = scatter_mean(pos_ctx_focal, dim=0, index=batch['current_atoms_batch'])
        residue_pos = batch['residue_pos'][:, 1, :]
        residue_index = torch.where(torch.norm(residue_pos - center_pos[batch['amino_acid_batch']], dim=1)<6)
        residue_emb = torch.zeros_like(node_hiddens)
        added = scatter_add(h_residue[residue_index], dim=0, index=batch['amino_acid_batch'][residue_index])
        residue_emb[:added.shape[0]] = added
        motif_hiddens = self.embedding(batch['current_wid'])
        pred_vecs = torch.cat([node_hiddens, motif_hiddens, residue_emb], dim=-1)
        pred_scores = torch.matmul(self.motif_mlp(pred_vecs), self.embedding.weight.transpose(1, 0))
        #print("pred_scores", pred_scores)
        #print("pred_scores_norm1", pred_scores.norm(1))
        #print("pred_scores_norm2", pred_scores.norm(2))
        
        if self.next_motif_loss == "KL":
            pred_loss_input = F.log_softmax(pred_scores, dim=1)
            sum_wid = torch.sum(batch['next_wid'], dim=-1)
            pred_loss_target = batch['next_wid'] / (sum_wid.reshape(-1, 1))
            # print("pred_loss_target", pred_loss_target)
            # pred_loss_target = F.softmax(batch['next_wid'], dim=1)
            pred_loss = self.pred_loss(pred_loss_input, pred_loss_target)
            loss_list[0] = pred_loss.item()
        else:
            pred_loss = self.pred_loss(pred_scores, batch['next_wid'])
            loss_list[0] = pred_loss.item()

        # attachment prediction
        if len(batch['cand_labels']) > 0:
            cand_mols = batch['cand_mols']
            cand_emb = self.comb_head(cand_mols.x, cand_mols.edge_index, cand_mols.edge_attr, cand_mols.batch)
            #attach_pred = self.attach_mlp(torch.cat([cand_emb, residue_emb[batch['cand_mols_batch']]], dim=-1))
            attach_pred = self.attach_mlp(cand_emb)
            comb_loss = self.comb_loss(attach_pred, batch['cand_labels'].view(attach_pred.shape).float())
            loss_list[1] = comb_loss.item()
        else:
            comb_loss = 0

        # focal prediction
        focal_protein_pred = self.focal_mlp(h_ctx_protein)
        focal_loss_protein = self.focal_loss(focal_protein_pred.reshape(-1), batch['protein_contact'].float())

        if h_ctx_ligand.numel() == 0:
            focal_loss = focal_loss_protein
        else:
            focal_ligand_pred = self.focal_mlp(h_ctx_ligand)
            focal_loss_ligand = self.focal_loss(focal_ligand_pred.reshape(-1), batch['ligand_frontier'].float())
            focal_loss = focal_loss_ligand + focal_loss_protein

        loss_list[2] = focal_loss.item()
        
        loss = pred_loss + comb_loss + focal_loss

        return loss, loss_list, residue_emb.detach()



    def get_probability(self, protein_pos, protein_atom_feature, ligand_pos, ligand_atom_feature, 
                    batch_protein, batch_ligand, batch):
        h_protein = self.protein_atom_emb(protein_atom_feature)
        h_ligand = self.ligand_atom_emb(ligand_atom_feature)
        
        h_ctx, pos_ctx, batch_ctx, mask_protein = compose_context_stable(h_protein=h_protein, h_ligand=h_ligand,
                                                                    pos_protein=protein_pos, pos_ligand=ligand_pos,
                                                                    batch_protein=batch_protein,
                                                                    batch_ligand=batch_ligand)
        h_ctx, h_residue = self.encoder(node_attr=h_ctx, pos=pos_ctx, batch=batch_ctx, X=batch['residue_pos'],
                                        S_id=batch['res_idx'], R=batch['amino_acid'], residue_batch=batch['amino_acid_batch'], atom2residue=batch['atom2residue'], mask=mask_protein)  # (N_p+N_l, H)
        h_ctx_ligand = h_ctx[~mask_protein]
        h_ctx_protein = h_ctx[mask_protein]

        # focal probability
        if h_ctx_ligand.numel() == 0:  
            focal_protein_pred = self.focal_mlp(h_ctx_protein)
            focus_protein_score = torch.sigmoid(focal_protein_pred)
            focus_protein_correspond_score = focus_protein_score[batch['protein_contact']]  
            focus_protein_probability = torch.sum(focus_protein_correspond_score).item() 
            focus_probability = focus_protein_probability
        else:
            focal_ligand_pred = self.focal_mlp(h_ctx_ligand)
            focus_ligand_score = torch.sigmoid(focal_ligand_pred)
            focus_ligand_correspond_score = focus_ligand_score[batch['ligand_frontier']]
            focus_ligand_probability = torch.sum(focus_ligand_correspond_score).item()  
            focus_probability = focus_ligand_probability
            
        # motif probability
        h_ctx_focal = h_ctx[batch['current_atoms']]
        pos_ctx_focal = pos_ctx[batch['current_atoms']]
        node_hiddens = scatter_add(h_ctx_focal, dim=0, index=batch['current_atoms_batch'])
        center_pos = scatter_mean(pos_ctx_focal, dim=0, index=batch['current_atoms_batch'])
        residue_pos = batch['residue_pos'][:, 1, :]
        residue_index = torch.where(torch.norm(residue_pos - center_pos[batch['amino_acid_batch']], dim=1)<6)
        residue_emb = torch.zeros_like(node_hiddens)
        added = scatter_add(h_residue[residue_index], dim=0, index=batch['amino_acid_batch'][residue_index])
        residue_emb[:added.shape[0]] = added
        motif_hiddens = self.embedding(batch['current_wid'])
        pred_vecs = torch.cat([node_hiddens, motif_hiddens, residue_emb], dim=-1)
        pred_scores = torch.matmul(self.motif_mlp(pred_vecs), self.embedding.weight.transpose(1, 0))
        motif_scores = F.softmax(pred_scores.squeeze(), dim=0)
        motif_probability = motif_scores[batch['next_wid'].squeeze() == 1].item()
        
        
        # attachment probability
        if len(batch['cand_labels']) == 0: 
            attach_probability = 1
        else:  
            cand_mols = batch['cand_mols']
            cand_emb = self.comb_head(cand_mols.x, cand_mols.edge_index, cand_mols.edge_attr, cand_mols.batch)
            #attach_pred = self.attach_mlp(torch.cat([cand_emb, residue_emb[batch['cand_mols_batch']]], dim=-1))
            attach_pred = self.attach_mlp(cand_emb)
            attach_score = torch.sigmoid(attach_pred)
            attach_probability = attach_score[batch['cand_labels']==1].item()
        
        return focus_probability, motif_probability, attach_probability
