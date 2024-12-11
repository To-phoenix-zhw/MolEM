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


class OrderingGenerator(Module):

    def __init__(self, config, protein_atom_feature_dim, ligand_atom_feature_dim, vocab, device):
        super().__init__()
        self.config = config
        self.vocab = vocab
        self.device = device
        self.protein_atom_emb = Linear(protein_atom_feature_dim, config.hidden_channels)
        self.ligand_atom_emb = Linear(ligand_atom_feature_dim+1, config.hidden_channels) 
        self.embedding = nn.Embedding(vocab.size() + 1, config.hidden_channels)
        self.encoder = get_encoder(config.encoder)

        self.ordering_mlp = MLP(in_dim=config.hidden_channels * 5, out_dim=1, num_layers=1)
        self.ordering_loss = nn.CrossEntropyLoss()


    def forward(self, batch_size, protein_pos, protein_atom_feature, ligand_pos, ligand_atom_feature, 
                    batch_protein, batch_ligand, batch):
        h_protein = self.protein_atom_emb(protein_atom_feature) 
        h_ligand = self.ligand_atom_emb(ligand_atom_feature)

        # Encode for ordering prediction
        h_ctx, pos_ctx, batch_ctx, mask_protein = compose_context_stable(h_protein=h_protein, h_ligand=h_ligand,
                                                                        pos_protein=protein_pos, pos_ligand=ligand_pos,
                                                                        batch_protein=batch_protein,
                                                                        batch_ligand=batch_ligand)
        h_ctx, h_residue = self.encoder(node_attr=h_ctx, pos=pos_ctx, batch=batch_ctx, X=batch['residue_pos'],
                                        S_id=batch['res_idx'], R=batch['amino_acid'], residue_batch=batch['amino_acid_batch'], atom2residue=batch['atom2residue'], mask=mask_protein)  # (N_p+N_l, H)
        h_ctx_ligand = h_ctx[~mask_protein]
        h_ctx_protein = h_ctx[mask_protein]
        
        # complete molecule
        ligand_representation = scatter_add(h_ctx_ligand, dim=0, index=batch_ligand)  
        protein_representation = scatter_add(h_ctx_protein, dim=0, index=batch_protein)  
        residue_representation = scatter_add(h_residue, dim=0, index=batch['amino_acid_batch']) 
        motif_hiddens = self.embedding(batch['ligand_wids']) 

        data_batch_inputs = []
        for d in range(batch_size): 
            inputs = []
            cur_protein_representation = protein_representation[d]
            cur_residue_representation = residue_representation[d]
            cur_ligand_representation = ligand_representation[d]
            cur_motif_hiddens = motif_hiddens[batch['ligand_wids_batch'] == d]
            for nid in range(sum(batch['ligand_wids_batch'] == d)): 
                cur_motif_atom_index = batch['motif_atom_index'][batch['motif_atoms_batch']==d]
                cur_motif_atoms = batch['motif_atoms'][batch['motif_atoms_batch']==d]
                token_atom_index = cur_motif_atoms[cur_motif_atom_index==nid] 
                token_atoms_representation = h_ctx_ligand[batch_ligand==d][token_atom_index].sum(dim=0) 
                # print(token_atoms_representation)
                pred_vecs = torch.cat([cur_protein_representation, cur_residue_representation, cur_ligand_representation, cur_motif_hiddens[nid], token_atoms_representation], dim=-1) 
                inputs.append(pred_vecs)
            data_batch_inputs.append(torch.stack(inputs))

        # ordering_prediction
        selected_pools = []
        for d in range(batch_size):
            data_inputs = data_batch_inputs[d]  # [18, 1280]
            pred_logits = self.ordering_mlp(data_inputs)  # [18, 1]
            pred_scores = F.softmax(pred_logits, dim=0)
            select_pool = torch.topk(pred_scores, pred_scores.shape[0], dim=0)  
            # print(pred_scores)
            # print(select_pool)
            selected_pools.append(select_pool)
            
        return selected_pools


    def get_loss(self, protein_pos, protein_atom_feature, ligand_pos, ligand_atom_feature, 
                batch_protein, batch_ligand, batch):
        self.device = protein_pos.device
        h_protein = self.protein_atom_emb(protein_atom_feature)
        h_ligand = self.ligand_atom_emb(ligand_atom_feature)


        # Encode for ordering prediction
        h_ctx, pos_ctx, batch_ctx, mask_protein = compose_context_stable(h_protein=h_protein, h_ligand=h_ligand,
                                                                         pos_protein=protein_pos, pos_ligand=ligand_pos,
                                                                         batch_protein=batch_protein,
                                                                         batch_ligand=batch_ligand)

        h_ctx, h_residue = self.encoder(node_attr=h_ctx, pos=pos_ctx, batch=batch_ctx, X=batch['residue_pos'],
                                        S_id=batch['res_idx'], R=batch['amino_acid'], residue_batch=batch['amino_acid_batch'], atom2residue=batch['atom2residue'], mask=mask_protein)  # (N_p+N_l, H)
        h_ctx_ligand = h_ctx[~mask_protein]
        h_ctx_protein = h_ctx[mask_protein]

        # complete molecule
        ligand_representation = scatter_add(h_ctx_ligand, dim=0, index=batch_ligand)  
        protein_representation = scatter_add(h_ctx_protein, dim=0, index=batch_protein) 
        residue_representation = scatter_add(h_residue, dim=0, index=batch['amino_acid_batch']) 
        motif_hiddens = self.embedding(batch['ligand_wids'])  

        data_batch_inputs = []
        for d in range(batch['ordering_label'].shape[0]): 
            inputs = []
            cur_protein_representation = protein_representation[d]
            cur_residue_representation = residue_representation[d]
            cur_ligand_representation = ligand_representation[d]
            cur_motif_hiddens = motif_hiddens[batch['ligand_wids_batch'] == d]
            for nid in range(sum(batch['ligand_wids_batch'] == d)):
                cur_motif_atom_index = batch['motif_atom_index'][batch['motif_atoms_batch']==d]
                cur_motif_atoms = batch['motif_atoms'][batch['motif_atoms_batch']==d]
                token_atom_index = cur_motif_atoms[cur_motif_atom_index==nid]
                token_atoms_representation = h_ctx_ligand[batch_ligand==d][token_atom_index].sum(dim=0) 
                # print(token_atoms_representation)
                pred_vecs = torch.cat([cur_protein_representation, cur_residue_representation, cur_ligand_representation, cur_motif_hiddens[nid], token_atoms_representation], dim=-1) 
                inputs.append(pred_vecs)
            data_batch_inputs.append(torch.stack(inputs))


        # ordering_prediction
        loss_total = 0
        acc_cnt = 0
        sample_cnt = batch['ordering_label'].shape[0]
        for d in range(sample_cnt):
            data_inputs = data_batch_inputs[d]  # [18, 1280]
            pred_logits = self.ordering_mlp(data_inputs)  # [18, 1]
            pred_label = torch.argmax(pred_logits, dim=0).item()
            true_label = batch['ordering_label'][d].item()
            # [1, 18], [1]
            acc_cnt += pred_label == true_label
#             print("pred_label", pred_label)
#             print("true_label", true_label)
#             print("pred_label == true_label", pred_label == true_label)
            loss_total += self.ordering_loss(pred_logits.transpose(0, 1), batch['ordering_label'][d].unsqueeze(0))
        loss = loss_total / batch['ordering_label'].shape[0]
#         print("acc_cnt", acc_cnt)

        return loss, acc_cnt, sample_cnt


