import copy
import torch
import numpy as np
import networkx as nx
from torch_geometric.data import Data, Batch
# from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset

FOLLOW_BATCH = ['protein_element', 'ligand_context_element', 'pos_real', 'pos_fake']


class ProteinLigandData(object):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def from_protein_ligand_dicts(protein_dict=None, ligand_dict=None, **kwargs):
        instance = ProteinLigandData(**kwargs)

        if protein_dict is not None:
            for key, item in protein_dict.items():
                instance['protein_' + key] = item

        if ligand_dict is not None:
            for key, item in ligand_dict.items():
                if key == 'moltree':
                    instance['moltree'] = item
                else:
                    instance['ligand_' + key] = item

        # instance['ligand_nbh_list'] = {i.item():[j.item() for k, j in enumerate(instance.ligand_bond_index[1]) if instance.ligand_bond_index[0, k].item() == i] for i in instance.ligand_bond_index[0]}
        return instance


def batch_from_data_list(data_list):
    return Batch.from_data_list(data_list, follow_batch=['ligand_element', 'protein_element'])


def torchify_dict(data):
    output = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            output[k] = torch.from_numpy(v)
        else:
            output[k] = v
    return output


def collate_mols(mol_dicts):
    data_batch = {}
    batch_size = len(mol_dicts)
    for mol in mol_dicts:
        if mol is None:
            return None
    for key in ['protein_pos', 'protein_atom_feature', 'ligand_context_pos', 'ligand_context_feature_full',
                'ligand_frontier', 'num_atoms', 'next_wid', 'current_wid', 'current_atoms', 'cand_labels',
                'protein_contact', 'res_idx', 'amino_acid', 'interaction']:
        data_batch[key] = torch.cat([mol_dict[key] for mol_dict in mol_dicts], dim=0)
    # residue pos
    data_batch['residue_pos'] = \
        torch.cat([torch.cat([mol_dict[key] for mol_dict in mol_dicts], dim=0).unsqueeze(0) for key in ['pos_N', 'pos_CA', 'pos_C', 'pos_O']], dim=0).permute(1,0,2)
    repeats = torch.cat([mol_dict['protein_atom2residue'].bincount() for mol_dict in mol_dicts])
    data_batch['atom2residue'] = torch.repeat_interleave(torch.arange(len(repeats)), repeats)

    # follow batch
    for key in ['protein_element', 'ligand_context_element', 'current_atoms', 'amino_acid', 'cand_mols']:
        repeats = torch.tensor([len(mol_dict[key]) for mol_dict in mol_dicts])
        data_batch[key + '_batch'] = torch.repeat_interleave(torch.arange(batch_size), repeats)


    offsets1 = torch.cat([torch.tensor([0]), torch.cumsum(data_batch['num_atoms'], dim=0)])[:-1]
    data_batch['current_atoms'] += torch.repeat_interleave(offsets1, data_batch['current_atoms_batch'].bincount())
    # cand mols: torch geometric Data
    cand_mol_list = []
    for data in mol_dicts:
        if len(data['cand_labels']) > 0:
            cand_mol_list.extend(data['cand_mols'])
    if len(cand_mol_list) > 0:
        data_batch['cand_mols'] = Batch.from_data_list(cand_mol_list)
    return data_batch


def collate_mols_simple(mol_dicts):
    data_batch = {}
    batch_size = len(mol_dicts)
    for key in ['protein_pos', 'protein_atom_feature', 'ligand_pos', 'ligand_atom_feature_full', 'vina']:
        data_batch[key] = torch.cat([mol_dict[key] for mol_dict in mol_dicts], dim=0).float()

    # follow batch
    for key in ['protein_element', 'ligand_element']:
        repeats = torch.tensor([len(mol_dict[key]) for mol_dict in mol_dicts])
        data_batch[key + '_batch'] = torch.repeat_interleave(torch.arange(batch_size), repeats)

    return data_batch



def collate_mols_base(mol_dicts):
    data_batch = {}
    batch_size = len(mol_dicts)
    for key in ['protein_pos', 'protein_atom_feature', 'ligand_context_pos', 'ligand_context_feature_full',
                'ligand_frontier', 'num_atoms', 'next_wid', 'current_wid', 'current_atoms', 'cand_labels',
                'ligand_pos_torsion', 'ligand_feature_torsion', 'true_sin', 'true_cos', 'true_three_hop',
                'dihedral_mask', 'protein_contact', 'true_dm', 'res_idx', 'amino_acid', 'interaction']:
        data_batch[key] = torch.cat([mol_dict[key] for mol_dict in mol_dicts], dim=0)
    # residue pos
    data_batch['residue_pos'] = \
        torch.cat([torch.cat([mol_dict[key] for mol_dict in mol_dicts], dim=0).unsqueeze(0) for key in ['pos_N', 'pos_CA', 'pos_C', 'pos_O']], dim=0).permute(1,0,2)
    repeats = torch.cat([mol_dict['protein_atom2residue'].bincount() for mol_dict in mol_dicts])
    data_batch['atom2residue'] = torch.repeat_interleave(torch.arange(len(repeats)), repeats)

    # unsqueeze dim0
    for key in ['xn_pos', 'yn_pos', 'ligand_torsion_xy_index', 'y_pos']:
        cat_list = [mol_dict[key].unsqueeze(0) for mol_dict in mol_dicts if len(mol_dict[key]) > 0]
        if len(cat_list) > 0:
            data_batch[key] = torch.cat(cat_list, dim=0)
        else:
            data_batch[key] = torch.tensor([])
    # follow batch
    for key in ['protein_element', 'ligand_context_element', 'current_atoms', 'amino_acid', 'cand_mols']:
        repeats = torch.tensor([len(mol_dict[key]) for mol_dict in mol_dicts])
        data_batch[key + '_batch'] = torch.repeat_interleave(torch.arange(batch_size), repeats)
    for key in ['ligand_element_torsion']:
        repeats = torch.tensor([len(mol_dict[key]) for mol_dict in mol_dicts if len(mol_dict[key]) > 0])
        if len(repeats) > 0:
            data_batch[key + '_batch'] = torch.repeat_interleave(torch.arange(len(repeats)), repeats)
        else:
            data_batch[key + '_batch'] = torch.tensor([])

    # distance matrix prediction
    p_idx, q_idx = torch.cartesian_prod(torch.arange(4), torch.arange(2)).chunk(2, dim=-1)
    p_idx, q_idx = p_idx.squeeze(-1), q_idx.squeeze(-1)
    protein_offsets = torch.cumsum(data_batch['protein_element_batch'].bincount(), dim=0)
    ligand_offsets = torch.cumsum(data_batch['ligand_context_element_batch'].bincount(), dim=0)
    protein_offsets, ligand_offsets = torch.cat([torch.tensor([0]), protein_offsets]), torch.cat([torch.tensor([0]), ligand_offsets])
    ligand_idx, protein_idx = [], []
    for i, mol_dict in enumerate(mol_dicts):
        if len(mol_dict['true_dm']) > 0:
            protein_idx.append(mol_dict['dm_protein_idx'][p_idx] + protein_offsets[i])
            ligand_idx.append(mol_dict['dm_ligand_idx'][q_idx] + ligand_offsets[i])
    if len(ligand_idx) > 0:
        data_batch['dm_ligand_idx'], data_batch['dm_protein_idx'] = torch.cat(ligand_idx), torch.cat(protein_idx)

    # structure refinement (alpha carbon - ligand atom)
    sr_ligand_idx, sr_protein_idx = [], []
    for i, mol_dict in enumerate(mol_dicts):
        if len(mol_dict['true_dm']) > 0:
            ligand_atom_index = torch.arange(len(mol_dict['ligand_context_pos']))
            p_idx, q_idx = torch.cartesian_prod(torch.arange(len(mol_dict['ligand_context_pos'])), torch.arange(len(mol_dict['protein_alpha_carbon_index']))).chunk(2, dim=-1)
            p_idx, q_idx = p_idx.squeeze(-1), q_idx.squeeze(-1)
            sr_ligand_idx.append(ligand_atom_index[p_idx] + ligand_offsets[i])
            sr_protein_idx.append(mol_dict['protein_alpha_carbon_index'][q_idx] + protein_offsets[i])
    if len(ligand_idx) > 0:
        data_batch['sr_ligand_idx'], data_batch['sr_protein_idx'] = torch.cat(sr_ligand_idx).long(), torch.cat(sr_protein_idx).long()

    # structure refinement (ligand atom - ligand atom)
    sr_ligand_idx0, sr_ligand_idx1 = [], []
    for i, mol_dict in enumerate(mol_dicts):
        if len(mol_dict['true_dm']) > 0:
            ligand_atom_index = torch.arange(len(mol_dict['ligand_context_pos']))
            p_idx, q_idx = torch.cartesian_prod(torch.arange(len(mol_dict['ligand_context_pos'])), torch.arange(len(mol_dict['ligand_context_pos']))).chunk(2, dim=-1)
            p_idx, q_idx = p_idx.squeeze(-1), q_idx.squeeze(-1)
            sr_ligand_idx0.append(ligand_atom_index[p_idx] + ligand_offsets[i])
            sr_ligand_idx1.append(ligand_atom_index[q_idx] + ligand_offsets[i])
    if len(ligand_idx) > 0:
        data_batch['sr_ligand_idx0'], data_batch['sr_ligand_idx1'] = torch.cat(sr_ligand_idx0).long(), torch.cat(sr_ligand_idx1).long()
    # index
    if len(data_batch['y_pos']) > 0:
        repeats = torch.tensor([len(mol_dict['ligand_element_torsion']) for mol_dict in mol_dicts if len(mol_dict['ligand_element_torsion']) > 0])
        offsets = torch.cat([torch.tensor([0]), torch.cumsum(repeats, dim=0)])[:-1]
        data_batch['ligand_torsion_xy_index'] += offsets.unsqueeze(1)

    offsets1 = torch.cat([torch.tensor([0]), torch.cumsum(data_batch['num_atoms'], dim=0)])[:-1]
    data_batch['current_atoms'] += torch.repeat_interleave(offsets1, data_batch['current_atoms_batch'].bincount())
    # cand mols: torch geometric Data
    cand_mol_list = []
    for data in mol_dicts:
        if len(data['cand_labels']) > 0:
            cand_mol_list.extend(data['cand_mols'])
    if len(cand_mol_list) > 0:
        data_batch['cand_mols'] = Batch.from_data_list(cand_mol_list)
    return data_batch



def collate_mols_vina(mol_dicts):
    data_batch = {}
    batch_size = len(mol_dicts)
    for mol in mol_dicts:
        if mol is None:
            return None
    for key in ['protein_pos', 'protein_atom_feature', 'ligand_context_pos', 'ligand_context_feature_full',
                'ligand_frontier', 'num_atoms', 'next_wid', 'current_wid', 'current_atoms', 'cand_labels',
                'protein_contact', 'res_idx', 'amino_acid', 'interaction']:
        data_batch[key] = torch.cat([mol_dict[key] for mol_dict in mol_dicts], dim=0)
    # residue pos
    data_batch['residue_pos'] = \
        torch.cat([torch.cat([mol_dict[key] for mol_dict in mol_dicts], dim=0).unsqueeze(0) for key in ['pos_N', 'pos_CA', 'pos_C', 'pos_O']], dim=0).permute(1,0,2)
    repeats = torch.cat([mol_dict['protein_atom2residue'].bincount() for mol_dict in mol_dicts])
    data_batch['atom2residue'] = torch.repeat_interleave(torch.arange(len(repeats)), repeats)

    # follow batch
    for key in ['protein_element', 'ligand_context_element', 'current_atoms', 'amino_acid', 'cand_mols']:
        repeats = torch.tensor([len(mol_dict[key]) for mol_dict in mol_dicts])
        data_batch[key + '_batch'] = torch.repeat_interleave(torch.arange(batch_size), repeats)


    offsets1 = torch.cat([torch.tensor([0]), torch.cumsum(data_batch['num_atoms'], dim=0)])[:-1]
    data_batch['current_atoms'] += torch.repeat_interleave(offsets1, data_batch['current_atoms_batch'].bincount())
    # cand mols: torch geometric Data
    cand_mol_list = []
    for data in mol_dicts:
        if len(data['cand_labels']) > 0:
            cand_mol_list.extend(data['cand_mols'])
    if len(cand_mol_list) > 0:
        data_batch['cand_mols'] = Batch.from_data_list(cand_mol_list)
    return data_batch


def collate_mols_learning(mol_dicts, vocab_size):
    data_batch = {}
    batch_size = len(mol_dicts)  

    for key in ['protein_pos', 'protein_atom_feature', 'ligand_context_pos', 'ligand_context_pos_crystal', 'ligand_context_feature_full',
                'ligand_frontier', 'num_atoms', 'current_wid', 'current_atoms', 'protein_contact', 'res_idx', 'amino_acid']:
        data_batch[key] = torch.cat([mol_dict[key] for mol_dict in mol_dicts], dim=0)

    # next wid
    next_wid_list = []
    for data in mol_dicts:
        labels = data['next_wid']
        labels = labels.unsqueeze(0)
        target = torch.zeros(labels.size(0), vocab_size+1).scatter_(1, labels, 1.).reshape(-1)  
        next_wid_list.append(target)
    data_batch['next_wid'] = torch.stack(next_wid_list, dim=0)

    # residue pos
    data_batch['residue_pos'] = \
        torch.cat([torch.cat([mol_dict[key] for mol_dict in mol_dicts], dim=0).unsqueeze(0) for key in ['pos_N', 'pos_CA', 'pos_C', 'pos_O']], dim=0).permute(1,0,2)
    repeats = torch.cat([mol_dict['protein_atom2residue'].bincount() for mol_dict in mol_dicts])
    data_batch['atom2residue'] = torch.repeat_interleave(torch.arange(len(repeats)), repeats)

    # follow batch
    for key in ['protein_element', 'ligand_context_element', 'current_atoms', 'amino_acid', 'cand_mols']:
        repeats = torch.tensor([len(mol_dict[key]) for mol_dict in mol_dicts])
        data_batch[key + '_batch'] = torch.repeat_interleave(torch.arange(batch_size), repeats)
    
    offsets1 = torch.cat([torch.tensor([0]), torch.cumsum(data_batch['num_atoms'], dim=0)])[:-1]
    data_batch['current_atoms'] += torch.repeat_interleave(offsets1, data_batch['current_atoms_batch'].bincount())

    # cand labels
    cand_label_list = []
    for data in mol_dicts:
        if len(data['cand_labels']) > 0:
            cand_label_list.extend(data['cand_labels'])
    data_batch['cand_labels'] = torch.tensor(cand_label_list)

    # cand mols: torch geometric Data
    cand_mol_list = []
    for data in mol_dicts:
        if len(data['cand_labels']) > 0:
            cand_mol_list.extend(data['cand_mols'])

    if len(cand_mol_list) > 0:
        data_batch['cand_mols'] = Batch.from_data_list(cand_mol_list)
    
    return data_batch




def collate_mols_ordering(mol_dicts, vocab_size):
    data_batch = {}
    batch_size = len(mol_dicts)   

    for key in ['protein_pos', 'protein_atom_feature', 'ligand_context_pos', 'ligand_pos', 'ligand_atom_feature_full', 'ordering_label', 
                'num_atoms', 'protein_contact', 'res_idx', 'amino_acid']:
        data_batch[key] = torch.cat([mol_dict[key] for mol_dict in mol_dicts], dim=0)

    # residue pos
    data_batch['residue_pos'] = \
        torch.cat([torch.cat([mol_dict[key] for mol_dict in mol_dicts], dim=0).unsqueeze(0) for key in ['pos_N', 'pos_CA', 'pos_C', 'pos_O']], dim=0).permute(1,0,2)
    repeats = torch.cat([mol_dict['protein_atom2residue'].bincount() for mol_dict in mol_dicts])
    data_batch['atom2residue'] = torch.repeat_interleave(torch.arange(len(repeats)), repeats)


    # follow batch
    for key in ['protein_element', 'ligand_element', 'amino_acid']:
        repeats = torch.tensor([len(mol_dict[key]) for mol_dict in mol_dicts])
        data_batch[key + '_batch'] = torch.repeat_interleave(torch.arange(batch_size), repeats)

    # orderings, chosen_motifs, ordering, moltree_motifs
    data_batch['ordering_idx'] = torch.cat([torch.tensor([mol_dict['ordering']]) for mol_dict in mol_dicts], dim=0)

    motif_atoms_list = []
    motif_atom_index_list = []
    motif_atoms_batch_repeats = []
    for mol_dict in mol_dicts:
        repeats = torch.tensor([len(node.clique) for node in mol_dict['moltree'].nodes])
        motif_atoms = torch.cat([torch.tensor(node.clique) for node in mol_dict['moltree'].nodes], dim=0)
        motif_atom_index = torch.repeat_interleave(torch.arange(len(mol_dict['moltree'].nodes)), repeats)
        motif_atoms_batch_repeats.append(len(motif_atom_index))
        motif_atoms_list.append(motif_atoms)
        motif_atom_index_list.append(motif_atom_index)
    motif_atoms_batch_repeats = torch.tensor(motif_atoms_batch_repeats)
    data_batch['motif_atoms_batch'] = torch.repeat_interleave(torch.arange(batch_size), motif_atoms_batch_repeats)  
    data_batch['motif_atom_index'] = torch.cat([motif_atom_index for motif_atom_index in motif_atom_index_list], dim=0)   
    data_batch['motif_atoms'] = torch.cat([motif_atoms for motif_atoms in motif_atoms_list], dim=0) 

    # wids
    repeats = torch.tensor([len(mol_dict['moltree'].nodes) for mol_dict in mol_dicts]) 
    data_batch['ligand_wids_batch'] = torch.repeat_interleave(torch.arange(batch_size), repeats) 

    wids_list = []
    for mol_dict in mol_dicts:
        wids = torch.tensor([mol_dict['moltree'].nodes[0].wid])  
        for k in range(1, len(mol_dict['moltree'].nodes)):  
            wids = torch.cat([wids, torch.tensor([mol_dict['moltree'].nodes[k].wid])], dim=0)  
        wids_list.append(wids)
    data_batch['ligand_wids'] = torch.cat([wids for wids in wids_list], dim=0) 

    # graph 
    data_graphs = []
    for mol_dict in mol_dicts:
        edges = []
        for node in mol_dict['moltree'].nodes:
            for n, nei in enumerate(node.neighbors):
                if node.nid < nei.nid:
                    edges.append((node.nid, nei.nid))
                else:
                    assert (nei.nid, node.nid) in edges      
        nodes_num = len(mol_dict['moltree'].nodes)
        pointList = list(range(nodes_num))
        linkList = edges
        G = nx.Graph()
        for node in pointList:
            G.add_node(node)
        for link in linkList:
            G.add_edge(link[0], link[1])
        data_graphs.append(G)
    data_batch['ligand_graphs'] = data_graphs


    return data_batch



def collate_mols_ordering0(mol_dicts, vocab_size):
    data_batch = {}
    batch_size = len(mol_dicts)  

    for key in ['protein_pos', 'protein_atom_feature', 'ligand_context_pos', 'ligand_pos', 'ligand_atom_feature_full', 'ordering_label', 
                'num_atoms', 'protein_contact', 'res_idx', 'amino_acid']:
        data_batch[key] = torch.cat([mol_dict[key] for mol_dict in mol_dicts], dim=0)

    # residue pos
    data_batch['residue_pos'] = \
        torch.cat([torch.cat([mol_dict[key] for mol_dict in mol_dicts], dim=0).unsqueeze(0) for key in ['pos_N', 'pos_CA', 'pos_C', 'pos_O']], dim=0).permute(1,0,2)
    repeats = torch.cat([mol_dict['protein_atom2residue'].bincount() for mol_dict in mol_dicts])
    data_batch['atom2residue'] = torch.repeat_interleave(torch.arange(len(repeats)), repeats)


    # follow batch
    for key in ['protein_element', 'ligand_element', 'amino_acid']:
        repeats = torch.tensor([len(mol_dict[key]) for mol_dict in mol_dicts])
        data_batch[key + '_batch'] = torch.repeat_interleave(torch.arange(batch_size), repeats)


    return data_batch