import sys
sys.path.append("..")
import copy
import os
import random
import torch
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from torch_geometric.transforms import Compose
from torch_geometric.nn.pool import knn_graph
from torch_geometric.utils.subgraph import subgraph
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.data import Data, Batch
from torch_scatter import scatter_add
import rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from meeko import MoleculePreparation
from meeko import PDBQTMolecule
from meeko import RDKitMolCreate
import networkx as nx
from itertools import combinations

from .data import ProteinLigandData
from .protein_ligand import ATOM_FAMILIES
from .chemutils import enumerate_assemble, list_filter, rand_rotate, get_clique_mol_simple
from .dihedral_utils import batch_dihedrals
from .misc import mksure_path
from .docking import *   


# jupyter
# from data import ProteinLigandData
# from protein_ligand import ATOM_FAMILIES
# from chemutils import enumerate_assemble, list_filter, rand_rotate
# from dihedral_utils import batch_dihedrals


# allowable node and edge features
allowable_features = {
    'possible_atomic_num_list': list(range(1, 119)),
    'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'possible_chirality_list': [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],
    'possible_hybridization_list': [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED
    ],
    'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6],
    'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'possible_bonds': [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'possible_bond_dirs': [  # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ]
}


def mol_to_graph_data_obj_simple(mol):
    """
    Converts rdkit mol object to graph Data object required by the pytorch
    geometric package. NB: Uses simplified atom and bond features, and represent
    as indices
    :param mol: rdkit mol object
    :return: graph data object with the attributes: x, edge_index, edge_attr
    """
    # atoms
    num_atom_features = 2  # atom type,  chirality tag
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = [allowable_features['possible_atomic_num_list'].index(
            atom.GetAtomicNum())] + [allowable_features[
                                         'possible_chirality_list'].index(atom.GetChiralTag())]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    num_bond_features = 2  # bond type, bond direction
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = [allowable_features['possible_bonds'].index(
                bond.GetBondType())] + [allowable_features[
                'possible_bond_dirs'].index(
                bond.GetBondDir())]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list),
                                 dtype=torch.long)
    else:  # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data




class RefineData(object):
    def __init__(self):
        super().__init__()

    def __call__(self, data):
        # delete H atom of pocket
        protein_element = data.protein_element
        is_H_protein = (protein_element == 1)
        if torch.sum(is_H_protein) > 0:
            not_H_protein = ~is_H_protein
            data.protein_atom_name = list(compress(data.protein_atom_name, not_H_protein))
            data.protein_atom_to_aa_type = data.protein_atom_to_aa_type[not_H_protein]
            data.protein_element = data.protein_element[not_H_protein]
            data.protein_is_backbone = data.protein_is_backbone[not_H_protein]
            data.protein_pos = data.protein_pos[not_H_protein]
        # delete H atom of ligand
        ligand_element = data.ligand_element
        is_H_ligand = (ligand_element == 1)
        if torch.sum(is_H_ligand) > 0:
            not_H_ligand = ~is_H_ligand
            data.ligand_atom_feature = data.ligand_atom_feature[not_H_ligand]
            data.ligand_element = data.ligand_element[not_H_ligand]
            data.ligand_pos = data.ligand_pos[not_H_ligand]
            # nbh
            index_atom_H = torch.nonzero(is_H_ligand)[:, 0]
            index_changer = -np.ones(len(not_H_ligand), dtype=np.int64)
            index_changer[not_H_ligand] = np.arange(torch.sum(not_H_ligand))
            new_nbh_list = [value for ind_this, value in zip(not_H_ligand, data.ligand_nbh_list.values()) if ind_this]
            data.ligand_nbh_list = {i: [index_changer[node] for node in neigh if node not in index_atom_H] for i, neigh
                                    in enumerate(new_nbh_list)}
            # bond
            ind_bond_with_H = np.array([(bond_i in index_atom_H) | (bond_j in index_atom_H) for bond_i, bond_j in
                                        zip(*data.ligand_bond_index)])
            ind_bond_without_H = ~ind_bond_with_H
            old_ligand_bond_index = data.ligand_bond_index[:, ind_bond_without_H]
            data.ligand_bond_index = torch.tensor(index_changer)[old_ligand_bond_index]
            data.ligand_bond_type = data.ligand_bond_type[ind_bond_without_H]

        return data


class FocalBuilder(object):
    def __init__(self, close_threshold=0.8, max_bond_length=2.4):
        self.close_threshold = close_threshold
        self.max_bond_length = max_bond_length
        super().__init__()

    def __call__(self, data: ProteinLigandData):
        # ligand_context_pos = data.ligand_context_pos
        # ligand_pos = data.ligand_pos
        ligand_masked_pos = data.ligand_masked_pos
        protein_pos = data.protein_pos
        context_idx = data.context_idx
        masked_idx = data.masked_idx
        old_bond_index = data.ligand_bond_index
        # old_bond_types = data.ligand_bond_type  # type: 0, 1, 2
        has_unmask_atoms = context_idx.nelement() > 0
        if has_unmask_atoms:
            # # get bridge bond index (mask-context bond)
            ind_edge_index_candidate = [
                (context_node in context_idx) and (mask_node in masked_idx)
                for mask_node, context_node in zip(*old_bond_index)
            ]  # the mask-context order is right
            bridge_bond_index = old_bond_index[:, ind_edge_index_candidate]
            # candidate_bond_types = old_bond_types[idx_edge_index_candidate]
            idx_generated_in_whole_ligand = bridge_bond_index[0]
            idx_focal_in_whole_ligand = bridge_bond_index[1]

            index_changer_masked = torch.zeros(masked_idx.max() + 1, dtype=torch.int64)
            index_changer_masked[masked_idx] = torch.arange(len(masked_idx))
            idx_generated_in_ligand_masked = index_changer_masked[idx_generated_in_whole_ligand]
            pos_generate = ligand_masked_pos[idx_generated_in_ligand_masked]

            data.idx_generated_in_ligand_masked = idx_generated_in_ligand_masked
            data.pos_generate = pos_generate

            index_changer_context = torch.zeros(context_idx.max() + 1, dtype=torch.int64)
            index_changer_context[context_idx] = torch.arange(len(context_idx))
            idx_focal_in_ligand_context = index_changer_context[idx_focal_in_whole_ligand]
            idx_focal_in_compose = idx_focal_in_ligand_context  # if ligand_context was not before protein in the compose, this was not correct
            data.idx_focal_in_compose = idx_focal_in_compose

            data.idx_protein_all_mask = torch.empty(0, dtype=torch.long)  # no use if has context
            data.y_protein_frontier = torch.empty(0, dtype=torch.bool)  # no use if has context

        else:  # # the initial atom. surface atoms between ligand and protein
            assign_index = radius(x=ligand_masked_pos, y=protein_pos, r=4., num_workers=16)
            if assign_index.size(1) == 0:
                dist = torch.norm(data.protein_pos.unsqueeze(1) - data.ligand_masked_pos.unsqueeze(0), p=2, dim=-1)
                assign_index = torch.nonzero(dist <= torch.min(dist) + 1e-5)[0:1].transpose(0, 1)
            idx_focal_in_protein = assign_index[0]
            data.idx_focal_in_compose = idx_focal_in_protein  # no ligand context, so all composes are protein atoms
            data.pos_generate = ligand_masked_pos[assign_index[1]]
            data.idx_generated_in_ligand_masked = torch.unique(assign_index[1])  # for real of the contractive transform

            data.idx_protein_all_mask = data.idx_protein_in_compose  # for input of initial frontier prediction
            y_protein_frontier = torch.zeros_like(data.idx_protein_all_mask,
                                                  dtype=torch.bool)  # for label of initial frontier prediction
            y_protein_frontier[torch.unique(idx_focal_in_protein)] = True
            data.y_protein_frontier = y_protein_frontier

        # generate not positions: around pos_focal ( with `max_bond_length` distance) but not close to true generated within `close_threshold`
        # pos_focal = ligand_context_pos[idx_focal_in_ligand_context]
        # pos_notgenerate = pos_focal + torch.randn_like(pos_focal) * self.max_bond_length  / 2.4
        # dist = torch.norm(pos_generate - pos_notgenerate, p=2, dim=-1)
        # ind_close = (dist < self.close_threshold)
        # while ind_close.any():
        #     new_pos_notgenerate = pos_focal[ind_close] + torch.randn_like(pos_focal[ind_close]) * self.max_bond_length  / 2.3
        #     dist[ind_close] = torch.norm(pos_generate[ind_close] - new_pos_notgenerate, p=2, dim=-1)
        #     pos_notgenerate[ind_close] = new_pos_notgenerate
        #     ind_close = (dist < self.close_threshold)
        # data.pos_notgenerate = pos_notgenerate

        return data


class AtomComposer(object):

    def __init__(self, protein_dim, ligand_dim, knn):
        super().__init__()
        self.protein_dim = protein_dim
        self.ligand_dim = ligand_dim
        self.knn = knn  # knn of compose atoms

    def __call__(self, data: ProteinLigandData):
        # fetch ligand context and protein from data
        ligand_context_pos = data['ligand_context_pos']
        ligand_context_feature_full = data['ligand_context_feature_full']
        protein_pos = data['protein_pos']
        protein_atom_feature = data['protein_atom_feature']
        len_ligand_ctx = len(ligand_context_pos)
        len_protein = len(protein_pos)

        # compose ligand context and protein. save idx of them in compose
        data['compose_pos'] = torch.cat([ligand_context_pos, protein_pos], dim=0)
        len_compose = len_ligand_ctx + len_protein
        ligand_context_feature_full_expand = torch.cat([
            ligand_context_feature_full,
            torch.zeros([len_ligand_ctx, self.protein_dim - self.ligand_dim], dtype=torch.long)
        ], dim=1)
        data['compose_feature'] = torch.cat([ligand_context_feature_full_expand, protein_atom_feature], dim=0)
        data['idx_ligand_ctx_in_compose'] = torch.arange(len_ligand_ctx, dtype=torch.long)  # can be delete
        data['idx_protein_in_compose'] = torch.arange(len_protein, dtype=torch.long) + len_ligand_ctx  # can be delete

        # build knn graph and bond type
        data = self.get_knn_graph(data, self.knn, len_ligand_ctx, len_compose, num_workers=16)
        return data

    @staticmethod
    def get_knn_graph(data: ProteinLigandData, knn, len_ligand_ctx, len_compose, num_workers=1, ):
        data['compose_knn_edge_index'] = knn_graph(data['compose_pos'], knn, flow='target_to_source', num_workers=num_workers)

        id_compose_edge = data['compose_knn_edge_index'][0,
                          :len_ligand_ctx * knn] * len_compose + data['compose_knn_edge_index'][1, :len_ligand_ctx * knn]
        id_ligand_ctx_edge = data['ligand_context_bond_index'][0] * len_compose + data['ligand_context_bond_index'][1]
        idx_edge = [torch.nonzero(id_compose_edge == id_) for id_ in id_ligand_ctx_edge]
        idx_edge = torch.tensor([a.squeeze() if len(a) > 0 else torch.tensor(-1) for a in idx_edge], dtype=torch.long)
        data['compose_knn_edge_type'] = torch.zeros(len(data['compose_knn_edge_index'][0]),
                                                 dtype=torch.long)  # for encoder edge embedding
        data['compose_knn_edge_type'][idx_edge[idx_edge >= 0]] = data['ligand_context_bond_type'][idx_edge >= 0]
        data['compose_knn_edge_feature'] = torch.cat([
            torch.ones([len(data['compose_knn_edge_index'][0]), 1], dtype=torch.long),
            torch.zeros([len(data['compose_knn_edge_index'][0]), 3], dtype=torch.long),
        ], dim=-1)
        data['compose_knn_edge_feature'][idx_edge[idx_edge >= 0]] = F.one_hot(data['ligand_context_bond_type'][idx_edge >= 0],
                                                                           num_classes=4)  # 0 (1,2,3)-onehot
        return data


class FeaturizeProteinAtom(object):

    def __init__(self):
        super().__init__()
        # self.atomic_numbers = torch.LongTensor([1, 6, 7, 8, 16, 34])    # H, C, N, O, S, Se
        self.atomic_numbers = torch.LongTensor([6, 7, 8, 16, 34])  # H, C, N, O, S, Se
        self.max_num_aa = 20

    @property
    def feature_dim(self):
        return self.atomic_numbers.size(0) + self.max_num_aa + 1

    def __call__(self, data: ProteinLigandData):
        element = data['protein_element'].view(-1, 1) == self.atomic_numbers.view(1, -1)  # (N_atoms, N_elements)
        amino_acid = F.one_hot(data['protein_atom_to_aa_type'], num_classes=self.max_num_aa)
        is_backbone = data['protein_is_backbone'].view(-1, 1).long()
        x = torch.cat([element, amino_acid, is_backbone], dim=-1)
        data['protein_atom_feature'] = x
        return data


class FeaturizeLigandAtom(object):

    def __init__(self):
        super().__init__()
        # self.atomic_numbers = torch.LongTensor([1,6,7,8,9,15,16,17])  # H C N O F P S Cl
        self.atomic_numbers = torch.LongTensor([6, 7, 8, 9, 15, 16, 17])  # C N O F P S Cl

    @property
    def num_properties(self):
        return len(ATOM_FAMILIES)

    @property
    def feature_dim(self):
        return self.atomic_numbers.size(0) + len(ATOM_FAMILIES)

    def __call__(self, data: ProteinLigandData):
        element = data['ligand_element'].view(-1, 1) == self.atomic_numbers.view(1, -1)  # (N_atoms, N_elements)
        x = torch.cat([element, data['ligand_atom_feature']], dim=-1)
        data['ligand_atom_feature_full'] = x
        return data


class FeaturizeLigandBond(object):

    def __init__(self):
        super().__init__()

    def __call__(self, data: ProteinLigandData):
        data['ligand_bond_feature'] = F.one_hot(data['ligand_bond_type'] - 1, num_classes=3)  # (1,2,3) to (0,1,2)-onehot

        neighbor_dict = {}
        # used in rotation angle prediction
        mol = data['moltree'].mol
        for i, atom in enumerate(mol.GetAtoms()):
            neighbor_dict[i] = [n.GetIdx() for n in atom.GetNeighbors()]
        data['ligand_neighbors'] = neighbor_dict
        return data


class LigandCountNeighbors(object):

    @staticmethod
    def count_neighbors(edge_index, symmetry, valence=None, num_nodes=None):
        assert symmetry == True, 'Only support symmetrical edges.'

        if num_nodes is None:
            num_nodes = maybe_num_nodes(edge_index)

        if valence is None:
            valence = torch.ones([edge_index.size(1)], device=edge_index.device)
        valence = valence.view(edge_index.size(1))

        return scatter_add(valence, index=edge_index[0], dim=0, dim_size=num_nodes).long()

    def __init__(self):
        super().__init__()

    def __call__(self, data):
        data['ligand_num_neighbors'] = self.count_neighbors(
            data['ligand_bond_index'],
            symmetry=True,
            num_nodes=data['ligand_element'].size(0),
        )
        data['ligand_atom_valence'] = self.count_neighbors(
            data['ligand_bond_index'],
            symmetry=True,
            valence=data['ligand_bond_type'],
            num_nodes=data['ligand_element'].size(0),
        )
        return data


class LigandRandomMask(object):

    def __init__(self, min_ratio=0.0, max_ratio=1.2, min_num_masked=1, min_num_unmasked=0):
        super().__init__()
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.min_num_masked = min_num_masked
        self.min_num_unmasked = min_num_unmasked

    def __call__(self, data: ProteinLigandData):
        ratio = np.clip(random.uniform(self.min_ratio, self.max_ratio), 0.0, 1.0)
        num_atoms = data.ligand_element.size(0)
        num_masked = int(num_atoms * ratio)

        if num_masked < self.min_num_masked:
            num_masked = self.min_num_masked
        if (num_atoms - num_masked) < self.min_num_unmasked:
            num_masked = num_atoms - self.min_num_unmasked

        idx = np.arange(num_atoms)
        np.random.shuffle(idx)
        idx = torch.LongTensor(idx)
        masked_idx = idx[:num_masked]
        context_idx = idx[num_masked:]

        data.ligand_masked_element = data.ligand_element[masked_idx]
        data.ligand_masked_feature = data.ligand_atom_feature[masked_idx]  # For Prediction
        data.ligand_masked_pos = data.ligand_pos[masked_idx]

        data.ligand_context_element = data.ligand_element[context_idx]
        data.ligand_context_feature_full = data.ligand_atom_feature_full[context_idx]  # For Input
        data.ligand_context_pos = data.ligand_pos[context_idx]

        data.ligand_context_bond_index, data.ligand_context_bond_feature = subgraph(
            context_idx,
            data.ligand_bond_index,
            edge_attr=data.ligand_bond_feature,
            relabel_nodes=True,
        )
        data.ligand_context_num_neighbors = LigandCountNeighbors.count_neighbors(
            data.ligand_context_bond_index,
            symmetry=True,
            num_nodes=context_idx.size(0),
        )

        # print(context_idx)
        # print(data.ligand_context_bond_index)

        # mask = torch.logical_and(
        #     (data.ligand_bond_index[0].view(-1, 1) == context_idx.view(1, -1)).any(dim=-1),
        #     (data.ligand_bond_index[1].view(-1, 1) == context_idx.view(1, -1)).any(dim=-1),
        # )
        # print(data.ligand_bond_index[:, mask])

        # print(data.ligand_context_num_neighbors)
        # print(data.ligand_num_neighbors[context_idx])

        data.ligand_frontier = data.ligand_context_num_neighbors < data.ligand_num_neighbors[context_idx]

        data._mask = 'random'

        return data





class LigandCompleteMask(object):

#     def __init__(self, min_ratio=0.0, max_ratio=1.2, min_num_masked=1, min_num_unmasked=0, num_masked_define=None, vocab=None):
    def __init__(self, vocab=None):
        super().__init__()
        self.vocab = vocab
        self.vocab_size = vocab.size()
#         self.num_masked_define = num_masked_define  


    def __call__(self, data):
        for i, node in enumerate(data['moltree'].nodes):  
            node.nid = i
            node.wid = self.vocab.get_index(node.smiles)
         
        edges = []
        for node in data['moltree'].nodes:
            for n, nei in enumerate(node.neighbors):
                if node.nid < nei.nid:
                    edges.append((node.nid, nei.nid))
                else:
                    assert (nei.nid, node.nid) in edges     
 
        nodes_num = len(data['moltree'].nodes)
        pointList = list(range(nodes_num))
        linkList = edges
        G = nx.Graph()
        for node in pointList:
            G.add_node(node)
        for link in linkList:
            G.add_edge(link[0], link[1])
        
 
        total_connected_subgraph = {}
        connected_subgraph_data_num = 0
        pre_suc_data_num = 0
        for k in range(1, nodes_num):
            connected_subgraph = []
            combins = [c for c in combinations(pointList, k)]
            connected_subgraph = []
            for comb in combins:
                cur_subgraph = G.subgraph(comb)   
            #     nx.draw_networkx(subgraph, with_labels=True)
                if nx.is_connected(cur_subgraph) == True:
                    cur_dict = {}
                    cur_dict["connected_subgraph"] = comb
                    connected_subgraph_data_num += 1
                    subgraph_nodes = list(cur_subgraph.nodes)
                    pre_suc_data = []
                    for node in subgraph_nodes:
        #                 print(node)
                        neis_in_subG = set(cur_subgraph.neighbors(node))
        #                 print(neis_in_subG)  
                        neis_in_G = set(G.neighbors(node)) 
        #                 print(neis_in_G)  
                        next_nodes = list(neis_in_G - neis_in_subG)
        #                 print(next_nodes)   
                        if len(next_nodes) != 0:
                            pre_suc_data.append({"focal": node, "next": next_nodes})
                            pre_suc_data_num += 1
                    cur_dict["pre_suc_data"] = pre_suc_data
                    connected_subgraph.append(cur_dict)
            total_connected_subgraph[k] = connected_subgraph
        
 
        data_zeroG = copy.deepcopy(data)
        context_idx = set()   
        context_idx = torch.LongTensor(list(context_idx))

        datas = []
 
        data_zeroG['current_wid'] = torch.tensor([self.vocab_size])
        data_zeroG['current_atoms'] = torch.tensor([data_zeroG['protein_contact_idx']])
     
        data_zeroG['next_wid'] = torch.tensor([data_zeroG['moltree'].nodes[n_].wid for n_ in range(nodes_num)])

        data_zeroG['ligand_context_element'] = data_zeroG['ligand_element'][context_idx]
        data_zeroG['ligand_context_feature_full'] = data_zeroG['ligand_atom_feature_full'][context_idx]  # For Input
        data_zeroG['pocket_center'] = torch.mean(data_zeroG['ligand_pos'], dim=0)  

        data_zeroG['ligand_context_pos'] = data_zeroG['ligand_pos'][context_idx]   # tensor([], size=(0,3))
        data_zeroG['ligand_context_pos_crystal'] = data_zeroG['ligand_pos'][context_idx]  

        data_zeroG['num_atoms'] = torch.tensor([len(context_idx) + len(data_zeroG['protein_pos'])])
        data_zeroG['protein_contact'] = torch.tensor(data_zeroG['protein_contact'])
        data_zeroG['cand_labels'], data_zeroG['cand_mols'] = torch.tensor([]), []

        data_zeroG['ligand_context_bond_index'], data_zeroG['ligand_context_bond_feature'] = subgraph(
            context_idx,
            data_zeroG['ligand_bond_index'],
            edge_attr=data_zeroG['ligand_bond_feature'],
            relabel_nodes=True,
        )
        data_zeroG['ligand_context_num_neighbors'] = LigandCountNeighbors.count_neighbors(
            data_zeroG['ligand_context_bond_index'],
            symmetry=True,
            num_nodes=context_idx.size(0),
        )
        data_zeroG['ligand_frontier'] = data_zeroG['ligand_context_num_neighbors'] < data_zeroG['ligand_num_neighbors'][context_idx]
        data_zeroG['_mask'] = 'complete'
        
        datas.append(data_zeroG)
        pre_suc_data_num += 1
        docked_failure = 0
 
        for K in range(1, nodes_num): 
            k_subgraph_num = len(total_connected_subgraph[K])
             
            for L in range(k_subgraph_num): 
                origin_data = total_connected_subgraph[K][L]  
                 
                context_motif_ids = origin_data['connected_subgraph']
                context_idx = set()
                for i in context_motif_ids:
                    context_idx = context_idx | set(data['moltree'].nodes[i].clique)
                context_idx = torch.LongTensor(list(context_idx))

                data_subG = copy.deepcopy(data)
                data_subG['ligand_context_element'] = data_subG['ligand_element'][context_idx]
                data_subG['ligand_context_feature_full'] = data_subG['ligand_atom_feature_full'][context_idx]  # For Input        
                data_subG['num_atoms'] = torch.tensor([len(context_idx) + len(data_subG['protein_pos'])])
                data_subG['protein_contact'] = torch.tensor(data_subG['protein_contact'])
                data_subG['pocket_center'] = torch.mean(data_subG['ligand_pos'], dim=0) 
                
                data_subG['ligand_context_bond_index'], data_subG['ligand_context_bond_feature'] = subgraph(
                    context_idx,
                    data_subG['ligand_bond_index'],
                    edge_attr=data_subG['ligand_bond_feature'],
                    relabel_nodes=True,
                )

                data_subG['ligand_context_num_neighbors'] = LigandCountNeighbors.count_neighbors(
                    data_subG['ligand_context_bond_index'],
                    symmetry=True,
                    num_nodes=context_idx.size(0),
                )
                
                data_subG['ligand_frontier'] = data_subG['ligand_context_num_neighbors'] < data_subG['ligand_num_neighbors'][context_idx]
                data_subG['_mask'] = 'complete'
                 
                try:
                    data_mol = copy.deepcopy(data_subG['moltree'].mol)
                    Chem.SanitizeMol(data_mol)
                    visible_mol = get_clique_mol_simple(data_subG['moltree'].mol, context_idx.numpy().tolist())
                    Chem.SanitizeMol(visible_mol)
                    tmp_dir = './train_vina_data/'
                    mksure_path(tmp_dir)
                    total_file = str(data_subG['id']) + '_' + get_random_id(5) 
                    mol_filename = tmp_dir + total_file + "_pre_docked_mol.sdf"

                    writer = Chem.SDWriter(mol_filename)
                    # writer.SetKekulize(False)
                    writer.write(visible_mol, confId=0)
                    writer.close()

                    protonated_lig = rdkit.Chem.AddHs(visible_mol) 
                    rdkit.Chem.AllChem.EmbedMolecule(protonated_lig)   
                    preparator = MoleculePreparation()
                    mol_setups = preparator.prepare(protonated_lig)
                    ligand_pdbqt_path = total_file + '_meeko'
                    preparator.write_pdbqt_file(tmp_dir + ligand_pdbqt_path + '.pdbqt')

                    protein_filename = "./data/crossdocked_pocket10/" + data_subG["protein_filename"]
                    vina_task = QVinaDockingTask.from_pdbqt_data(
                        protein_filename, 
                        ligand_pdbqt_path,
                        tmp_dir=tmp_dir,
                        task_id=total_file,
                        center=data_subG['pocket_center']   
                    )
                    vina_results = vina_task.run_sync()
                    docked_mol = vina_results[0]['rdmol']
                    vina_score = vina_results[0]['affinity']
 
                    hit_ats = data_mol.GetSubstructMatches(docked_mol)  
                    list_order = []
                    for hit in hit_ats:
                        if len(set(hit) & set(context_idx.numpy())) == len(context_idx):
                            list_order = hit
                    if len(list_order) == 0:
                        print("Inconsistency") 
                        docked_failure += 1
                        continue
                        # return None
                    order_dict = {}
                    for i in range(len(list_order)):
                        order_dict[list_order[i]] = i

                    after_docked_all_pos = np.array([docked_mol.GetConformer().GetAtomPosition(atom.GetIdx()) for atom in docked_mol.GetAtoms()])
                #             print("list_order", list_order)
                #             print("order_dict", order_dict)
                #             print("context_idx", context_idx)
                #             print("order_dict[int(context_idx[0])]", order_dict[int(context_idx[0])])
                    docked_pos = np.expand_dims(after_docked_all_pos[order_dict[int(context_idx[0])]], 0)
                    for idx in context_idx[1:]:
                        docked_idx = order_dict[int(idx)]
                        cur_docked_pos = np.expand_dims(after_docked_all_pos[docked_idx], 0)
                        docked_pos = np.concatenate([docked_pos, cur_docked_pos],axis=0)
 
                    writer = Chem.SDWriter(tmp_dir + total_file + "_docked_mol.sdf")
                    # writer.SetKekulize(False)
                    writer.write(docked_mol, confId=0)
                    writer.close()

                    data_subG['ligand_context_pos'] = torch.tensor(docked_pos, dtype=data_subG['ligand_pos'].dtype, device=data_subG['ligand_pos'].device)  
                except Exception as ex:
                    print("Exception%s"%ex)
                    docked_failure += 1
                    continue
                #     return None
                 
                data_subG['ligand_context_pos_crystal'] = data_subG['ligand_pos'][context_idx]  
 
                unit_nums = len(origin_data['pre_suc_data'])
                for u in range(unit_nums):
        #             u = 0
                    data_ = copy.deepcopy(data_subG)
                    unit_data = origin_data['pre_suc_data'][u]
                    data_['current_wid'] = torch.tensor([data_['moltree'].nodes[unit_data["focal"]].wid])
                    data_['next_wid'] = torch.tensor([data_['moltree'].nodes[n_].wid for n_ in unit_data['next']])   # For Prediction
                    current_atoms = data_['moltree'].nodes[unit_data["focal"]].clique
                    data_['current_atoms'] = torch.cat([torch.where(context_idx == i)[0] for i in current_atoms]) + len(data_['protein_pos'])
                    
 
                    cand_labels_n, cand_mols_n = [], []
                    for n_ in unit_data['next']:
                        cand_labels, cand_mols = enumerate_assemble(data_['moltree'].mol, context_idx.tolist(),
                                                                    data_['moltree'].nodes[unit_data['focal']],
                                                                    data_['moltree'].nodes[n_])
                        cand_labels_n.extend(cand_labels)
                        cand_mols_n.extend(cand_mols) 
                    data_['cand_labels'] = cand_labels_n
                    data_['cand_mols'] = [mol_to_graph_data_obj_simple(mol) for mol in cand_mols_n]
                    
                    datas.append(data_)   

        return datas, pre_suc_data_num, docked_failure



class LigandPartialMask(object):
 
    def __init__(self, C=10, vocab=None): 
        super().__init__()
        self.vocab = vocab
        self.vocab_size = vocab.size()
        self.C = C 


    def __call__(self, data):
        for i, node in enumerate(data['moltree'].nodes):  
            node.nid = i
            node.wid = self.vocab.get_index(node.smiles)
         
        edges = []
        for node in data['moltree'].nodes:
            for n, nei in enumerate(node.neighbors):
                if node.nid < nei.nid:
                    edges.append((node.nid, nei.nid))
                else:
                    assert (nei.nid, node.nid) in edges     
 
        nodes_num = len(data['moltree'].nodes)
        pointList = list(range(nodes_num))
        linkList = edges
        G = nx.Graph()
        for node in pointList:
            G.add_node(node)
        for link in linkList:
            G.add_edge(link[0], link[1])
         
        total_connected_subgraph = {}
        connected_subgraph_data_num = 0
        pre_suc_data_num = 0
        for k in range(1, 8):  # nodes_num
            # k = 8
            connected_subgraph = [] 
            combins = [c for c in combinations(pointList, k)]
            for comb in combins:
                cur_subgraph = G.subgraph(comb)   
                if nx.is_connected(cur_subgraph) == True:
                    cur_dict = {}
                    cur_dict["connected_subgraph"] = comb
                    connected_subgraph_data_num += 1
                    subgraph_nodes = list(cur_subgraph.nodes)
                    pre_suc_data = []
                    for node in subgraph_nodes:
        #                 print(node)
                        neis_in_subG = set(cur_subgraph.neighbors(node))
        #                 print(neis_in_subG)  
                        neis_in_G = set(G.neighbors(node))  
        #                 print(neis_in_G)  
                        next_nodes = list(neis_in_G - neis_in_subG) 
                        if len(next_nodes) != 0:
                            pre_suc_data.append({"focal": node, "next": next_nodes})
                            pre_suc_data_num += 1
                    cur_dict["pre_suc_data"] = pre_suc_data
                    connected_subgraph.append(cur_dict)
                    if len(connected_subgraph) >= self.C:
                        break
            total_connected_subgraph[k] = connected_subgraph
            #break
         
        data_zeroG = copy.deepcopy(data)
        context_idx = set()   
        context_idx = torch.LongTensor(list(context_idx))

        datas = []
 
        data_zeroG['current_wid'] = torch.tensor([self.vocab_size])
        data_zeroG['current_atoms'] = torch.tensor([data_zeroG['protein_contact_idx']])
 
        data_zeroG['next_wid'] = torch.tensor([data_zeroG['moltree'].nodes[n_].wid for n_ in range(nodes_num)])

        data_zeroG['ligand_context_element'] = data_zeroG['ligand_element'][context_idx]
        data_zeroG['ligand_context_feature_full'] = data_zeroG['ligand_atom_feature_full'][context_idx]  # For Input
        data_zeroG['pocket_center'] = torch.mean(data_zeroG['ligand_pos'], dim=0)   

        data_zeroG['ligand_context_pos'] = data_zeroG['ligand_pos'][context_idx]   # tensor([], size=(0,3))
        data_zeroG['ligand_context_pos_crystal'] = data_zeroG['ligand_pos'][context_idx]  

        data_zeroG['num_atoms'] = torch.tensor([len(context_idx) + len(data_zeroG['protein_pos'])])
        data_zeroG['protein_contact'] = torch.tensor(data_zeroG['protein_contact'])
        data_zeroG['cand_labels'], data_zeroG['cand_mols'] = torch.tensor([]), []

        data_zeroG['ligand_context_bond_index'], data_zeroG['ligand_context_bond_feature'] = subgraph(
            context_idx,
            data_zeroG['ligand_bond_index'],
            edge_attr=data_zeroG['ligand_bond_feature'],
            relabel_nodes=True,
        )
        data_zeroG['ligand_context_num_neighbors'] = LigandCountNeighbors.count_neighbors(
            data_zeroG['ligand_context_bond_index'],
            symmetry=True,
            num_nodes=context_idx.size(0),
        )
        data_zeroG['ligand_frontier'] = data_zeroG['ligand_context_num_neighbors'] < data_zeroG['ligand_num_neighbors'][context_idx]
        data_zeroG['_mask'] = 'complete'
        data_zeroG['visible_nodes_num'] = 0
        data_zeroG['visible_mol'] = ''   
        data_zeroG['subgraph_id'] = 0
        
        datas.append(data_zeroG)
        pre_suc_data_num += 1
        docked_failure = 0
 
        for K in range(1, 8):   
            k_subgraph_num = len(total_connected_subgraph[K])
              
            for L in range(k_subgraph_num):  
                origin_data = total_connected_subgraph[K][L]   
                 
                context_motif_ids = origin_data['connected_subgraph']
                context_idx = set()
                for i in context_motif_ids:
                    context_idx = context_idx | set(data['moltree'].nodes[i].clique)
                context_idx = torch.LongTensor(list(context_idx))

                data_subG = copy.deepcopy(data)
                data_subG['ligand_context_element'] = data_subG['ligand_element'][context_idx]
                data_subG['ligand_context_feature_full'] = data_subG['ligand_atom_feature_full'][context_idx]  # For Input        
                data_subG['num_atoms'] = torch.tensor([len(context_idx) + len(data_subG['protein_pos'])])
                data_subG['protein_contact'] = torch.tensor(data_subG['protein_contact'])
                data_subG['pocket_center'] = torch.mean(data_subG['ligand_pos'], dim=0)   
                data_subG['visible_nodes_num'] = K
                data_subG['subgraph_id'] = L
        
                data_subG['ligand_context_bond_index'], data_subG['ligand_context_bond_feature'] = subgraph(
                    context_idx,
                    data_subG['ligand_bond_index'],
                    edge_attr=data_subG['ligand_bond_feature'],
                    relabel_nodes=True,
                )

                data_subG['ligand_context_num_neighbors'] = LigandCountNeighbors.count_neighbors(
                    data_subG['ligand_context_bond_index'],
                    symmetry=True,
                    num_nodes=context_idx.size(0),
                )
                
                data_subG['ligand_frontier'] = data_subG['ligand_context_num_neighbors'] < data_subG['ligand_num_neighbors'][context_idx]
                data_subG['_mask'] = 'complete'
                 
                try:
                    data_mol = copy.deepcopy(data_subG['moltree'].mol)
                    Chem.SanitizeMol(data_mol)
                    visible_mol = get_clique_mol_simple(data_subG['moltree'].mol, context_idx.numpy().tolist())
                    Chem.SanitizeMol(visible_mol)
                    tmp_dir = './train_vina_data/'
                    mksure_path(tmp_dir)
                    total_file = str(data_subG['id']) + '_' + get_random_id(5)  
                    mol_filename = tmp_dir + total_file + "_pre_docked_mol.sdf"

                    writer = Chem.SDWriter(mol_filename)
                    # writer.SetKekulize(False)
                    writer.write(visible_mol, confId=0)
                    writer.close()

                    protonated_lig = rdkit.Chem.AddHs(visible_mol)   
                    rdkit.Chem.AllChem.EmbedMolecule(protonated_lig) 
                    preparator = MoleculePreparation()
                    mol_setups = preparator.prepare(protonated_lig)
                    ligand_pdbqt_path = total_file + '_meeko'
                    preparator.write_pdbqt_file(tmp_dir + ligand_pdbqt_path + '.pdbqt')

                    protein_filename = "./data/crossdocked_pocket10/" + data_subG["protein_filename"]
                    vina_task = QVinaDockingTask.from_pdbqt_data(
                        protein_filename, 
                        ligand_pdbqt_path,
                        tmp_dir=tmp_dir,
                        task_id=total_file,
                        center=data_subG['pocket_center']    
                    )
                    vina_results = vina_task.run_sync()
                    docked_mol = vina_results[0]['rdmol']
                    vina_score = vina_results[0]['affinity']
                     
                    hit_ats = data_mol.GetSubstructMatches(docked_mol)  
                    list_order = []
                    for hit in hit_ats:
                        if len(set(hit) & set(context_idx.numpy())) == len(context_idx):
                            list_order = hit
                    if len(list_order) == 0:
                        print("Inconsistency") 
                        docked_failure += 1
                        continue
                        # return None
                    order_dict = {}
                    for i in range(len(list_order)):
                        order_dict[list_order[i]] = i

                    after_docked_all_pos = np.array([docked_mol.GetConformer().GetAtomPosition(atom.GetIdx()) for atom in docked_mol.GetAtoms()])
 
                    docked_pos = np.expand_dims(after_docked_all_pos[order_dict[int(context_idx[0])]], 0)
                    for idx in context_idx[1:]:
                        docked_idx = order_dict[int(idx)]
                        cur_docked_pos = np.expand_dims(after_docked_all_pos[docked_idx], 0)
                        docked_pos = np.concatenate([docked_pos, cur_docked_pos],axis=0)
 
                    writer = Chem.SDWriter(tmp_dir + total_file + "_docked_mol.sdf")
                    # writer.SetKekulize(False)
                    writer.write(docked_mol, confId=0)
                    writer.close()

                    data_subG['ligand_context_pos'] = torch.tensor(docked_pos, dtype=data_subG['ligand_pos'].dtype, device=data_subG['ligand_pos'].device)  
                    data_subG['visible_mol'] = docked_mol  
                except Exception as ex:
                    print("Exception%s"%ex)
                    docked_failure += 1
                    continue
                #     return None
                 
                data_subG['ligand_context_pos_crystal'] = data_subG['ligand_pos'][context_idx]  
 
                unit_nums = len(origin_data['pre_suc_data'])
                for u in range(unit_nums):
        #             u = 0
                    data_ = copy.deepcopy(data_subG)
                    unit_data = origin_data['pre_suc_data'][u]
                    data_['current_wid'] = torch.tensor([data_['moltree'].nodes[unit_data["focal"]].wid])
                    data_['next_wid'] = torch.tensor([data_['moltree'].nodes[n_].wid for n_ in unit_data['next']])   # For Prediction
                    current_atoms = data_['moltree'].nodes[unit_data["focal"]].clique
                    data_['current_atoms'] = torch.cat([torch.where(context_idx == i)[0] for i in current_atoms]) + len(data_['protein_pos'])
                    
 
                    cand_labels_n, cand_mols_n = [], []
                    for n_ in unit_data['next']:
                        cand_labels, cand_mols = enumerate_assemble(data_['moltree'].mol, context_idx.tolist(),
                                                                    data_['moltree'].nodes[unit_data['focal']],
                                                                    data_['moltree'].nodes[n_])
                        cand_labels_n.extend(cand_labels)
                        cand_mols_n.extend(cand_mols)
        #             for c_mol in cand_mols_n:
        #                 display(c_mol)
                    data_['cand_labels'] = cand_labels_n
                    data_['cand_mols'] = [mol_to_graph_data_obj_simple(mol) for mol in cand_mols_n]
                    
                    datas.append(data_) 
        return datas, pre_suc_data_num, connected_subgraph_data_num, docked_failure




class LigandPreOrderingMask(object):

    def __init__(self, vocab=None): 
        super().__init__()
        self.vocab = vocab
        self.vocab_size = vocab.size()


    def __call__(self, data, ordering):
        for i, node in enumerate(data['moltree'].nodes):  
            node.nid = i
            node.wid = self.vocab.get_index(node.smiles)
 
        edges = []
        for node in data['moltree'].nodes:
            for n, nei in enumerate(node.neighbors):
                if node.nid < nei.nid:
                    edges.append((node.nid, nei.nid))
                else:
                    assert (nei.nid, node.nid) in edges     
 
        nodes_num = len(data['moltree'].nodes)
        pointList = list(range(nodes_num))
        linkList = edges
        G = nx.Graph()
        for node in pointList:
            G.add_node(node)
        for link in linkList:
            G.add_edge(link[0], link[1])

        combins = [ordering[0:k] for k in range(1, len(ordering))]

        total_connected_subgraph = {}
        connected_subgraph_data_num = 0
        pre_suc_data_num = 0

 
        for c_idx, comb in enumerate(combins):
        #     print("c_idx, comb", c_idx, comb)
            actual_unique_next = ordering[c_idx+1] 
            cur_subgraph = G.subgraph(comb) 
        #     nx.draw_networkx(subgraph, with_labels=True)
            if nx.is_connected(cur_subgraph) == True:   
                cur_dict = {}
                cur_dict["connected_subgraph"] = comb
                connected_subgraph_data_num += 1
                subgraph_nodes = list(cur_subgraph.nodes)
                pre_suc_data = []
                for node in subgraph_nodes:
        #                 print(node)
                    neis_in_subG = set(cur_subgraph.neighbors(node))
        #                 print(neis_in_subG)  
                    neis_in_G = set(G.neighbors(node)) 
        #                 print(neis_in_G)  
                    next_nodes = list(neis_in_G - neis_in_subG)
        #             print("next_nodes", next_nodes)   
                    if len(next_nodes) != 0 and actual_unique_next in next_nodes:
                        next_nodes = [actual_unique_next]
                        pre_suc_data.append({"focal": node, "next": next_nodes})
                        pre_suc_data_num += 1
                        break
                if len(pre_suc_data) == 0:  
                    raise Exception('Ordering invalid!')
                cur_dict["pre_suc_data"] = pre_suc_data
                total_connected_subgraph[len(comb)] = [cur_dict]   
            else:
                raise Exception('Ordering invalid!')

            #break
 
        data_zeroG = copy.deepcopy(data)
        context_idx = set()   
        context_idx = torch.LongTensor(list(context_idx))

        datas = []
 
        data_zeroG['current_wid'] = torch.tensor([self.vocab_size])
        data_zeroG['current_atoms'] = torch.tensor([data_zeroG['protein_contact_idx']]) 
        data_zeroG['next_wid'] = torch.tensor([data_zeroG['moltree'].nodes[ordering[0]].wid])

        data_zeroG['ligand_context_element'] = data_zeroG['ligand_element'][context_idx]
        data_zeroG['ligand_context_feature_full'] = data_zeroG['ligand_atom_feature_full'][context_idx]  # For Input
        data_zeroG['pocket_center'] = torch.mean(data_zeroG['ligand_pos'], dim=0)   

        data_zeroG['ligand_context_pos'] = data_zeroG['ligand_pos'][context_idx]   # tensor([], size=(0,3))
        data_zeroG['ligand_context_pos_crystal'] = data_zeroG['ligand_pos'][context_idx]  

        data_zeroG['num_atoms'] = torch.tensor([len(context_idx) + len(data_zeroG['protein_pos'])])
        data_zeroG['protein_contact'] = torch.tensor(data_zeroG['protein_contact'])
        data_zeroG['cand_labels'], data_zeroG['cand_mols'] = torch.tensor([]), []

        data_zeroG['ligand_context_bond_index'], data_zeroG['ligand_context_bond_feature'] = subgraph(
            context_idx,
            data_zeroG['ligand_bond_index'],
            edge_attr=data_zeroG['ligand_bond_feature'],
            relabel_nodes=True,
        )
        data_zeroG['ligand_context_num_neighbors'] = LigandCountNeighbors.count_neighbors(
            data_zeroG['ligand_context_bond_index'],
            symmetry=True,
            num_nodes=context_idx.size(0),
        )
        data_zeroG['ligand_frontier'] = data_zeroG['ligand_context_num_neighbors'] < data_zeroG['ligand_num_neighbors'][context_idx]
        data_zeroG['_mask'] = 'complete'
        data_zeroG['visible_nodes_num'] = 0
        data_zeroG['visible_mol'] = ''  
        data_zeroG['subgraph_id'] = 0
        
        datas.append(data_zeroG)
        pre_suc_data_num += 1
        docked_failure = 0

 
        for K in range(1, 1+len(total_connected_subgraph.keys())):  # nodes_num
 
            k_subgraph_num = len(total_connected_subgraph[K])
            
 
            for L in range(k_subgraph_num):  
                origin_data = total_connected_subgraph[K][L]   
        
                context_motif_ids = origin_data['connected_subgraph']
                context_idx = set()
                for i in context_motif_ids:
                    context_idx = context_idx | set(data['moltree'].nodes[i].clique)
                context_idx = torch.LongTensor(list(context_idx))

                data_subG = copy.deepcopy(data)
                data_subG['ligand_context_element'] = data_subG['ligand_element'][context_idx]
                data_subG['ligand_context_feature_full'] = data_subG['ligand_atom_feature_full'][context_idx]  # For Input        
                data_subG['num_atoms'] = torch.tensor([len(context_idx) + len(data_subG['protein_pos'])])
                data_subG['protein_contact'] = torch.tensor(data_subG['protein_contact'])
                data_subG['pocket_center'] = torch.mean(data_subG['ligand_pos'], dim=0)   
                data_subG['visible_nodes_num'] = K
                data_subG['subgraph_id'] = L
        
                data_subG['ligand_context_bond_index'], data_subG['ligand_context_bond_feature'] = subgraph(
                    context_idx,
                    data_subG['ligand_bond_index'],
                    edge_attr=data_subG['ligand_bond_feature'],
                    relabel_nodes=True,
                )

                data_subG['ligand_context_num_neighbors'] = LigandCountNeighbors.count_neighbors(
                    data_subG['ligand_context_bond_index'],
                    symmetry=True,
                    num_nodes=context_idx.size(0),
                )
                
                data_subG['ligand_frontier'] = data_subG['ligand_context_num_neighbors'] < data_subG['ligand_num_neighbors'][context_idx]
                data_subG['_mask'] = 'complete'
                 
                try:
                    data_mol = copy.deepcopy(data_subG['moltree'].mol)
                    Chem.SanitizeMol(data_mol)
                    visible_mol = get_clique_mol_simple(data_subG['moltree'].mol, context_idx.numpy().tolist())
                    Chem.SanitizeMol(visible_mol)
                    tmp_dir = './train_vina_data/'
                    mksure_path(tmp_dir)
                    total_file = str(data_subG['id']) + '_' + get_random_id(5)  
                    mol_filename = tmp_dir + total_file + "_pre_docked_mol.sdf"

                    writer = Chem.SDWriter(mol_filename)
                    # writer.SetKekulize(False)
                    writer.write(visible_mol, confId=0)
                    writer.close()

                    protonated_lig = rdkit.Chem.AddHs(visible_mol)  
                    rdkit.Chem.AllChem.EmbedMolecule(protonated_lig)   
                    preparator = MoleculePreparation()
                    mol_setups = preparator.prepare(protonated_lig)
                    ligand_pdbqt_path = total_file + '_meeko'
                    preparator.write_pdbqt_file(tmp_dir + ligand_pdbqt_path + '.pdbqt')

                    protein_filename = "./data/crossdocked_pocket10/" + data_subG["protein_filename"]
                    vina_task = QVinaDockingTask.from_pdbqt_data(
                        protein_filename, 
                        ligand_pdbqt_path,
                        tmp_dir=tmp_dir,
                        task_id=total_file,
                        center=data_subG['pocket_center']   
                    )
                    vina_results = vina_task.run_sync()
                    docked_mol = vina_results[0]['rdmol']
                    vina_score = vina_results[0]['affinity']
                     
                    hit_ats = data_mol.GetSubstructMatches(docked_mol)  
                    list_order = []
                    for hit in hit_ats:
                        if len(set(hit) & set(context_idx.numpy())) == len(context_idx):
                            list_order = hit
                    if len(list_order) == 0:
                        print("Inconsistency") 
                        return [], 0, 0, 1
                        # continue
                        # return None
                    order_dict = {}
                    for i in range(len(list_order)):
                        order_dict[list_order[i]] = i

                    after_docked_all_pos = np.array([docked_mol.GetConformer().GetAtomPosition(atom.GetIdx()) for atom in docked_mol.GetAtoms()])
 
                    docked_pos = np.expand_dims(after_docked_all_pos[order_dict[int(context_idx[0])]], 0)
                    for idx in context_idx[1:]:
                        docked_idx = order_dict[int(idx)]
                        cur_docked_pos = np.expand_dims(after_docked_all_pos[docked_idx], 0)
                        docked_pos = np.concatenate([docked_pos, cur_docked_pos],axis=0)
 
                    writer = Chem.SDWriter(tmp_dir + total_file + "_docked_mol.sdf")
                    # writer.SetKekulize(False)
                    writer.write(docked_mol, confId=0)
                    writer.close()

                    data_subG['ligand_context_pos'] = torch.tensor(docked_pos, dtype=data_subG['ligand_pos'].dtype, device=data_subG['ligand_pos'].device)   
                    data_subG['visible_mol'] = docked_mol  
                except Exception as ex:
                    print("Exception%s"%ex)
                    # docked_failure += 1
                    return [], 0, 0, 1
                    # continue
                #     return None
                 
                data_subG['ligand_context_pos_crystal'] = data_subG['ligand_pos'][context_idx]  
 
                unit_nums = len(origin_data['pre_suc_data'])
                for u in range(unit_nums):
        #             u = 0
                    data_ = copy.deepcopy(data_subG)
                    unit_data = origin_data['pre_suc_data'][u]
                    data_['current_wid'] = torch.tensor([data_['moltree'].nodes[unit_data["focal"]].wid])
                    data_['next_wid'] = torch.tensor([data_['moltree'].nodes[n_].wid for n_ in unit_data['next']])   # For Prediction
                    current_atoms = data_['moltree'].nodes[unit_data["focal"]].clique
                    data_['current_atoms'] = torch.cat([torch.where(context_idx == i)[0] for i in current_atoms]) + len(data_['protein_pos'])
                    
 
                    cand_labels_n, cand_mols_n = [], []
                    for n_ in unit_data['next']:
                        cand_labels, cand_mols = enumerate_assemble(data_['moltree'].mol, context_idx.tolist(),
                                                                    data_['moltree'].nodes[unit_data['focal']],
                                                                    data_['moltree'].nodes[n_])
                        cand_labels_n.extend(cand_labels)
                        cand_mols_n.extend(cand_mols)
        #             for c_mol in cand_mols_n:
        #                 display(c_mol)
                    data_['cand_labels'] = cand_labels_n
                    data_['cand_mols'] = [mol_to_graph_data_obj_simple(mol) for mol in cand_mols_n]
                    
                    datas.append(data_) 

        return datas, pre_suc_data_num, connected_subgraph_data_num, docked_failure







class LigandOrderingMask(object):

    def __init__(self, vocab=None): 
        super().__init__()
        self.vocab = vocab
        self.vocab_size = vocab.size()


    def __call__(self, data, ordering):
        for i, node in enumerate(data['moltree'].nodes):  
            node.nid = i
            node.wid = self.vocab.get_index(node.smiles)
 
        edges = []
        for node in data['moltree'].nodes:
            for n, nei in enumerate(node.neighbors):
                if node.nid < nei.nid:
                    edges.append((node.nid, nei.nid))
                else:
                    assert (nei.nid, node.nid) in edges     
 
        nodes_num = len(data['moltree'].nodes)
        pointList = list(range(nodes_num))
        linkList = edges
        G = nx.Graph()
        for node in pointList:
            G.add_node(node)
        for link in linkList:
            G.add_edge(link[0], link[1])

        combins = [ordering[0:k] for k in range(1, len(ordering))]

        total_connected_subgraph = {}
        connected_subgraph_data_num = 0
        pre_suc_data_num = 0

        for comb in combins:
            cur_subgraph = G.subgraph(comb)    
            if nx.is_connected(cur_subgraph) == True:  
                cur_dict = {}
                cur_dict["connected_subgraph"] = comb
                connected_subgraph_data_num += 1
                subgraph_nodes = list(cur_subgraph.nodes)
                pre_suc_data = []
                for node in subgraph_nodes:
        #                 print(node)
                    neis_in_subG = set(cur_subgraph.neighbors(node))
        #                 print(neis_in_subG)  
                    neis_in_G = set(G.neighbors(node))  
        #                 print(neis_in_G)  
                    next_nodes = list(neis_in_G - neis_in_subG)
        #                 print(next_nodes)   
                    if len(next_nodes) != 0:
                        pre_suc_data.append({"focal": node, "next": next_nodes})
                        pre_suc_data_num += 1
                cur_dict["pre_suc_data"] = pre_suc_data
                total_connected_subgraph[len(comb)] = [cur_dict]   
            else:
                raise Exception('Ordering invalid!')

            #break
 
        data_zeroG = copy.deepcopy(data)
        context_idx = set()  
        context_idx = torch.LongTensor(list(context_idx))

        datas = []
 
        data_zeroG['current_wid'] = torch.tensor([self.vocab_size])
        data_zeroG['current_atoms'] = torch.tensor([data_zeroG['protein_contact_idx']])
 
        data_zeroG['next_wid'] = torch.tensor([data_zeroG['moltree'].nodes[n_].wid for n_ in range(nodes_num)])

        data_zeroG['ligand_context_element'] = data_zeroG['ligand_element'][context_idx]
        data_zeroG['ligand_context_feature_full'] = data_zeroG['ligand_atom_feature_full'][context_idx]  # For Input
        data_zeroG['pocket_center'] = torch.mean(data_zeroG['ligand_pos'], dim=0)  

        data_zeroG['ligand_context_pos'] = data_zeroG['ligand_pos'][context_idx]   # tensor([], size=(0,3))
        data_zeroG['ligand_context_pos_crystal'] = data_zeroG['ligand_pos'][context_idx]  

        data_zeroG['num_atoms'] = torch.tensor([len(context_idx) + len(data_zeroG['protein_pos'])])
        data_zeroG['protein_contact'] = torch.tensor(data_zeroG['protein_contact'])
        data_zeroG['cand_labels'], data_zeroG['cand_mols'] = torch.tensor([]), []

        data_zeroG['ligand_context_bond_index'], data_zeroG['ligand_context_bond_feature'] = subgraph(
            context_idx,
            data_zeroG['ligand_bond_index'],
            edge_attr=data_zeroG['ligand_bond_feature'],
            relabel_nodes=True,
        )
        data_zeroG['ligand_context_num_neighbors'] = LigandCountNeighbors.count_neighbors(
            data_zeroG['ligand_context_bond_index'],
            symmetry=True,
            num_nodes=context_idx.size(0),
        )
        data_zeroG['ligand_frontier'] = data_zeroG['ligand_context_num_neighbors'] < data_zeroG['ligand_num_neighbors'][context_idx]
        data_zeroG['_mask'] = 'complete'
        data_zeroG['visible_nodes_num'] = 0
        data_zeroG['visible_mol'] = ''  
        data_zeroG['subgraph_id'] = 0
        
        datas.append(data_zeroG)
        pre_suc_data_num += 1
        docked_failure = 0
 
        for K in range(1, 1+len(total_connected_subgraph.keys())):  # nodes_num
 
            k_subgraph_num = len(total_connected_subgraph[K])
 
            for L in range(k_subgraph_num):  
                origin_data = total_connected_subgraph[K][L]   
                 
                context_motif_ids = origin_data['connected_subgraph']
                context_idx = set()
                for i in context_motif_ids:
                    context_idx = context_idx | set(data['moltree'].nodes[i].clique)
                context_idx = torch.LongTensor(list(context_idx))

                data_subG = copy.deepcopy(data)
                data_subG['ligand_context_element'] = data_subG['ligand_element'][context_idx]
                data_subG['ligand_context_feature_full'] = data_subG['ligand_atom_feature_full'][context_idx]  # For Input        
                data_subG['num_atoms'] = torch.tensor([len(context_idx) + len(data_subG['protein_pos'])])
                data_subG['protein_contact'] = torch.tensor(data_subG['protein_contact'])
                data_subG['pocket_center'] = torch.mean(data_subG['ligand_pos'], dim=0)  
                data_subG['visible_nodes_num'] = K
                data_subG['subgraph_id'] = L
        
                data_subG['ligand_context_bond_index'], data_subG['ligand_context_bond_feature'] = subgraph(
                    context_idx,
                    data_subG['ligand_bond_index'],
                    edge_attr=data_subG['ligand_bond_feature'],
                    relabel_nodes=True,
                )

                data_subG['ligand_context_num_neighbors'] = LigandCountNeighbors.count_neighbors(
                    data_subG['ligand_context_bond_index'],
                    symmetry=True,
                    num_nodes=context_idx.size(0),
                )
                
                data_subG['ligand_frontier'] = data_subG['ligand_context_num_neighbors'] < data_subG['ligand_num_neighbors'][context_idx]
                data_subG['_mask'] = 'complete'
                 
                try:
                    data_mol = copy.deepcopy(data_subG['moltree'].mol)
                    Chem.SanitizeMol(data_mol)
                    visible_mol = get_clique_mol_simple(data_subG['moltree'].mol, context_idx.numpy().tolist())
                    Chem.SanitizeMol(visible_mol)
                    tmp_dir = './train_vina_data/'
                    mksure_path(tmp_dir)
                    total_file = str(data_subG['id']) + '_' + get_random_id(5)  
                    mol_filename = tmp_dir + total_file + "_pre_docked_mol.sdf"

                    writer = Chem.SDWriter(mol_filename)
                    # writer.SetKekulize(False)
                    writer.write(visible_mol, confId=0)
                    writer.close()

                    protonated_lig = rdkit.Chem.AddHs(visible_mol) 
                    rdkit.Chem.AllChem.EmbedMolecule(protonated_lig)  
                    preparator = MoleculePreparation()
                    mol_setups = preparator.prepare(protonated_lig)
                    ligand_pdbqt_path = total_file + '_meeko'
                    preparator.write_pdbqt_file(tmp_dir + ligand_pdbqt_path + '.pdbqt')

                    protein_filename = "./data/crossdocked_pocket10/" + data_subG["protein_filename"]
                    vina_task = QVinaDockingTask.from_pdbqt_data(
                        protein_filename, 
                        ligand_pdbqt_path,
                        tmp_dir=tmp_dir,
                        task_id=total_file,
                        center=data_subG['pocket_center']  
                    )
                    vina_results = vina_task.run_sync()
                    docked_mol = vina_results[0]['rdmol']
                    vina_score = vina_results[0]['affinity']
                     
                    hit_ats = data_mol.GetSubstructMatches(docked_mol)   
                    list_order = []
                    for hit in hit_ats:
                        if len(set(hit) & set(context_idx.numpy())) == len(context_idx):
                            list_order = hit
                    if len(list_order) == 0:
                        print("Inconsistency") 
                        docked_failure += 1
                        continue
                        # return None
                    order_dict = {}
                    for i in range(len(list_order)):
                        order_dict[list_order[i]] = i

                    after_docked_all_pos = np.array([docked_mol.GetConformer().GetAtomPosition(atom.GetIdx()) for atom in docked_mol.GetAtoms()])
    
                    docked_pos = np.expand_dims(after_docked_all_pos[order_dict[int(context_idx[0])]], 0)
                    for idx in context_idx[1:]:
                        docked_idx = order_dict[int(idx)]
                        cur_docked_pos = np.expand_dims(after_docked_all_pos[docked_idx], 0)
                        docked_pos = np.concatenate([docked_pos, cur_docked_pos],axis=0)
 
                    writer = Chem.SDWriter(tmp_dir + total_file + "_docked_mol.sdf")
                    # writer.SetKekulize(False)
                    writer.write(docked_mol, confId=0)
                    writer.close()

                    data_subG['ligand_context_pos'] = torch.tensor(docked_pos, dtype=data_subG['ligand_pos'].dtype, device=data_subG['ligand_pos'].device)  
                    data_subG['visible_mol'] = docked_mol  
                except Exception as ex:
                    print("Exception%s"%ex)
                    docked_failure += 1
                    continue
                #     return None
                 
                data_subG['ligand_context_pos_crystal'] = data_subG['ligand_pos'][context_idx]  
 
                unit_nums = len(origin_data['pre_suc_data'])
                for u in range(unit_nums):
        #             u = 0
                    data_ = copy.deepcopy(data_subG)
                    unit_data = origin_data['pre_suc_data'][u]
                    data_['current_wid'] = torch.tensor([data_['moltree'].nodes[unit_data["focal"]].wid])
                    data_['next_wid'] = torch.tensor([data_['moltree'].nodes[n_].wid for n_ in unit_data['next']])   # For Prediction
                    current_atoms = data_['moltree'].nodes[unit_data["focal"]].clique
                    data_['current_atoms'] = torch.cat([torch.where(context_idx == i)[0] for i in current_atoms]) + len(data_['protein_pos'])
                     
                    cand_labels_n, cand_mols_n = [], []
                    for n_ in unit_data['next']:
                        cand_labels, cand_mols = enumerate_assemble(data_['moltree'].mol, context_idx.tolist(),
                                                                    data_['moltree'].nodes[unit_data['focal']],
                                                                    data_['moltree'].nodes[n_])
                        cand_labels_n.extend(cand_labels)
                        cand_mols_n.extend(cand_mols)
        #             for c_mol in cand_mols_n:
        #                 display(c_mol)
                    data_['cand_labels'] = cand_labels_n
                    data_['cand_mols'] = [mol_to_graph_data_obj_simple(mol) for mol in cand_mols_n]
                    
                    datas.append(data_)
   

        return datas, pre_suc_data_num, connected_subgraph_data_num, docked_failure






class OrderingTransform(object):

    def __init__(self, ordering_nums=10, vocab=None):
        super().__init__()
        self.ordering_nums = ordering_nums
        self.vocab = vocab
        self.vocab_size = vocab.size()

    @staticmethod
    def get_perm_motif(moltree, vocab, root=''):  
        for i, node in enumerate(moltree.nodes):
            node.nid = i
            node.wid = vocab.get_index(node.smiles)
        
        traverse_nodes = [root]    
        nodes_perm = []
        visited = {traverse_nodes[0]}
        while len(traverse_nodes) > 0:
            current = traverse_nodes.pop(0)
            nodes_perm.append(current)
            next_candid = []
            for motif in moltree.nodes[current].neighbors:
                if motif.nid in visited: continue
                next_candid.append(motif.nid)
                visited.add(motif.nid)
            
            random.shuffle(next_candid)
            traverse_nodes += next_candid

        return nodes_perm

    def __call__(self, data): 
        orderings = []
        motif_nodes_nids = list(range(len(data['moltree'].nodes)))   
        starting_points = random.sample(motif_nodes_nids, min(self.ordering_nums, len(motif_nodes_nids))) 
     
        cnt = 0  
        starting_idx = 0
        while(len(orderings) < self.ordering_nums):
            cnt += 1
            if starting_idx < len(starting_points):
                nodes_perm = self.get_perm_motif(data['moltree'], self.vocab, starting_points[starting_idx])
                starting_idx += 1
            else: 
                nodes_perm = self.get_perm_motif(data['moltree'], self.vocab, np.random.choice(motif_nodes_nids))
            in_it = False
            for ordering in orderings:
                if ordering == nodes_perm:
                    in_it = True
            if in_it == False:
                orderings.append(nodes_perm)
            if cnt == 1000:
                break
        
        data['orderings'] = orderings   

        num_motifs = len(nodes_perm)
        context_idx = torch.LongTensor(list(range(data['ligand_element'].shape[0])))   

    
 
        data['pocket_center'] = torch.mean(data['ligand_pos'], dim=0)   
    
        try:
            data_mol = copy.deepcopy(data['moltree'].mol)
            Chem.SanitizeMol(data_mol)
            tmp_dir = './train_vina_data/'
            mksure_path(tmp_dir)
            total_file = str(data['id']) + '_' + get_random_id(5)  
            mol_filename = tmp_dir + total_file + "_pre_docked_mol.sdf"

            writer = Chem.SDWriter(mol_filename)
            # writer.SetKekulize(False)
            writer.write(data_mol, confId=0)
            writer.close()

            protonated_lig = rdkit.Chem.AddHs(data_mol)  
            rdkit.Chem.AllChem.EmbedMolecule(protonated_lig)  
            preparator = MoleculePreparation()
            mol_setups = preparator.prepare(protonated_lig)
            ligand_pdbqt_path = total_file + '_meeko'
            preparator.write_pdbqt_file(tmp_dir + ligand_pdbqt_path + '.pdbqt')

            protein_filename = "./data/crossdocked_pocket10/" + data["protein_filename"]
            vina_task = QVinaDockingTask.from_pdbqt_data(
                protein_filename, 
                ligand_pdbqt_path,
                tmp_dir=tmp_dir,
                task_id=total_file,
                center=data['pocket_center']   
            )
            vina_results = vina_task.run_sync()
            docked_mol = vina_results[0]['rdmol']
            vina_score = vina_results[0]['affinity']

            hit_ats = data_mol.GetSubstructMatches(docked_mol) 
            list_order = []
            for hit in hit_ats:
                if len(set(hit) & set(context_idx.numpy())) == len(context_idx):
                    list_order = hit
            if len(list_order) == 0:
                print("Inconsistency")
                return None
            order_dict = {}
            for i in range(len(list_order)):
                order_dict[list_order[i]] = i

            after_docked_all_pos = np.array([docked_mol.GetConformer().GetAtomPosition(atom.GetIdx()) for atom in docked_mol.GetAtoms()])
            docked_pos = np.expand_dims(after_docked_all_pos[order_dict[int(context_idx[0])]], 0)
            for idx in context_idx[1:]:
                docked_idx = order_dict[int(idx)]
                cur_docked_pos = np.expand_dims(after_docked_all_pos[docked_idx], 0)
                docked_pos = np.concatenate([docked_pos, cur_docked_pos],axis=0)
 
            writer = Chem.SDWriter(tmp_dir + total_file + "_docked_mol.sdf")
         
            writer.write(docked_mol, confId=0)
            writer.close()

            data['ligand_context_pos'] = torch.tensor(docked_pos, dtype=data['ligand_pos'].dtype, device=data['ligand_pos'].device)  
        except Exception as ex:
            print("Exception%s"%ex)
            return None

        # print("vina done")     
        data['num_atoms'] = torch.tensor([len(context_idx) + len(data['protein_pos'])])

 
        ordering_datas = {}
        for ith_ordering, ordering in enumerate(data['orderings']):
            # print(ordering)
            datas = []
            gen_motif = []
            gen_motif_atoms = []
            for o_idx in range(len(ordering)):
                cur_motif = ordering[o_idx]
                # print(cur_motif)
                cur_data = data.copy()
                gen_signal = torch.zeros([data['ligand_atom_feature_full'].shape[0], 1])
            #     print(gen_signal)
                if o_idx == 0:
                    pass
                else:  
                    gen_motif_atoms.extend(cur_data['moltree'].nodes[gen_motif[o_idx-1]].clique)
                    gen_signal[gen_motif_atoms, 0] = 1

                cur_data['ordering_label'] = torch.LongTensor([cur_motif])
                # print(cur_data['ordering_label'])
                cur_data['ligand_atom_feature_full'] = torch.cat((cur_data['ligand_atom_feature_full'], gen_signal), dim=1)
                # print(cur_data['ligand_atom_feature_full'])
                cur_data['ordering'] = ith_ordering   
                cur_data['chosen_motifs'] = gen_motif.copy()

                gen_motif.append(cur_motif)
                datas.append(cur_data)
            ordering_datas[ith_ordering] = datas

        return ordering_datas



class LigandMaskAll(LigandBFSMask):

    def __init__(self, vocab):
        super().__init__(min_ratio=1.0, vocab=vocab)


class LigandMixedMask(object):

    def __init__(self, min_ratio=0.0, max_ratio=1.2, min_num_masked=1, min_num_unmasked=0, p_random=0.5, p_bfs=0.25,
                 p_invbfs=0.25):
        super().__init__()

        self.t = [
            LigandRandomMask(min_ratio, max_ratio, min_num_masked, min_num_unmasked),
            LigandBFSMask(min_ratio, max_ratio, min_num_masked, min_num_unmasked, inverse=False),
            LigandBFSMask(min_ratio, max_ratio, min_num_masked, min_num_unmasked, inverse=True),
        ]
        self.p = [p_random, p_bfs, p_invbfs]

    def __call__(self, data):
        f = random.choices(self.t, k=1, weights=self.p)[0]
        return f(data)





def get_mask(cfg, vocab):
    if cfg.type == 'pre_ordering':
        return LigandPreOrderingMask(
            vocab=vocab
        )
    elif cfg.type == 'ordering':
        return LigandOrderingMask(
            vocab=vocab
        )
    elif cfg.type == 'em':
        return OrderingTransform(
            ordering_nums=cfg.C,
            vocab=vocab
        )
    elif cfg.type == 'canonical':
        return LigandCanonicalMask(
            vocab=vocab,
            canonical=cfg.canonical
        )
    elif cfg.type == 'complete':
        return LigandCompleteMask(
            vocab=vocab
        )
    elif cfg.type == 'partial':
        return LigandPartialMask(
            vocab=vocab,
            C = cfg.C
        )
    elif cfg.type == 'random':
        return LigandRandomMask(
            min_ratio=cfg.min_ratio,
            max_ratio=cfg.max_ratio,
            min_num_masked=cfg.min_num_masked,
            min_num_unmasked=cfg.min_num_unmasked,
        )
    elif cfg.type == 'mixed':
        return LigandMixedMask(
            min_ratio=cfg.min_ratio,
            max_ratio=cfg.max_ratio,
            min_num_masked=cfg.min_num_masked,
            min_num_unmasked=cfg.min_num_unmasked,
            p_random=cfg.p_random,
            p_bfs=cfg.p_bfs,
            p_invbfs=cfg.p_invbfs,
        )
    elif cfg.type == 'all':
        return LigandMaskAll()
    else:
        raise NotImplementedError('Unknown mask: %s' % cfg.type)


def kabsch(A, B):
    # Input:
    #     Nominal  A Nx3 matrix of points
    #     Measured B Nx3 matrix of points
    # Returns R,t
    # R = 3x3 rotation matrix (B to A)
    # t = 3x1 translation vector (B to A)
    assert len(A) == len(B)
    N = A.shape[0]  # total points
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    # center the points
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))
    H = np.transpose(BB) * AA
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T * U.T
    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T * U.T
    t = -R * centroid_B.T + centroid_A.T
    return R, t
