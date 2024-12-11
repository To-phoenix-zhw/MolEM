import os
import shutil
import argparse
import random
import torch
import numpy as np
import math
import rmsd
from torch_geometric.data import Batch
from tqdm.auto import tqdm
from rdkit import Chem
from rdkit.Geometry import Point3D
from torch.utils.data import DataLoader
from rdkit.Chem.rdchem import BondType
from rdkit.Chem import ChemicalFeatures, rdMolDescriptors
from rdkit import RDConfig
from rdkit.Chem.Descriptors import MolLogP, qed

from models.MoleculeGenerator import MoleculeGenerator
from utils.transforms import *
from utils.datasets import get_dataset
from utils.misc import *
from utils.data import *
from utils.mol_tree import *
from utils.chemutils import *
from utils.dihedral_utils import *
from utils.sascorer import compute_sa_score
from models.common import compose_context_stable
from torch_scatter import scatter_add, scatter_mean


parser = argparse.ArgumentParser()
parser.add_argument('--begin', type=int, default=0)
parser.add_argument('--end', type=int, default=100)
parser.add_argument('--data_root', type=str, default='./em/pre_orderings/')
parser.add_argument('--vocab_path', type=str, default='./utils/vocab.txt')
parser.add_argument('--device', type=str, default='cuda:0') 
parser.add_argument('--config', type=str, default='./configs/sampleMolecule.yml')
parser.add_argument('--outdir', type=str, default='./em/ordering_score/0/')
parser.add_argument('--dataset', type=str, default='training')  
parser.add_argument('--ckpt', type=str, default='')
args = parser.parse_args()
print(args)

# Load vocab
vocab = []
for line in open(args.vocab_path):
    p1, _, p3 = line.partition(':')
    vocab.append(p1)
vocab = Vocab(vocab)

# Load configs
config = load_config(args.config)
config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
seed_all(config.sample.seed)

# Data
protein_featurizer = FeaturizeProteinAtom()
ligand_featurizer = FeaturizeLigandAtom()

# Model (Main)
ckpt = torch.load(args.ckpt, map_location=args.device)
model = MoleculeGenerator(
    config.model,
    protein_atom_feature_dim=protein_featurizer.feature_dim,
    ligand_atom_feature_dim=ligand_featurizer.feature_dim,
    vocab=vocab,
    device=args.device).to(args.device)
model.load_state_dict(ckpt['model'])


with torch.no_grad():
    model.eval()
    ordering_path = args.data_root + args.dataset + '/pre_orderings.csv'
    ordering_pd = pd.read_csv(ordering_path)

    mol_ordering = []
    for i in range(args.begin, args.end):
        order_idx_data = ordering_pd[ordering_pd['mol_id'] == i]
        if len(order_idx_data) == 0:
            continue
        # print("cur_molecule", i)
        cur_orderings = eval(order_idx_data['orderings'].item())
        ordering_probabilities = []
        log_ordering_probabilities = []
        for o_idx in range(len(cur_orderings)):
            cur_ordering = cur_orderings[o_idx]
            # print("cur_ordering", o_idx, cur_ordering)
            data_dir = args.data_root + 'pre_orderings/' + args.dataset + "/" + str(i) + "/" + str(o_idx) + "/"
            data_paths = []
            for root, dirs, files in os.walk(data_dir):
                data_paths = files
            
            ordering_valid = True
            for d_idx in range(len(data_paths)):
                check_valid = data_paths[d_idx].split('.')[0].split('_')[1] == data_paths[d_idx].split('.')[0].split('_')[2]
                if check_valid == False:  # unreasonable conformation
                    ordering_valid = False
                    break
            if len(data_paths) == 0 or ordering_valid == False:  # No data, or not a correct order.
                ordering_probabilities.append(float('-inf'))
                log_ordering_probabilities.append(float('-inf'))
                continue
            data_paths.sort(key=lambda l: int(l.split('.')[0].split('_')[1]))
            step_probabilities = []
            focus_probabilities= []
            motif_probabilities = []
            attach_probabilities = []
            for step, data_path in enumerate(data_paths):
                load_data_path = data_dir + data_path
                cur_load_data = torch.load(load_data_path)
                cur_batch = collate_mols_learning([cur_load_data], vocab.size())
                for key in cur_batch:
                    cur_batch[key] = cur_batch[key].to(args.device)
                focus_probability, motif_probability, attach_probability = model.get_probability(
                    protein_pos=cur_batch['protein_pos'],
                    protein_atom_feature=cur_batch['protein_atom_feature'].float(),
                    ligand_pos=cur_batch['ligand_context_pos'],
                    ligand_atom_feature=cur_batch['ligand_context_feature_full'].float(),
                    batch_protein=cur_batch['protein_element_batch'],
                    batch_ligand=cur_batch['ligand_context_element_batch'],
                    batch=cur_batch)
                # step probability
                step_probability = focus_probability * motif_probability * attach_probability
                step_probabilities.append(step_probability)
                focus_probabilities.append(focus_probability)
                motif_probabilities.append(motif_probability)
                attach_probabilities.append(attach_probability)
                print("step", step)
                print("step_probability", step_probability)
                print("focus_probability, motif_probability, attach_probability", focus_probability, motif_probability, attach_probability)
            ordering_probability = torch.prod(torch.tensor(step_probabilities)).item()
            log_ordering_probability = torch.sum(torch.log(torch.tensor(step_probabilities))).item()
            print("ordering_probability", ordering_probability)
            print("log_ordering_probability", log_ordering_probability)
            ordering_probabilities.append(ordering_probability)
            log_ordering_probabilities.append(log_ordering_probability)
            print("-"*8)
        print("ordering_probabilities", ordering_probabilities)
        print("log_ordering_probabilities", log_ordering_probabilities)
        inf_mask = np.logical_not(np.isinf(log_ordering_probabilities))
        if sum(inf_mask) == 0:  # All the docks failed.
            print("all invalid!!!")
        else:
            assert len(log_ordering_probabilities) == len(cur_orderings)
            assert len(ordering_probabilities) == len(cur_orderings)
            max_log_probability = max(log_ordering_probabilities) 
            max_ordering_idx = log_ordering_probabilities.index(max_log_probability) 
            print("max_ordering_idx", max_ordering_idx)
            print("max_probability", ordering_probabilities[max_ordering_idx], max_log_probability)
            cur_dict = {'mol_id': i, 'ordering_idx': max_ordering_idx, 'ordering': cur_orderings[max_ordering_idx]}
            print(cur_dict)
            mol_ordering.append(cur_dict)
        print("*"*8)
    save_path = args.outdir + args.dataset + "/"
    mksure_path(save_path)
    pd.DataFrame(mol_ordering).to_csv(save_path + str(args.begin) + '_' + str(args.end) + ".csv", index=False)
