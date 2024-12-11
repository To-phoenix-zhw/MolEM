import os
import shutil
import argparse
import random
import torch
import numpy as np
import math
import rmsd
import time
from torch_geometric.data import Batch
from tqdm.auto import tqdm
from rdkit import Chem
from rdkit.Geometry import Point3D
from torch.utils.data import DataLoader
from rdkit.Chem.rdchem import BondType
from rdkit.Chem import ChemicalFeatures, rdMolDescriptors
from rdkit import RDConfig
from rdkit.Chem.Descriptors import MolLogP, qed

from models.OrderingGenerator import OrderingGenerator
from models.common import compose_context_stable
from utils.transforms import *
from utils.datasets import get_dataset
from utils.datasets.pl import PLDataset
from utils.misc import *
from utils.data import *
from utils.mol_tree import *
from utils.chemutils import *
from utils.dihedral_utils import *
from utils.sascorer import compute_sa_score
from utils.docking import *  




def ordering_gen(batch_size, batch, model):
    generated_ordering = {}   
    gen_motif_atoms = {}   
    finished = torch.zeros(batch_size).bool()  
    
    while finished.all() != True:   
        print("finished", finished)
        selected_pools = model(batch_size=batch_size, protein_pos=batch['protein_pos'],
                                protein_atom_feature=batch['protein_atom_feature'].float(),
                                ligand_pos=batch['ligand_context_pos'],
                                ligand_atom_feature=batch['ligand_atom_feature_full'].float(),
                                batch_protein=batch['protein_element_batch'],
                                batch_ligand=batch['ligand_element_batch'], batch=batch)
        for d in range(batch_size):
            print("data %d in batch" % d)
            if finished[d] == True:  
                print("finished")
                continue
            select_pool = selected_pools[d]
            print("select_pool", select_pool)
            if d in generated_ordering.keys():   
                already_generated_orders = generated_ordering[d]  
                print("already_generated_orderings", already_generated_orders)
                cur_motif_atom_index = batch['motif_atom_index'][batch['motif_atoms_batch']==d]
                if len(already_generated_orders) == cur_motif_atom_index.max().item(): 
                    cur_index = (set(range(cur_motif_atom_index.max().item() + 1)) - set(already_generated_orders)).pop()
                    generated_ordering[d].append(cur_index)  
                    finished[d] = True
                    print("add the last motif", cur_index)
                    continue
                    
                if len(already_generated_orders) == cur_motif_atom_index.max().item() + 1:  
                    finished[d] = True
                    print("total motifs", cur_motif_atom_index.max().item() + 1)
                    continue

                select_state = False
                for select_index in select_pool.indices:   
                    cur_index = select_index.item()
                    print("cur_index", cur_index)
                    if cur_index in already_generated_orders:   
                        print("already generated")
                        continue
                    # Connectivity
                    cur_comb = already_generated_orders.copy()
                    cur_comb.append(cur_index)   
                    complete_G = batch['ligand_graphs'][d]
                    cur_subgraph = complete_G.subgraph(cur_comb)   
                    if nx.is_connected(cur_subgraph) == False:   
                        print("not connected")
                        continue
 
                    generated_ordering[d].append(cur_index)
                    print("add_motif", cur_index)
                    select_state = True
                    break
                if select_state == False:  
                    print("invalid")
                    finished[d] = True
                    del generated_ordering[d]  
            else:   
                if finished[d] == True:
                    continue
                else:
                    print("already_generated_orderings", [])
                    print("add_motif", select_pool.indices[0].item())
                    generated_ordering[d] = [select_pool.indices[0].item()]  
            # print("-"*8)


        for d in range(batch_size):
            if finished[d] == True:
                continue
            cur_motif_atom_index = batch['motif_atom_index'][batch['motif_atoms_batch']==d]
            cur_motif_atoms = batch['motif_atoms'][batch['motif_atoms_batch']==d]
                
            gen_motif = generated_ordering[d]
            nid = gen_motif[-1]  

            # print(gen_motif)
            if d in gen_motif_atoms.keys():
                gen_motif_atoms[d] = torch.cat((gen_motif_atoms[d], cur_motif_atoms[cur_motif_atom_index==nid])) 
            else: 
                gen_motif_atoms[d] = cur_motif_atoms[cur_motif_atom_index==nid] 

            cur_data_ligand_atom_feature_full = batch['ligand_atom_feature_full'][batch['ligand_element_batch'] == d]
            gen_signal = torch.zeros([cur_data_ligand_atom_feature_full.shape[0]])
            gen_signal[gen_motif_atoms[d]] = 1
            cur_data_ligand_atom_feature_full[:, -1] = gen_signal   
            batch['ligand_atom_feature_full'][batch['ligand_element_batch'] == d] = cur_data_ligand_atom_feature_full
            # print(gen_motif_atoms[d])
            # print(gen_signal)
            # print(batch['ligand_atom_feature_full'][batch['ligand_element_batch'] == d])
        #     break
    
    return generated_ordering  
        




if __name__ == "__main__":
    time_start = time.time()  

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/sampleOrdering.yml')
    parser.add_argument('--data_root', type=str, default='./em/')
    parser.add_argument('--dataset', type=str, default='training') 
    parser.add_argument('--begin', type=int, default=0)  
    parser.add_argument('--end', type=int, default=100)   
    parser.add_argument('--device', type=str, default='cuda:0')  
    parser.add_argument('--outdir', type=str, default='./em/ordering_predict/0/')
    parser.add_argument('--vocab_path', type=str, default='./utils/vocab.txt')
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
    ordering_data_path = args.data_root + args.dataset + '/'
    data_dir_lst = os.listdir(ordering_data_path)
    data_dir_lst.remove('pre_orderings.csv')
    data_dir_lst.sort(key=lambda l: int(l))
    # print("data_dir_lst", data_dir_lst)

    # Model (Main)
    print(f'Loading main model...\n {args.ckpt}')
    # ckpt = torch.load(config.model.checkpoint, map_location=args.device)
    ckpt = torch.load(args.ckpt, map_location=args.device)
    model = OrderingGenerator(
        config.model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim,
        vocab=vocab,
        device=args.device).to(args.device)
    model.load_state_dict(ckpt['model'])


    molecule_generated_orderings = []
    with torch.no_grad():
        model.eval()
        for i in range(args.begin, args.end, config.sample.batch_size):
            print("i", i)
            batch_mol_ids = []
            data_paths = []
            mol_dicts = []
            for j in range(i, i+config.sample.batch_size):
                print("j", j)
                print(len(data_dir_lst))
                print("args.end", args.end)
                print("int(data_dir_lst[args.end-1])", int(data_dir_lst[args.end-1]))
                if j >= len(data_dir_lst) or int(data_dir_lst[j]) > int(data_dir_lst[args.end-1]):   
                    print("break")
                    break
                print("data_dir_lst[j]", data_dir_lst[j], type(data_dir_lst[j]))
                cur_mol_id = str(data_dir_lst[j])
                batch_mol_ids.append(cur_mol_id)
                cur_ordering_id = str(0)  
                data_path = ordering_data_path + cur_mol_id + '/' + cur_ordering_id + '/0.pt'  
                data_paths.append(data_path) 
                mol_dicts.append(torch.load(data_path))
            print("batch_mol_ids", batch_mol_ids)
            cur_batch_size = len(batch_mol_ids) 
            if cur_batch_size == 0:
                break
            batch = collate_mols_ordering(mol_dicts, vocab.size())
            for key in batch:
                if isinstance(batch[key], list):
                    continue
            #     print(batch[key])
                batch[key] = batch[key].to(args.device)
            generated_ordering = ordering_gen(cur_batch_size, batch, model)
            for batch_index, gen_ordering in generated_ordering.items():
                cur_dict = {"mol_id": batch_mol_ids[batch_index], "ordering": gen_ordering}
                molecule_generated_orderings.append(cur_dict)

    mksure_path(args.outdir + args.dataset)
    pd.DataFrame(molecule_generated_orderings).to_csv(args.outdir + args.dataset + '/orderings.csv', index=False) 
    
    time_end = time.time()  
    time_sum = time_end - time_start   
    print(time_sum)
