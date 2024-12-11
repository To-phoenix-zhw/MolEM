import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import shutil
import argparse
from tqdm.auto import tqdm
import torch
from torch.nn.utils import clip_grad_norm_
import torch.utils.tensorboard
from kmeans_pytorch import kmeans
import numpy as np
from models.MoleculeGenerator import MoleculeGenerator
from utils.datasets import *
from utils.misc import *
from utils.train import *
from utils.data import *
from utils.mol_tree import *
from utils.transforms import *
from torch.utils.data import DataLoader

def mksure_path(dirs_or_files):
    if not os.path.exists(dirs_or_files):
        os.makedirs(dirs_or_files)


parser = argparse.ArgumentParser()
parser.add_argument('--begin', type=int, default=0)
parser.add_argument('--end', type=int, default=100)
parser.add_argument('--data_root', type=str, default='./em/ordering_predict/0/')
parser.add_argument('--vocab_path', type=str, default='./utils/vocab.txt')
parser.add_argument('--dataset', type=str, default='training')  
parser.add_argument('--config', type=str, default='./configs/ordering_prediction_construction.yml')
args = parser.parse_args()
print(args)

# Load vocab
vocab = []
for line in open(args.vocab_path):
    p, _, _ = line.partition(':')
    vocab.append(p)
print(len(vocab))
vocab = Vocab(vocab)


# Load configs
config = load_config(args.config)
print(config)
config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
seed_all(config.train.seed)


# Transforms
protein_featurizer = FeaturizeProteinAtom()
ligand_featurizer = FeaturizeLigandAtom()
transform_basic = Compose([
    LigandCountNeighbors(),
    protein_featurizer,
    ligand_featurizer,
    FeaturizeLigandBond(),
])
masking = get_mask(config.train.transform.mask, vocab)


# Datasets
# logger.info('Loading dataset...')
dataset, subsets = get_dataset(config=config.dataset, transform=transform_basic)  
train_set, val_set = subsets['train'], subsets['val']
ordering_path = args.data_root + args.dataset + '/orderings.csv'
ordering_pd = pd.read_csv(ordering_path)



# C=10
if args.dataset == 'training':
    train_total_data = 0
    train_total_docked_failure = 0
    train_subgraph_num = 0
    train_factual_num = 0
    print("Preparing training data.")
    for i in range(args.begin, args.end):
        print("i", i)
        order_idx_data = ordering_pd[ordering_pd['mol_id'] == i]
        if len(order_idx_data) == 0:
            train_total_docked_failure += 1 
            continue
        cur_ordering = eval(order_idx_data['ordering'].item())
        cur_data = train_set[i]  # transform
        datas_i, pre_suc_data_num_i, connected_subgraph_data_num_i, docked_failure_i = masking(cur_data, cur_ordering)
        train_subgraph_num += connected_subgraph_data_num_i
        train_total_data += pre_suc_data_num_i
        train_total_docked_failure += docked_failure_i
        print("i, pre_suc_data_num_i, connected_subgraph_data_num_i, docked_failure_i", i, pre_suc_data_num_i, connected_subgraph_data_num_i, docked_failure_i)
        for j in range(len(datas_i)):
            data_ij = datas_i[j]
            if data_ij is None:
                continue
            cur_node_nums = data_ij['visible_nodes_num']
            cur_subgraph_id = data_ij['subgraph_id']
            mksure_path(args.data_root + "training/")
            torch.save(data_ij, args.data_root + "training/%d_%d_%d_%d.pt"%(i,j,cur_node_nums,cur_subgraph_id))  
            train_factual_num += 1
        print("train_factual_num", train_factual_num)
        
    print("train_total_data, train_subgraph_num, train_total_docked_failure", train_total_data, train_subgraph_num, train_total_docked_failure)



if args.dataset == 'validation':
    val_total_data = 0
    val_total_docked_failure = 0
    val_subgraph_num = 0
    val_factual_num = 0
    print("Preparing validation data.")
    for i in range(args.begin, args.end):
        print("i", i)
        order_idx_data = ordering_pd[ordering_pd['mol_id'] == i]
        if len(order_idx_data) == 0:
            val_total_docked_failure += 1  
            continue
        cur_ordering = eval(order_idx_data['ordering'].item())
        cur_data = val_set[i]  # transform
        datas_i, pre_suc_data_num_i, connected_subgraph_data_num_i, docked_failure_i = masking(cur_data, cur_ordering)
        val_total_data += pre_suc_data_num_i
        val_subgraph_num += connected_subgraph_data_num_i
        val_total_docked_failure += docked_failure_i
        print("i, pre_suc_data_num_i, connected_subgraph_data_num_i, docked_failure_i", i, pre_suc_data_num_i, connected_subgraph_data_num_i, docked_failure_i)
        for j in range(len(datas_i)):
            data_ij = datas_i[j]
            if data_ij is None:
                continue
            cur_node_nums = data_ij['visible_nodes_num']
            cur_subgraph_id = data_ij['subgraph_id']
            mksure_path(args.data_root + "validation/")
            torch.save(data_ij, args.data_root +"validation/%d_%d_%d_%d.pt"%(i,j,cur_node_nums,cur_subgraph_id))
            val_factual_num += 1
        print("val_factual_num", val_factual_num)

    print("val_total_data, val_subgraph_num, val_total_docked_failure", val_total_data, val_subgraph_num, val_total_docked_failure)

print("Done.")
