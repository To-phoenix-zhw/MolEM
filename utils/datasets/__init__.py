import torch
from torch.utils.data import Subset
from .pl import PocketLigandPairDataset, PocketLigandPairDataset_for_contrastive_data, PocketLigandPairDataset_for_contrastive_data_Call, PLDataset, PocketLigandPairCompleteDataset, PocketLigandPairPartialDataset
import random

def get_dataset(config, *args, **kwargs):
    name = config.name  # pl
    if name == 'pl':
        root = config.path  # ./data/crossdocked_pocket10
        dataset = PocketLigandPairDataset(root, *args, **kwargs)
    elif name == 'pl_prepared': 
        if 'train_path' in config:
            train_files = config.train_path
            val_files = config.val_path
        else:
            train_files = kwargs['train_file_path']  
            val_files = kwargs['val_file_path']  
        train_dataset = PLDataset(train_files)
        val_dataset = PLDataset(val_files)
        return train_dataset, val_dataset
    else:
        raise NotImplementedError('Unknown dataset: %s' % name)
    
    if 'split' in config:   
        split_by_name = torch.load(config.split) 
        split = {k: [dataset.name2id[n] for n in names if n in dataset.name2id] for k, names in split_by_name.items()} 
        print("split len", len(split["train"]), len(split["test"]), len(split["val"]))  # 
        subsets = {k:Subset(dataset, indices=v) for k, v in split.items()}
        return dataset, subsets
    else:
        return dataset
