import os
import pickle
import lmdb
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm
import numpy as np
import pandas as pd

from ..protein_ligand import PDBProtein, parse_sdf_file
from ..data import ProteinLigandData, torchify_dict
from ..mol_tree import MolTree


def reset_moltree_root(moltree, ligand_pos, protein_pos):
    ligand2 = np.sum(np.square(ligand_pos), 1, keepdims=True)
    protein2 = np.sum(np.square(protein_pos), 1, keepdims=True)
    dist = np.add(np.add(-2 * np.dot(ligand_pos, protein_pos.T), ligand2), protein2.T)
    min_dist = np.min(dist, 1)
    avg_min_dist = []
    for node in moltree.nodes:
        avg_min_dist.append(np.min(min_dist[node.clique]))
    root = np.argmin(avg_min_dist)
    if root > 0:
        moltree.nodes[0], moltree.nodes[root] = moltree.nodes[root], moltree.nodes[0]
    contact_idx = np.argmin(np.min(dist[moltree.nodes[0].clique], 0))
    contact_protein = torch.tensor(np.min(dist, 0) < 4 ** 2)

    return moltree, contact_protein, torch.tensor([contact_idx])


def from_protein_ligand_dicts(protein_dict=None, ligand_dict=None, residue_dict=None):
    instance = {}

    if protein_dict is not None:
        for key, item in protein_dict.items():
            instance['protein_' + key] = item

    if residue_dict is not None:
        for key, item in residue_dict.items():
            instance[key] = item

    if ligand_dict is not None:
        for key, item in ligand_dict.items():
            if key == 'moltree':
                instance['moltree'] = item
            else:
                instance['ligand_' + key] = item
    return instance


class PocketLigandPairDataset(Dataset):

    def __init__(self, raw_path, transform=None):
        super().__init__()
        self.raw_path = raw_path.rstrip('/')
        self.index_path = os.path.join(self.raw_path, 'index.pkl')
        self.processed_path = os.path.join(os.path.dirname(self.raw_path),
                                           os.path.basename(self.raw_path) + '_processed.lmdb')
        self.name2id_path = os.path.join(os.path.dirname(self.raw_path),
                                         os.path.basename(self.raw_path) + '_name2id.pt')
        self.transform = transform
        self.db = None

        self.keys = None

#print('pl', self.processed_path)
        if not os.path.exists(self.processed_path):
            self._process()
            self._precompute_name2id()

        self.name2id = torch.load(self.name2id_path)

    def _connect_db(self):
        """
            Establish read-only database connection
        """
        assert self.db is None, 'A connection has already been opened.'
        self.db = lmdb.open(
            self.processed_path,
            map_size=10 * (1024 * 1024 * 1024),  # 10GB
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.db.begin() as txn:
            self.keys = list(txn.cursor().iternext(values=False))

    def _close_db(self):
        self.db.close()
        self.db = None
        self.keys = None

    def _process(self):
        db = lmdb.open(
            self.processed_path,
            map_size=10 * (1024 * 1024 * 1024),  # 10GB
            create=True,
            subdir=False,
            readonly=False,  # Writable
        )
        with open(self.index_path, 'rb') as f:
            index = pickle.load(f)

        atoms = np.load('./atoms_hbond.npy', allow_pickle=True).item()
        # atoms = np.load('./utils/atoms_hbond.npy', allow_pickle=True).item()
        os.system("pwd")  
        print(len(atoms))
        num_skipped = 0
        with db.begin(write=True, buffers=True) as txn:
            for i, (pocket_fn, ligand_fn, _, rmsd_str) in enumerate(tqdm(index)):
#print('pl,liuxg',i)
#if i >1000:break
                if pocket_fn is None: continue
#if True:
                try:
                    pocket_dict = PDBProtein(os.path.join(self.raw_path, pocket_fn)).to_dict_atom()
                    residue_dict = PDBProtein(os.path.join(self.raw_path, pocket_fn)).to_dict_residue()
                    ligand_dict = parse_sdf_file(os.path.join(self.raw_path, ligand_fn))
                    ligand_dict['moltree'], pocket_dict['contact'], pocket_dict['contact_idx'] = reset_moltree_root(ligand_dict['moltree'],
                                                                                        ligand_dict['pos'],
                                                                                        pocket_dict['pos'])
                    if pocket_fn[:-3]+ligand_fn[:-3] in atoms:
                        ligand_dict['inter_atoms'] = atoms[pocket_fn[:-3]+ligand_fn[:-3]]
                    else:
                        ligand_dict['inter_atoms'] = []
                    data = from_protein_ligand_dicts(
                        protein_dict=torchify_dict(pocket_dict),
                        ligand_dict=torchify_dict(ligand_dict),
                        residue_dict=torchify_dict(residue_dict)
                    )
                    data['protein_filename'] = pocket_fn
                    data['ligand_filename'] = ligand_fn
                    txn.put(
                        key=str(i).encode(),
                        value=pickle.dumps(data)
                    )
                except:
#else:
                    num_skipped += 1
                    print('Skipping (%d) %s' % (num_skipped, ligand_fn,))
                    continue
        db.close()

    def _precompute_name2id(self):
        name2id = {}
        for i in tqdm(range(self.__len__()), 'Indexing'):
            try:
                data = self.__getitem__(i)
            except AssertionError as e:
                print(i, e)
                continue
            name = (data['protein_filename'], data['ligand_filename'])
            name2id[name] = i
        torch.save(name2id, self.name2id_path)

    def __len__(self):
        if self.db is None:
            self._connect_db()
        return len(self.keys)

    def __getitem__(self, idx):
        if self.db is None:
            self._connect_db()
        key = self.keys[idx]
        data = pickle.loads(self.db.begin().get(key))
        data['id'] = idx
        print("getitem, idx", idx)
#         print()
        assert data['protein_pos'].size(0) > 0
        if self.transform is not None:
            data = self.transform(data)
        return data





class PocketLigandPairDataset_for_contrastive_data(Dataset):

    def __init__(self, raw_path, transform=None):
        super().__init__()
        self.raw_path = raw_path.rstrip('/')
        self.index_path = os.path.join(self.raw_path, 'index.pkl')
        self.processed_path = os.path.join(os.path.dirname(self.raw_path),
                                           os.path.basename(self.raw_path) + '_processed.lmdb')
        self.name2id_path = os.path.join(os.path.dirname(self.raw_path),
                                         os.path.basename(self.raw_path) + '_name2id.pt')
        self.transform = transform
        self.db = None

        self.keys = None

#print('pl', self.processed_path)
        if not os.path.exists(self.processed_path):
            self._process()
            self._precompute_name2id()

        self.name2id = torch.load(self.name2id_path)

    def _connect_db(self):
        """
            Establish read-only database connection
        """
        assert self.db is None, 'A connection has already been opened.'
        self.db = lmdb.open(
            self.processed_path,
            map_size=10 * (1024 * 1024 * 1024),  # 10GB
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.db.begin() as txn:
            self.keys = list(txn.cursor().iternext(values=False))

    def _close_db(self):
        self.db.close()
        self.db = None
        self.keys = None

    def _process(self):
        db = lmdb.open(
            self.processed_path,
            map_size=10 * (1024 * 1024 * 1024),  # 10GB
            create=True,
            subdir=False,
            readonly=False,  # Writable
        )
        with open(self.index_path, 'rb') as f:
            index = pickle.load(f)

        atoms = np.load('./atoms_hbond.npy', allow_pickle=True).item()
        # atoms = np.load('./utils/atoms_hbond.npy', allow_pickle=True).item()
        os.system("pwd")  
        print(len(atoms))
        num_skipped = 0
        with db.begin(write=True, buffers=True) as txn:
            for i, (pocket_fn, ligand_fn, _, rmsd_str) in enumerate(tqdm(index)):
#print('pl,liuxg',i)
#if i >1000:break
                if pocket_fn is None: continue
#if True:
                try:
                    pocket_dict = PDBProtein(os.path.join(self.raw_path, pocket_fn)).to_dict_atom()
                    residue_dict = PDBProtein(os.path.join(self.raw_path, pocket_fn)).to_dict_residue()
                    ligand_dict = parse_sdf_file(os.path.join(self.raw_path, ligand_fn))
                    ligand_dict['moltree'], pocket_dict['contact'], pocket_dict['contact_idx'] = reset_moltree_root(ligand_dict['moltree'],
                                                                                        ligand_dict['pos'],
                                                                                        pocket_dict['pos'])
                    if pocket_fn[:-3]+ligand_fn[:-3] in atoms:
                        ligand_dict['inter_atoms'] = atoms[pocket_fn[:-3]+ligand_fn[:-3]]
                    else:
                        ligand_dict['inter_atoms'] = []
                    data = from_protein_ligand_dicts(
                        protein_dict=torchify_dict(pocket_dict),
                        ligand_dict=torchify_dict(ligand_dict),
                        residue_dict=torchify_dict(residue_dict)
                    )
                    data['protein_filename'] = pocket_fn
                    data['ligand_filename'] = ligand_fn
                    txn.put(
                        key=str(i).encode(),
                        value=pickle.dumps(data)
                    )
                except:
#else:
                    num_skipped += 1
                    print('Skipping (%d) %s' % (num_skipped, ligand_fn,))
                    continue
        db.close()

    def _precompute_name2id(self):
        name2id = {}
        for i in tqdm(range(self.__len__()), 'Indexing'):
            try:
                data = self.__getitem__(i)
            except AssertionError as e:
                print(i, e)
                continue
            name = (data['protein_filename'], data['ligand_filename'])
            name2id[name] = i
        torch.save(name2id, self.name2id_path)

    def __len__(self):
        if self.db is None:
            self._connect_db()
        return len(self.keys)

    def __getitem__(self, idx):
        if self.db is None:
            self._connect_db()
        key = self.keys[idx]
        data = pickle.loads(self.db.begin().get(key))
        data['id'] = idx
#         print("getitem, idx", idx)
#         print()
        assert data['protein_pos'].size(0) > 0
        if self.transform is not None:
            data = self.transform(data)
        while data is None:   
            print("data is None")   
            data = pickle.loads(self.db.begin().get(key))  
            data['id'] = idx  
            data = self.transform(data)   
        return data





class PocketLigandPairDataset_for_contrastive_data_Call(Dataset):

    def __init__(self, raw_path, transform=None):
        super().__init__()
        self.raw_path = raw_path.rstrip('/')
        self.index_path = os.path.join(self.raw_path, 'index.pkl')
        self.processed_path = os.path.join(os.path.dirname(self.raw_path),
                                           os.path.basename(self.raw_path) + '_processed.lmdb')
        self.name2id_path = os.path.join(os.path.dirname(self.raw_path),
                                         os.path.basename(self.raw_path) + '_name2id.pt')
        self.transform = transform
        self.db = None

        self.keys = None

#print('pl', self.processed_path)
        if not os.path.exists(self.processed_path):
            self._process()
            self._precompute_name2id()

        self.name2id = torch.load(self.name2id_path)

    def _connect_db(self):
        """
            Establish read-only database connection
        """
        assert self.db is None, 'A connection has already been opened.'
        self.db = lmdb.open(
            self.processed_path,
            map_size=10 * (1024 * 1024 * 1024),  # 10GB
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.db.begin() as txn:
            self.keys = list(txn.cursor().iternext(values=False))

    def _close_db(self):
        self.db.close()
        self.db = None
        self.keys = None

    def _process(self):
        db = lmdb.open(
            self.processed_path,
            map_size=10 * (1024 * 1024 * 1024),  # 10GB
            create=True,
            subdir=False,
            readonly=False,  # Writable
        )
        with open(self.index_path, 'rb') as f:
            index = pickle.load(f)

        atoms = np.load('./atoms_hbond.npy', allow_pickle=True).item()
        # atoms = np.load('./utils/atoms_hbond.npy', allow_pickle=True).item()
        os.system("pwd") 
        print(len(atoms))
        num_skipped = 0
        with db.begin(write=True, buffers=True) as txn:
            for i, (pocket_fn, ligand_fn, _, rmsd_str) in enumerate(tqdm(index)):
#print('pl,liuxg',i)
#if i >1000:break
                if pocket_fn is None: continue
#if True:
                try:
                    pocket_dict = PDBProtein(os.path.join(self.raw_path, pocket_fn)).to_dict_atom()
                    residue_dict = PDBProtein(os.path.join(self.raw_path, pocket_fn)).to_dict_residue()
                    ligand_dict = parse_sdf_file(os.path.join(self.raw_path, ligand_fn))
                    ligand_dict['moltree'], pocket_dict['contact'], pocket_dict['contact_idx'] = reset_moltree_root(ligand_dict['moltree'],
                                                                                        ligand_dict['pos'],
                                                                                        pocket_dict['pos'])
                    if pocket_fn[:-3]+ligand_fn[:-3] in atoms:
                        ligand_dict['inter_atoms'] = atoms[pocket_fn[:-3]+ligand_fn[:-3]]
                    else:
                        ligand_dict['inter_atoms'] = []
                    data = from_protein_ligand_dicts(
                        protein_dict=torchify_dict(pocket_dict),
                        ligand_dict=torchify_dict(ligand_dict),
                        residue_dict=torchify_dict(residue_dict)
                    )
                    data['protein_filename'] = pocket_fn
                    data['ligand_filename'] = ligand_fn
                    txn.put(
                        key=str(i).encode(),
                        value=pickle.dumps(data)
                    )
                except:
#else:
                    num_skipped += 1
                    print('Skipping (%d) %s' % (num_skipped, ligand_fn,))
                    continue
        db.close()

    def _precompute_name2id(self):
        name2id = {}
        for i in tqdm(range(self.__len__()), 'Indexing'):
            try:
                data = self.__getitem__(i)
            except AssertionError as e:
                print(i, e)
                continue
            name = (data['protein_filename'], data['ligand_filename'])
            name2id[name] = i
        torch.save(name2id, self.name2id_path)

    def __len__(self):
        if self.db is None:
            self._connect_db()
        return len(self.keys)

    def __getitem__(self, idx):
        if self.db is None:
            self._connect_db()
        key = self.keys[idx]
        data = pickle.loads(self.db.begin().get(key))
        data['id'] = idx
#         print("getitem, idx", idx)
#         print()
        assert data['protein_pos'].size(0) > 0
        if self.transform is not None:
            datas, data_baselines, num_motifs, docked_failure = self.transform(data)
        return datas, data_baselines, num_motifs, docked_failure



class PocketLigandPairCompleteDataset(Dataset):

    def __init__(self, raw_path, transform=None):
        super().__init__()
        self.raw_path = raw_path.rstrip('/')
        self.index_path = os.path.join(self.raw_path, 'index.pkl')
        self.processed_path = os.path.join(os.path.dirname(self.raw_path),
                                           os.path.basename(self.raw_path) + '_processed.lmdb')
        self.name2id_path = os.path.join(os.path.dirname(self.raw_path),
                                         os.path.basename(self.raw_path) + '_name2id.pt')
        self.transform = transform
        self.db = None

        self.keys = None

#print('pl', self.processed_path)
        if not os.path.exists(self.processed_path):
            self._process()
            self._precompute_name2id()

        self.name2id = torch.load(self.name2id_path)

    def _connect_db(self):
        """
            Establish read-only database connection
        """
        assert self.db is None, 'A connection has already been opened.'
        self.db = lmdb.open(
            self.processed_path,
            map_size=10 * (1024 * 1024 * 1024),  # 10GB
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.db.begin() as txn:
            self.keys = list(txn.cursor().iternext(values=False))

    def _close_db(self):
        self.db.close()
        self.db = None
        self.keys = None

    def _process(self):
        db = lmdb.open(
            self.processed_path,
            map_size=10 * (1024 * 1024 * 1024),  # 10GB
            create=True,
            subdir=False,
            readonly=False,  # Writable
        )
        with open(self.index_path, 'rb') as f:
            index = pickle.load(f)

        atoms = np.load('./atoms_hbond.npy', allow_pickle=True).item()
        # atoms = np.load('./utils/atoms_hbond.npy', allow_pickle=True).item()
        os.system("pwd")
        print(len(atoms))
        num_skipped = 0
        with db.begin(write=True, buffers=True) as txn:
            for i, (pocket_fn, ligand_fn, _, rmsd_str) in enumerate(tqdm(index)):
#print('pl,liuxg',i)
#if i >1000:break
                if pocket_fn is None: continue
#if True:
                try:
                    pocket_dict = PDBProtein(os.path.join(self.raw_path, pocket_fn)).to_dict_atom()
                    residue_dict = PDBProtein(os.path.join(self.raw_path, pocket_fn)).to_dict_residue()
                    ligand_dict = parse_sdf_file(os.path.join(self.raw_path, ligand_fn))
                    ligand_dict['moltree'], pocket_dict['contact'], pocket_dict['contact_idx'] = reset_moltree_root(ligand_dict['moltree'],
                                                                                        ligand_dict['pos'],
                                                                                        pocket_dict['pos'])
                    if pocket_fn[:-3]+ligand_fn[:-3] in atoms:
                        ligand_dict['inter_atoms'] = atoms[pocket_fn[:-3]+ligand_fn[:-3]]
                    else:
                        ligand_dict['inter_atoms'] = []
                    data = from_protein_ligand_dicts(
                        protein_dict=torchify_dict(pocket_dict),
                        ligand_dict=torchify_dict(ligand_dict),
                        residue_dict=torchify_dict(residue_dict)
                    )
                    data['protein_filename'] = pocket_fn
                    data['ligand_filename'] = ligand_fn
                    txn.put(
                        key=str(i).encode(),
                        value=pickle.dumps(data)
                    )
                except:
#else:
                    num_skipped += 1
                    print('Skipping (%d) %s' % (num_skipped, ligand_fn,))
                    continue
        db.close()

    def _precompute_name2id(self):
        name2id = {}
        for i in tqdm(range(self.__len__()), 'Indexing'):
            try:
                data = self.__getitem__(i)
            except AssertionError as e:
                print(i, e)
                continue
            name = (data['protein_filename'], data['ligand_filename'])
            name2id[name] = i
        torch.save(name2id, self.name2id_path)

    def __len__(self):
        if self.db is None:
            self._connect_db()
        return len(self.keys)

    def __getitem__(self, idx):
        if self.db is None:
            self._connect_db()
        key = self.keys[idx]
        data = pickle.loads(self.db.begin().get(key))
        data['id'] = idx
#         print("getitem, idx", idx)
#         print()
        assert data['protein_pos'].size(0) > 0
        if self.transform is not None:
            datas, pre_suc_data_num, docked_failure = self.transform(data)
        return datas, pre_suc_data_num, docked_failure




class PocketLigandPairPartialDataset(Dataset):

    def __init__(self, raw_path, transform=None):
        super().__init__()
        self.raw_path = raw_path.rstrip('/')
        self.index_path = os.path.join(self.raw_path, 'index.pkl')
        self.processed_path = os.path.join(os.path.dirname(self.raw_path),
                                           os.path.basename(self.raw_path) + '_processed.lmdb')
        self.name2id_path = os.path.join(os.path.dirname(self.raw_path),
                                         os.path.basename(self.raw_path) + '_name2id.pt')
        self.transform = transform
        self.db = None

        self.keys = None

#print('pl', self.processed_path)
        if not os.path.exists(self.processed_path):
            self._process()
            self._precompute_name2id()

        self.name2id = torch.load(self.name2id_path)

    def _connect_db(self):
        """
            Establish read-only database connection
        """
        assert self.db is None, 'A connection has already been opened.'
        self.db = lmdb.open(
            self.processed_path,
            map_size=10 * (1024 * 1024 * 1024),  # 10GB
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.db.begin() as txn:
            self.keys = list(txn.cursor().iternext(values=False))

    def _close_db(self):
        self.db.close()
        self.db = None
        self.keys = None

    def _process(self):
        db = lmdb.open(
            self.processed_path,
            map_size=10 * (1024 * 1024 * 1024),  # 10GB
            create=True,
            subdir=False,
            readonly=False,  # Writable
        )
        with open(self.index_path, 'rb') as f:
            index = pickle.load(f)

        atoms = np.load('./atoms_hbond.npy', allow_pickle=True).item()
        # atoms = np.load('./utils/atoms_hbond.npy', allow_pickle=True).item()
        os.system("pwd")
        print(len(atoms))
        num_skipped = 0
        with db.begin(write=True, buffers=True) as txn:
            for i, (pocket_fn, ligand_fn, _, rmsd_str) in enumerate(tqdm(index)):
#print('pl,liuxg',i)
#if i >1000:break
                if pocket_fn is None: continue
#if True:
                try:
                    pocket_dict = PDBProtein(os.path.join(self.raw_path, pocket_fn)).to_dict_atom()
                    residue_dict = PDBProtein(os.path.join(self.raw_path, pocket_fn)).to_dict_residue()
                    ligand_dict = parse_sdf_file(os.path.join(self.raw_path, ligand_fn))
                    ligand_dict['moltree'], pocket_dict['contact'], pocket_dict['contact_idx'] = reset_moltree_root(ligand_dict['moltree'],
                                                                                        ligand_dict['pos'],
                                                                                        pocket_dict['pos'])
                    if pocket_fn[:-3]+ligand_fn[:-3] in atoms:
                        ligand_dict['inter_atoms'] = atoms[pocket_fn[:-3]+ligand_fn[:-3]]
                    else:
                        ligand_dict['inter_atoms'] = []
                    data = from_protein_ligand_dicts(
                        protein_dict=torchify_dict(pocket_dict),
                        ligand_dict=torchify_dict(ligand_dict),
                        residue_dict=torchify_dict(residue_dict)
                    )
                    data['protein_filename'] = pocket_fn
                    data['ligand_filename'] = ligand_fn
                    txn.put(
                        key=str(i).encode(),
                        value=pickle.dumps(data)
                    )
                except:
#else:
                    num_skipped += 1
                    print('Skipping (%d) %s' % (num_skipped, ligand_fn,))
                    continue
        db.close()

    def _precompute_name2id(self):
        name2id = {}
        for i in tqdm(range(self.__len__()), 'Indexing'):
            try:
                data = self.__getitem__(i)
            except AssertionError as e:
                print(i, e)
                continue
            name = (data['protein_filename'], data['ligand_filename'])
            name2id[name] = i
        torch.save(name2id, self.name2id_path)

    def __len__(self):
        if self.db is None:
            self._connect_db()
        return len(self.keys)

    def __getitem__(self, idx):
        if self.db is None:
            self._connect_db()
        key = self.keys[idx]
        data = pickle.loads(self.db.begin().get(key))
        data['id'] = idx
        print("getitem, idx", idx)
#         print()
        assert data['protein_pos'].size(0) > 0
        if self.transform is not None:
            datas, pre_suc_data_num, connected_subgraph_data_num, docked_failure = self.transform(data)
        return datas, pre_suc_data_num, connected_subgraph_data_num, docked_failure





class PLDataset(Dataset):
    def __init__(self, data_file_path, transform=None, target_transform=None):
        self.data_file_path = data_file_path
        self.data_files = pd.read_csv(data_file_path)
        self.transform = transform
        self.target_transform = target_transform
        

    def __len__(self):
        #if 'val_data' in self.data_file_path:
            #return 10000
        return len(self.data_files)
    
    
    def __getitem__(self, idx):
        return torch.load(self.data_files.iloc[idx].values[0])



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    args = parser.parse_args()

    PocketLigandPairDataset(args.path)
