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
from utils.docking import *  

_fscores = None

ATOM_FAMILIES = ['Acceptor', 'Donor', 'Aromatic', 'Hydrophobe', 'LumpedHydrophobe', 'NegIonizable', 'PosIonizable',
                 'ZnBinder']
ATOM_FAMILIES_ID = {s: i for i, s in enumerate(ATOM_FAMILIES)}

STATUS_RUNNING = 'running'
STATUS_FINISHED = 'finished'
STATUS_FAILED = 'failed'



from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
IPythonConsole.ipython_3d = True
import py3Dmol
from rdkit.Chem import rdDepictor
from rdkit.Chem import rdDistGeom
import rdkit
print(rdkit.__version__)



from meeko import MoleculePreparation
from meeko import PDBQTMolecule
from meeko import RDKitMolCreate


import time


def mksure_path(dirs_or_files):
    if not os.path.exists(dirs_or_files):
        os.makedirs(dirs_or_files)


def get_feat(mol):
    fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
    atomic_numbers = torch.LongTensor([6, 7, 8, 9, 15, 16, 17])  # C N O F P S Cl
    ptable = Chem.GetPeriodicTable()
    Chem.SanitizeMol(mol)
    feat_mat = np.zeros([mol.GetNumAtoms(), len(ATOM_FAMILIES)], dtype=np.int_)  
    for feat in factory.GetFeaturesForMol(mol):
        feat_mat[feat.GetAtomIds(), ATOM_FAMILIES_ID[feat.GetFamily()]] = 1
    ligand_element = torch.tensor([ptable.GetAtomicNumber(atom.GetSymbol()) for atom in mol.GetAtoms()])
    element = ligand_element.view(-1, 1) == atomic_numbers.view(1, -1)  # (N_atoms, N_elements)
    return torch.cat([element, torch.tensor(feat_mat)], dim=-1).float()



def get_file(file_path: str, suffix: str, res_file_path: list) -> list:
    for file in os.listdir(file_path):

        if os.path.isdir(os.path.join(file_path, file)):
            get_file(os.path.join(file_path, file), suffix, res_file_path)
        else:
            res_file_path.append(os.path.join(file_path, file))

    return res_file_path if suffix == '' or suffix is None else list(filter(lambda x: x.endswith(suffix), res_file_path))



def atom_equal(a1, a2):
    return a1.GetSymbol() == a2.GetSymbol() and a1.GetFormalCharge() == a2.GetFormalCharge()



def get_random_id(length=30):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length)) 




def find_reference(protein_pos, focal_id):
    # Select three reference protein atoms
    d = torch.norm(protein_pos - protein_pos[focal_id], dim=1)
    reference_idx = torch.topk(d, k=4, largest=False)[1]
    reference_pos = protein_pos[reference_idx]
    return reference_pos, reference_idx


def SetAtomNum(mol, atoms):
    for atom in mol.GetAtoms():
        if atom.GetIdx() in atoms:
            atom.SetAtomMapNum(1)
        else:
            atom.SetAtomMapNum(0)
    return mol


def SetMolPos(mol_list, pos_list):
    for i in range(len(pos_list)):
        mol = mol_list[i]
        conf = mol.GetConformer()
        pos = np.array(pos_list[i].cpu())
        if mol.GetNumAtoms() == len(pos):
            for node in range(mol.GetNumAtoms()):
                conf.SetAtomPosition(node, pos[node])
    return mol_list


def lipinski(mol):
    count = 0
    if qed(mol) <= 5:
        count += 1
    if Chem.Lipinski.NumHDonors(mol) <= 5:
        count += 1
    if Chem.Lipinski.NumHAcceptors(mol) <= 10:
        count += 1
    if Chem.Descriptors.ExactMolWt(mol) <= 500:
        count += 1
    if Chem.Lipinski.NumRotatableBonds(mol) <= 5:
        count += 1
    return count



def show_mol(mol):
#     print('\t'.join(['id', 'num', 'symbol', 'MapNum', 'degree', 'charge', 'hybrid', 'TotalNumHs']))
    display(mol)
    print('\t'.join(['id', 'num', 'symbol', 'MapNum', 'degree', 'charge', 'hybrid']))
    for atom in mol.GetAtoms():
        print(atom.GetIdx(), end='\t')  # id
        print(atom.GetAtomicNum(), end='\t')  # num
        print(atom.GetSymbol(), end='\t')  # symbol
        print(atom.GetAtomMapNum(), end='\t')  # MapNum
        print(atom.GetDegree(), end='\t')  # degree
        print(atom.GetFormalCharge(), end='\t')  # charge
        print(atom.GetHybridization())  # hybrid 
#         print(atom.GetTotalNumHs())  # TotalNumHs



def ligand_gen(batch, model, vocab, config, center, argsckpt, argsdata_id, batch_num, protein_filename, tmp_dir):
    pos_list = []
    feat_list = []
    vina_list = [] 
    mol_list = []
    motif_id = [0 for _ in range(config.sample.batch_size)]
    finished = torch.zeros(config.sample.batch_size).bool()
    
    curmodel_num = argsckpt.split('/')[-1].split('.')[0]
    curtask = argsdata_id
    cur_batch = batch_num
    
    record_mol = []
    record_vina = []
    for i in range(config.sample.max_steps):
        if torch.sum(finished) == config.sample.batch_size:
            #mol_list = SetMolPos(mol_list, pos_list)
            return mol_list, pos_list, vina_list 
        if i == 0:  
            focal_pred, mask_protein, h_ctx, pos_ctx, h_residue = model(protein_pos=batch['protein_pos'],
                                            protein_atom_feature=batch['protein_atom_feature'].float(),
                                            ligand_pos=batch['ligand_context_pos'],
                                            ligand_atom_feature=batch['ligand_context_feature_full'].float(),
                                            batch_protein=batch['protein_element_batch'],
                                            batch_ligand=batch['ligand_context_element_batch'], batch=batch)
            protein_atom_feature = batch['protein_atom_feature'].float()
            focal_protein = focal_pred[mask_protein]
            h_ctx_protein = h_ctx[mask_protein]
            pos_ctx_protein = pos_ctx[mask_protein]
            focus_score = torch.sigmoid(focal_protein)
            can_focus = focus_score > 0.1
            slice_idx = torch.cat([torch.tensor([0]).to(h_ctx.device), torch.cumsum(batch['protein_element_batch'].bincount(), dim=0)])
            focal_id = []
            for j in range(len(slice_idx) - 1):
                focus = focus_score[slice_idx[j]:slice_idx[j + 1]]
                focal_id.append(torch.argmax(focus.reshape(-1).float()).item() + slice_idx[j].item())
            focal_id = torch.tensor(focal_id)
            h_ctx_focal = h_ctx_protein[focal_id]
            pos_ctx_focal = pos_ctx_protein[focal_id]
            current_wid = torch.tensor([vocab.size()] * config.sample.batch_size)
            next_motif_wid = model.forward_motif(h_ctx_focal, pos_ctx_focal, current_wid.to(h_ctx_focal.device)
                                                 ,torch.arange(config.sample.batch_size).to(h_ctx_focal.device)
                                                 ,h_residue, batch['residue_pos'], batch['amino_acid_batch'])
            for wid in next_motif_wid:
                mol_list.append(Chem.MolFromSmiles(vocab.get_smiles(wid)))

            # docking
            docked_mol_list = []
            for j in range(config.sample.batch_size):
                cur_sample = j                
                cur_step = i
                total_file = str(curmodel_num) + "_" + str(curtask) + "_" + str(cur_batch) + "_" + str(cur_sample) + "_" + str(cur_step)
                mol_filename = tmp_dir + total_file + "_pre_docked_mol.sdf"
                
                writer = Chem.SDWriter(mol_filename)
                # writer.SetKekulize(False)
                writer.write(mol_list[j], confId=0)
                writer.close()
                
                try:
                    protonated_lig = rdkit.Chem.AddHs(mol_list[j])  
                    rdkit.Chem.AllChem.EmbedMolecule(protonated_lig)  
                    
                    preparator = MoleculePreparation()
                    mol_setups = preparator.prepare(protonated_lig)
                    
                    ligand_pdbqt_path = total_file + '_meeko'
                    preparator.write_pdbqt_file(tmp_dir + ligand_pdbqt_path + '.pdbqt')
                except:
                    print("Structure error!")
                    return (record_mol, record_vina)

                vina_task = QVinaDockingTask.from_pdbqt_data(
                    protein_filename, 
                    ligand_pdbqt_path,
                    tmp_dir=tmp_dir,
                    task_id=total_file,
                    center=center  
                )
                vina_results = vina_task.run_sync()
                try:
                    docked_mol = vina_results[0]['rdmol']
                    vina_score = vina_results[0]['affinity']
                    docked_mol_list.append(docked_mol)
                except:
                    print("error: Vina error")
                    return record_mol, record_vina
                vina_list.append(vina_score)

                
                match_sub = docked_mol.HasSubstructMatch(mol_list[j])
                if match_sub != True:
                    print("error: Get substructure match error")
                    return record_mol, record_vina
                
                list_order = list(docked_mol.GetSubstructMatch(mol_list[j]))
                after_docked_all_atoms = np.array([atom.GetSymbol() for atom in docked_mol.GetAtoms()])
                after_docked_all_pos = np.array([docked_mol.GetConformer().GetAtomPosition(atom.GetIdx()) for atom in docked_mol.GetAtoms()])
                q_atoms = after_docked_all_atoms[list_order]
                q_coord = after_docked_all_pos[list_order]
                
                try:
                    AllChem.EmbedMolecule(mol_list[j])  
                    conf = mol_list[j].GetConformer()  
                except:
                    print("error: First fragment error")
                    return record_mol, record_vina
                
                pos = q_coord 

                if mol_list[j].GetNumAtoms() == len(pos):
                    for node in range(mol_list[j].GetNumAtoms()):
                        conf.SetAtomPosition(node, pos[node])
                ligand_pos, ligand_feat = torch.tensor(mol_list[j].GetConformer().GetPositions()).to(h_ctx_focal.device), get_feat(mol_list[j]).to(h_ctx_focal.device)
                feat_list.append(ligand_feat)
                pos_list.append(ligand_pos)
                record_mol.append(mol_list[j])
                record_vina.append(vina_list[j])
                assert mol_list[j].GetNumAtoms() == len(pos_list[j])

                writer = Chem.SDWriter(tmp_dir + total_file + "_docked_mol.sdf")
                # writer.SetKekulize(False)
                writer.write(mol_list[j], confId=0)
                writer.close()
                        
            atom_to_motif = [{} for _ in range(config.sample.batch_size)]   
            motif_to_atoms = [{} for _ in range(config.sample.batch_size)]   
            motif_wid = [{} for _ in range(config.sample.batch_size)]  
            for j in range(config.sample.batch_size):
                for k in range(mol_list[j].GetNumAtoms()): 
                    atom_to_motif[j][k] = 0 

            for j in range(config.sample.batch_size):
                motif_to_atoms[j][0] = list(np.arange(mol_list[j].GetNumAtoms()))  
                motif_wid[j][0] = next_motif_wid[j].item()  
            
            
        else:  
            repeats = torch.tensor([len(pos) for pos in pos_list]).to(center.device)
            ligand_batch = torch.repeat_interleave(torch.arange(config.sample.batch_size).to(repeats.device), repeats)
            focal_pred, mask_protein, h_ctx, pos_ctx, h_residue = model(protein_pos=batch['protein_pos'].float().to(repeats.device),
                                                    protein_atom_feature=batch['protein_atom_feature'].float().to(repeats.device),
                                                    ligand_pos=torch.cat(pos_list, dim=0).float().to(repeats.device),
                                                    ligand_atom_feature=torch.cat(feat_list, dim=0).float().to(repeats.device),
                                                    batch_protein=batch['protein_element_batch'],
                                                    batch_ligand=ligand_batch, batch=batch)
            focal_ligand = focal_pred[~mask_protein]
            h_ctx_ligand = h_ctx[~mask_protein]
            pos_ctx_ligand = pos_ctx[~mask_protein]
            focus_score = torch.sigmoid(focal_ligand)
            can_focus = focus_score > 0.1
            slice_idx = torch.cat([torch.tensor([0]).to(repeats.device), torch.cumsum(repeats, dim=0)])

            current_atoms_batch, current_atoms = [], []
            for j in range(len(slice_idx) - 1):
                focus = focus_score[slice_idx[j]:slice_idx[j + 1]]
                if torch.sum(can_focus[slice_idx[j]:slice_idx[j + 1]]) > 0 and ~finished[j]:
                    sample_focal_atom = torch.multinomial(focus.reshape(-1).float(), 1)
                    focal_motif = atom_to_motif[j][sample_focal_atom.item()]
                    motif_id[j] = focal_motif
                else:
                    finished[j] = True

                current_atoms.extend((np.array(motif_to_atoms[j][motif_id[j]]) + slice_idx[j].item()).tolist())
                current_atoms_batch.extend([j] * len(motif_to_atoms[j][motif_id[j]]))
                mol_list[j] = SetAtomNum(mol_list[j], motif_to_atoms[j][motif_id[j]])   
            
            
            # second step: next motif prediction
            current_wid = [motif_wid[j][motif_id[j]] for j in range(len(mol_list))]
            current_atoms = torch.tensor(current_atoms)
            next_motif_wid = model.forward_motif(h_ctx_ligand[current_atoms], pos_ctx_ligand[current_atoms],
                                                 torch.tensor(current_wid).to(h_ctx_focal.device),
                                                 torch.tensor(current_atoms_batch).to(h_ctx_focal.device), h_residue, batch['residue_pos'],batch['amino_acid_batch'])
            # assemble
            try:
                next_motif_smiles = [vocab.get_smiles(id) for id in next_motif_wid]
            except:
                print('get vocab id error')
                return (record_mol, record_vina)
            
    
            new_mol_list, new_atoms, one_atom_attach, intersection, attach_fail = model.forward_attach(mol_list, next_motif_smiles)   
            attach_fail = attach_fail.to(next_motif_wid.device)
            
            
            # docking
            docked_mol_list = []
            for j in range(config.sample.batch_size):
                if attach_fail[j] or finished[j]:  
                    continue  
                cur_sample = j
                cur_step = i
                total_file = str(curmodel_num) + "_" + str(curtask) + "_" + str(cur_batch) + "_" + str(cur_sample) + "_" + str(cur_step)
                Chem.SanitizeMol(new_mol_list[j])

                mol_filename = tmp_dir + total_file + "_pre_docked_mol.sdf"

                writer = Chem.SDWriter(mol_filename)
                # writer.SetKekulize(False)
                writer.write(new_mol_list[j], confId=0)
                writer.close()
       
                try:
                    protonated_lig = rdkit.Chem.AddHs(new_mol_list[j])
                    rdkit.Chem.AllChem.EmbedMolecule(protonated_lig)
                    preparator = MoleculePreparation()
                    mol_setups = preparator.prepare(protonated_lig)

                    ligand_pdbqt_path = total_file + '_meeko'
                    preparator.write_pdbqt_file(tmp_dir + ligand_pdbqt_path + '.pdbqt')
                except:
                    print("Structure error!")
                    return (record_mol, record_vina)
        
        
        
                vina_task = QVinaDockingTask.from_pdbqt_data(
                    protein_filename, 
                    ligand_pdbqt_path,
                    tmp_dir=tmp_dir,
                    task_id=total_file,
                    center=center   
                )

                vina_results = vina_task.run_sync()
                try:
                    docked_mol = vina_results[0]['rdmol']
                    docked_mol_list.append(docked_mol)
                    vina_score = vina_results[0]['affinity']
                except:
                    print("error: Vina error")
                    return record_mol, record_vina
                
                match_sub = docked_mol.HasSubstructMatch(new_mol_list[j])
                if match_sub != True:
                    print("error: Get substructure match error")
                    return record_mol, record_vina

                list_order = list(docked_mol.GetSubstructMatch(new_mol_list[j]))   
                
                before_docked_all_atoms = np.array([atom.GetSymbol() for atom in new_mol_list[j].GetAtoms()])
                after_docked_all_atoms = np.array([atom.GetSymbol() for atom in docked_mol.GetAtoms()])
                after_docked_all_pos = np.array([docked_mol.GetConformer().GetAtomPosition(atom.GetIdx()) for atom in docked_mol.GetAtoms()])
                before_size = before_docked_all_atoms.shape[0]
                after_size = after_docked_all_atoms.shape[0]
                if not before_size == after_size:
                    print("error: Structures not same size")
             
                q_atoms = copy.deepcopy(after_docked_all_atoms)
                q_coord = copy.deepcopy(after_docked_all_pos)

                q_atoms = q_atoms[list_order]
                q_coord = q_coord[list_order]
                
         
                AllChem.EmbedMolecule(new_mol_list[j])  
                try:
                    conf = new_mol_list[j].GetConformer()
                except:
                    print("Embed error")
                    return record_mol, record_vina
                pos = q_coord 
                if new_mol_list[j].GetNumAtoms() == len(pos):
                    for node in range(new_mol_list[j].GetNumAtoms()):
                        conf.SetAtomPosition(node, pos[node])
                pos_list[j] = torch.tensor(pos).to(next_motif_wid.device)
                feat_list[j] = get_feat(new_mol_list[j]).to(next_motif_wid.device)
                vina_list[j] = vina_score
                mol_list[j] = new_mol_list[j]
                
                record_mol.append(mol_list[j])
                record_vina.append(vina_list[j])
                
                assert mol_list[j].GetNumAtoms() == len(pos_list[j])
                
               
                writer = Chem.SDWriter(tmp_dir + total_file + "_docked_mol.sdf")
                # writer.SetKekulize(False)
                writer.write(mol_list[j], confId=0)
                writer.close()

            
            # update motif2atoms and atom2motif
            for j in range(len(mol_list)):
                if attach_fail[j] or finished[j]:   
                    continue
                motif_to_atoms[j][i] = new_atoms[j]  
                motif_wid[j][i] = next_motif_wid[j].item()  
                for k in new_atoms[j]: 
                    atom_to_motif[j][k] = i
                    

    return record_mol, pos_list, record_vina



if __name__ == "__main__":
    time_start = time.time()   

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/sampleMolecule.yml')
    parser.add_argument('-i', '--data_id', type=int, default=11)
    parser.add_argument('--device', type=str, default='cuda:0')  
    parser.add_argument('--outdir', type=str, default='./outputs')
    parser.add_argument('--vocab_path', type=str, default='./utils/vocab.txt')
    parser.add_argument('--ckpt', type=str, default='')
    args = parser.parse_args()



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


    # Logging
    log_dir = get_new_log_dir(args.outdir, prefix='%s-%d' % (config_name, args.data_id))
    logger = get_logger('sample', log_dir)
    logger.info(args)
    logger.info(config)
    shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))


    # Data
    logger.info('Loading data...')
    protein_featurizer = FeaturizeProteinAtom()
    ligand_featurizer = FeaturizeLigandAtom()
    masking = LigandMaskAll(vocab)
    transform = Compose([
        LigandCountNeighbors(),
        protein_featurizer,
        ligand_featurizer,
        FeaturizeLigandBond(),
        masking,
    ])
    dataset, subsets = get_dataset(
        config=config.dataset,
        transform=transform,
    )



    # Model (Main)
    logger.info(f'Loading main model...\n {args.ckpt}')
    # ckpt = torch.load(config.model.checkpoint, map_location=args.device)
    ckpt = torch.load(args.ckpt, map_location=args.device)
    model = MoleculeGenerator(
        config.model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim,
        vocab=vocab,
        device=args.device).to(args.device)
    model.load_state_dict(ckpt['model'])


    testset = subsets['test']
    data = testset[args.data_id]
    center = data['pocket_center'].to(args.device)
    test_set = [data for _ in range(config.sample.num_samples)]

    with open(os.path.join(log_dir, 'pocket_info.txt'), 'a') as f:
        f.write(data['protein_filename'] + '\n')


    protein_filename = "./data/crossdocked_pocket10/" + data["protein_filename"]


    # my code goes here
    sample_loader = DataLoader(test_set, batch_size=config.sample.batch_size,
                            shuffle=False, num_workers=config.sample.num_workers,
                            collate_fn=collate_mols)

    
    data_list = []
    vina_scores = []
    with torch.no_grad():
        model.eval()
        for idx, batch in enumerate(tqdm(sample_loader)):
            tmp_dir = log_dir + '/' + str(idx) + '/'
            mksure_path(tmp_dir)
            for key in batch:
                batch[key] = batch[key].to(args.device)
            gen_situation = ligand_gen(batch, model, vocab, config, center, args.ckpt, args.data_id, idx, protein_filename, tmp_dir)
            while len(gen_situation) != 3:
                record_mol, record_vina = gen_situation
                if len(record_mol) == 0 or len(record_vina) == 0:   
                    gen_situation = ligand_gen(batch, model, vocab, config, center, args.ckpt, args.data_id, idx, protein_filename, tmp_dir)
                    continue
                min_value = min(record_vina) 
                min_idx = record_vina.index(min_value) 
                cur_gen_mol = record_mol[min_idx]
                cur_vina = record_vina[min_idx]
                
                if cur_vina < -5:  
                    break
                gen_situation = ligand_gen(batch, model, vocab, config, center, args.ckpt, args.data_id, idx, protein_filename, tmp_dir)
            
            if len(gen_situation) == 3:
                gen_data, pos_list, vina_list = gen_situation

                min_value = min(vina_list) 
                min_idx = vina_list.index(min_value)  
                cur_gen_mol = gen_data[min_idx]
                cur_vina = vina_list[min_idx]
            else:
                pass
            # print("cur_gen_mol")
            # show_mol(cur_gen_mol)
            print("cur_vina", cur_vina)
            data_list.append(cur_gen_mol)  
            vina_scores.append(cur_vina)
            print([Chem.MolToSmiles(mol) for mol in data_list])
            smiles = [Chem.MolFromSmiles(Chem.MolToSmiles(mol)) for mol in data_list]
            qed_list = [qed(mol) for mol in smiles]
            logp_list = [MolLogP(mol) for mol in smiles]
            sa_list = [compute_sa_score(mol) for mol in smiles]
            Lip_list = [lipinski(mol) for mol in smiles]
            print('Vina score %.6f | QED %.6f | LogP %.6f | SA %.6f | Lipinski %.6f \n' % (
            np.average(vina_scores), np.average(qed_list), np.average(logp_list), np.average(sa_list), np.average(Lip_list)))
            print("SA", sa_list)
            print("Vina", vina_scores)
    #         SetMolPos(data_list, pos_list)

            with open(os.path.join(log_dir, 'SMILES.txt'), 'a') as smiles_f:
                for i, mol in enumerate(data_list):
                    smiles_f.write(Chem.MolToSmiles(mol) + '\n')
                    writer = Chem.SDWriter(os.path.join(log_dir, '%d.sdf' % i))
                    # writer.SetKekulize(False)
                    writer.write(mol, confId=0)
                    writer.close()
    
    
    time_end = time.time()  
    time_sum = time_end - time_start  
    print(time_sum)
