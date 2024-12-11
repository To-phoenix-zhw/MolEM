import os
import subprocess
import random
import string
from easydict import EasyDict
from rdkit import Chem
from rdkit.Chem.rdForceFieldHelpers import UFFOptimizeMolecule
import pandas as pd
from .reconstruct import reconstruct_from_generated
from meeko import PDBQTMolecule
from meeko import RDKitMolCreate

def get_random_id(length=30):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length)) 


def load_pdb(path):
    with open(path, 'r') as f:
        return f.read()


def parse_qvina_outputs(docked_pdbqt_path):
    pdbqt_docked_mol = PDBQTMolecule.from_file(docked_pdbqt_path, poses_to_read=0, skip_typing=True)
    dockedmol_list = RDKitMolCreate.from_pdbqt_mol(pdbqt_docked_mol)
    dockedmol = dockedmol_list[0]

    vina_score = pdbqt_docked_mol.score
    docked_mol = Chem.RemoveHs(dockedmol)   

    results = []
    results.append(EasyDict({
        'rdmol': docked_mol,
        'affinity': vina_score,
    }))

    return results

class BaseDockingTask(object):

    def __init__(self, pdb_block, ligand_pdbqt_path):
        super().__init__()
        self.pdb_block = pdb_block
        self.ligand_pdbqt_path = ligand_pdbqt_path

    def run(self):
        raise NotImplementedError()
    
    def get_results(self):
        raise NotImplementedError()



def try_func():
    print("try")




class QVinaDockingTask(BaseDockingTask):

    @classmethod
    def from_generated_data(cls, data, protein_root='./data/crossdocked', **kwargs):
        protein_fn = os.path.join(
            os.path.dirname(data.ligand_filename),
            os.path.basename(data.ligand_filename)[:10] + '.pdb'
        )
        protein_path = os.path.join(protein_root, protein_fn)
        with open(protein_path, 'r') as f:
            pdb_block = f.read()
        ligand_rdmol = reconstruct_from_generated(data)
        return cls(pdb_block, ligand_rdmol, **kwargs)

    @classmethod
    def from_original_data(cls, data, ligand_root='./data/crossdocked_pocket10', protein_root='./data/crossdocked', **kwargs):
        protein_fn = os.path.join(
            os.path.dirname(data.ligand_filename),
            os.path.basename(data.ligand_filename)[:10] + '.pdb'
        )
        protein_path = os.path.join(protein_root, protein_fn)
        with open(protein_path, 'r') as f:
            pdb_block = f.read()

        ligand_path = os.path.join(ligand_root, data.ligand_filename)
        ligand_rdmol = next(iter(Chem.SDMolSupplier(ligand_path)))
        return cls(pdb_block, ligand_rdmol, **kwargs)
    
    @classmethod    
    def from_data(cls, protein_path, ligand_path, ligand_root='./data/crossdocked_pocket10', protein_root='./data/crossdocked_pocket10', **kwargs):
        with open(protein_path, 'r') as f:
            pdb_block = f.read()

        ligand_rdmol = next(iter(Chem.SDMolSupplier(ligand_path)))
        return cls(pdb_block, ligand_rdmol, **kwargs)
    

    @classmethod
    def from_pdbqt_data(cls, protein_path, ligand_pdbqt_path, ligand_root='./data/crossdocked_pocket10', protein_root='./data/crossdocked_pocket10', **kwargs):
        with open(protein_path, 'r') as f:
            pdb_block = f.read()
        
        return cls(pdb_block, ligand_pdbqt_path, **kwargs)

    def __init__(self, pdb_block, ligand_pdbqt_path, conda_env='adt', tmp_dir='./tmp', use_uff=True, center=None, task_id=''):
        super().__init__(pdb_block, ligand_pdbqt_path)
        self.conda_env = conda_env
        self.tmp_dir = os.path.realpath(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)

        if task_id == '':
            self.task_id = get_random_id()
        else:
            self.task_id = task_id
#         print("self.task_id", self.task_id)

        self.receptor_id = self.task_id + '_receptor'
        self.ligand_id = ligand_pdbqt_path

        self.receptor_path = os.path.join(self.tmp_dir, self.receptor_id + '.pdb')
        with open(self.receptor_path, 'w') as f:
            f.write(pdb_block)

        self.center = center

        self.proc = None
        self.results = None
        self.output = None
        self.docked_pdbqt_path = None

    def run(self, exhaustiveness=16):
        commands = """
eval "$(conda shell.bash hook)"
conda activate {env}
cd {tmp}
# Prepare receptor (PDB->PDBQT)
prepare_receptor4.py -r {receptor_id}.pdb
qvina2 \
    --receptor {receptor_id}.pdbqt \
    --ligand {ligand_id}.pdbqt \
    --center_x {center_x:.4f} \
    --center_y {center_y:.4f} \
    --center_z {center_z:.4f} \
    --size_x 20 --size_y 20 --size_z 20 \
    --exhaustiveness {exhaust}
        """.format(
            receptor_id = self.receptor_id,
            ligand_id = self.ligand_id,
            env = self.conda_env, 
            tmp = self.tmp_dir, 
            exhaust = exhaustiveness,
            center_x = self.center[0],
            center_y = self.center[1],
            center_z = self.center[2],
        )

        self.docked_pdbqt_path = os.path.join(self.tmp_dir, '%s_out.pdbqt' % self.ligand_id)
        self.proc = subprocess.Popen(
            '/bin/bash', 
            shell=False, 
            stdin=subprocess.PIPE, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )

        self.proc.stdin.write(commands.encode('utf-8'))
        self.proc.stdin.close()

        # return commands


    def run_sync(self):
        # print(pd.isna(float('nan')))
        # print("run_sync")
#         print("self.center", self.center)
        # print("pd.isna(float(self.center[0]))", pd.isna(float(self.center[0])))
        # print("pd.isna(float(self.center[1]))", pd.isna(float(self.center[0])))
        # print("pd.isna(float(self.center[2]))", pd.isna(float(self.center[0])))
        if (pd.isna(float(self.center[0])) or pd.isna(float(self.center[1]))) or (pd.isna(float(self.center[2]))):
#             print("[]")
            return []

        self.run()
        while self.get_results() is None:
            pass
        results = self.get_results()
#         print("results", results)
        if results is None or len(results) == 0:
            return results
#         print('Best affinity:', results[0]['affinity'])
        return results

    def get_results(self):
        if self.proc is None:   # Not started
            return None
        elif self.proc.poll() is None:  # In progress
            return None
        else:
            if self.output is None:
                self.output = self.proc.stdout.readlines()
                try:
                    self.results = parse_qvina_outputs(self.docked_pdbqt_path)
                except:
                    print('[Error] Vina output error: %s' % self.docked_pdbqt_path)
                    return []
            return self.results

