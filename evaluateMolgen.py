import argparse
import pandas as pd
from utils.mol_tree import *
from utils.misc import *
from utils.transforms import *
from utils.datasets import get_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='./configs/sampleMolecule.yml')
parser.add_argument('--vocab_path', type=str, default='./utils/vocab.txt')
parser.add_argument('--data_root', type=str, default='./logs/elaboration_100_2024_07_04__15_47_27/0/generated_mols')
parser.add_argument('--task', type=int, default=100)
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

# Data
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
testset = subsets['test']


standard_vina_score = pd.read_csv("standard_vina_score.csv")
tasks = list(range(0, args.task, 1))

total_dict = []
best_vina_scores = []
for argsdata_id in tasks:
    try:
        print(argsdata_id)
        data = testset[argsdata_id]
        protein_filename = "./data/crossdocked_pocket10" + "/" + data["protein_filename"]

        assert os.path.basename(standard_vina_score.iloc[argsdata_id]["protein_filename"]) == os.path.basename(protein_filename)
        output_path = args.data_root + '/test_' + str(argsdata_id) + '_pdb'

        log_path = output_path + '/sample_molecule.log'
        with open(log_path) as fp:
            log_str = fp.read()
        cur_vina_scores = eval(re.findall('Vina \[(.*)]', log_str)[-1])
        if isinstance((cur_vina_scores), float):
            cur_vina_scores = [cur_vina_scores]
        else:
            cur_vina_scores = list(cur_vina_scores)
        print(cur_vina_scores)
        std_vina_score = standard_vina_score.iloc[argsdata_id]["vina_score"]
        best_vina_score = min(cur_vina_scores)
        metrics = re.findall('Vina score (.*) \| QED (.*) \| LogP (.*) \| SA (.*) \| Lipinski (.*) ', log_str)[-1]
        avg_vina_score = eval(metrics[0])

        high_aff = sum(np.array(cur_vina_scores) <= std_vina_score)/len(cur_vina_scores)
        best_vina_scores.append(best_vina_score)
        cur_dict = {"id": argsdata_id, "protein_filename": protein_filename, 
                    "generated_mol": len(cur_vina_scores),
                    "std_vina_score": std_vina_score,
                    "best_vina_score": best_vina_score, "avg_vina_score": avg_vina_score, 
                    "high_affinity": high_aff,
                "QED": metrics[1], "LogP": metrics[2], "SA": metrics[3], "Lipinski": metrics[4]}
        total_dict.append(cur_dict)
    except:
        cur_dict = {"id": argsdata_id, "protein_filename": protein_filename,
                    "generated_mol": 0,
                    "std_vina_score": "",
                    "best_vina_score": "", "avg_vina_score": "", 
                    "high_affinity": "",
                "QED": "", "LogP": "", "SA": "", "Lipinski": ""}
        total_dict.append(cur_dict) 
pd.DataFrame(total_dict).to_csv(args.data_root + "/generated_metrics.csv", index=False)
high_affinities = sum(np.array(best_vina_scores) <= std_vina_score)/len(tasks)
print("high_affinities", high_affinities)
