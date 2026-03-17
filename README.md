# MolEM: a unified generative framework for molecular graphs and sequential orders

Official implementation of the paper:

**MolEM: a unified generative framework for molecular graphs and sequential orders**  
Briefings in Bioinformatics (2025)

Paper: https://academic.oup.com/bib/article/26/2/bbaf094/8101319

## Overview

MolEM is a generative framework for molecular design that jointly models:

- **molecular graph generation**
- **latent ordering**

The key idea is that **multiple sequential orders can generate the same molecular graph**.  
MolEM treats these sequential orders as latent variables and learns them through an **Expectation-Maximization (EM)** procedure.

The framework alternates between:

1. **Ordering Generator**
   - predicts orders

2. **Molecular Generator**
   - generates molecular graphs conditioned on the predicted order

These two models are trained iteratively.


---

## Installation

The code has been implemented in the following environment.

| Package | Version  |
| ------- | -------- |
| Python  | 3.8      |
| PyTorch | 1.10.1   |
| CUDA    | 11.3     |
| RDKit   | 2024.3.5 |
| meeko   | v0.5.0   |

You can create the environment as follows.

```shell
conda env create -f environment.yaml
conda activate MolEM
```


Install Autodocktools:

```shell
git clone git@github.com:Valdes-Tresanco-MS/AutoDockTools_py3.git
cd AutoDockTools_py3/
pip install .

pip install meeko
```

Install qvina:

```shell
conda env create -f env_adt.yml
conda install -c conda-forge qvina
```


---

## Data Organization

MolEM expects the following directory structure:

```
em/
 ├── training/
 │   ├── pre_orderings.csv
 │   ├── mol_000/
 │   ├── mol_001/
 │   └── ...
 │
 └── validation/
 ├── pre_orderings.csv
 ├── mol_000/
 ├── mol_001/
 └── ...
```


Each molecule directory contains preprocessed graph data stored as `.pt` files.

---

## Running

We provides an **end-to-end EM training pipeline** implemented in: `MolEM.py`

Example:

```shell
python MolEM.py 
 --config ./configs/elaboration.yml 
 --data_root ./em/ 
 --device cuda 
 --gpu_id 0
```


Main arguments:

| Argument | Description |
|------|------|
| `--config` | training configuration file |
| `--data_root` | root directory of dataset |
| `--device` | `cuda` or `cpu` |
| `--gpu_id` | GPU index |

---

## Pretrained Models

We provide pretrained checkpoints obtained from the EM training procedure.

The following models are included in the `checkpoints/` directory:

| Model           | Description                                    |
| --------------- | ---------------------------------------------- |
| `ordergen_4.pt` | Ordering generator after the 4th EM iteration  |
| `molgen_4.pt`   | Molecular generator after the 4th EM iteration |

These checkpoints correspond to the models obtained at **EM iteration 4** during training.

Users can directly use these models for evaluation or molecule generation without running the full EM training pipeline.

---

## EM Training Procedure

Each EM iteration consists of the following steps.

### 1. Select orderings

For the first iteration, fragment orderings are randomly selected.  
In later iterations, the ordering with the highest likelihood under the molecular generator is chosen.

### 2. Train ordering generator


python train_ordering.py

### 3. Predict orderings


python sample_ordering.py

### 4. Build molecule generation dataset


python contrastive_data_building_for_orderings.py

### 5. Train molecular generator


python train_molecule_generation.py

### 6. Generate molecules


bash sample_molecule.sh

### 7. Evaluate generated molecules


python evaluateMolgen.py

### 8. Score candidate orderings


python scoreOrderings.py


The best orderings are then used for the next EM iteration.


---

## Citation

If you find this repo useful, please cite our paper.

> @article{zhang2025molem,
> title={MolEM: a unified generative framework for molecular graphs and sequential orders},
>   author={Zhang, Hanwen and Xiong, Deng and Liu, Xianggen and Lv, Jiancheng},
>   journal={Briefings in bioinformatics},
>   volume={26},
>   number={2},
>   pages={bbaf094},
>   year={2025},
>   publisher={Oxford University Press}
>   }

## Contact

If you encounter problems running the code, please open an issue in the repository.

If you encounter any problems during the setup of environment or the execution of the framework, do not hesitate to contact [hanwenzhang@stu.scu.edu.cn](mailto:hanwenzhang@stu.scu.edu.cn). You could also create an issue under the repository: https://github.com/To-phoenix-zhw/MolEM.
