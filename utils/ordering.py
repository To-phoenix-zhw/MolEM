import sys
sys.path.append("..")
import random
import numpy as np
import pandas as pd
import torch
from .misc import get_file, mksure_path

def get_total_mol_ordering(file_path): 
    mol_ordering_dict = {}
    for i in range(len(file_path)):
        mol_id = file_path[i].split('/')[-3] 
        ordering_id = file_path[i].split('/')[-2] 
        if mol_id in mol_ordering_dict:
            mol_ordering_dict[mol_id].add(ordering_id)
        else:
            mol_ordering_dict[mol_id] = {ordering_id}
            
    for key, values in mol_ordering_dict.items():
        l = list(values)
        l.sort()
        mol_ordering_dict[key] = l   
    
    return mol_ordering_dict

 
def random_choose_ordering(mol_ordering_dict):
    mol_ordering_dict_list = []
    for mol_id, ordering_ids in mol_ordering_dict.items():
        chosen_ordering = random.choice(ordering_ids)  
        cur_dict = {'mol_id': mol_id, 'ordering_id': chosen_ordering}
        mol_ordering_dict_list.append(cur_dict)
    return mol_ordering_dict_list

 
def get_dynamic_data_ordering(mol_ordering_df, root_path):
    data_paths = []
    mol_orderings = []
    for i in range(len(mol_ordering_df)):
        cur_mol_id = str(mol_ordering_df.iloc[i]["mol_id"])
        cur_ordering_id = str(mol_ordering_df.iloc[i]["ordering_id"])
        dir_path = root_path + cur_mol_id + '/' + cur_ordering_id + '/'
#         print(dir_path)
        data_path = get_file(dir_path, 'pt', [])
#         print(torch.load(dir_path + '0.pt'))
        cur_ordering = torch.load(dir_path + '0.pt')["orderings"][int(cur_ordering_id)]
        mol_orderings.append({"mol_id": cur_mol_id, "ordering_id": cur_ordering_id, "ordering": cur_ordering})
        data_paths.extend(data_path)
    return data_paths, mol_orderings


def store_data_orderings(
    data_root, 
    em_iteration_times, 
    dynamic_data_ordering, 
    ordering_lst,
    mode
):
    dynamic_ordering_root = data_root + 'ordering_input/' + str(em_iteration_times) + '/' 
    mksure_path(dynamic_ordering_root)
    pd.DataFrame(dynamic_data_ordering, columns=['path']).to_csv(dynamic_ordering_root + mode + '_data.csv', index=False)
    pd.DataFrame(ordering_lst).to_csv(data_root + 'ordering_input/' + str(em_iteration_times) + '/' + mode + '_ordering.csv', index=False)



def store_data_and_orderings(
    data_root, 
    em_iteration_times, 
    train_dynamic_data_ordering, 
    val_dynamic_data_ordering, 
    train_ordering_lst, 
    val_ordering_lst
):
    dynamic_ordering_root = data_root + 'ordering_input/' + str(em_iteration_times) + '/' 
    mksure_path(dynamic_ordering_root)
    # print(dynamic_ordering_root)
    pd.DataFrame(train_dynamic_data_ordering, columns=['path']).to_csv(dynamic_ordering_root + 'train_data.csv', index=False)
    pd.DataFrame(val_dynamic_data_ordering, columns=['path']).to_csv(dynamic_ordering_root + 'val_data.csv', index=False)
    pd.DataFrame(train_ordering_lst).to_csv(data_root + 'ordering_input/' + str(em_iteration_times) + '/train_ordering.csv', index=False)
    pd.DataFrame(val_ordering_lst).to_csv(data_root + 'ordering_input/' + str(em_iteration_times) + '/val_ordering.csv', index=False)


def compare_orderings(compare_whether_pre):
    cnt_in_pre = 0
    cnt_same_as_input = 0
    for i in range(len(compare_whether_pre)):
        mol_id = compare_whether_pre.iloc[i]["mol_id"]
        input_ordering = compare_whether_pre.iloc[i]["input_ordering"]
        output_ordering = compare_whether_pre.iloc[i]["output_ordering"]
        pre_orderings = compare_whether_pre.iloc[i]["orderings"]
    #     print(mol_id, output_ordering, pre_orderings)
        if output_ordering == input_ordering:
            print(mol_id)
            print("equal")
            print(output_ordering)
            print(input_ordering)
            cnt_same_as_input += 1
            cnt_in_pre += 1
            continue
        if output_ordering in pre_orderings:
            print(mol_id)
            print(output_ordering)
            print(pre_orderings)
            print("in pre")
            print(1)
            cnt_in_pre += 1
    #     break
    return cnt_in_pre, cnt_same_as_input