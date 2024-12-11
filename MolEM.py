import os
import pandas as pd
import random
import torch
import torch.utils.tensorboard
import argparse
import shutil
import time
from utils.misc import *
from utils.ordering import *


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='./configs/elaboration.yml')
parser.add_argument('--data_root', type=str,default='./em/')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--gpu_id', type=str, default='2')
parser.add_argument('--logdir', type=str, default='./logs')
parser.add_argument('--vocab_path', type=str, default='./utils/vocab.txt')
parser.add_argument('--continue_iteration', type=int, default=0)
parser.add_argument('--log_dir', type=str, default='')
args = parser.parse_args()


# Load configs
config = load_config(args.config)
config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
seed_all(config.train.seed)


# Logging
if args.continue_iteration == 0:
    log_dir = get_new_log_dir(args.logdir, prefix=config_name)
    logger = get_logger('EM', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    logger.info(args)
    logger.info(config)
    logger.info(log_dir)
    shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
else:
    log_dir = args.log_dir
    logger = get_logger('EM', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
print("log_dir", log_dir)

data_roots = ['./em/']  # different paths storing data
train_pre_orderings_dfs = []
for data_root in data_roots:
    print(data_root)
    train_dir_path = data_root + 'training/'
    train_pre_orderings_path = train_dir_path + 'pre_orderings.csv'
    train_pre_orderings_df = pd.read_csv(train_pre_orderings_path)
    train_pre_orderings_df['loc'] = data_root 
    train_pre_orderings_df[['mol_id']] = train_pre_orderings_df[['mol_id']].astype(str)
    train_pre_orderings_dfs.append(train_pre_orderings_df.iloc[0:config.dataset.data_scale])

time_record = time.time()
train_file_paths = []
for i in range(len(train_pre_orderings_dfs)):
    cur_df = train_pre_orderings_dfs[i]
    file_paths = []
    for j in range(len(cur_df)):
        mol_id = cur_df.iloc[j]["mol_id"]
        mol_loc = cur_df.iloc[j]["loc"]
        train_mol_loc = mol_loc + 'training/' + mol_id + '/'
        train_file_path = get_file(train_mol_loc, 'pt', [])
        print(len(train_file_path))
        file_paths.extend(train_file_path)
    train_file_paths.append(file_paths)

print('time cost: %.4fs' % (time.time() - time_record))

total_train_files = 0
for path in train_file_paths:
    print(len(path))
    total_train_files += len(path)
print("total_train_files", total_train_files)

val_dir_path = args.data_root + 'validation/'
val_file_paths = get_file(val_dir_path, 'pt', [])
val_pre_orderings_path = val_dir_path + 'pre_orderings.csv'
val_pre_orderings_df = pd.read_csv(val_pre_orderings_path)
print("total_val_files", len(val_file_paths))

train_mol_ordering_dicts = []
for file_paths in train_file_paths:
    train_mol_ordering_dict = get_total_mol_ordering(file_paths)  
    train_mol_ordering_dicts.append(train_mol_ordering_dict)

val_mol_ordering_dict = get_total_mol_ordering(val_file_paths)

for em_iteration_times in range(args.continue_iteration, config.train.em_iterations):
    time_total_em_record = time.time()
    print("*"*4 + "EM ITERATION: %d" % (em_iteration_times) + "*"*4)
    if em_iteration_times == 0:
        logger.info('[EM iteration] Times %d: random choose ordering' % (em_iteration_times))  
        train_mol_ordering_df = []
        for mol_ordering_dict in train_mol_ordering_dicts:
            mol_ordering_df = pd.DataFrame(random_choose_ordering(mol_ordering_dict))
            train_mol_ordering_df.append(mol_ordering_df)
        val_mol_ordering_df = pd.DataFrame(random_choose_ordering(val_mol_ordering_dict))
    else:
        logger.info('[EM iteration] Times %d: choose ordering based on molecule_generator' % (em_iteration_times))  
        train_mol_ordering_df = []
        for data_root in data_roots:
            ordering_file_path = get_file(data_root + 'ordering_score/' + str(em_iteration_times-1) + '/training/', 'csv', [])
            print(ordering_file_path)
            first_df = False
            for o_idx in range(len(ordering_file_path)):
                try:
                    cur_df = pd.read_csv(ordering_file_path[o_idx], usecols=['mol_id', 'ordering_idx'])
                except pd.errors.EmptyDataError:  
                    print("Empty csv.")
                    continue
                if first_df == False:
                    first_df = True
                    mol_ordering_df = cur_df
                else:
                    mol_ordering_df = pd.concat([mol_ordering_df, cur_df])
            mol_ordering_df.sort_values(by="mol_id" , inplace=True, ascending=True) 
            mol_ordering_df.reset_index(drop=True, inplace=True)    
            mol_ordering_df.rename(columns={'ordering_idx': 'ordering_id'}, inplace=True)
            print(mol_ordering_df)
            train_mol_ordering_df.append(mol_ordering_df)

        ordering_file_path = get_file(args.data_root + 'ordering_score/' + str(em_iteration_times-1) + '/validation/', 'csv', [])[0]
        val_mol_ordering_df = pd.read_csv(ordering_file_path, usecols=['mol_id', 'ordering_idx'])
        val_mol_ordering_df.rename(columns={'ordering_idx': 'ordering_id'}, inplace=True)

    print(train_mol_ordering_df)
    print(val_mol_ordering_df)

    train_dynamic_data_ordering, train_ordering_lst = [], []
    for i, mol_ordering_df in enumerate(train_mol_ordering_df):
        cur_dir_path = data_roots[i] + 'training/'
        dynamic_data_ordering, ordering_lst = get_dynamic_data_ordering(mol_ordering_df, cur_dir_path)
        print(len(dynamic_data_ordering), len(ordering_lst))
        train_dynamic_data_ordering.append(dynamic_data_ordering)
        train_ordering_lst.append(ordering_lst)
    val_dynamic_data_ordering, val_ordering_lst = get_dynamic_data_ordering(val_mol_ordering_df, val_dir_path)
    for i, dynamic_data_ordering in enumerate(train_dynamic_data_ordering):
        store_data_orderings(data_roots[i], em_iteration_times, dynamic_data_ordering, train_ordering_lst[i], 'train')
    store_data_orderings(args.data_root, em_iteration_times, val_dynamic_data_ordering, val_ordering_lst, 'val')
    

    train_data_file = pd.read_csv(data_roots[0] + 'ordering_input/' + str(em_iteration_times) + '/train_data.csv')
    for i in range(1, len(data_roots)):
        train_data_file = pd.concat([train_data_file, pd.read_csv(data_roots[i] + 'ordering_input/' + str(em_iteration_times) + '/train_data.csv')])
    train_data_file.reset_index(drop=True, inplace=True)
    train_data_file.to_csv(args.data_root + 'ordering_input/' + str(em_iteration_times) + '/train_data.csv', index=False)
    
    
    
    """
    OrderingGenerator
    """
    logger.info('[EM iteration] Times %d: training ordering_generator' % (em_iteration_times))
    mksure_path(log_dir + '/' + str(em_iteration_times) + '/')
    commands = """python train_ordering.py \
    --max_iters {max_iters} \
    --logdir {logdir} \
    --train_file_path {train_file_path} \
    --val_file_path {val_file_path} \
    --device {device}""".format(
        max_iters = config.train.o_gen_max_iters,   
        logdir = log_dir + '/' + str(em_iteration_times) + '/',
        train_file_path = args.data_root + 'ordering_input/' + str(em_iteration_times) + '/train_data.csv',
        val_file_path = args.data_root + 'ordering_input/' + str(em_iteration_times) + '/val_data.csv',
        device = args.device,
    )
    if args.device == 'cuda':
        commands = "CUDA_VISIBLE_DEVICES=" + args.gpu_id + " " + commands
    print(commands)
    time_record = time.time()
    output = commands_execute(commands)
    ordering_model_path = standardise_str(output)

    ordering_best_model_path, ordering_best_iter, ordering_best_val_loss, ordering_best_acc = analysis_checkpoints_performance(
        ordering_model_path, 
        '\[Validate\] Iter (.*) \| Loss (\-?\d+\.?\d*) \| acc (\-?\d+\.?\d*)',
        3
    )
    logger.info('[EM iteration] Ordering_generator: Iter %05d | Loss %.6f | Accuracy %.6f' % (ordering_best_iter, ordering_best_val_loss, ordering_best_acc))
    logger.info('time cost: %.4fs' % (time.time() - time_record))
    print("ordering_best_model_path", ordering_best_model_path)

   
    for data_root in data_roots:
        mksure_path(data_root + 'ordering_predict/' + str(em_iteration_times) + '/')

    for data_set in ['training', 'validation']:
        logger.info('[EM iteration] Times %d: sampling ordering_generator for %s set' % (em_iteration_times, data_set))
        if data_set == 'training':
            time_record = time.time()
            for d_idx, data_root in enumerate(data_roots):
                commands_set = """python sample_ordering.py \
                --data_root {data_root} \
                --dataset {data_set} \
                --end {end} \
                --outdir {outdir} \
                --ckpt {ckpt} \
                --device {device} > {output_log}""".format(
                    data_root = data_root,
                    data_set = data_set,
                    end = config.dataset.data_scale,   
                    outdir = data_root + 'ordering_predict/' + str(em_iteration_times) + '/',
                    ckpt = ordering_best_model_path,
                    device = args.device,
                    output_log = ordering_model_path + '/' + 'sample_' + data_set + '_set_' + str(d_idx) + '.log',
                )
                if args.device == 'cuda':
                    commands_set = "CUDA_VISIBLE_DEVICES=" + args.gpu_id + " " + commands_set
                print(commands_set)
                commands_execute(commands_set)
            logger.info('time cost: %.4fs' % (time.time() - time_record))

        else:
            commands_set = """python sample_ordering.py \
            --data_root {data_root} \
            --dataset {data_set} \
            --end {end} \
            --outdir {outdir} \
            --ckpt {ckpt} \
            --device {device} > {output_log}""".format(
                data_root = args.data_root,
                data_set = data_set,
                end = len(val_pre_orderings_df),
                outdir = args.data_root + 'ordering_predict/' + str(em_iteration_times) + '/',
                ckpt = ordering_best_model_path,
                device = args.device,
                output_log = ordering_model_path + '/' + 'sample_' + data_set + '_set.log',
            )
            if args.device == 'cuda':
                commands_set = "CUDA_VISIBLE_DEVICES=" + args.gpu_id + " " + commands_set
            print(commands_set)
            time_record = time.time()
            commands_execute(commands_set)
            logger.info('time cost: %.4fs' % (time.time() - time_record))

    """
    Ordering Analysis
    """
    input_train_ordering = pd.read_csv(args.data_root + 'ordering_input/' + str(em_iteration_times) + '/train_ordering.csv')
    input_val_ordering = pd.read_csv(args.data_root + 'ordering_input/' + str(em_iteration_times) + '/val_ordering.csv')
    input_train_ordering.rename(columns={'ordering': 'input_ordering'}, inplace=True)
    input_val_ordering.rename(columns={'ordering': 'input_ordering'}, inplace=True)
    input_train_ordering[['mol_id']] = input_train_ordering[['mol_id']].astype(str)
    input_val_ordering[['mol_id']] = input_val_ordering[['mol_id']].astype(str)

    output_train_ordering = pd.read_csv(data_roots[0] + 'ordering_predict/' + str(em_iteration_times) + '/training/orderings.csv')
    for d_idx in range(1, len(data_roots)) :
        output_train_ordering = pd.concat([output_train_ordering, pd.read_csv(data_roots[d_idx] + 'ordering_predict/' + str(em_iteration_times) + '/training/orderings.csv')])
    output_train_ordering.rename(columns={'ordering': 'output_ordering'}, inplace=True)
    output_train_ordering.reset_index(drop=True, inplace=True)
    output_train_ordering[['mol_id']] = output_train_ordering[['mol_id']].astype(str)
    output_val_ordering = pd.read_csv(args.data_root + 'ordering_predict/' + str(em_iteration_times) + '/validation/orderings.csv')
    output_val_ordering.rename(columns={'ordering': 'output_ordering'}, inplace=True)
    output_val_ordering[['mol_id']] = output_val_ordering[['mol_id']].astype(str)
    
    all_pre_train_orderings = train_pre_orderings_dfs[0]
    for d_idx in range(1, len(train_pre_orderings_dfs)):
        all_pre_train_orderings = pd.concat([all_pre_train_orderings, train_pre_orderings_dfs[d_idx]])
    all_pre_train_orderings.reset_index(drop=True, inplace=True)
    all_pre_train_orderings[['mol_id']] = all_pre_train_orderings[['mol_id']].astype(str)
    all_pre_val_orderings = pd.read_csv(args.data_root + 'validation/pre_orderings.csv')
    all_pre_val_orderings[['mol_id']] = all_pre_val_orderings[['mol_id']].astype(str)
    
    compare_training_whether_pre = pd.merge(output_train_ordering,all_pre_train_orderings,how='left',on='mol_id')
    compare_training_whether_pre = pd.merge(compare_training_whether_pre,input_train_ordering,how='left',on='mol_id')
    compare_validation_whether_pre = pd.merge(output_val_ordering,all_pre_val_orderings,how='left',on='mol_id')
    compare_validation_whether_pre = pd.merge(compare_validation_whether_pre,input_val_ordering,how='left',on='mol_id')

    cnt_training_in_pre, cnt_training_same_as_input = compare_orderings(compare_training_whether_pre)
    cnt_validation_in_pre, cnt_validation_same_as_input = compare_orderings(compare_validation_whether_pre)

    logger.info('[EM iteration] Times %d: sampling ordering analysis, training set --- %f in predefined orderings, %f the same as input' % (em_iteration_times, cnt_training_in_pre/len(compare_training_whether_pre), cnt_training_same_as_input/len(compare_training_whether_pre)))
    logger.info('[EM iteration] Times %d: sampling ordering analysis, validation set --- %f in predefined orderings, %f the same as input' % (em_iteration_times, cnt_validation_in_pre/len(compare_validation_whether_pre), cnt_validation_same_as_input/len(compare_validation_whether_pre)))


    """
    Construct data based on the OrderingGenerator
    """
    time_record = time.time()
    data_prepare_bash_file = "data_building_for_orderings_" + str(em_iteration_times) + ".sh"
    with open(data_prepare_bash_file, 'w') as f:
        print("#! /bin/bash", file=f)
    pids = []
    trainingset_log_file = []
    for data_set in ['training', 'validation']:
        logger.info('[EM iteration] Times %d: according to ordering prediction, build %s dataset for molecule_generator' % (em_iteration_times, data_set))
        if data_set == 'validation':
            commands_set = """nohup python -u contrastive_data_building_for_orderings.py \
            --begin {begin} \
            --end {end} \
            --data_root {data_root} \
            --dataset {data_set} \
            > {output_log}""".format(
                begin = config.dataset.begin,
                end = config.dataset.end,  
                data_root = args.data_root + 'ordering_predict/' + str(em_iteration_times) + '/',   
                data_set = data_set,
                output_log = ordering_model_path + '/' + 'build_' + data_set + '_set.log',
            )
            if args.device == 'cuda':
                commands_set = "CUDA_VISIBLE_DEVICES=" + args.gpu_id + " " + commands_set
            with open(data_prepare_bash_file, 'a+') as f:
                print(commands_set + ' 2>&1 &', file=f)
                print("pidv=$!", file=f)
            print(commands_set)  
            pids.append("pidv")
        else:
            for d_idx in range(len(data_roots)):
                order_predict_df = pd.read_csv(data_roots[d_idx] + 'ordering_predict/' + str(em_iteration_times) + '/training/orderings.csv')
                order_begin = order_predict_df['mol_id'].min()
                order_end = order_predict_df['mol_id'].max() + 1
                for sub_idx in range(order_begin, order_end, config.dataset.building_fragment):
                    commands_set = """nohup python -u contrastive_data_building_for_orderings.py \
                    --begin {begin} \
                    --end {end} \
                    --data_root {data_root} \
                    --dataset {data_set} \
                    > {output_log}""".format(
                        begin = sub_idx,
                        end = min(sub_idx + config.dataset.building_fragment, order_end),  
                        data_root = data_roots[d_idx] + 'ordering_predict/' + str(em_iteration_times) + '/',   
                        data_set = data_set,
                        output_log = ordering_model_path + '/' + 'build_' + data_set + '_set_' + str(d_idx) + "_" + str(sub_idx) + '.log',
                    )
                    if args.device == 'cuda':
                        if d_idx <= 1:
                            commands_set = "CUDA_VISIBLE_DEVICES=" + args.gpu_id + " " + commands_set
                        else:  # Distributed
                            commands_set = "CUDA_VISIBLE_DEVICES=" + str(int(args.gpu_id) + 1)  + " " + commands_set
                    with open(data_prepare_bash_file, 'a+') as f:
                        print(commands_set + ' 2>&1 &', file=f)
                        print("pid" + str(d_idx) + "_" + str(sub_idx) + "=$!", file=f)
                    print(commands_set)   
                    pids.append("pid" + str(d_idx) + "_" + str(sub_idx))
                    trainingset_log_file.append(ordering_model_path + '/' + 'build_' + data_set + '_set_' + str(d_idx) + "_" + str(sub_idx) + '.log')

    with open(data_prepare_bash_file, 'a+') as f:
        for pid in pids:
            print("wait $" + pid, file=f)
    commands_set = "bash " + data_prepare_bash_file
    commands_execute(commands_set)
    logger.info('time cost: %.4fs' % (time.time() - time_record))

    validationset_log_file = ordering_model_path + '/' + 'build_validation_set.log'

    for log_file in trainingset_log_file:
        with open(log_file) as fp:
            log_str = fp.read()
        train_data_situation = re.findall('train_total_data, train_subgraph_num, train_total_docked_failure (\-?\d+\.?\d*) (\-?\d+\.?\d*) (\-?\d+\.?\d*)', log_str)[0]

    with open(validationset_log_file) as fp:
        log_str = fp.read()
    val_data_situation = re.findall('val_total_data, val_subgraph_num, val_total_docked_failure (\-?\d+\.?\d*) (\-?\d+\.?\d*) (\-?\d+\.?\d*)', log_str)[0]

    builded_training_file_path = []
    builded_training_total_file_path = []
    for data_root in data_roots:
        cur_file_path = get_file(data_root + 'ordering_predict/' + str(em_iteration_times) + '/training/', 'pt', [])
        builded_training_file_path.append(cur_file_path)
        builded_training_total_file_path.extend(cur_file_path)
    builded_validation_file_path = get_file(args.data_root + 'ordering_predict/' + str(em_iteration_times) + '/validation/', 'pt', [])
    logger.info('[EM iteration] Times %d: dataset builded for molecule_generator, training %d, validation %d' % (em_iteration_times, len(builded_training_total_file_path), len(builded_validation_file_path)))

    pd.DataFrame(builded_training_total_file_path, columns=['path']).to_csv(args.data_root + 'ordering_predict/' + str(em_iteration_times) + '/train_data.csv', index=False)
    pd.DataFrame(builded_validation_file_path, columns=['path']).to_csv(args.data_root + 'ordering_predict/' + str(em_iteration_times) + '/val_data.csv', index=False)

    """
    MoleculeGenerator
    """
    commands = """python train_molecule_generation.py \
    --max_iters {max_iters} \
    --logdir {logdir} \
    --train_file_path {train_file_path} \
    --val_file_path {val_file_path} \
    --device {device}""".format(
        max_iters = config.train.m_gen_max_iters,
        logdir = log_dir + '/' + str(em_iteration_times) + '/',
        train_file_path = args.data_root + 'ordering_predict/' + str(em_iteration_times) + '/train_data.csv',
        val_file_path = args.data_root + 'ordering_predict/' + str(em_iteration_times) + '/val_data.csv',
        device = args.device,
    )
    if args.device == 'cuda':
        commands = "CUDA_VISIBLE_DEVICES=" + args.gpu_id + " " + commands

    print(commands)
    time_record = time.time()
    logger.info('[EM iteration] Times %d: training molecule_generator' % (em_iteration_times))
    output = commands_execute(commands)
    

    molecule_generation_model_path = standardise_str(output)
    molgen_best_model_path, molgen_best_model_iteration, molgen_best_model_val_loss, \
    molgen_pred_loss, molgen_attach_loss, molgen_focal_loss = analysis_checkpoints_performance(
        molecule_generation_model_path, 
        '\[Validate\] Iter (.*) \| Loss (\-?\d+\.?\d*) \| Loss\(Pred\) (\-?\d+\.?\d*) \| Loss\(attach\) (\-?\d+\.?\d*) \| Loss\(Focal\) (\-?\d+\.?\d*)',
        5
    )
    logger.info('[EM iteration] Molecule_generator: Iter %05d | Loss %.6f | Loss(Pred) %.6f | Loss(attach) %.6f | Loss(Focal) %.6f' % (molgen_best_model_iteration, molgen_best_model_val_loss, molgen_pred_loss, molgen_attach_loss, molgen_focal_loss))
    logger.info('time cost: %.4fs' % (time.time() - time_record))
    print("molgen_best_model_path", molgen_best_model_path)

    mksure_path(log_dir + '/' + str(em_iteration_times) + '/' + 'generated_mols/')
    commands_test = """bash sample_molecule.sh {gpu} {device} {outdir} {ckpt}""".format(
        gpu = args.gpu_id,
        device = args.device,
        outdir = log_dir + '/' + str(em_iteration_times) + '/' + 'generated_mols',
        ckpt = molgen_best_model_path,
    )
    print(commands_test)
    time_record = time.time()
    logger.info('[EM iteration] Times %d: sampling molecule_generator' % (em_iteration_times))
    commands_execute(commands_test)
    logger.info('time cost: %.4fs' % (time.time() - time_record))

    """
    Evaluation
    """
    commands_evaluation = """python evaluateMolgen.py \
    --data_root {data_root} \
    --task {task}""".format(
        data_root = log_dir + '/' + str(em_iteration_times) + '/' + 'generated_mols',
        task = config.dataset.end,   
    )
    print(commands_evaluation)
    time_record = time.time()
    logger.info('[EM iteration] Times %d: evaluating molecule_generator' % (em_iteration_times))
    commands_execute(commands_evaluation)

    metric_file = log_dir + '/' + str(em_iteration_times) + '/' + 'generated_mols/generated_metrics.csv'
    metric_pd = pd.read_csv(metric_file)
    mol_gen_metrics = metric_pd.mean(numeric_only=True)
    logger.info('[EM iteration] Times %d: evaluate molecule_generator | average generated molecules %d | best vina score %.6f | avg vina score %.6f | high affinity %.6f | QED %.6f | LogP %.6f | SA %.6f | Lipinski %.6f' 
                % (em_iteration_times, mol_gen_metrics['generated_mol'], mol_gen_metrics['best_vina_score'], mol_gen_metrics['avg_vina_score'],
                mol_gen_metrics['high_affinity'], mol_gen_metrics['QED'], mol_gen_metrics['LogP'], mol_gen_metrics['SA'], mol_gen_metrics['Lipinski']))
    logger.info('time cost: %.4fs' % (time.time() - time_record))

    """
    Score orderings
    """
    for data_root in data_roots:
        mksure_path(data_root + 'ordering_score/' + str(em_iteration_times) + '/')
    time_record = time.time()
    order_score_bash_file = "order_score_" + str(em_iteration_times) + ".sh"
    with open(order_score_bash_file, 'w') as f:
        print("#! /bin/bash", file=f)
    pids = []

    for data_set in ['training', 'validation']:
        logger.info('[EM iteration] Times %d: scoring predefined orderings for %s set' % (em_iteration_times, data_set))
        if data_set == 'validation':
            commands_score_orderings = """python scoreOrderings.py \
            --data_root {data_root} \
            --dataset {data_set} \
            --begin {begin} \
            --end {end} \
            --outdir {outdir} \
            --ckpt {ckpt} \
            --device {device} > {output_log}""".format(
                data_root = args.data_root,
                data_set = data_set, 
                begin = config.dataset.begin,
                end = config.dataset.end,
                outdir = args.data_root + 'ordering_score/' + str(em_iteration_times) + '/',
                ckpt = molgen_best_model_path,
                device = args.device,
                output_log = molecule_generation_model_path + '/' + 'score_' + data_set + '_set.log',
            )
            if args.device == 'cuda':
                commands_score_orderings = "CUDA_VISIBLE_DEVICES=" + args.gpu_id + " " + commands_score_orderings
            with open(order_score_bash_file, 'a+') as f:
                print(commands_score_orderings + ' 2>&1 &', file=f)
                print("pidv=$!", file=f)
            print(commands_score_orderings) 
            pids.append("pidv")
            
        else:
            for d_idx, data_root in enumerate(data_roots):
                order_predict_df = pd.read_csv(data_root + 'ordering_predict/' + str(em_iteration_times) + '/training/orderings.csv')
                order_begin = order_predict_df['mol_id'].min()
                order_end = order_predict_df['mol_id'].max() + 1

                for sub_idx in range(order_begin, order_end, config.dataset.building_fragment*2):
                    commands_score_orderings = """python scoreOrderings.py \
                    --data_root {data_root} \
                    --dataset {data_set} \
                    --begin {begin} \
                    --end {end} \
                    --outdir {outdir} \
                    --ckpt {ckpt} \
                    --device {device} > {output_log}""".format(
                        data_root = data_root,
                        data_set = data_set, 
                        begin = sub_idx,
                        end = min(sub_idx + config.dataset.building_fragment*2, order_end),  
                        outdir = data_root + 'ordering_score/' + str(em_iteration_times) + '/',
                        ckpt = molgen_best_model_path,
                        device = args.device,
                        output_log = molecule_generation_model_path + '/' + 'score_' + data_set + '_set_' + str(d_idx) + "_" + str(sub_idx) + '.log',
                    )
                    if args.device == 'cuda':
                        if d_idx <= 1:
                            commands_score_orderings = "CUDA_VISIBLE_DEVICES=" + args.gpu_id + " " + commands_score_orderings
                        else:  # Distributed
                            commands_score_orderings = "CUDA_VISIBLE_DEVICES=" + str(int(args.gpu_id) + 1)  + " " + commands_score_orderings
                    with open(order_score_bash_file, 'a+') as f:
                        print(commands_score_orderings + ' 2>&1 &', file=f)
                        print("pid" + str(d_idx) + "_" + str(sub_idx) + "=$!", file=f)
                    print(commands_score_orderings)
                    pids.append("pid" + str(d_idx) + "_" + str(sub_idx))
            
    with open(order_score_bash_file, 'a+') as f:
        for pid in pids:
            print("wait $" + pid, file=f)
    commands_score_orderings = "bash " + order_score_bash_file
    commands_execute(commands_score_orderings)
    logger.info('time cost: %.4fs' % (time.time() - time_record))

    logger.info('Total time cost in em iteration: %.4fs' % (time.time() - time_total_em_record))
