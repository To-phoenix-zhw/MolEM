import os
import re
import time
import random
import logging
import torch
import numpy as np
import pandas as pd
import yaml
import subprocess
from easydict import EasyDict
from logging import Logger
from tqdm.auto import tqdm


class BlackHole(object):
    def __setattr__(self, name, value):
        pass
    def __call__(self, *args, **kwargs):
        return self
    def __getattr__(self, name):
        return self


def load_config(path):
    with open(path, 'r') as f:
        return EasyDict(yaml.safe_load(f))


def get_logger(name, log_dir=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s::%(name)s::%(levelname)s] %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_dir is not None:
        file_handler = logging.FileHandler(os.path.join(log_dir, 'log.txt'))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_new_log_dir(root='./logs', prefix='', tag=''):
    fn = time.strftime('%Y_%m_%d__%H_%M_%S', time.localtime())
    if prefix != '':
        fn = prefix + '_' + fn
    if tag != '':
        fn = fn + '_' + tag
    log_dir = os.path.join(root, fn)
    os.makedirs(log_dir)
    return log_dir


def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def log_hyperparams(writer, args):
    from torch.utils.tensorboard.summary import hparams
    vars_args = {k:v if isinstance(v, str) else repr(v) for k, v in vars(args).items()}
    exp, ssi, sei = hparams(vars_args, {})
    writer.file_writer.add_summary(exp)
    writer.file_writer.add_summary(ssi)
    writer.file_writer.add_summary(sei)


def int_tuple(argstr):
    return tuple(map(int, argstr.split(',')))


def str_tuple(argstr):
    return tuple(argstr.split(','))


def mksure_path(dirs_or_files):
    if not os.path.exists(dirs_or_files):
        os.makedirs(dirs_or_files)
        

def get_file(file_path: str, suffix: str, res_file_path: list) -> list:

    for file in os.listdir(file_path):

        if os.path.isdir(os.path.join(file_path, file)):
            get_file(os.path.join(file_path, file), suffix, res_file_path)
        else:
            res_file_path.append(os.path.join(file_path, file))
 
    return res_file_path if suffix == '' or suffix is None else list(filter(lambda x: x.endswith(suffix), res_file_path))


def extract_float_using_regex(string):
    pattern = r"\d+\.\d+" 
    match = re.search(pattern, string)
    if match:
        return match.group()
    else:
        return None



def get_data_in_domain(file_path, data_domain):
    data_file_path = []
    for i in range(len(file_path)):
        mol_id = int(file_path[i].split('/')[-3]) 
        if mol_id in data_domain:
            data_file_path.append(file_path[i])
    return data_file_path


def standardise_str(input_str):
    input_str = input_str.rstrip().decode("utf-8")
    return input_str



def commands_execute(command):
    proc = subprocess.Popen(
        '/bin/bash', 
        shell=False, 
        stdin=subprocess.PIPE, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE
    )
    proc.stdin.write(command.encode('utf-8'))
#     proc.stdin.close()
    if proc is None:   # Not started
        raise Exception('Process not started!')
    stdout, stderr = proc.communicate()
    # output = stdout.readlines()
    # while proc.poll() is None:  # In progress
    #     continue   
    # output = proc.stdout.readlines()
    return stdout


def analysis_checkpoints_performance(model_path, key_str, mode): 
    log_path = model_path + '/log.txt'
    checkpoints_path = model_path + '/checkpoints'
    
    checkpoints = get_file(checkpoints_path, 'pt', []) 
    ckpts_pd = pd.DataFrame(checkpoints, columns=['checkpoint'])
    ckpts_pd["iteration"] = ckpts_pd['checkpoint'].map(lambda x: int(x.split('/')[-1].split('_')[0]))
    ckpts_pd["val_loss"] = ckpts_pd['checkpoint'].map(lambda x: float(extract_float_using_regex(x)))
    
    best_idx = ckpts_pd["val_loss"].argmin()
    best_model_path = ckpts_pd.iloc[best_idx]["checkpoint"]
    best_model_iteration = ckpts_pd.iloc[best_idx]["iteration"]
    best_model_val_loss = ckpts_pd.iloc[best_idx]["val_loss"]
    
    with open(log_path) as fp:
        log_str = fp.read()
    val_iter_loss = re.findall(key_str, log_str)
    iters = [int(i[0]) for i in val_iter_loss]
    losses = [float(i[1]) for i in val_iter_loss]
    best_log_idx = iters.index(best_model_iteration)
    assert losses[best_log_idx] == best_model_val_loss   
 
    if mode == 3:
        accs = [float(i[2]) for i in val_iter_loss]
        return best_model_path, iters[best_log_idx], losses[best_log_idx], accs[best_log_idx]
    elif mode == 5:
        predlosses = [float(i[2]) for i in val_iter_loss]
        attachlosses = [float(i[3]) for i in val_iter_loss]
        focallosses = [float(i[4]) for i in val_iter_loss]
        return best_model_path, iters[best_log_idx], losses[best_log_idx], predlosses[best_log_idx], attachlosses[best_log_idx], focallosses[best_log_idx]
    else:
        raise Exception('Mode error!')