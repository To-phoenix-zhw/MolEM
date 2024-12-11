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
from models.OrderingGenerator import OrderingGenerator
from utils.datasets import *
from utils.misc import *
from utils.train import *
from utils.data import *
from utils.mol_tree import *
from utils.transforms import *
from torch.utils.data import DataLoader

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/train_ordering.yml')
    parser.add_argument('--max_iters', type=int)
    parser.add_argument('--train_file_path', type=str)
    parser.add_argument('--val_file_path', type=str)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--logdir', type=str, default='./logs')
    parser.add_argument('--vocab_path', type=str, default='./utils/vocab.txt')
    parser.add_argument('--early_stopping', type=int, default=50000)
    args = parser.parse_args()

    # Load configs
    config = load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    seed_all(config.train.seed)

    # Logging
    log_dir = get_new_log_dir(args.logdir, prefix=config_name)
    print(log_dir)   
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = get_logger('train', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    logger.info(args)
    logger.info(config)
    logger.info(log_dir)
    shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
    shutil.copytree('./models', os.path.join(log_dir, 'models'))


    # Load vocab
    vocab = []
    for line in open(args.vocab_path):
        p, _, _ = line.partition(':')
        vocab.append(p)
    vocab = Vocab(vocab)
    logger.info('Vocab length: %d' % (vocab.size()))
    

    
    # Transforms
    protein_featurizer = FeaturizeProteinAtom()
    ligand_featurizer = FeaturizeLigandAtom()

    # Datasets
    logger.info('Loading dataset...')
    train_dataset, val_dataset = get_dataset(config=config.dataset, train_file_path=args.train_file_path, val_file_path=args.val_file_path)
    logger.info('Dataset length: %d %d ' % (len(train_dataset), len(val_dataset)))

    # Model
    logger.info('Building model...')
    #ckpt = torch.load(config.model.checkpoint, map_location=args.device)
    model = OrderingGenerator(
        config.model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim,
        vocab=vocab,
        device=args.device).to(args.device).cuda()
    #model.load_state_dict(ckpt['model'])

    # Optimizer and scheduler
    optimizer = get_optimizer(config.train.optimizer, model)
    scheduler = get_scheduler(config.train.scheduler, optimizer)


    def train(it, batch):
        model.train()
        optimizer.zero_grad() 
        for key in batch:
            if isinstance(batch[key], list):
                continue
            batch[key] = batch[key].to(args.device)
        protein_noise = torch.randn_like(batch['protein_pos']) * config.train.pos_noise_std
        ligand_noise = torch.randn_like(batch['ligand_context_pos']) * config.train.pos_noise_std


        loss, acc_cnt, sample_cnt = model.get_loss(
            protein_pos=batch['protein_pos'] + protein_noise.cuda(),
            protein_atom_feature=batch['protein_atom_feature'].float(),
            ligand_pos=batch['ligand_context_pos'] + ligand_noise.cuda(),
            ligand_atom_feature=batch['ligand_atom_feature_full'].float(),
            batch_protein=batch['protein_element_batch'],
            batch_ligand=batch['ligand_element_batch'],
            batch=batch)
        loss.backward()
        orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
        optimizer.step()

        # logger.info('[Train] Iter %d | Loss %.6f | Orig_grad_norm %.6f' % (it, loss.item(), orig_grad_norm))
        logger.info('[Train] Iter %d | Loss %.6f | Orig_grad_norm %.6f | batch acc %.6f | acc_cnt %d | sample_cnt %d' % (it, loss.item(), orig_grad_norm, acc_cnt/sample_cnt, acc_cnt, sample_cnt))

        writer.add_scalar('train/loss', loss, it)
        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
        writer.add_scalar('train/grad', orig_grad_norm, it)
        writer.add_scalar('train/batch_acc', acc_cnt/sample_cnt, it)
        writer.flush()

        return acc_cnt, sample_cnt



    def validate(it):
        sum_loss, sum_n = 0, 0
        sum_acc, sum_sample = 0, 0
        sum_pred_loss, sum_attach_loss, sum_focal_loss = 0, 0, 0

        val_loader = DataLoader(val_dataset, batch_size=config.train.batch_size, shuffle=False,
                                num_workers=config.train.num_workers, collate_fn=lambda x: collate_mols_ordering(x, vocab.size()))
        with torch.no_grad():
            model.eval()
            for batch in tqdm(val_loader, desc='Validate'):
                for key in batch:
                    if isinstance(batch[key], list):
                        continue
                    batch[key] = batch[key].to(args.device)
                loss, acc_cnt, sample_cnt = model.get_loss(
                    protein_pos=batch['protein_pos'],
                    protein_atom_feature=batch['protein_atom_feature'].float(),
                    ligand_pos=batch['ligand_context_pos'],
                    ligand_atom_feature=batch['ligand_atom_feature_full'].float(),
                    batch_protein=batch['protein_element_batch'],
                    batch_ligand=batch['ligand_element_batch'],
                    batch=batch)
                sum_loss += loss.item()
                sum_acc += acc_cnt
                sum_sample += sample_cnt
                sum_n += 1
        avg_loss = sum_loss / sum_n
        avg_acc = sum_acc / sum_sample

        if config.train.scheduler.type == 'plateau':
            scheduler.step(avg_loss)
        elif config.train.scheduler.type == 'warmup_plateau':
            scheduler.step_ReduceLROnPlateau(avg_loss)
        else:
            scheduler.step()

        # logger.info('[Validate] Iter %05d | Loss %.6f' % (it, avg_loss))
        logger.info('[Validate] Iter %05d | Loss %.6f | acc %.6f | sum_acc %d | sum_sample %d' % (it, avg_loss, avg_acc, sum_acc, sum_sample))

        writer.add_scalar('val/loss', avg_loss, it)
        writer.add_scalar('val/acc', avg_acc, it)
        writer.flush()
        return avg_loss



    it = 1
    best_it = 0   
    stop = False
    best_val_loss = 1000000
    while it < args.max_iters + 1 and stop == False: 
        train_loader = DataLoader(train_dataset, batch_size=config.train.batch_size, shuffle=True,
                            num_workers=config.train.num_workers, collate_fn=lambda x: collate_mols_ordering(x, vocab.size()))
        sum_train_acc, sum_train_sample = 0, 0  
        for batch in train_loader:
            if it == args.max_iters + 1:  
                break
            train_batch_acc_cnt, train_batch_sample_cnt = train(it, batch)
            sum_train_acc += train_batch_acc_cnt
            sum_train_sample += train_batch_sample_cnt
            if it % config.train.val_freq == 0 or it == args.max_iters:
                val_loss = validate(it)
                ckpt_path = os.path.join(ckpt_dir, '%d_%.6f.pt' % (it, val_loss))
                torch.save({
                    'config': config,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'iteration': it,
                }, ckpt_path)
                 
                if val_loss < best_val_loss:  
                    best_val_loss = val_loss  
                    best_it = it   
                else:  
                    if it > best_it + args.early_stopping:  
                        stop = True
                        break  
            it += 1
        
        logger.info('[Train] Iter %d | epoch acc %.6f | sum_acc %d | sum_sample %d' % (it-1, sum_train_acc/sum_train_sample, sum_train_acc, sum_train_sample))
