import os
import logging
import random
import argparse
import warnings
import sys
import pathlib
from sys import platform

import pandas as pd
from addict import Dict
sys.path.insert(0, './')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp

from SPICE.fixmatch.utils import net_builder, get_logger, count_parameters
from SPICE.fixmatch.train_utils import TBLog, get_SGD, get_cosine_schedule_with_warmup
from SPICE.fixmatch.models.fixmatch.rfixmatch_v1 import FixMatch
from SPICE.fixmatch.datasets.ssl_dataset_robust import SSL_Dataset
from SPICE.fixmatch.datasets.data_utils import get_data_loader

# Import utils
parent_dir = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))
import utils
from utils_data import OCTDatasetSSL

# Img size and moco_dim (nb of classes) values based on the dataset
img_size_dict = {'stl10': 96,
                 'cifar10': 32,
                 'cifar100': 32}
num_cluster_dict = {'stl10': 10,
                    'cifar10': 10,
                    'cifar100': 100}

mean, std = {}, {}
mean['cifar10'] = [x / 255 for x in [125.3, 123.0, 113.9]]
mean['cifar100'] = [x / 255 for x in [129.3, 124.1, 112.4]]
mean['stl10'] = [0.485, 0.456, 0.406]
mean['npy'] = [0.485, 0.456, 0.406]
mean['npy224'] = [0.485, 0.456, 0.406]
mean['oct'] = [149.888, 149.888, 149.888]

std['cifar10'] = [x / 255 for x in [63.0, 62.1, 66.7]]
std['cifar100'] = [x / 255 for x in [68.2,  65.4,  70.4]]
std['stl10'] = [0.229, 0.224, 0.225]
std['npy'] = [0.229, 0.224, 0.225]
std['npy224'] = [0.229, 0.224, 0.225]
std['oct'] = [11.766, 11.766, 11.766]

def main(args):
    '''
    For (Distributed)DataParallelism,
    main(args) spawn each process (main_worker) to each GPU.
    '''
    
    save_path = os.path.join(args.save_dir, args.save_name)
    if os.path.exists(save_path) and not args.overwrite:
        raise Exception('already existing model: {}'.format(save_path))
    if args.resume:
        if args.load_path is None:
            raise Exception('Resume of training requires --load_path in the args')
        if os.path.abspath(save_path) == os.path.abspath(args.load_path) and not args.overwrite:
            raise Exception('Saving & Loading pathes are same. \
                            If you want over-write, give --overwrite in the argument.')
        
    if args.seed is not None:
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
        
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
    
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    
    #distributed: true if manually selected or if world_size > 1
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed 
    ngpus_per_node = torch.cuda.device_count() # number of gpus of each node
    
    if args.multiprocessing_distributed:
        # now, args.world_size means num of total processes in all nodes
        args.world_size = ngpus_per_node * args.world_size 
        
        #args=(,) means the arguments of main_worker
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args)) 
    else:
        main_worker(args.gpu, ngpus_per_node, args)
    

def main_worker(gpu, ngpus_per_node, args):
    '''
    main_worker is conducted on each GPU.
    '''
    
    global best_acc1
    args.gpu = gpu
    
    # random seed has to be set for the syncronization of labeled data sampling in each process.
    assert args.seed is not None
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.deterministic = True
    

    # SET UP FOR DISTRIBUTED TRAINING
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu # compute global rank
        
        # set distributed group:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    
    
    #SET save_path and logger
    save_path = os.path.join(args.save_dir, args.save_name)
    logger_level = "WARNING"
    tb_log = None
    if args.rank % ngpus_per_node == 0:
        tb_log = TBLog(save_path, 'tensorboard')
        logger_level = "INFO"
    
    logger = get_logger(args.save_name, save_path, logger_level)
    logger.warning(f"USE GPU: {args.gpu} for training")

    # SET FixMatch: class FixMatch in models.fixmatch
    args.bn_momentum = 1.0 - args.ema_m
    if args.net in ['WideResNet', 'WideResNet_stl10', 'WideResNet_tiny', 'resnet18', 'resnet18_cifar', 'resnet34']:
        _net_builder = net_builder(args.net,
                                   args.net_from_name,
                                   {'depth': args.depth,
                                    'widen_factor': args.widen_factor,
                                    'leaky_slope': args.leaky_slope,
                                    'bn_momentum': args.bn_momentum,
                                    'dropRate': args.dropout})
    else:
        raise TypeError

    model = FixMatch(_net_builder,
                     args.num_classes,
                     args.ema_m,
                     args.T,
                     args.p_cutoff,
                     args.ulb_loss_ratio,
                     args.hard_label,
                     num_eval_iter=args.num_eval_iter,
                     tb_log=tb_log,
                     logger=logger)

    logger.info(f'Number of Trainable Params: {count_parameters(model.train_model)}')
        

    # SET Optimizer & LR Scheduler
    ## construct SGD and cosine lr scheduler
    optimizer = get_SGD(model.train_model, 'SGD', args.lr, args.momentum, args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                args.num_train_iter,
                                                num_warmup_steps=args.num_train_iter*0)
    ## set SGD and cosine lr on FixMatch 
    model.set_optimizer(optimizer, scheduler)
    
    
    # SET Devices for (Distributed) DataParallel
    if not torch.cuda.is_available():
        raise Exception('ONLY GPU TRAINING IS SUPPORTED')
    elif args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            
            '''
            batch_size: batch_size per node -> batch_size per gpu
            workers: workers per node -> workers per gpu
            '''
            args.batch_size = int(args.batch_size / ngpus_per_node)
            model.train_model.cuda(args.gpu)
            model.train_model = torch.nn.parallel.DistributedDataParallel(model.train_model,
                                                                          device_ids=[args.gpu])
            model.eval_model.cuda(args.gpu)
            
        else:
            # if arg.gpu is None, DDP will divide and allocate batch_size
            # to all available GPUs if device_ids are not set.
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
            
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model.train_model = model.train_model.cuda(args.gpu)
        model.eval_model = model.eval_model.cuda(args.gpu)
        
    else:
        model.train_model = torch.nn.DataParallel(model.train_model).cuda()
        model.eval_model = torch.nn.DataParallel(model.eval_model).cuda()
    
    logger.info(f"model_arch: {model}")
    logger.info(f"Arguments: {args}")
    
    cudnn.benchmark = True

    # Construct Dataset & DataLoader
    if args.dataset == 'oct':
        reliable_labels = np.load(args.label_file).astype(np.long)
        reliable_labels = np.where(reliable_labels >= 0)[0] # shape: (10,)
        # DEBUG: Resample from current map df
        reliable_labels = pd.read_csv(args.map_df_paths['train']).groupby('label').sample(2).index.to_numpy()
        # print(reliable_labels)
        lb_dset = OCTDatasetSSL(root=args.data_dir, split='train', map_df_paths=args.map_df_paths, labels_dict=args.labels_dict,
                                reliable_label_idxs=reliable_labels, return_just_reliable=True,
                                use_strong_transform=False, strong_transforms=None, one_hot=False)
        ulb_dset = OCTDatasetSSL(root=args.data_dir, split='train', map_df_paths=args.map_df_paths,
                                labels_dict=args.labels_dict,
                                reliable_label_idxs=reliable_labels, return_just_reliable=False,
                                use_strong_transform=True, strong_transforms=None, one_hot=False)

    else:
        train_dset = SSL_Dataset(name=args.dataset, train=True, label_file=args.label_file, all=args.all, unlabeled=args.unlabeled,
                                 num_classes=args.num_classes, data_dir=args.data_dir)
        lb_dset, ulb_dset = train_dset.get_ssl_dset(args.num_labels)
        # lb_dset and ulb_dset: BasicDataset

    if args.dataset == 'oct':
        eval_dset = OCTDatasetSSL(root=args.data_dir, split='valid', map_df_paths=args.map_df_paths,
                                labels_dict=args.labels_dict,
                                reliable_label_idxs=None, return_just_reliable=False,
                                use_strong_transform=False, strong_transforms=None, one_hot=False)

    else:
        oct_args = None
        _eval_dset = SSL_Dataset(name=args.dataset, train=False, label_file=None, all=args.all, unlabeled=False,
                                 num_classes=args.num_classes, data_dir=args.data_dir, oct_args=oct_args)
        eval_dset = _eval_dset.get_dset()
        # eval_dset: BasicDataset
    
    loader_dict = {}
    dset_dict = {'train_lb': lb_dset, 'train_ulb': ulb_dset, 'eval': eval_dset}
    
    loader_dict['train_lb'] = get_data_loader(dset_dict['train_lb'],
                                              args.batch_size,
                                              data_sampler = args.train_sampler,
                                              num_iters=args.num_train_iter,
                                              num_workers=args.num_workers, 
                                              distributed=args.distributed)
    
    loader_dict['train_ulb'] = get_data_loader(dset_dict['train_ulb'],
                                               args.batch_size*args.uratio,
                                               data_sampler = args.train_sampler,
                                               num_iters=args.num_train_iter,
                                               num_workers=4*args.num_workers,
                                               distributed=args.distributed)
    
    loader_dict['eval'] = get_data_loader(dset_dict['eval'],
                                          args.eval_batch_size, 
                                          num_workers=args.num_workers)
    
    ## set DataLoader on FixMatch
    model.set_data_loader(loader_dict)
    
    #If args.resume, load checkpoints from args.load_path
    """
    resume = '{}/{}/model_last.pth'.format(args.save_dir, args.save_name)
    if os.path.exists(resume):
        args.load_path = resume
        # args.resume = True
    """
    if args.resume:
        model.load_model(args.load_path)
    
    # START TRAINING of FixMatch
    trainer = model.train
    for epoch in range(args.epoch):
        trainer(args, logger=logger)
        
    if not args.multiprocessing_distributed or \
                (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        model.save_model('latest_model.pth', save_path)
        
    logging.warning(f"GPU {args.rank} training is FINISHED")
    

if __name__ == "__main__":
    # Set up the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path',
                        help='Path to the config file',
                        type=str)

    args = parser.parse_args()

    if args.config_path is None:
        args.config_path = pathlib.Path('../../config.yaml')
    config_file = pathlib.Path(args.config_path)

    if not config_file.exists():
        print(f'Config file not found at {args.config_path}')
        raise SystemExit(1)

    configs = utils.load_configs(config_file)
    if platform == "linux" or platform == "linux2":
        dataset_root = pathlib.Path(configs['data']['dataset_root_linux'])
        dataset_path = pathlib.Path(configs['SPICE']['MoCo']['dataset_path_linux'])
    elif platform == "win32":
        dataset_root = pathlib.Path(configs['data']['dataset_root_windows'])
        dataset_path = pathlib.Path(configs['SPICE']['MoCo']['dataset_path_windows'])
    ds_split = configs['data']['ds_split']
    labels = configs['data']['labels']
    ascan_per_group = configs['data']['ascan_per_group']
    use_mini_dataset = configs['data']['use_mini_dataset']
    # oct_img_root = pathlib.Path(f'OCT_lab_data/{ascan_per_group}mscans')
    img_size_dict['oct'] = (512, ascan_per_group)
    num_cluster_dict['oct'] = len(labels)

    # Saving and loading of the model
    moco_dataset_name = configs['SPICE']['MoCo']['dataset_name']
    args.save_name = configs['SPICE']['semi']['model_name']
    args.save_dir = f"{configs['SPICE']['MoCo']['save_folder']}/{moco_dataset_name}/{args.save_name}"
    args.resume = configs['SPICE']['semi']['resume']
    args.load_path = pathlib.Path(configs['SPICE']['MoCo']['save_folder']).joinpath(moco_dataset_name).joinpath(
        configs['SPICE']['semi']['model_name']).joinpath('model_last.pth.tar')
    args.overwrite = configs['SPICE']['semi']['overwrite']

    # Training configuration of FixMatch
    args.epoch = configs['SPICE']['semi']['epoch']
    args.num_train_iter = eval(configs['SPICE']['semi']['num_train_iter'])
    args.num_eval_iter = configs['SPICE']['semi']['num_eval_iter']
    args.num_labels = configs['SPICE']['semi']['num_labels']
    args.batch_size = configs['SPICE']['semi']['batch_size']
    args.uratio = configs['SPICE']['semi']['uratio']
    args.eval_batch_size = configs['SPICE']['semi']['eval_batch_size']
    args.hard_label = configs['SPICE']['semi']['hard_label']
    args.T = configs['SPICE']['semi']['T']
    args.p_cutoff = configs['SPICE']['semi']['p_cutoff']
    args.ema_m = configs['SPICE']['semi']['ema_m']
    args.ulb_loss_ratio = configs['SPICE']['semi']['ulb_loss_ratio']

    # Optimizer configurations
    args.lr = configs['SPICE']['semi']['lr']
    args.momentum = configs['SPICE']['semi']['momentum']
    args.weight_decay = eval(configs['SPICE']['semi']['weight_decay'])
    args.amp = configs['SPICE']['semi']['amp']

    # Backbone net configurations
    args.net = configs['SPICE']['semi']['net']
    args.net_from_name = configs['SPICE']['semi']['net_from_name']
    args.depth = configs['SPICE']['semi']['depth']
    args.widen_factor = configs['SPICE']['semi']['widen_factor']
    args.leaky_slope = configs['SPICE']['semi']['leaky_slope']
    args.dropout = configs['SPICE']['semi']['dropout']

    # Data configurations
    args.data_dir = dataset_path.joinpath(
        'OCT_lab_data' if moco_dataset_name == 'oct' else moco_dataset_name)
    args.dataset = moco_dataset_name
    args.label_file = f"{configs['SPICE']['MoCo']['save_folder']}/{moco_dataset_name}/{configs['SPICE']['local_consistency']['model_name']}/labels_reliable.npy"
    args.all = configs['SPICE']['semi']['all']
    args.unlabeled = configs['SPICE']['semi']['unlabeled']
    args.train_sampler = configs['SPICE']['semi']['train_sampler']
    args.num_classes = num_cluster_dict[moco_dataset_name]
    args.seed = configs['training']['random_seed']
    args.labels_dict = {i: lbl for i, lbl in enumerate(labels)}
    args.map_df_paths = {
        split: dataset_root.joinpath(f"{ascan_per_group}mscans").joinpath(
            f"{split}{'Mini' if use_mini_dataset else ''}_mapping_{ascan_per_group}scans.csv")
        for split in ['train', 'valid', 'test']}

    # Multi-GPU and distributed training
    # More info on values: https://stackoverflow.com/a/76828907
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    args.num_workers = configs['SPICE']['MoCo']['num_workers']
    args.world_size = configs['SPICE']['MoCo']['world_size']
    args.rank = configs['SPICE']['MoCo']['rank']
    args.dist_url = configs['SPICE']['MoCo']['dist_url']
    args.dist_backend = configs['SPICE']['MoCo']['dist_backend']
    args.gpu = None if configs['SPICE']['MoCo']['gpu_id'] == 'None' else configs['SPICE']['MoCo']['gpu_id']
    args.multiprocessing_distributed = configs['SPICE']['MoCo']['multiprocessing_distributed']

    # python ./tools/train_semi.py
    # --unlabeled 1
    # --num_classes 10
    # --label_file ./results/stl10/eval/labels_reliable.npy
    # --save_dir ./results/stl10/spice_semi
    # --save_name semi
    # --batch_size 64
    # --net WideResNet_stl10
    # --data_dir ./datasets/stl10
    # --dataset stl10



    main(args)
