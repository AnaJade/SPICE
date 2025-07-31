import argparse
import pathlib
import builtins
import math
import os
import random
import shutil
import time
import warnings
import sys
from sys import platform
from addict import Dict
sys.path.insert(0, './')
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
from SPICE.spice.config import Config
from SPICE.spice.data.build_dataset import build_dataset
from SPICE.spice.model.sim2sem import Sim2Sem
from SPICE.spice.solver import make_lr_scheduler, make_optimizer
from SPICE.spice.utils.miscellaneous import mkdir, save_config
from SPICE.spice.utils.evaluation import calculate_acc, calculate_nmi, calculate_ari
from SPICE.spice.utils.load_model_weights import load_model_weights
from SPICE.spice.utils.logger import setup_logger
import logging
from SPICE.spice.utils.comm import get_rank
import numpy as np

# Import utils
parent_dir = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))
import utils

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

# Img size and moco_dim (nb of classes) values based on the dataset
img_size_dict = {'stl10': 96,
                 'cifar10': 32,
                 'cifar100': 32}
num_cluster_dict = {'stl10': 10,
                    'cifar10': 10,
                    'cifar100': 100}

# Set up the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--config_path',
                    help='Path to the config file',
                    type=str)
"""
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument(
    "--config-file",
    default="./configs/stl10/spice_self.py",
    metavar="FILE",
    help="path to config file",
    type=str,
)

parser.add_argument(
    "--all",
    default=1,
    type=int,
)
"""


def main():
    args = parser.parse_args()
    if args.config_path is None:
        if platform == "linux" or platform == "linux2":
            args.config_path = pathlib.Path('../../config.yaml')
        elif platform == "win32":
            args.config_path = pathlib.Path('../../config_windows.yaml')
    config_file = pathlib.Path(args.config_path)
    if not config_file.exists():
        print(f'Config file not found at {args.config_path}')
        raise SystemExit(1)
    configs = utils.load_configs(config_file)
    dataset_root = pathlib.Path(configs['data']['dataset_root'])
    ds_split = configs['data']['ds_split']
    labels = configs['data']['labels']
    ascan_per_group = configs['data']['ascan_per_group']
    use_mini_dataset = configs['data']['use_mini_dataset']
    img_size_dict['oct'] = (512, ascan_per_group)
    num_cluster_dict['oct'] = len(labels)
    ### Convert config file values to the args variable equivalent (match the format of the existing code)
    print("Assigning config values to corresponding args variables...")
    cfg = Dict()
    cfg.labels_dict = {i: lbl for i, lbl in enumerate(labels)}
    cfg.map_df_paths = {
        split: dataset_root.joinpath(f"{split}{'Mini' if use_mini_dataset else ''}_mapping_{ascan_per_group}scans.csv")
        for split in ['train', 'valid', 'test']}
    cfg.all = configs['SPICE']['SPICE_self']['all']
    cfg.model_name = configs['SPICE']['SPICE_self']['model_name']
    moco_dataset_name = configs['SPICE']['MoCo']['dataset_name']
    cfg.pre_model = pathlib.Path(configs['SPICE']['MoCo']['save_folder']).joinpath(moco_dataset_name).joinpath(
        'moco').joinpath(
        configs['SPICE']['embedding']['weight'])
    cfg.embedding = pathlib.Path(configs['SPICE']['MoCo']['save_folder']).joinpath(moco_dataset_name).joinpath(
        configs['SPICE']['embedding']['model_name']).joinpath('feas_moco_512_l2.npy')
    cfg.resume = pathlib.Path(configs['SPICE']['MoCo']['save_folder']).joinpath(moco_dataset_name).joinpath(
        configs['SPICE']['SPICE_self']['model_name']).joinpath(
        'checkpoint_last.pth.tar') if configs['SPICE']['SPICE_self']['resume'] else False
    cfg.model_type = configs['SPICE']['MoCo']['base_model']
    cfg.num_head = configs['SPICE']['SPICE_self']['num_head']
    cfg.num_workers = configs['SPICE']['MoCo']['num_workers']
    cfg.device_id = None if configs['SPICE']['MoCo']['gpu_id'] == 'None' else configs['SPICE']['MoCo']['gpu_id']
    cfg.num_train = configs['SPICE']['SPICE_self']['num_train']
    cfg.num_cluster = num_cluster_dict[moco_dataset_name]
    cfg.batch_size = configs['SPICE']['SPICE_self']['batch_size']
    cfg.target_sub_batch_size = configs['SPICE']['SPICE_self']['target_sub_batch_size']
    cfg.train_sub_batch_size = configs['SPICE']['SPICE_self']['train_sub_batch_size']
    cfg.batch_size_test = configs['SPICE']['SPICE_self']['batch_size_test']
    cfg.num_trans_aug = configs['SPICE']['SPICE_self']['num_trans_aug']
    cfg.num_repeat = configs['SPICE']['SPICE_self']['num_repeat']
    cfg.fea_dim = configs['SPICE']['SPICE_self']['fea_dim']
    cfg.att_conv_dim = num_cluster_dict[moco_dataset_name]
    cfg.att_size = configs['SPICE']['SPICE_self']['att_size']
    cfg.center_ratio = configs['SPICE']['SPICE_self']['center_ratio']
    cfg.sim_center_ratio = configs['SPICE']['SPICE_self']['sim_center_ratio']
    cfg.epochs = configs['SPICE']['SPICE_self']['epochs']
    cfg.seed = configs['training']['random_seed']

    # Multiprocessing
    # More info on values: https://stackoverflow.com/a/76828907
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    cfg.world_size = configs['SPICE']['MoCo']['world_size']
    cfg.workers = configs['SPICE']['MoCo']['num_workers']
    cfg.rank = configs['SPICE']['MoCo']['rank']
    cfg.dist_url = configs['SPICE']['MoCo']['dist_url']
    cfg.dist_backend = configs['SPICE']['MoCo']['dist_backend']
    cfg.gpu = None if configs['SPICE']['MoCo']['gpu_id'] == 'None' else configs['SPICE']['MoCo']['gpu_id']
    cfg.multiprocessing_distributed = configs['SPICE']['MoCo']['multiprocessing_distributed']

    cfg.start_epoch = configs['SPICE']['SPICE_self']['start_epoch']
    cfg.print_freq = configs['SPICE']['SPICE_self']['print_freq']
    cfg.test_freq = configs['SPICE']['SPICE_self']['test_freq']
    cfg.eval_ent = configs['SPICE']['SPICE_self']['eval_ent']
    cfg.eval_ent_weight = configs['SPICE']['SPICE_self']['eval_ent_weight']

    # Data train
    cfg.data_train = Dict(dict(
        type=f'{moco_dataset_name}_emb', # "stl10_emb",
        root_folder= pathlib.Path(configs['SPICE']['MoCo']['dataset_path']).joinpath(moco_dataset_name), # "./datasets/stl10",
        embedding=cfg.embedding,
        split='train', # "train+test",
        ims_per_batch=cfg.batch_size,
        shuffle=True,
        aspect_ratio_grouping=False,
        train=True,
        show=False,
        trans1=Dict(dict(
            aug_type="weak",
            crop_size=96,
            normalize=Dict(dict(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])),
        )),

        trans2=Dict(dict(
            aug_type="scan",
            crop_size=img_size_dict[moco_dataset_name],
            normalize=Dict(dict(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])),
            num_strong_augs=4,
            cutout_kwargs=Dict(dict(n_holes=1,
                               length=32,
                               random=True))
        )),
    ))
    cfg.data_train.labels_dict = {i: lbl for i, lbl in enumerate(labels)}
    cfg.data_train.map_df_paths = {
        split: dataset_root.joinpath(f"{split}{'Mini' if use_mini_dataset else ''}_mapping_{ascan_per_group}scans.csv")
        for split in ['train', 'valid', 'test']}


    # Data test
    cfg.data_test = Dict(dict(
        type=f'{moco_dataset_name}_emb', # "stl10_emb",
        root_folder=pathlib.Path(configs['SPICE']['MoCo']['dataset_path']).joinpath(
        'OCT_lab_data' if moco_dataset_name == 'oct' else moco_dataset_name), # pathlib.Path(configs['SPICE']['MoCo']['dataset_path']).joinpath(moco_dataset_name),# "./datasets/stl10",
        embedding=cfg.embedding,
        split='train' , # "train+test",
        shuffle=False,
        ims_per_batch=50,
        aspect_ratio_grouping=False,
        train=False,
        show=False,
        trans1=Dict(dict(
            aug_type="test",
            normalize=Dict(dict(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])),
        )),
        trans2=Dict(dict(
            aug_type="test",
            normalize=Dict(dict(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])),
        )),

    ))
    cfg.data_test.labels_dict = {i: lbl for i, lbl in enumerate(labels)}
    cfg.data_test.map_df_paths = {
        split: dataset_root.joinpath(f"{ascan_per_group}mscans").joinpath(
            f"{split}{'Mini' if use_mini_dataset else ''}_mapping_{ascan_per_group}scans.csv")
        for split in ['train', 'valid', 'test']}

    # Model
    cfg.model = Dict(dict(
        feature=Dict(dict(
            type=cfg.model_type,
            num_classes=cfg.num_cluster,
            in_channels=3,
            in_size=img_size_dict[moco_dataset_name],
            batchnorm_track=True,
            test=False,
            feature_only=True
        )),

        head=Dict(dict(type="sem_multi",
                  multi_heads=[Dict(dict(classifier=Dict(dict(type="mlp", num_neurons=[cfg.fea_dim, cfg.fea_dim, cfg.num_cluster], last_activation="softmax")),
                                    feature_conv=None,
                                    num_cluster=cfg.num_cluster,
                                    loss_weight=Dict(dict(loss_cls=1)),
                                    iter_start=cfg.epochs,
                                    iter_up=cfg.epochs,
                                    iter_down=cfg.epochs,
                                    iter_end=cfg.epochs,
                                    ratio_start=1.0,
                                    ratio_end=1.0,
                                    center_ratio=cfg.center_ratio,
                                    ))]*cfg.num_head,
                  )),
        model_type="moco",
        pretrained=cfg.pre_model,
        freeze_conv=True,
    ))

    # Solver
    cfg.solver = Dict(dict(
        type="adam",
        base_lr=0.005,
        bias_lr_factor=1,
        weight_decay=0,
        weight_decay_bias=0,
        target_sub_batch_size=cfg.target_sub_batch_size,
        batch_size=cfg.batch_size,
        train_sub_batch_size=cfg.train_sub_batch_size,
        num_repeat=cfg.num_repeat,
    ))

    # Results
    cfg.results.output_dir = f"{configs['SPICE']['MoCo']['save_folder']}/{moco_dataset_name}/{cfg.model_name}"

    ###################
    # Old code
    # args = parser.parse_args()
    # cfg = Config.fromfile(args.config_file)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.device_id)

    output_dir = cfg.results.output_dir
    if output_dir:
        mkdir(output_dir)

    output_config_path = os.path.join(output_dir, 'config.py')
    # save_config(cfg, output_config_path)

    # cfg.all = args.all
    if cfg.all:
        cfg.data_train.split = "train+test"
        cfg.data_train.all = True
        cfg.data_test.split = "train+test"
        cfg.data_test.all = True
    else:
        cfg.data_train.split = "train"
        cfg.data_train.all = False
        cfg.data_train.train = True
        cfg.data_test.split = "train"
        cfg.data_test.all = False
        cfg.data_test.train = True

    if cfg.seed is not None:
        random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if cfg.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if cfg.dist_url == "env://" and cfg.world_size == -1:
        cfg.world_size = int(os.environ["WORLD_SIZE"])

    cfg.distributed = cfg.world_size > 1 or cfg.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if cfg.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        cfg.world_size = ngpus_per_node * cfg.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, cfg.copy()))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, cfg)


def main_worker(gpu, ngpus_per_node, cfg):
    cfg.gpu = gpu
    # logger = logging.getLogger("{}.trainer".format(cfg.logger_name))
    logger_name = "spice"
    cfg.logger_name = logger_name

    logger = setup_logger(logger_name, cfg.results.output_dir, get_rank())

    # suppress printing if not master
    if cfg.multiprocessing_distributed and cfg.gpu != 0:
        def print_pass(*cfg):
            pass
        builtins.print = print_pass

    if cfg.gpu is not None:
        logger.info("Use GPU: {} for training".format(cfg.gpu))

    if cfg.distributed:
        if cfg.dist_url == "env://" and cfg.rank == -1:
            cfg.rank = int(os.environ["RANK"])
        if cfg.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            cfg.rank = cfg.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=cfg.dist_backend, init_method=cfg.dist_url,
                                world_size=cfg.world_size, rank=cfg.rank)
    # create model
    model = Sim2Sem(**cfg.model)
    logger.info(model)

    if cfg.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if cfg.gpu is not None:
            torch.cuda.set_device(cfg.gpu)
            model.cuda(cfg.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            cfg.batch_size = int(cfg.batch_size / ngpus_per_node)
            cfg.workers = int((cfg.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif cfg.gpu is not None:
        torch.cuda.set_device(cfg.gpu)
        model = model.cuda(cfg.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    optimizer = make_optimizer(cfg, model)
    scheduler = None

    if "lr_type" in cfg.solver:
        scheduler = make_lr_scheduler(cfg, optimizer)

    # optionally resume from a checkpoint
    if cfg.model.pretrained is not None:
        load_model_weights(model, cfg.model.pretrained, cfg.model.model_type, cfg.gpu)

    if cfg.resume:
        if os.path.isfile(cfg.resume):
            logger.info("=> loading checkpoint '{}'".format(cfg.resume))
            if cfg.gpu is None:
                checkpoint = torch.load(cfg.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(cfg.gpu)
                checkpoint = torch.load(cfg.resume, map_location=loc)
            cfg.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(cfg.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(cfg.resume))

    # Load similarity model

    cudnn.benchmark = True

    # Data loading code
    train_dataset = build_dataset(cfg.data_train)

    if cfg.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.target_sub_batch_size, shuffle=(train_sampler is None),
        num_workers=cfg.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    dataset_val = build_dataset(cfg.data_test)
    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=cfg.batch_size_test, shuffle=False, num_workers=1)

    best_acc = -2
    best_nmi = -1
    best_ari = -1
    best_head = -1
    best_epoch = -1
    min_loss = 1e10
    loss_head = -1
    loss_acc = -2
    loss_nmi = -1
    loss_ari = -1
    loss_epoch = -1
    eval_ent = cfg.eval_ent
    eval_ent_weight = cfg.eval_ent_weight
    for epoch in range(cfg.start_epoch, cfg.epochs):
        if cfg.distributed:
            train_sampler.set_epoch(epoch)
        if scheduler is not None:
            scheduler.step()

        # train for one epoch
        train(train_loader, model, optimizer, epoch, cfg)

        if not cfg.multiprocessing_distributed or (cfg.multiprocessing_distributed
                and cfg.rank % ngpus_per_node == 0 and (epoch+1) % cfg.test_freq == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, is_best=False, filename='{}/checkpoint_{:04d}.pth.tar'.format(cfg.results.output_dir, epoch))
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, is_best=False, filename='{}/checkpoint_last.pth.tar'.format(cfg.results.output_dir))
            if (epoch+1) == cfg.epochs:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, is_best=False, filename='{}/checkpoint_final.pth.tar'.format(cfg.results.output_dir))
            model.eval()

            loss_fn = nn.CrossEntropyLoss()
            num_heads = len(cfg.model.head.multi_heads)
            gt_labels = []
            pred_labels = []
            scores_all = []
            accs = []
            aris = []
            nmis = []
            feas_sim = []
            for h in range(num_heads):
                pred_labels.append([])
                scores_all.append([])

            for _, (images, _, embs, labels, idx) in enumerate(val_loader):
                images = images.to(cfg.gpu, non_blocking=True)
                with torch.no_grad():
                    scores = model(images, forward_type="sem")

                feas_sim.append(embs)

                assert len(scores) == num_heads
                for h in range(num_heads):
                    pred_idx = scores[h].argmax(dim=1)
                    pred_labels[h].append(pred_idx)
                    scores_all[h].append(scores[h])

                gt_labels.append(labels)

            gt_labels = torch.cat(gt_labels).long().cpu().numpy()
            feas_sim = torch.cat(feas_sim, dim=0)
            feas_sim = feas_sim.to(cfg.gpu, non_blocking=True)
            losses = []

            for h in range(num_heads):
                scores_all[h] = torch.cat(scores_all[h], dim=0)
                pred_labels[h] = torch.cat(pred_labels[h], dim=0)

            idx_select, gt_cluster_labels = model(feas_sim=feas_sim, scores=scores_all, epoch=epoch,
                                                  forward_type="sim2sem")

            for h in range(num_heads):
                pred_labels_h = pred_labels[h].long().cpu().numpy()

                pred_scores_select = scores_all[h][idx_select[h].cpu()]
                gt_labels_select = gt_cluster_labels[h]
                loss = loss_fn(pred_scores_select.cpu(), gt_labels_select)

                if eval_ent:
                    probs = scores_all[h].mean(dim=0)
                    probs = torch.clamp(probs, min=1e-8)
                    ent = -(probs * torch.log(probs)).sum()
                    loss = loss - eval_ent_weight * ent

                try:
                    acc = calculate_acc(pred_labels_h, gt_labels)
                except:
                    acc = -1

                nmi = calculate_nmi(pred_labels_h, gt_labels)

                ari = calculate_ari(pred_labels_h, gt_labels)

                accs.append(acc)
                nmis.append(nmi)
                aris.append(ari)

                losses.append(loss.item())

            accs = np.array(accs)
            nmis = np.array(nmis)
            aris = np.array(aris)
            losses = np.array(losses)

            best_acc_real = accs.max()
            head_real = np.where(accs == best_acc_real)
            head_real = head_real[0][0]
            best_nmi_real = nmis[head_real]
            best_ari_real = aris[head_real]
            logger.info("Real: ACC: {}, NMI: {}, ARI: {}, head: {}".format(best_acc_real, best_nmi_real, best_ari_real, head_real))

            head_loss = np.where(losses == losses.min())[0]
            head_loss = head_loss[0]
            best_acc_loss = accs[head_loss]
            best_nmi_loss = nmis[head_loss]
            best_ari_loss = aris[head_loss]
            logger.info("Loss: ACC: {}, NMI: {}, ARI: {}, head: {}".format(best_acc_loss, best_nmi_loss, best_ari_loss, head_loss))
            if best_acc_real > best_acc:
                best_acc = best_acc_real
                best_nmi = best_nmi_real
                best_ari = best_ari_real
                best_epoch = epoch
                best_head = np.array(accs).argmax()

                state_dict = model.state_dict()
                state_dict_save = {}
                for k in list(state_dict.keys()):
                    if not k.startswith('module.head'):
                        state_dict_save[k] = state_dict[k]
                    # print(k)
                    if k.startswith('module.head.head_{}'.format(best_head)):
                        state_dict_save['module.head.head_0.{}'.format(k[len('module.head.head_{}.'.format(best_head))::])] = state_dict[k]

                torch.save(state_dict_save, '{}/checkpoint_best.pth.tar'.format(cfg.results.output_dir))
                # save_checkpoint({
                #     'epoch': epoch + 1,
                #     'state_dict': model.state_dict(),
                #     'optimizer': optimizer.state_dict(),
                # }, is_best=False, filename='{}/checkpoint_best.pth.tar'.format(cfg.results.output_dir))

            if min_loss > losses.min():
                min_loss = losses.min()
                loss_head = head_loss
                loss_epoch = epoch
                loss_acc = best_acc_loss
                loss_nmi = best_nmi_loss
                loss_ari = best_ari_loss

                state_dict = model.state_dict()
                state_dict_save = {}
                for k in list(state_dict.keys()):
                    if not k.startswith('module.head'):
                        state_dict_save[k] = state_dict[k]
                    # print(k)
                    if k.startswith('module.head.head_{}'.format(loss_head)):
                        state_dict_save['module.head.head_0.{}'.format(k[len('module.head.head_{}.'.format(loss_head))::])] = state_dict[k]

                torch.save(state_dict_save, '{}/checkpoint_select.pth.tar'.format(cfg.results.output_dir))

            model.train()

            logger.info("FINAL -- Best ACC: {}, Best NMI: {}, Best ARI: {}, epoch: {}, head: {}".format(best_acc, best_nmi, best_ari, best_epoch, best_head))
            logger.info("FINAL -- Select ACC: {}, Select NMI: {}, Select ARI: {}, epoch: {}, head: {}".format(loss_acc, loss_nmi, loss_ari, loss_epoch, loss_head))


def train(train_loader, model, optimizer, epoch, cfg):
    logger = logging.getLogger("{}.trainer".format(cfg.logger_name))

    info = []
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    info.append(batch_time)
    info.append(data_time)
    num_heads = len(cfg.model.head.multi_heads)
    losses = []
    for h in range(num_heads):
        losses_h = AverageMeter('Loss_{}'.format(h), ':.4e')
        losses.append(losses_h)
        info.append(losses_h)
    lr = AverageMeter('lr', ':.6f')
    lr.update(optimizer.param_groups[0]["lr"])
    info.append(lr)

    progress = ProgressMeter(
        len(train_loader),
        info,
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    target_sub_batch_size = cfg.solver.target_sub_batch_size
    batch_size = cfg.solver.batch_size
    train_sub_batch_size = cfg.solver.train_sub_batch_size

    num_repeat = cfg.solver.num_repeat

    num_imgs_all = len(train_loader.dataset)

    iters_end = batch_size // target_sub_batch_size
    num_iters_l = num_imgs_all // batch_size
    for ii in range(num_iters_l):
        end = time.time()
        model.eval()
        scores = []
        for h in range(num_heads):
            scores.append([])

        images_trans_l = []
        feas_sim = []

        for _, (images_ori_l_batch, images_trans_l_batch, feas_sim_batch, _, idx_l_batch) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            # print(images_ori_l_batch.shape)

            # Generate ground truth.

            # Select samples and estimate the ground-truth relationship between samples.
            images_ori_l_batch = images_ori_l_batch.to(cfg.gpu, non_blocking=True)
            with torch.no_grad():
                scores_nl = model(images_ori_l_batch, forward_type="sem")

            assert num_heads == len(scores_nl)

            for h in range(num_heads):
                scores[h].append(scores_nl[h].detach())

            images_trans_l.append(images_trans_l_batch)
            feas_sim.append(feas_sim_batch)

            if len(feas_sim) >= iters_end:
                train_loader.sampler.set_epoch(train_loader.sampler.epoch + 1)
                break

        for h in range(num_heads):
            scores[h] = torch.cat(scores[h], dim=0)

        images_trans_l = torch.cat(images_trans_l)
        feas_sim = torch.cat(feas_sim)

        feas_sim = feas_sim.to(cfg.gpu).to(torch.float32)

        idx_select, gt_cluster_labels = model(feas_sim=feas_sim, scores=scores, epoch=epoch, forward_type="sim2sem")

        # Move indices to cpu
        idx_select = [idx.cpu() for idx in idx_select]
        images_trans = []
        for h in range(num_heads):
            images_trans.append(images_trans_l[idx_select[h], :, :, :])

        num_imgs = images_trans[0].shape[0]

        # Train with the generated ground truth
        model.train()
        img_idx = list(range(num_imgs))
        # Select a set of images for training.

        num_train = num_imgs
        # train_sub_iters = int(torch.ceil(float(num_train) / train_sub_batch_size))
        train_sub_iters = num_train // train_sub_batch_size

        for n in range(num_repeat):
            random.shuffle(img_idx)

            for i in range(train_sub_iters):
                start_idx = i * train_sub_batch_size
                end_idx = min((i + 1) * train_sub_batch_size, num_train)
                img_idx_i = img_idx[start_idx:end_idx]

                imgs_i = []
                targets_i = []

                for h in range(num_heads):
                    imgs_i.append(images_trans[h][img_idx_i, :, :, :].to(cfg.gpu, non_blocking=True))
                    targets_i.append(gt_cluster_labels[h][img_idx_i].to(cfg.gpu, non_blocking=True))

                loss_dict = model(imgs_i, target=targets_i, forward_type="loss")

                loss = sum(loss for loss in loss_dict.values())
                loss_mean = loss / num_heads

                optimizer.zero_grad()
                loss_mean.backward()
                optimizer.step()

                for h in range(num_heads):
                    # measure accuracy and record loss
                    losses[h].update(loss_dict['head_{}'.format(h)].item(), imgs_i[0].size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if ii % cfg.print_freq == 0:
            logger.info(progress.display(ii))


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
