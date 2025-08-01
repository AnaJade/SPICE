#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
import sys
sys.path.insert(0, './')
import pathlib
import numpy as np
import pandas as pd
from sys import platform

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.models as models
from SPICE.spice.model.feature_modules.cluster_resnet import ClusterResNet
from SPICE.spice.model.feature_modules.resnet_stl import resnet18
from SPICE.spice.model.feature_modules.resnet_cifar import resnet18_cifar

import SPICE.moco.loader
import SPICE.moco.builder
from SPICE.moco.stl10 import STL10
from SPICE.moco.cifar import CIFAR10, CIFAR100

# Import utils
parent_dir = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))
import utils
from utils_data import OCTDataset

model_names = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))

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

# Set up the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--config_path',
                    help='Path to the config file',
                    type=str)

def main():
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
    labels = configs['data']['labels']
    ascan_per_group = configs['data']['ascan_per_group']
    use_mini_dataset = configs['data']['use_mini_dataset']
    # oct_img_root = pathlib.Path(f'OCT_lab_data/{ascan_per_group}mscans')
    img_size_dict['oct'] = (512, ascan_per_group)
    num_cluster_dict['oct'] = len(labels)

    ### Convert config file values to the args variable equivalent (match the format of the existing code)
    print("Assigning config values to corresponding args variables...")
    # Dataset
    args.labels_dict = {i: lbl for i, lbl in enumerate(labels)}
    args.map_df_paths = {
        split: dataset_root.joinpath(f"{ascan_per_group}mscans").joinpath(
            f"{split}{'Mini' if use_mini_dataset else ''}_mapping_{ascan_per_group}scans.csv")
        for split in ['train', 'valid', 'test']}

    args.data_type = configs['SPICE']['MoCo']['dataset_name']
    args.data = pathlib.Path(dataset_path).joinpath('OCT_lab_data' if args.data_type == 'oct' else args.data_type)
    print(f"args.data: {args.data}")

    args.all = configs['SPICE']['MoCo']['use_all']
    args.img_size = img_size_dict[args.data_type]

    # Training params
    args.seed = configs['training']['random_seed']
    args.save_folder = pathlib.Path(configs['SPICE']['MoCo']['save_folder']).joinpath(args.data_type).joinpath('moco')
    args.save_freq = configs['SPICE']['MoCo']['save_freq']
    args.arch = configs['SPICE']['MoCo']['base_model']
    args.workers = configs['SPICE']['MoCo']['num_workers']
    args.epochs = configs['SPICE']['MoCo']['max_epochs']
    args.start_epoch = configs['SPICE']['MoCo']['start_epoch']
    args.batch_size = configs['SPICE']['MoCo']['batch_size']
    args.lr = configs['SPICE']['MoCo']['lr']
    args.schedule = configs['SPICE']['MoCo']['lr_schedule']
    args.momentum = configs['SPICE']['MoCo']['momentum']
    args.weight_decay = configs['SPICE']['MoCo']['weight_decay']
    args.print_freq = configs['SPICE']['MoCo']['print_freq']
    args.resume = pathlib.Path(args.save_folder).joinpath('checkpoint_last.pth.tar') if configs['SPICE']['MoCo']['resume'] else False

    # Multiprocessing
    # More info on values: https://stackoverflow.com/a/76828907
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    args.world_size = configs['SPICE']['MoCo']['world_size']
    args.rank = configs['SPICE']['MoCo']['rank']
    args.dist_url = configs['SPICE']['MoCo']['dist_url']
    args.dist_backend = configs['SPICE']['MoCo']['dist_backend']
    args.gpu = None if configs['SPICE']['MoCo']['gpu_id'] == 'None' else configs['SPICE']['MoCo']['gpu_id']
    args.multiprocessing_distributed = configs['SPICE']['MoCo']['multiprocessing_distributed']

    # moco specific configs:
    args.moco_dim = num_cluster_dict[args.data_type]
    args.moco_k = configs['SPICE']['MoCo']['moco_queue_size']
    args.moco_m = configs['SPICE']['MoCo']['moco_momentum']
    args.moco_t = configs['SPICE']['MoCo']['moco_softmax_temp']
    # options for moco v2
    args.mlp = configs['SPICE']['MoCo']['moco2_mlp']
    args.aug_plus = configs['SPICE']['MoCo']['moco2_aug_plus']
    args.cos = configs['SPICE']['MoCo']['moco2_cos']

    # Save folder
    if not args.save_folder.is_dir():
        args.save_folder.mkdir(parents=True)

    # Set all random seeds
    print("Setting random seed...")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.deterministic = True
    warnings.warn('You have chosen to seed training. '
                  'This will turn on the CUDNN deterministic setting, '
                  'which can slow down your training considerably! '
                  'You may see unexpected behavior when restarting '
                  'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if "env://" in args.dist_url and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    print("Running main worker function...")
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if "env://" in args.dist_url and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        # Sample init: https://stackoverflow.com/a/76828907
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.arch == "clusterresnet":
        base_model = ClusterResNet
    elif args.arch == "resnet18":
        base_model = resnet18
    elif args.arch == "resnet18_cifar":
        base_model = resnet18_cifar
    else:
        base_model = models.__dict__[args.arch]
    model = SPICE.moco.builder.MoCo(
        base_model,
        args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp)
    # print(model)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        # Will raise ValueError: Default process group has not been initialized, please make sure to call init_process_group.
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    normalize = transforms.Normalize(mean=mean[args.data_type],
                                     std=std[args.data_type])
    if args.aug_plus:
        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        augmentation = [
            transforms.RandomResizedCrop(args.img_size, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([SPICE.moco.loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
    else:
        # MoCo v1's aug: the same as InstDisc https://arxiv.org/abs/1805.01978
        augmentation = [
            transforms.RandomResizedCrop(args.img_size, scale=(0.2, 1.)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]

    print(f"Creating {args.data_type} dataset...")
    if args.data_type == 'stl10':
        if args.all:
            split = "train+test+unlabeled"
        else:
            split = "train+unlabeled"
        train_dataset = STL10(args.data, split=split,
                              transform=SPICE.moco.loader.TwoCropsTransform(transforms.Compose(augmentation)),
                              download=True)
    elif args.data_type == 'cifar10':
        train_dataset = CIFAR10(args.data, all=args.all,
                                transform=SPICE.moco.loader.TwoCropsTransform(transforms.Compose(augmentation)),
                                download=True)
    elif args.data_type == 'cifar100':
        train_dataset = CIFAR100(args.data, all=args.all,
                                 transform=SPICE.moco.loader.TwoCropsTransform(transforms.Compose(augmentation)),
                                 download=True)
    elif args.data_type == 'oct':
        train_dataset = OCTDataset(args.data, 'train', args.map_df_paths, args.labels_dict,
                                   SPICE.moco.loader.TwoCropsTransform(transforms.Compose(augmentation)))
    else:
        raise TypeError

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
    #     num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    # Added to solve OSError: [Errno 22] Invalid argument
    # Source: https://discuss.pytorch.org/t/oserror-errno-22-invalid-argument-pickle-unpicklingerror-pickle-data-was-truncated/138469/2
    # Documentation: https://docs.pytorch.org/docs/stable/notes/windows.html#multiprocessing-error-without-if-clause-protection
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        sampler=train_sampler, drop_last=True)

    print("Starting train loop...")
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            if (epoch+1) % args.save_freq == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                }, is_best=False, filename='{}/checkpoint_{:04d}.pth.tar'.format(args.save_folder, epoch))

                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                }, is_best=False, filename='{}/checkpoint_last.pth.tar'.format(args.save_folder))

            if (epoch+1) == args.epochs:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                }, is_best=False, filename='{}/checkpoint_final.pth.tar'.format(args.save_folder))


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)

        # compute output
        output, target = model(im_q=images[0], im_k=images[1])
        loss = criterion(output, target)

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images[0].size(0))
        top1.update(acc1[0], images[0].size(0))
        top5.update(acc5[0], images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


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


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            correct_k = correct[:k].reshape(k*correct.shape[1]).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
