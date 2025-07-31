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
import matplotlib.pyplot as plt

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

if __name__ == '__main__':
    config_path = pathlib.Path('../../config_windows.yaml')
    configs = utils.load_configs(config_path)
    dataset_root = pathlib.Path(configs['data']['dataset_root'])
    labels = configs['data']['labels']
    labels_dict = {i: lbl for i, lbl in enumerate(labels)}
    ascan_per_group = configs['data']['ascan_per_group']
    use_mini_dataset = configs['data']['use_mini_dataset']
    map_df_paths = {
        split: dataset_root.joinpath(f"{ascan_per_group}mscans").joinpath(f"{split}{'Mini' if use_mini_dataset else ''}_mapping_{ascan_per_group}scans.csv")
        for split in ['train', 'valid', 'test']}

    print(f"Creating dataset...")
    split = "train+unlabeled"
    dataset_path = r'C:\Users\anaja\OneDrive\Documents\Ecole\TUHH\Semester 6\Masterarbeit\stl10'
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    augmentation = [
        transforms.RandomResizedCrop(96, scale=(0.2, 1.)),
        transforms.RandomGrayscale(p=0.2),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]

    train_dataset = STL10(dataset_path, split=split, transform=SPICE.moco.loader.TwoCropsTransform(transforms.Compose(augmentation)), download=False)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=False, sampler=None, drop_last=True)

    for i, (images, labels) in enumerate(train_loader):
        if i > 3:
            break
        print(type(images)) # [img t1 tensor torch.float32 [batch size, 3, 96, 96], img t2 tensor torch.float32 [batch_size, 3, 96, 96]]
        print(type(labels)) # [int], tensor torch.int64

    # OCT dataset
    split = 'train'
    dataset_train = OCTDataset(dataset_root, split, map_df_paths, labels_dict, SPICE.moco.loader.TwoCropsTransform(transforms.Compose(augmentation)))
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=4, shuffle=False, sampler=None, drop_last=True)

    for i, (images, labels) in enumerate(dataloader_train):
        if i > 3:
            break
        print(type(images)) # [img t1 tensor torch.float32 [batch size, 3, 96, 96], img t2 tensor torch.float32 [batch_size, 3, 96, 96]]
        print(type(labels)) # [int], tensor torch.i
        print(images[0].shape)
        print(type(images[0]))