import argparse
import os
import sys
from sys import platform
import pathlib
from addict import Dict
sys.path.insert(0, './')

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from SPICE.spice.config import Config
from SPICE.spice.data.build_dataset import build_dataset
from SPICE.spice.model.build_model_sim import build_model_sim
from SPICE.spice.utils.miscellaneous import mkdir, save_config
import numpy as np
from SPICE.spice.utils.load_model_weights import load_model_weights

# Import utils
parent_dir = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))
import utils

"""
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument(
    "--config-file",
    default="./configs/stl10/embedding.py",
    metavar="FILE",
    help="path to config file",
    type=str,
)
"""

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
    cfg = Dict()

    ### Convert config file values to the args variable equivalent (match the format of the existing code)
    print("Assigning config values to corresponding args variables...")
    # Dataset
    dataset_root = pathlib.Path(configs['data']['dataset_root'])
    ds_split = configs['data']['ds_split']
    labels = configs['data']['labels']
    ascan_per_group = configs['data']['ascan_per_group']
    use_mini_dataset = configs['data']['use_mini_dataset']
    img_size_dict['oct'] = (512, ascan_per_group)
    num_cluster_dict['oct'] = len(labels)
    cfg.labels_dict = {i: lbl for i, lbl in enumerate(labels)}
    cfg.map_df_paths = {
        split: dataset_root.joinpath(f"{split}{'Mini' if use_mini_dataset else ''}_mapping_{ascan_per_group}scans.csv")
        for split in ['train', 'valid', 'test']}

    moco_dataset_name = configs['SPICE']['MoCo']['dataset_name']
    cfg.model_name = configs['SPICE']['embedding']['model_name']
    cfg.weight = pathlib.Path(configs['SPICE']['MoCo']['save_folder']).joinpath(moco_dataset_name).joinpath('moco').joinpath(
        configs['SPICE']['embedding']['weight'])
    cfg.model_type = configs['SPICE']['MoCo']['base_model']
    cfg.num_cluster = num_cluster_dict[moco_dataset_name]
    cfg.batch_size = configs['SPICE']['embedding']['batch_size']
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

    # data_test
    split = 'train' # "train+test"
    cfg.data_test.type=moco_dataset_name
    # cfg.data_test.root_folder=pathlib.Path(configs['SPICE']['MoCo']['dataset_path']).joinpath(moco_dataset_name)
    cfg.data_test.root_folder=pathlib.Path(configs['SPICE']['MoCo']['dataset_path']).joinpath(
        'OCT_lab_data' if moco_dataset_name == 'oct' else moco_dataset_name)
    cfg.data_test.split=split
    cfg.data_test.shuffle=configs['SPICE']['embedding']['data_test']['shuffle']
    cfg.data_test.ims_per_batch=configs['SPICE']['embedding']['data_test']['imgs_per_batch']
    cfg.data_test.aspect_ratio_grouping=configs['SPICE']['embedding']['data_test']['aspect_ratio_grouping']
    cfg.data_test.train=configs['SPICE']['embedding']['data_test']['train']
    cfg.data_test.show=configs['SPICE']['embedding']['data_test']['show']
    cfg.data_test.trans1=Dict(dict(
        aug_type="test",
        normalize=Dict(dict(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])),
    ))
    cfg.data_test.trans2=Dict(dict(
        aug_type="test",
        normalize=Dict(dict(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])),
    ))
    cfg.data_test.labels_dict = {i: lbl for i, lbl in enumerate(labels)}
    cfg.data_test.map_df_paths = {
        split: dataset_root.joinpath(f"{ascan_per_group}mscans").joinpath(
            f"{split}{'Mini' if use_mini_dataset else ''}_mapping_{ascan_per_group}scans.csv")
        for split in ['train', 'valid', 'test']}
    cfg.data_test.preload_data = False

    # model_sim
    cfg.model_sim.type=cfg.model_type
    cfg.model_sim.num_classes=configs['SPICE']['embedding']['model_sim']['num_classes']
    cfg.model_sim.in_channels=configs['SPICE']['embedding']['model_sim']['in_channels']
    cfg.model_sim.in_size=img_size_dict[moco_dataset_name]
    cfg.model_sim.batchnorm_track=configs['SPICE']['embedding']['model_sim']['batchnorm_track']
    cfg.model_sim.test=configs['SPICE']['embedding']['model_sim']['test']
    cfg.model_sim.feature_only=configs['SPICE']['embedding']['model_sim']['features_only']
    cfg.model_sim.pretrained=cfg.weight
    cfg.model_sim.model_type=configs['SPICE']['embedding']['model_sim']['model_type']

    # results
    cfg.results.output_dir = f"{configs['SPICE']['MoCo']['save_folder']}/{moco_dataset_name}/{cfg.model_name}"

    # Create output dir if necessary
    output_dir = cfg.results.output_dir
    if output_dir:
        mkdir(output_dir)

    output_config_path = os.path.join(output_dir, 'config.py')
    # save_config(cfg, output_config_path) # TODO: find a way to make this work (not so important)

    if cfg.gpu is not None:
        print("Use GPU: {}".format(cfg.gpu))

    # create model
    model_sim = build_model_sim(cfg.model_sim)
    # print(model_sim)

    torch.cuda.set_device(cfg.gpu)
    model_sim = model_sim.cuda(cfg.gpu)

    # Load similarity model
    if cfg.model_sim.pretrained is not None:
        load_model_weights(model_sim, cfg.model_sim.pretrained, cfg.model_sim.model_type)

    cudnn.benchmark = True

    dataset_val = build_dataset(cfg.data_test)
    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=cfg.batch_size, shuffle=False, num_workers=1)

    model_sim.eval()

    pool = nn.AdaptiveAvgPool2d(1)

    feas_sim = []
    for _, (images, _, labels, idx) in enumerate(val_loader):
        images = images.to(cfg.gpu, non_blocking=True)
        print(images.shape)
        with torch.no_grad():
            feas_sim_i = model_sim(images)
            if len(feas_sim_i.shape) == 4:
                feas_sim_i = pool(feas_sim_i)
                feas_sim_i = torch.flatten(feas_sim_i, start_dim=1)
            feas_sim_i = nn.functional.normalize(feas_sim_i, dim=1)
            feas_sim.append(feas_sim_i.cpu())

    feas_sim = torch.cat(feas_sim, dim=0)
    feas_sim = feas_sim.numpy()

    np.save("{}/feas_moco_512_l2.npy".format(cfg.results.output_dir), feas_sim)


if __name__ == '__main__':
    main()
