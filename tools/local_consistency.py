import argparse
import pathlib
import random
import os
import sys
import pandas as pd
from sys import platform
from addict import Dict
sys.path.insert(0, './')

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
from SPICE.spice.config import Config
from SPICE.spice.data.build_dataset import build_dataset
from SPICE.spice.model.sim2sem import Sim2Sem
from SPICE.spice.utils.miscellaneous import mkdir, save_config
import numpy as np
from SPICE.spice.utils.evaluation import calculate_acc, calculate_nmi, calculate_ari

# Import utils
parent_dir = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))
import utils

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

"""
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument(
    "--config-file",
    default="./configs/stl10/eval.py",
    metavar="FILE",
    help="path to config file",
    type=str,
)
parser.add_argument(
    "--embedding",
    default="./results/stl10/embedding/feas_moco_512_l2.npy",
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

    # Cfg file values
    cfg.model_name = configs['SPICE']['local_consistency']['model_name']
    moco_dataset_name = configs['SPICE']['MoCo']['dataset_name']
    cfg.embedding = pathlib.Path(configs['SPICE']['MoCo']['save_folder']).joinpath(moco_dataset_name).joinpath(
        configs['SPICE']['embedding']['model_name']).joinpath('feas_moco_512_l2.npy')
    cfg.weight = pathlib.Path(configs['SPICE']['MoCo']['save_folder']).joinpath(moco_dataset_name).joinpath(
        configs['SPICE']['SPICE_self']['model_name']).joinpath('checkpoint_select.pth.tar')
    cfg.model_type = configs['SPICE']['MoCo']['base_model']
    cfg.num_cluster = num_cluster_dict[moco_dataset_name]
    cfg.batch_size = configs['SPICE']['local_consistency']['batch_size']
    cfg.fea_dim = configs['SPICE']['SPICE_self']['fea_dim']
    cfg.center_ratio = configs['SPICE']['SPICE_self']['center_ratio']
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

    # Data test
    cfg.data_test = Dict(dict(
        type=f'{moco_dataset_name}_emb', # "stl10_emb",
        root_folder=pathlib.Path(configs['SPICE']['MoCo']['dataset_path']).joinpath(
        'OCT_lab_data' if moco_dataset_name == 'oct' else moco_dataset_name), # "./datasets/stl10",
        embedding=None,
        split='train', # "train+test",
        shuffle=False,
        ims_per_batch=configs['SPICE']['local_consistency']['batch_size_test'],
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
                  multi_heads=[Dict(dict(classifier=Dict(dict(type="mlp", num_neurons=[cfg.fea_dim, cfg.fea_dim, cfg.num_cluster],
                                                    last_activation="softmax")),
                                    feature_conv=None,
                                    num_cluster=cfg.num_cluster,
                                    ratio_start=1,
                                    ratio_end=1,
                                    center_ratio=cfg.center_ratio,
                                    ))] * 1,
                  ratio_confident=0.90,
                  num_neighbor=100,
                  )),
        model_type="moco_select",
        pretrained=cfg.weight,
        head_id=3,
        freeze_conv=True,
        )
    )

    # Output dir
    cfg.results.output_dir = f"{configs['SPICE']['MoCo']['save_folder']}/{moco_dataset_name}/{cfg.model_name}"

    ### Old code
    # cfg = Config.fromfile(args.config_file)
    # cfg.embedding = args.embedding

    output_dir = cfg.results.output_dir
    if output_dir:
        mkdir(output_dir)

    output_config_path = os.path.join(output_dir, 'config.py')
    # save_config(cfg, output_config_path)

    if cfg.gpu is not None:
        print("Use GPU: {}".format(cfg.gpu))

    # create model
    model = Sim2Sem(**cfg.model)
    print(model)

    torch.cuda.set_device(cfg.gpu)
    model = model.cuda(cfg.gpu)

    state_dict = torch.load(cfg.model.pretrained)
    if 'state_dict' in state_dict.keys():
        state_dict = state_dict['state_dict']

    for k in list(state_dict.keys()):
        # Initialize the feature module with encoder_q of moco.
        if k.startswith('module.'):
            # remove prefix
            # state_dict["module.{}".format(k[len('module.encoder_q.'):])] = state_dict[k]
            state_dict["{}".format(k[len('module.'):])] = state_dict[k]

        # delete renamed or unused k
        del state_dict[k]

    model.load_state_dict(state_dict)
    cudnn.benchmark = True

    # Data loading code
    dataset_val = build_dataset(cfg.data_test)
    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=cfg.batch_size, shuffle=False, num_workers=1)

    model.eval()

    num_heads = len(cfg.model.head.multi_heads)
    assert num_heads == 1
    gt_labels = []
    pred_labels = []
    scores_all = []

    for _, (images, _, labels, idx) in enumerate(val_loader):
        images = images.to(cfg.gpu, non_blocking=True)
        with torch.no_grad():
            scores = model(images, forward_type="sem")

        assert len(scores) == num_heads

        pred_idx = scores[0].argmax(dim=1)
        pred_labels.append(pred_idx)
        scores_all.append(scores[0])

        gt_labels.append(labels)

    gt_labels = torch.cat(gt_labels).long().cpu().numpy()
    feas_sim = torch.from_numpy(np.load(cfg.embedding))

    pred_labels = torch.cat(pred_labels).long().cpu().numpy()
    scores = torch.cat(scores_all).cpu()

    try:
        acc = calculate_acc(pred_labels, gt_labels)
    except:
        acc = -1

    nmi = calculate_nmi(pred_labels, gt_labels)
    ari = calculate_ari(pred_labels, gt_labels)

    print("ACC: {}, NMI: {}, ARI: {}".format(acc, nmi, ari))

    # Save original labels and scores
    preds = pd.DataFrame(np.concat((np.stack((pred_labels, gt_labels), axis=1), scores), axis=1),
                         columns=['pred', 'label'] + [f'score_{i}' for i in range(10)])
    preds_best = pd.concat([preds[preds['label'] == i].sort_values(f'score_{i}', ascending=False).head(20) for i in range(10)], axis=0)

    # DEBUG: lower consistency thresholds
    # model.head.ratio_confident = 0.9 # OG value: 0.9
    # model.head.score_th = 0.99 # OG value: 0.99

    idx_select, labels_select = model(feas_sim=feas_sim, scores=scores, forward_type="local_consistency")
    # DEBUG: Create dummy idx_select and labels_select
    """
    preds_best.loc[(preds_best['label'] == 4) & (preds_best['score_4'] >= 0.1), 'pred'] = 4  # Create at least one correctly predicted 4 label
    img_select = preds_best.reset_index().groupby('label').first().reset_index().set_index('index')
    idx_select = torch.tensor(img_select.sort_index().index.to_list())
    labels_select = torch.tensor(img_select.sort_index()['pred'].to_list())
    """

    # gt_labels_select = gt_labels[idx_select]
    gt_labels_select = gt_labels[[int(i) for i in idx_select]]

    acc = calculate_acc(labels_select.numpy(), gt_labels_select)
    print('ACC of local consistency: {}, number of samples: {}'.format(acc, len(gt_labels_select)))

    labels_correct = np.zeros([feas_sim.shape[0]]) - 100
    labels_correct[idx_select] = labels_select

    np.save("{}/labels_reliable.npy".format(cfg.results.output_dir), labels_correct)


if __name__ == '__main__':
    main()
