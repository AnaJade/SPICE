import sys
import pathlib
from SPICE.spice.data.stl10 import STL10
from SPICE.spice.data.transformations import get_train_transformations
from SPICE.spice.data.stl10_embedding import STL10EMB
from SPICE.spice.data.cifar import CIFAR10, CIFAR20
from SPICE.spice.data.imagenet import ImageNetSubEmb, ImageNetSubEmbLMDB, TImageNetEmbLMDB
from SPICE.spice.data.npy import NPYEMB

parent_dir = pathlib.Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(parent_dir))
from utils_data import OCTDataset, OCTDataset2Trans

def build_dataset(data_cfg):
    type = data_cfg.type

    dataset = None

    train_trans1 = get_train_transformations(data_cfg.trans1)
    train_trans2 = get_train_transformations(data_cfg.trans2)

    if type == "stl10":
        dataset = STL10(root=data_cfg.root_folder,
                        split=data_cfg.split,
                        show=data_cfg.show,
                        transform1=train_trans1,
                        transform2=train_trans2,
                        download=False)
    elif type == "stl10_emb":
        dataset = STL10EMB(root=data_cfg.root_folder,
                           split=data_cfg.split,
                           show=data_cfg.show,
                           transform1=train_trans1,
                           transform2=train_trans2,
                           download=False,
                           embedding=data_cfg.embedding)
    elif type == "npy_emb":
        dataset = NPYEMB(root=data_cfg.root,
                         show=data_cfg.show,
                         transform1=train_trans1,
                         transform2=train_trans2,
                         embedding=data_cfg.embedding)
    elif type == "cifar10":
        dataset = CIFAR10(root=data_cfg.root_folder,
                          all=data_cfg.all,
                          train=data_cfg.train,
                          transform1=train_trans1,
                          transform2=train_trans2,
                          target_transform=None,
                          download=False,
                          embedding=data_cfg.embedding,
                          show=data_cfg.show,
                          )
    elif type == "cifar100":
        dataset = CIFAR20(root=data_cfg.root_folder,
                          all=data_cfg.all,
                          train=data_cfg.train,
                          transform1=train_trans1,
                          transform2=train_trans2,
                          target_transform=None,
                          download=False,
                          embedding=data_cfg.embedding,
                          show=data_cfg.show,
                          )
    elif type == 'imagenet':
        dataset = ImageNetSubEmb(subset_file=data_cfg.subset_file,
                                 embedding=data_cfg.embedding,
                                 split=data_cfg.split,
                                 transform1=train_trans1,
                                 transform2=train_trans2)
    elif type == 'imagenet_lmdb':
        dataset = ImageNetSubEmbLMDB(lmdb_file=data_cfg.lmdb_file,
                                     meta_info_file=data_cfg.meta_info_file,
                                     embedding=data_cfg.embedding,
                                     split=data_cfg.split,
                                     transform1=train_trans1,
                                     transform2=train_trans2,
                                     resize=data_cfg.resize)
    elif type == 'timagenet_lmdb':
        dataset = TImageNetEmbLMDB(lmdb_file=data_cfg.lmdb_file,
                                   meta_info_file=data_cfg.meta_info_file,
                                   embedding=data_cfg.embedding,
                                   transform1=train_trans1,
                                   transform2=train_trans2)
    elif type == 'oct':
        dataset = OCTDataset2Trans(root=data_cfg.root_folder,
                                   split=data_cfg.split,
                                   map_df_paths=data_cfg.map_df_paths,
                                   show=data_cfg.show,
                                   labels_dict=data_cfg.labels_dict,
                                   transform1=train_trans1,
                                   transform2=train_trans2,
                                   preload_data=data_cfg.preload_data)
    elif type == 'oct_emb':
        dataset = OCTDataset2Trans(root=data_cfg.root_folder,
                                   split=data_cfg.split,
                                   map_df_paths=data_cfg.map_df_paths,
                                   show=data_cfg.show,
                                   labels_dict=data_cfg.labels_dict,
                                   transform1=train_trans1,
                                   transform2=train_trans2,
                                   embedding=data_cfg.embedding,
                                   preload_data=False)
    else:
        assert TypeError

    return dataset