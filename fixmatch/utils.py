import os
import time
from torch.utils.tensorboard import SummaryWriter
import logging

def setattr_cls_from_kwargs(cls, kwargs):
    #if default values are in the cls,
    #overlap the value by kwargs
    for key in kwargs.keys():
        if hasattr(cls, key):
            print(f"{key} in {cls} is overlapped by kwargs: {getattr(cls,key)} -> {kwargs[key]}")
        setattr(cls, key, kwargs[key])

        
def test_setattr_cls_from_kwargs():
    class _test_cls:
        def __init__(self):
            self.a = 1
            self.b = 'hello'
    test_cls = _test_cls()
    config = {'a': 3, 'b': 'change_hello', 'c':5}
    setattr_cls_from_kwargs(test_cls, config)
    for key in config.keys():
        print(f"{key}:\t {getattr(test_cls, key)}")
        
        
def net_builder(net_name, from_name: bool, net_conf=None):
    """
    return **class** of backbone network (not instance).
    Args
        net_name: 'WideResNet' or network names in torchvision.models
        from_name: If True, net_buidler takes models in torch.vision models. Then, net_conf is ignored.
        net_conf: When from_name is False, net_conf is the configuration of backbone network (now, only WRN is supported).
    """
    if from_name:
        import torchvision.models as models
        model_name_list = sorted(name for name in models.__dict__
                                if name.islower() and not name.startswith("__")
                                and callable(models.__dict__[name]))

        if net_name not in model_name_list:
            assert Exception(f"[!] Networks\' Name is wrong, check net config, \
                               expected: {model_name_list}  \
                               received: {net_name}")
        else:
            return models.__dict__[net_name]
        
    else:
        if net_name == 'WideResNet':
            import SPICE.fixmatch.models.nets.wrn as net
            builder = getattr(net, 'build_WideResNet')()
        elif net_name == 'WideResNet_stl10':
            import SPICE.fixmatch.models.nets.wrn_stl10 as net
            builder = getattr(net, 'build_WideResNet')()
        elif net_name == 'ClusterResNet':
            import SPICE.fixmatch.models.nets.cluster_resnet as net
            builder = getattr(net, 'build_ClusterResNet')()
        elif net_name == 'WideResNet_tiny':
            import SPICE.fixmatch.models.nets.wrn_tiny as net
            builder = getattr(net, 'build_WideResNet')()
        elif net_name == 'resnet18':
            import SPICE.fixmatch.models.nets.resnet_stl as net
            builder = getattr(net, 'build_ResNet18')()
        elif net_name == 'resnet18_cifar':
            import SPICE.fixmatch.models.nets.resnet_cifar as net
            builder = getattr(net, 'build_ResNet18')()
        elif net_name == 'resnet34':
            import SPICE.fixmatch.models.nets.imagenet as net
            builder = getattr(net, 'build_ResNet34')()
        else:
            assert Exception("Not Implemented Error")
            
        setattr_cls_from_kwargs(builder, net_conf)
        return builder.build

    
def test_net_builder(net_name, from_name, net_conf=None):
    builder = net_builder(net_name, from_name, net_conf)
    print(f"net_name: {net_name}, from_name: {from_name}, net_conf: {net_conf}")
    print(builder)

    
def get_logger(name, save_path=None, level='INFO'):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))
    
    log_format = logging.Formatter('[%(asctime)s %(levelname)s] %(message)s')
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(log_format)
    logger.addHandler(streamHandler)
    
    if not save_path is None:
        os.makedirs(save_path, exist_ok=True)
        fileHandler = logging.FileHandler(os.path.join(save_path, 'log.txt'))
        fileHandler.setFormatter(log_format)
        logger.addHandler(fileHandler)
    
    return logger


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
