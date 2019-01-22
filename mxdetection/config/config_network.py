import numpy as np
from easydict import EasyDict as edict

# network params
#     config.network.net_type
#     config.network.pretrained_prefix
#     config.network.pretrained_epoch
#     config.network.input_mean
#     config.network.input_scale
#     config.network.image_stride
def add_network_params(config, net_type_layer):
    if 'network' not in config:
        config.network = edict()
    home_dir = config.hdfs_remote_home_dir
    net_type, num_layer = net_type_layer.split('#')
    config.network.net_type = net_type
    # pretrained params
    # config.network.pretrained_prefix
    # config.network.pretrained_epoch
    # config.network.input_mean
    # config.network.input_scale
    # config.network.body_params
    config.network.backbone_params = dict()
    config = add_resnet_params(config, home_dir, net_type, int(num_layer))
    # other params
    if 'image_stride' not in config.network:
        config.network.image_stride = 0
    return config


def add_resnet_params(config, home_dir, net_type, num_layer):
    config.network.net_type = net_type
    config.network.num_layer = num_layer
    if net_type == 'resnet':
        config.network.pretrained_prefix = home_dir + '/common/models/resnet-%d' % num_layer
        config.network.pretrained_epoch = 0
        config.network.input_mean = np.array([0, 0, 0], dtype=np.float32)
        config.network.input_scale = 1.0
    elif net_type == 'resnet_v1':
        config.network.pretrained_prefix = home_dir + '/common/models/resnet-v1-%d' % num_layer
        config.network.pretrained_epoch = 0
        config.network.input_mean = np.array([103.94, 116.78, 123.68], dtype=np.float32)
        config.network.input_scale = 0.017
    elif net_type == 'resnet_v1_channel025':
        config.network.pretrained_prefix = home_dir + '/common/models/resnet-v1-%d-channel0.25' % num_layer
        config.network.pretrained_epoch = 0
        config.network.input_mean = np.array([103.94, 116.78, 123.68], dtype=np.float32)
        config.network.input_scale = 0.017
        config.network.backbone_params['net_type'] = 'resnet_v1'
        config.network.backbone_params['res1_use_pooling'] = False
        config.network.body_params['filter_multiplier'] = 0.25
        config.network.backbone_params['filter_multiplier'] = 0.25
    elif net_type == 'resnet_v2':
        config.network.pretrained_prefix = home_dir + '/common/models/resnet-v2-%d' % num_layer
        config.network.pretrained_epoch = 0
        config.network.input_mean = np.array([103.94, 116.78, 123.68], dtype=np.float32)
        config.network.input_scale = 0.017
    elif net_type == 'resnext_g32':
        config.network.pretrained_prefix = home_dir + '/common/models/resnext-%d-g32' % num_layer
        config.network.pretrained_epoch = 0
        config.network.input_mean = np.array([103.94, 116.78, 123.68], dtype=np.float32)
        config.network.input_scale = 1.0
        config.network.backbone_params['net_type'] = 'resnext'
        config.network.backbone_params['num_group'] = 32
    elif net_type == 'resnext_g64':
        config.network.pretrained_prefix = home_dir + '/common/models/resnext-%d-g64' % num_layer
        config.network.pretrained_epoch = 0
        config.network.input_mean = np.array([103.94, 116.78, 123.68], dtype=np.float32)
        config.network.input_scale = 1.0
        config.network.backbone_params['net_type'] = 'resnext'
        config.network.backbone_params['num_group'] = 64
    else:
        raise ValueError("unknown net type {}".format(net_type))
    return config
