import numpy as np
import os
from easydict import EasyDict as edict


def add_home_dir(config):
    file_path = os.path.abspath(__file__)
    config.local_home_dir = file_path.split(config.person_name)[0] + config.person_name
    config.hdfs_home_dir = '/opt/hdfs/user/' + config.person_name
    config.hdfs_remote_home_dir = 'hdfs://hobot-bigdata/user/' + config.person_name
    return config

def add_train_params(config):
    if 'TRAIN' not in config:
        config.TRAIN = edict()
    config.TRAIN.gpus = '0,1,2,3'
    config.TRAIN.image_batch_size = 1
    config.TRAIN.num_workers = 1
    config.TRAIN.use_prefetchiter = False
    config.TRAIN.bn_use_sync = False
    config.TRAIN.bn_use_global_stats = True
    config.TRAIN.do_eval_during_training = False
    config.TRAIN.absorb_bn = False
    # solver params
    config.TRAIN.solver = edict()
    config.TRAIN.solver.optimizer = 'sgd'
    config.TRAIN.solver._lr = 0.00125  # 0.00125 * 16 = 0.02 refer to fpn paper
    config.TRAIN.solver.lr_step = '1'
    config.TRAIN.solver.num_epoch = 2
    config.TRAIN.solver.frequent = 100
    config.TRAIN.solver.load_epoch = None
    # warm up params
    config.TRAIN.solver.warmup = False
    config.TRAIN.solver.warmup_linear = True
    config.TRAIN.solver.warmup_step_ratio = 1.0
    config.TRAIN.solver.warmup_lr = 0
    # filter_strategy params
    config.TRAIN.filter_strategy = edict()
    # aug_strategy params
    config.TRAIN.aug_strategy = edict()
    config.TRAIN.aug_strategy.shuffle = True
    config.TRAIN.aug_strategy.aspect_grouping = True
    config.TRAIN.aug_strategy.flip = False
    config.TRAIN.aug_strategy.rotated_angle_range = 0
    config.TRAIN.aug_strategy.scales = [(600, 1000)]
    return config

def modify_lr_params(config, use_warmup=True):
    num_gpus = len(config.TRAIN.gpus.split(','))
    config.TRAIN.solver.lr = config.TRAIN.solver._lr * config.TRAIN.image_batch_size * num_gpus * config.TRAIN.num_workers
    if use_warmup and config.TRAIN.solver.lr > 0.0001:
        config.TRAIN.solver.warmup = True
        config.TRAIN.solver.warmup_lr = 0.0001
    else:
        config.TRAIN.solver.warmup = False
    return config

def add_test_params(config):
    if 'TEST' not in config:
        config.TEST = edict()
    config.TEST.gpus = '0,1,2,3'
    config.TEST.use_gt_rois = False
    config.TEST.filter_strategy = edict()
    config.TEST.aug_strategy = edict()   
    config.TEST.aug_strategy.scales = [(600, 1000)]
    config.TEST.params_allow_missing = False
    config.TEST.ms_testing = False
    config.TEST.aug_strategy.flip = False
    config.TEST.aug_strategy.bbox_vote = False
    config.TEST.aug_strategy.bbox_vote_thresh = 0.9
    config.TEST.max_per_image = 100
    return config

def add_network_params(config, net_type_layer):
    if 'network' not in config:
        config.network = edict()
    home_dir = config.hdfs_remote_home_dir
    net_type, num_layer = net_type_layer.split('#')
    config.network.net_type = net_type
    config.network.num_layer = int(num_layer)
    if net_type == 'resnet':
        config.network.pretrained_prefix = home_dir + '/common/models/resnet-%d' % config.network.num_layer
        config.network.pretrained_epoch = 0
        config.network.input_mean = np.array([0, 0, 0], dtype=np.float32)
        config.network.input_scale = 1.0
    config.network.add_nonlocal = False
    config.network.use_fpn_augmentation = False
    # other params
    if 'image_stride' not in config.network:
        config.network.image_stride = 0
    return config

def get_dataset_info(dataset_name, imageset_name, dataset_type):
    dataset_info = dict()
    dataset_info['dataset_name'] = dataset_name
    dataset_info['imageset_name'] = imageset_name
    dataset_info['roidb_name'] = '%s_%s_gt_roidb' % (imageset_name, dataset_type)
    return dataset_info

def add_dataset_params(config, train_dataset_list=None, test_dataset_list=None):
    if 'dataset' not in config:
        config.dataset = edict()
    # train
    config.dataset.train_roidb_path_list = []
    config.dataset.train_imglst_path_list = []
    config.dataset.train_imgidx_path_list = []
    config.dataset.train_imgrec_path_list = []
    if train_dataset_list is not None:
        for dataset_info in train_dataset_list:
            roidb_path, imglst_path, imgidx_path, imgrec_path = get_roidb_image_path(config, dataset_info)
            config.dataset.train_roidb_path_list.append(roidb_path)
            config.dataset.train_imglst_path_list.append(imglst_path)
            config.dataset.train_imgidx_path_list.append(imgidx_path)
            config.dataset.train_imgrec_path_list.append(imgrec_path)
    # test
    config.dataset.test_roidb_path_list = []
    config.dataset.test_imglst_path_list = []
    config.dataset.test_imgidx_path_list = []
    config.dataset.test_imgrec_path_list = []
    if test_dataset_list is not None:
        for dataset_info in test_dataset_list:
            roidb_path, imglst_path, imgidx_path, imgrec_path = get_roidb_image_path(config, dataset_info)
            config.dataset.test_roidb_path_list.append(roidb_path)
            config.dataset.test_imglst_path_list.append(imglst_path)
            config.dataset.test_imgidx_path_list.append(imgidx_path)
            config.dataset.test_imgrec_path_list.append(imgrec_path)
    return config

def get_roidb_image_path(config, dataset_info):
    imageset_name = dataset_info['imageset_name']
    dataset_type = dataset_info['roidb_name']
    roidb_path = os.path.join('../', 'roidbs', '%s.pkl' % (dataset_type))
    imglst_path = os.path.join('../', 'images_lst_rec', '%s.lst' % imageset_name)
    imgidx_path = os.path.join('../', 'images_lst_rec', '%s.idx' % imageset_name)
    imgrec_path = os.path.join('../', 'images_lst_rec', '%s.rec' % imageset_name)
    return roidb_path, imglst_path, imgidx_path, imgrec_path

def add_test_coco_anno_path(config, dataset_type):
    sub_anno_path = '../annotations/%s_val2017.json'
    config.dataset.test_coco_anno_path = dict()
    if dataset_type == 'det':
        config.dataset.test_coco_anno_path['det'] = sub_anno_path % 'det'
    elif dataset_type == 'instances':
        config.dataset.test_coco_anno_path['instances'] = sub_anno_path % 'instances'
    elif dataset_type == 'kps':
        config.dataset.test_coco_anno_path['det'] = sub_anno_path % 'person_keypoints'
    elif dataset_type == 'seg':
        config.dataset.test_coco_anno_path['det'] = sub_anno_path % 'stuff'
    else:
        assert False
    return config
