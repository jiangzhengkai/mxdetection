import os
from easydict import EasyDict as edict
from mxdetection.config.config_common import *
from mxdetection.config.config_fpn import add_fpn_params
from mxdetection.config.config_network import add_network_params
from mxdetection.config.config_mask import add_mask_params

config = edict()
config.person_name = 'zhengkai.jiang'
config.root_output_dir = 'experiments_mask_res50'
config = add_home_dir(config)

# train params
config = add_train_params(config)
config.TRAIN.gpus = '2,3'
config.TRAIN.image_batch_size = 2
config.TRAIN.use_prefetchiter = True
config.TRAIN.bn_use_global_stats = True
config.TRAIN.do_eval_during_training = True
config.TRAIN.solver.optimizer = 'sgd'
config.TRAIN.solver.lr_step = '14,17'
config.TRAIN.solver.num_epoch = 18
config.TRAIN.filter_strategy.remove_empty_boxes = True
# config.TRAIN.aug_strategy.scales = [(416, 1024), (448, 1024), (480, 1024), (512, 1024), (544, 1024), (576, 1024), (608, 1024)]
# config.TRAIN.aug_strategy.scales = [(608, 1333), (640, 1333), (672, 1333), (704, 1333), (736, 1333), (768, 1333), (800, 1333)]
# config.TRAIN.aug_strategy.scales = [(512, 1173), (544, 1173), (576, 1173), (608, 1173), (640, 1173), (672, 1173), (704, 1173)]
config.TRAIN.aug_strategy.scales = [(600, 1000)]
config.TRAIN.aug_strategy.flip = True
config = modify_lr_params(config, use_warmup=True)

# test params
config = add_test_params(config)
config.TEST.gpus = '0,1,2,3,4,5,6,7'
config.TEST.aug_strategy.scales = [(600, 1000)]
config.TEST.load_sym_from_file = False

# dataset params
dataset_type = 'instance'
train_dataset_list = list()
test_dataset_list = list()
train_dataset_list.append(get_dataset_info('coco2017', 'train2017', dataset_type))
test_dataset_list.append(get_dataset_info('coco2017', 'val2017', dataset_type))
config = add_dataset_params(config=config,
                            train_dataset_list=train_dataset_list,
                            test_dataset_list=test_dataset_list)
config.dataset.num_classes = 81
config = add_test_coco_anno_path(config, dataset_type)

# net params
config = add_network_params(config, 'resnet#50')
config.network.task_type = 'fpn_rpn_rcnn_mask'  # 'fpn_rpn_rcnn', 'frcnn_rpn_rcnn', 'fpn_only_rpn', 'frcnn_only_rpn'

config.network.sym = 'mxdetection.models.detectors.TwoStageDetector.TwoStageDetector'
config.network.backbone = 'mxdetection.models.backbones.backbone.Backbone'
config.network.neck = 'mxdetection.models.necks.fpn.FPN'
config.network.rpn_head = 'mxdetection.models.rpn_heads.rpn_head.RpnHead'
config.network.bbox_roi_extractor = 'mxdetection.models.roi_extractors.bbox_rois_extractor.Bbox_rois_extractor'
config.network.bbox_head = 'mxdetection.models.bbox_heads.bbox_head.BboxHead'

if 'mask' in config.network.task_type:
    config.network.mask_head = 'mxdetection.models.mask_heads.mask_head.MaskHead'
    config.network.mask_roi_extractor = 'mxdetection.models.roi_extractors.mask_rois_extractor.Mask_rois_extractor'
else:
    config.network.mask_head = None

# det params
if 'fpn' in config.network.task_type:
    config = add_fpn_params(config)
elif 'frcnn' in config.network.task_type:
    config = add_frcnn_params(config)

if 'mask' in config.network.task_type:
    config = add_mask_params(config)
    config.TRAIN.mask_roi_batch_size = int(config.TRAIN.rcnn_batch_rois * config.TRAIN.rcnn_fg_fraction)

# experiments params
config.experiments = 'symbol_fpn_detection_resnet50'

config.TRAIN.model_prefix = os.path.join(config.hdfs_remote_home_dir, config.root_output_dir, config.experiments, 'models') + '/fpn_detection'
