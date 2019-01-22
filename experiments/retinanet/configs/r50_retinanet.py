from mxdetection.config.config_common import *
from mxdetection.config.config_fpn_retinanet import add_fpn_retinanet_params
from mxdetection.config.config_network import add_network_params
import os
from easydict import EasyDict as edict

config = edict()
config.person_name = 'zhengkai.jiang'
config.root_output_dir = 'experiments_retinanet'
config = add_home_dir(config)

# train params
config = add_train_params(config)
config.TRAIN.gpus = '0,1,2,3,4,5,6,7'
config.TRAIN.image_batch_size = 2
config.TRAIN.use_prefetchiter = True
config.TRAIN.bn_use_global_stats = True
config.TRAIN.do_eval_during_training = True
config.TRAIN.solver.optimizer = 'sgd'
config.TRAIN.solver.lr_step = '14,17'
config.TRAIN.solver.num_epoch = 18
config.TRAIN.filter_strategy.remove_empty_boxes = True
config.TRAIN.aug_strategy.scales = [(600, 1000)]
config.TRAIN.aug_strategy.flip = True
config = modify_lr_params(config, use_warmup=True)

# test params
config = add_test_params(config)
config.TEST.gpus = '0,1,2,3,4,5,6,7'
config.TEST.aug_strategy.scales = [(600, 1000)]
config.TEST.load_sym_from_file = False

# dataset params
dataset_type = 'instances'
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
config.network.task_type = 'fpn_retinanet'
config.network.sym = 'mxdetection.models.detectors.SingleStageDetector.SingleStageDetector'
config.network.backbone = 'mxdetection.models.backbones.backbone.Backbone'
config.network.neck = 'mxdetection.models.necks.fpn.FPN'
config.network.bbox_head = 'mxdetection.models.single_stage_heads.retina_head.RetinaHead'

# det params
config = add_fpn_retinanet_params(config)
config.TRAIN.retinanet_cls_loss_type = 'focal_loss_cross_entropy'
config.TRAIN.retinanet_bg_fg_ratio = -1

# experiments params
config.experiments = 'symbol_retinanet_detection_resnet50'
config.TRAIN.model_prefix = '/job_data/retinanet_detection'


#config.TRAIN.model_prefix = os.path.join(config.hdfs_remote_home_dir, config.root_output_dir, config.experiments, 'models') + '/retinanet_detection'
