import os
import random
import logging
import pprint
import numpy as np
import mxnet as mx

from .module import MutableModule
from mxdetection.utils.callback import Speedometer
from mxdetection.datasets.load_roidb import load_roidb
from mxdetection.core.loss.metric import get_eval_metrics
from mxdetection.utils.lr_scheduler import get_warmupmf_scheduler
from mxdetection.models.utils.symbol_common import get_symbol_function
from mxdetection.utils.utils import create_logger, get_kv, load_param


def train_net(config, TrainDataIter, **kwargs):
    ctx = [mx.gpu(int(i)) for i in config.TRAIN.gpus.split(',')]
    # set random seed
    seed = 6
    mx.random.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # init kv and log_name
    if config.TRAIN.num_workers > 1:
        kv = mx.kvstore.create('dist_sync')
        log_name = 'train_test_%d_%s.log' % (kv.rank, config.network.task_type)
    else:
        kv = get_kv(len(ctx))
        log_name = 'train_test_%s.log' % config.network.task_type
    
    # get logger
    logger = create_logger(log_name)
    
    # init dist
    if config.TRAIN.bn_use_sync:
        assert not config.TRAIN.bn_use_global_stats
        from mxnet.device_sync import init_device_sync
        init_device_sync(ctx)

    # get_symbol
    config.network.symbol_network = get_symbol_function(config.network.sym)(config).get_train_symbol()
    
    # load training data
    imglst_path_list = config.dataset.train_imglst_path_list
    roidb = load_roidb(roidb_path_list=config.dataset.train_roidb_path_list,
                       imglst_path_list=imglst_path_list,
                       filter_strategy=config.TRAIN.filter_strategy)
    train_data = TrainDataIter(roidb=roidb, config=config, batch_size=config.TRAIN.image_batch_size * len(ctx), ctx=ctx)
    if config.TRAIN.use_prefetchiter:
        from mxdetection.datasets.loader.base_iter import PrefetchingIter
        train_data = PrefetchingIter(data_iter=train_data, num_workers=8, max_queue_size=8)

    # load initialized params
    if config.TRAIN.solver.load_epoch is not None:
        logger.info('continue training from %s-%04d.params' % (config.TRAIN.model_prefix, config.TRAIN.solver.load_epoch))
        arg_params, aux_params = load_param(config.TRAIN.model_prefix, config.TRAIN.solver.load_epoch)
    elif config.network.pretrained_prefix is not None:
        logger.info('init model from %s-%04d.params' % (config.network.pretrained_prefix, config.network.pretrained_epoch))
        arg_params, aux_params = load_param(config.network.pretrained_prefix, config.network.pretrained_epoch)
    else:
        arg_params = dict()
        aux_params = dict()
    if 'arg_params' in kwargs:
        arg_params.update(kwargs['arg_params'])
    if 'aux_params' in kwargs:
        aux_params.update(kwargs['aux_params'])
    if len(arg_params) == 0:
        arg_params = None
    if len(aux_params) == 0:
        aux_params = None
        
    # load initializer
    if 'initializer' in kwargs:
        initializer = kwargs['initializer']
    else:
        initializer = mx.init.Normal(sigma=0.01)
    
    # load optimizer
    config.TRAIN.num_examples = train_data.size
    config.TRAIN.batch_size = train_data.batch_size
    config, lr_scheduler = get_warmupmf_scheduler(config)
    optimizer_params = {'rescale_grad': 1.0 / (config.TRAIN.num_workers * len(ctx)),
                        'learning_rate': config.TRAIN.solver.lr,
                        'lr_scheduler': lr_scheduler,
                        'begin_num_update': config.TRAIN.solver.begin_num_update}
    if config.TRAIN.solver.optimizer == 'sgd':
        optimizer_params['momentum'] = 0.9
    if 'wd' in config.TRAIN:
        optimizer_params['wd'] = config.TRAIN.wd
    logger.info('optimizer_params:{}\n'.format(pprint.pformat(optimizer_params)))
    
    # load loss for eval_metrics
    if 'eval_info_list' in kwargs:
        eval_info_list = kwargs['eval_info_list']
    else:
        eval_info_list = get_eval_info_list(config)
    eval_metrics = get_eval_metrics(eval_info_list)
    
    # create module
    mod = MutableModule(symbol=config.network.symbol_network,
                        data_names=train_data.data_name,
                        label_names=train_data.label_name,
                        logger=logger,
                        context=ctx,
                        max_data_shapes=train_data.max_data_shape,
                        max_label_shapes=train_data.max_label_shape)
    
    # get batch and epoch and callback
    batch_end_callback = [Speedometer(config.TRAIN.batch_size, frequent=config.TRAIN.solver.frequent)]
    epoch_end_callback = [mx.callback.do_checkpoint(config.TRAIN.model_prefix)]
    
    
    # start training
    logger.info('config:{}\n'.format(pprint.pformat(config)))
    mod.fit(train_data=train_data,
            eval_metric=eval_metrics,
            epoch_end_callback=epoch_end_callback,
            batch_end_callback=batch_end_callback,
            kvstore=kv,
            optimizer=config.TRAIN.solver.optimizer,
            optimizer_params=optimizer_params,
            arg_params=arg_params,
            aux_params=aux_params,
            initializer=initializer,
            allow_missing=True,
            begin_epoch=config.TRAIN.solver.begin_epoch,
            num_epoch=config.TRAIN.solver.num_epoch)
    
   
def get_eval_info_list(config):
    eval_info_list = []
    if 'retina' in config.network.task_type:
        from mxdetection.models.single_stage_heads.retinanet_symbol_utils import get_retinanet_eval_info_list
        eval_info_list.extend(get_retinanet_eval_info_list(config))
    if 'rpn' in config.network.task_type:
        from mxdetection.models.rpn_heads.rpn_symbol_utils import get_rpn_eval_info_list
        eval_info_list.extend(get_rpn_eval_info_list(config))
    if 'rpn_rcnn' in config.network.task_type:
        from mxdetection.models.bbox_heads.rcnn_symbol_utils import get_rcnn_eval_info_list
        eval_info_list.extend(get_rcnn_eval_info_list(config))
    if 'mask' in config.network.task_type:
        from mxdetection.models.mask_heads.mask_symbol_utils import get_mask_eval_info_list
        eval_info_list.extend(get_mask_eval_info_list(config))
    assert len(eval_info_list) > 0
    return eval_info_list 
    
