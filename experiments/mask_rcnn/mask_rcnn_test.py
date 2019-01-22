import init_paths
import os
import pprint
import mxnet as mx
from configs.r50_mask_rcnn import config
from mxdetection.datasets.loader.mask_iter import FPNMaskTestIter
from mxdetection.utils.utils import create_logger, copy_file
from mxdetection.datasets.load_roidb_eval import load_coco_test_roidb_eval
from mxdetection.models.predictors.fpn_mask_predict import FPNMaskPredictor


def main():
    pid = os.getpid()
    if config.TRAIN.num_workers > 1:
        config.TRAIN.model_prefix += "-%d" % (config.TRAIN.num_workers - 1)
    test_ctx = [mx.gpu(int(i)) for i in config.TEST.gpus.split(',')]

    log_name = 'test_pid%d.log' % pid
    # get logger
    logger = create_logger(log_name)

    config.TEST.filter_strategy.remove_empty_boxes = True
    config.TEST.filter_strategy.max_num_images = 5000
    test_roidb, eval_func = load_coco_test_roidb_eval(config)

    logger.info('config:{}\n'.format(pprint.pformat(config)))
    test_data = FPNMaskTestIter(roidb=test_roidb, config=config, batch_size=len(test_ctx))
    begin_test_epoch = config.TRAIN.solver.num_epoch
    end_test_epoch = config.TRAIN.solver.num_epoch + 1
    for epoch in range(begin_test_epoch, end_test_epoch):
        predictor = FPNMaskPredictor(config=config,
                                 prefix=config.TRAIN.model_prefix,
                                 epoch=epoch,
                                 provide_data=test_data.provide_data,
                                 max_data_shape=test_data.max_data_shape,
                                 ctx=test_ctx)

        predictor.predict_fpn_mask(test_data)


if __name__ == '__main__':
    main()
