import init_paths
import pprint
import os
from configs.r50_retinanet import config
from mxdetection.datasets.loader.retina_iter import RetinanetTestIter
from mxdetection.models.predictors.retinanet_predict import RetinanetPredictor
from mxdetection.utils.utils import create_logger
from mxdetection.datasets.load_roidb_eval import load_coco_test_roidb_eval
import mxnet as mx

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
    test_data = RetinanetTestIter(roidb=test_roidb, config=config, batch_size=len(test_ctx))
    begin_test_epoch = config.TRAIN.solver.num_epoch
    end_test_epoch = config.TRAIN.solver.num_epoch + 1
    for epoch in range(begin_test_epoch, end_test_epoch):
        predictor = RetinanetPredictor(config=config,
                                       prefix=config.TRAIN.model_prefix,
                                       epoch=epoch,
                                       provide_data=test_data.provide_data,
                                       max_data_shape=test_data.max_data_shape,
                                       ctx=test_ctx)
        predictor.predict_data(test_data)

if __name__ == '__main__':
    main()
