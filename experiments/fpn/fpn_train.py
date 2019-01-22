import init_paths
from configs.r50_fpn import config
from mxdetection.datasets.loader.fpn_iter import FPNIter
from mxdetection.apis.train_net import train_net
from mxdetection.models.predictors.fpn_predict import eval_during_training_func
from mxdetection.datasets.load_roidb_eval import load_coco_test_roidb_eval

def main(config):
    epoch_end_callback = []
    if config.TRAIN.do_eval_during_training:
        test_roidb, eval_func = load_coco_test_roidb_eval(config)
        epoch_end_callback = eval_during_training_func(roidb=test_roidb, eval_func=eval_func, config=config)
        
    train_net(config=config,
              TrainDataIter=FPNIter,
              epoch_end_callback=epoch_end_callback)

if __name__ == '__main__':
    main(config)
