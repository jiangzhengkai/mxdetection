import logging
import os
import numpy as np
import mxnet as mx
from mxdetection.datasets.loader.retina_iter import RetinanetTestIter
from mxdetection.core.bbox.nms.nms import py_nms_wrapper, py_softnms_wrapper
from .retinanet_predict_utils import retinanet_predict
from mxdetection.models.predictors.base_predictor import BasePredictor
from mxdetection.utils.utils import save_roidb

def eval_during_training_func(roidb, eval_func, config):
    test_ctx = [mx.gpu(int(i)) for i in config.TEST.gpus.split(',')]
    test_data = RetinanetTestIter(roidb=roidb, config=config, batch_size=len(test_ctx))
    def _callback(iter_no, sym, arg, aux):
        if iter_no + 1 == config.TRAIN.solver.num_epoch or True:
            predictor = RetinanetPredictor(config, config.TRAIN.model_prefix, iter_no + 1,
                                           test_data.provide_data, test_data.max_data_shape, test_ctx)
            predictor.predict_data(test_data, eval_func=eval_func, alg='alg-pid%d' % os.getpid())
    return _callback



class RetinanetPredictor(BasePredictor):
    def __init__(self, config, prefix, epoch, provide_data, max_data_shape, ctx=mx.cpu()):
        super(RetinanetPredictor, self).__init__(config, prefix, epoch, provide_data, max_data_shape, ctx=ctx,
                                                 allow_missing=config.TEST.params_allow_missing)
        if self.config.TEST.retinanet_nms_method == 'softnms':
            self.nms_func = py_softnms_wrapper(self.config.TEST.retinanet_softnms)
        else:
            self.nms_func = py_nms_wrapper(self.config.TEST.retinanet_nms)

    def predict_data(self, test_data, eval_func=None, alg='alg',
                     save_roidb_path=None, vis=False, **vis_kwargs):
        assert len(self.config.TEST.aug_strategy.scales) == 1
        assert self.config.network.retinanet_num_branch == 1

        num_classes = self.config.dataset.num_classes
        all_boxes = [[[] for _ in xrange(test_data.size)] for _ in xrange(num_classes)]
        k = 0
        for data_batch, need_forward in test_data:
            if k % 100 == 0:
                logging.info('{}/{}'.format(k, test_data.size))
            outputs = self.predict(data_batch, need_forward)
            for i in range(len(outputs)):
                outputs_i = [outputs[i][j].asnumpy() for j in range(len(outputs[i]))]
                local_vars_i = test_data.extra_local_vars[i]
                retinanet_output = retinanet_predict(outputs_i, local_vars_i, self.config)
                for j in range(1, num_classes):
                    keep_j = np.where(retinanet_output[:, -1] == j)[0]
                    all_boxes[j][k + i] = retinanet_output[keep_j, :5]
                if vis:
                    self.vis_results(local_vars=local_vars_i,
                                     det_results=retinanet_output,
                                     **vis_kwargs)
            k += test_data.batch_size
        test_data.reset()
        if save_roidb_path is not None:
            results = dict()
            results['all_boxes'] = all_boxes
            save_roidb(results, save_roidb_path)
        if eval_func is not None:
            eval_func(all_boxes=all_boxes, alg=alg)
