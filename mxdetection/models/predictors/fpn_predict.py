import logging
import os
import numpy as np
import mxnet as mx
from mxdetection.datasets.loader.fpn_iter import FPNTestIter
from .fpn_predict_utils import rpn_predict, rcnn_predict
from .base_predictor import BasePredictor
from mxdetection.utils.utils import save_roidb

def eval_during_training_func(roidb, eval_func, config):
    test_ctx = [mx.gpu(int(i)) for i in config.TEST.gpus.split(',')]
    test_data = FPNTestIter(roidb=roidb, config=config, batch_size=len(test_ctx))
    
    def _callback(iter_no, sym, arg, aux):
        if iter_no + 1 == config.TRAIN.sover.num_epoch or True:
            predictor = FPNpredictor(config, config.TRAIN.model_prefix, iter_no + 1,
                                     test_data.provide_data, test_data.max_data_shape, test_ctx)
            predictor.predict_data(test_data, eval_func=eval_func, alg='alg-pid%d' % os.getpid())
    return _callback

class FPNPredictor(BasePredictor):
    def __init__(self, config, prefix, epoch, provide_data, max_data_shape, ctx=mx.cpu()):
        super(FPNPredictor, self).__init__(config, prefix, epoch, provide_data, max_data_shape, ctx=ctx,
                                           allow_missing=config.TEST.params_allow_missing)

    def predict_data(self, test_data, eval_func=None, alg='alg',
                     save_roidb_path=None, vis=False, **vis_kwargs):
        assert len(self.config.TEST.aug_strategy.scales) == 1
        num_branch = self.config.network.rpn_rcnn_num_branch
        if num_branch > 1:
            assert self.config.dataset.num_classes == 2
            num_rpn_classes = num_branch + 1
            num_rcnn_classes = num_branch + 1
        else:
            num_rpn_classes = 2
            num_rcnn_classes = self.config.dataset.num_classes
        all_proposals = [[[] for _ in xrange(test_data.size)] for _ in xrange(num_rpn_classes)]
        all_boxes = [[[] for _ in xrange(test_data.size)] for _ in xrange(num_rcnn_classes)]

        k = 0
        for data_batch, need_forward in test_data:
            if k % 100 == 0:
                logging.info('{}/{}'.format(k, test_data.size))
            outputs = self.predict(data_batch, need_forward)
            for i in range(len(outputs)):
                outputs_i = [outputs[i][j].asnumpy() for j in range(len(outputs[i]))]
                local_vars_i = test_data.extra_local_vars[i]
                if self.config.TEST.rpn_do_test:
                    rpn_outputs_i = [outputs_i[_] for _ in range(num_branch * 2)]
                    proposals = rpn_predict(rpn_outputs_i, local_vars_i, self.config)
                    for j in range(1, num_rpn_classes):
                        proposals_j = proposals[:, (j - 1) * 5:j * 5]
                        keep_j = np.where(np.sum(proposals_j, axis=1) != -500)[0]
                        all_proposals[j][k + i] = proposals_j[keep_j]
                if 'rpn_rcnn' in self.config.network.task_type:
                    rcnn_outputs_i = [outputs_i[_] for _ in range(len(outputs_i) - num_branch, len(outputs_i))]
                    rcnn_output = rcnn_predict(rcnn_outputs_i, local_vars_i, self.config)
                    for j in range(1, num_rcnn_classes):
                        keep_j = np.where(rcnn_output[:, -1] == j)[0]
                        all_boxes[j][k + i] = rcnn_output[keep_j, :5]
                    if vis:
                        self.vis_results(local_vars=local_vars_i,
                                         det_results=rcnn_output,
                                         **vis_kwargs)
            k += test_data.batch_size
        test_data.reset()
        if save_roidb_path is not None:
            results = dict()
            results['all_boxes'] = all_boxes
            save_roidb(results, save_roidb_path)
        if eval_func is not None:
            eval_func(all_proposals=all_proposals,
                      all_boxes=all_boxes,
                      alg=alg)

