import logging
import os
import numpy as np
import mxnet as mx
from mxdetection.datasets.loader.mask_iter import FPNMaskTestIter
from .fpn_mask_predict_utils import e2e_mask_predict
from .base_predictor import BasePredictor
from mxdetection.utils.utils import save_roidb

def eval_during_training_func(roidb, eval_func, config):
    test_ctx = [mx.gpu(int(i)) for i in config.TEST.gpus.split(',')]
    test_data = FPNMaskTestIter(roidb=roidb, config=config, batch_size=len(test_ctx))
    def _callback(iter_no, sym, arg, aux):
        if iter_no + 1 == config.TRAIN.solver.num_epoch:
            predictor = FPNMaskPredictor(config, config.TRAIN.model_prefix, iter_no + 1,
                                              test_data.provide_data, test_data.max_data_shape, test_ctx)
            predictor.predict_fpn_mask(test_data, eval_func)
    return _callback

class FPNMaskPredictor(BasePredictor):
    def __init__(self, config, prefix, epoch, provide_data, max_data_shape, ctx=mx.cpu()):
        super(FPNMaskPredictor, self).__init__(config, prefix, epoch, provide_data, max_data_shape, ctx=ctx,
                                                    allow_missing=config.network.params_allow_missing)
        self.new_roidb = []
    def predict_fpn_mask(self, test_data, eval_func=None, alg='alg', save_roidb_path=None, vis=False, **vis_kwargs):
        k = 0
        num_classes = self.config.dataset.num_classes
        all_boxes = [[[] for _ in xrange(test_data.size)] for _ in xrange(num_classes)]
        all_mask_boxes = [[[] for _ in xrange(test_data.size)] for _ in xrange(num_classes)]
        for data_batch, need_forward in test_data:
            if k % 100 == 0:
                logging.info('{}/{}'.format(k, test_data.size))
            outputs = self.predict(data_batch, need_forward)
            for i in range(len(outputs)):
                outputs_i = [outputs[i][j].asnumpy() for j in range(len(outputs[i]))]
                local_vars_i = test_data.extra_local_vars[i]
                res_dict = e2e_mask_predict(outputs_i, local_vars_i, self.config)
                if len(res_dict['det_results']) > 0:
                    for j in range(1, num_classes):
                        keep_j = np.where(res_dict['det_results'][:, -1] == j)[0]
                        all_boxes[j][k + i] = res_dict['det_results'][keep_j, :5]
                        if 'mask_boxes' in res_dict['mask_results'] and len(res_dict['mask_results']['mask_boxes']) > 0:
                            all_mask_boxes[j][k + i] = res_dict['mask_results']['mask_boxes'][keep_j, :]
                            all_masks[j][k + i] = res_dict['mask_results']['masks'][keep_j, :]
                if save_roidb_path is not None:
                    roi_rec = local_vars_i['roi_rec'].copy()
                    for name in res_dict:
                        roi_rec[name] = res_dict[name]
                    self.new_roidb.append(roi_rec)
                if vis:
                    self.vis_results(local_vars=local_vars_i,
                                     det_results=res_dict['det_results'],
                                     mask_results=res_dict['mask_results'],
                                     **vis_kwargs)
            k += test_data.batch_size
        test_data.reset()
        if save_roidb_path is not None:
            self.save_roidb(self.new_roidb, save_roidb_path)
            self.new_roidb = []
        if eval_func is not None:
            eval_func(all_boxes=all_boxes,
                      all_mask_boxes=all_mask_boxes,
                      all_masks=all_masks,
                      alg=alg)



