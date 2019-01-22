import mxnet as mx
import logging
from rcnn_symbol_utils import get_rcnn_grad_scales
from mxdetection.ops.operator_py.box_annotator_ohem import box_annotator_ohem

class BboxHead(object):
    def __init__(self, config):
        self.config = config

    def get_symbol(self, roi_feats, is_train, suffix='', **kwargs):
        num_classes = self.config.dataset.num_classes
        num_bbox_classes = self.config.dataset.num_classes

        rcnn_fc6 = mx.sym.FullyConnected(data=roi_feats, name='rcnn_fc6%s' % suffix, num_hidden=1024, no_bias=False)
        rcnn_fc6 = mx.sym.relu(data=rcnn_fc6, name='rcnn_relu6%s' % suffix)
        rcnn_fc7 = mx.sym.FullyConnected(data=rcnn_fc6, name='rcnn_fc7%s' % suffix, num_hidden=1024, no_bias=False)
        rcnn_fc7 = mx.sym.relu(data=rcnn_fc7, name='rcnn_relu7%s' % suffix)

        rcnn_cls_score = mx.sym.FullyConnected(data=rcnn_fc7, name='rcnn_cls_score%s' % suffix, num_hidden=num_classes,
                                               no_bias=False)
        rcnn_bbox_pred = mx.sym.FullyConnected(data=rcnn_fc7, name='rcnn_bbox_pred%s' % suffix,
                                               num_hidden=num_bbox_classes * 4, no_bias=False)

        if 'input_dict' in kwargs:
            input_dict = kwargs['input_dict']
            logging.info('rcnn_fc6{}: {}'.format(suffix, rcnn_fc6.infer_shape(**input_dict)[1]))
            logging.info('rcnn_fc7{}: {}'.format(suffix, rcnn_fc7.infer_shape(**input_dict)[1]))
            logging.info('rcnn_cls_score{}: {}'.format(suffix, rcnn_cls_score.infer_shape(**input_dict)[1]))
            logging.info('rcnn_bbox_pred{}: {}'.format(suffix, rcnn_bbox_pred.infer_shape(**input_dict)[1]))

        return self.get_rcnn_train_test(rcnn_cls_score=rcnn_cls_score, rcnn_bbox_pred=rcnn_bbox_pred,
                                        is_train=is_train, suffix=suffix, **kwargs)

    def get_rcnn_train_test(self, rcnn_cls_score, rcnn_bbox_pred, is_train=True,
                            rcnn_label=None, rcnn_bbox_target=None, rcnn_bbox_weight=None, suffix='', **kwargs):
        num_classes = self.config.dataset.num_classes
        num_bbox_classes = self.config.dataset.num_classes

        if is_train:
            if self.config.TRAIN.rcnn_enable_ohem:
                rcnn_cls_prob_ohem = mx.sym.SoftmaxActivation(data=rcnn_cls_score)
                rcnn_bbox_loss_ohem = rcnn_bbox_weight * mx.sym.smooth_l1(data=(rcnn_bbox_pred - rcnn_bbox_target), scalar=1.0)
                rcnn_label, rcnn_bbox_weight = box_annotator_ohem(rcnn_cls_prob=rcnn_cls_prob_ohem,
                                                                  rcnn_bbox_loss=rcnn_bbox_loss_ohem,
                                                                  rcnn_label=rcnn_label,
                                                                  rcnn_bbox_weight=rcnn_bbox_weight,
                                                                  roi_per_img=self.config.TRAIN.rcnn_batch_rois,
                                                                  batch_size=self.config.TRAIN.image_batch_size)
                if 'input_dict' in kwargs:
                    input_dict = kwargs['input_dict']
                    logging.info('rcnn_label_ohem{}: {}'.format(suffix, rcnn_label.infer_shape(**input_dict)[1]))
                    logging.info('rcnn_bbox_weight_ohem{}: {}'.format(suffix, rcnn_bbox_weight.infer_shape(**input_dict)[1]))
            grad_scales = get_rcnn_grad_scales(self.config)
            rcnn_cls_prob = mx.sym.SoftmaxOutput(data=rcnn_cls_score, label=rcnn_label, normalization='valid',
                                                 use_ignore=True, ignore_label=-1, grad_scale=grad_scales[0])
            rcnn_bbox_loss_t = rcnn_bbox_weight * mx.sym.smooth_l1(data=(rcnn_bbox_pred - rcnn_bbox_target), scalar=1.0)
            rcnn_bbox_loss = mx.sym.MakeLoss(data=rcnn_bbox_loss_t, grad_scale=grad_scales[1])

            rcnn_label = mx.sym.Reshape(data=rcnn_label, shape=(self.config.TRAIN.image_batch_size, -1))
            rcnn_cls_prob = mx.sym.Reshape(data=rcnn_cls_prob,
                                           shape=(self.config.TRAIN.image_batch_size, -1, num_classes))
            rcnn_bbox_loss = mx.sym.Reshape(data=rcnn_bbox_loss,
                                            shape=(self.config.TRAIN.image_batch_size, -1, num_bbox_classes * 4))
            return [rcnn_label, rcnn_cls_prob, rcnn_bbox_loss]
        else:
            rcnn_cls_prob = mx.sym.SoftmaxActivation(data=rcnn_cls_score)
            return [rcnn_cls_prob, rcnn_bbox_pred]


