import mxnet as mx
import logging
import numpy  as np
import math
from mxdetection.models.utils import symbol_common as sym
from mxdetection.ops.operator_py.detection_post_processing import dpp
from retinanet_symbol_utils import get_retinanet_grad_scales

class RetinaHead(object):
    def __init__(self, config):
        self.config = config

    def get_symbol(self, conv_feat, im_info, is_train=True, **kwargs):
        return self.get_fpn_retinanet_net(conv_feat=conv_feat, im_info=im_info, is_train=is_train, **kwargs)

    def get_fpn_retinanet_net(self, conv_feat, im_info, is_train, suffix='', **kwargs):
        input_dict = kwargs['input_dict']
        if self.config.TRAIN.retinanet_cls_loss_type == 'focal_loss_cross_entropy':
            num_classes = self.config.dataset.num_classes - 1
        else:
            num_classes = self.config.dataset.num_classes

        retinanet_output_list = []
        for i, stride in enumerate(self.config.network.retinanet_feat_strides):
            num_anchors = len(self.config.network.retinanet_anchor_ratios) * len(self.config.network.retinanet_anchor_scales[i])
            # cls
            retinanet_cls_conv_stride = conv_feat['stride%d' % stride]
            for j in range(self.config.network.retinanet_num_convs):
                retinanet_cls_conv_stride = sym.conv(data=retinanet_cls_conv_stride,
                                                     name='retinanet_cls_conv_stride%d_n%d%s' % (stride, j, suffix),
                                                     num_filter=self.config.network.retinanet_num_filter, kernel=3)
                retinanet_cls_conv_stride = sym.relu(data=retinanet_cls_conv_stride,
                                                     name='retinanet_cls_relu_stride%d_n%d%s' % (stride, j, suffix))
            if self.config.TRAIN.retinanet_cls_loss_type == 'focal_loss_cross_entropy':
                bias_value = -np.log((1 - 0.01) / 0.01)
                # bias_value = np.zeros((num_classes, num_anchors), dtype=np.float32)
                # bias_value[0, :] = np.log((num_classes - 1) * (1 - 0.01) / 0.01)
                # bias_value = bias_value.reshape((-1,)).tolist()
            else:
                bias_value = 0
            bias_init = mx.init.Constant(value=bias_value)
            bias = mx.sym.Variable('retinanet_cls_score_stride%d%s_bias' % (stride, suffix),
                                   attr={'__init__': bias_init.dumps()})
            retinanet_cls_score = sym.conv(data=retinanet_cls_conv_stride,
                                           name='retinanet_cls_score_stride%d%s' % (stride, suffix),
                                           num_filter=num_anchors * num_classes,
                                           kernel=self.config.network.retinanet_cls_bbox_kernel,
                                           bias=bias)
            # bbox
            if self.config.network.retinanet_share_cls_bbox_tower:
                retinanet_bbox_conv_stride = retinanet_cls_conv_stride
            else:
                retinanet_bbox_conv_stride = conv_feat['stride%d' % stride]
                for j in range(self.config.network.retinanet_num_convs):
                    retinanet_bbox_conv_stride = sym.conv(data=retinanet_bbox_conv_stride,
                                                          name='retinanet_bbox_conv_stride%d_n%d%s' % (
                                                          stride, j, suffix),
                                                          num_filter=self.config.network.retinanet_num_filter, kernel=3)
                    retinanet_bbox_conv_stride = sym.relu(data=retinanet_bbox_conv_stride,
                                                          name='retinanet_bbox_relu_stride%d_n%d%s' % (
                                                          stride, j, suffix))
            retinanet_bbox_pred = sym.conv(data=retinanet_bbox_conv_stride,
                                           name='retinanet_bbox_pred_stride%d%s' % (stride, suffix),
                                           num_filter=num_anchors * 4,
                                           kernel=self.config.network.retinanet_cls_bbox_kernel)

            retinanet_bbox_pred = mx.sym.reshape(data=retinanet_bbox_pred, shape=(0, -4, num_anchors, 4, -2))
            retinanet_cls_score = mx.sym.reshape(data=retinanet_cls_score, shape=(0, -4, num_anchors, num_classes, -2))
            retinanet_output = mx.symbol.concat(*[retinanet_bbox_pred, retinanet_cls_score], dim=2)
            retinanet_output = mx.sym.reshape(data=retinanet_output, shape=(0, -3, -2),
                                              name='retinanet_output_stride%d%s' % (stride, suffix))
            retinanet_output_list.append(retinanet_output)
            logging.info('retinanet_cls_conv_stride{}{}: {}'.format(stride, suffix, retinanet_cls_conv_stride.infer_shape(**input_dict)[1]))
            logging.info('retinanet_output_stride{}{}: {}'.format(stride, suffix, retinanet_output.infer_shape(**input_dict)[1]))

        rois = self.get_proposal_rois(data=retinanet_output_list, im_info=im_info, num_classes=num_classes,
                                      name='dpp%s' % suffix, config=self.config, is_train=is_train)
        logging.info('rois{}: {}'.format(suffix, rois.infer_shape(**input_dict)[1]))

        if is_train:
            retinanet_cls_score_list = []
            retinanet_bbox_pred_list = []
            for i in range(len(retinanet_output_list)):
                num_anchors = len(self.config.network.retinanet_anchor_ratios) * len(self.config.network.retinanet_anchor_scales[i])
                retinanet_cls_bbox = mx.sym.reshape(data=retinanet_output_list[i], shape=(0, -4, num_anchors, 4 + num_classes, -2))
                retinanet_bbox_pred = mx.sym.slice_axis(data=retinanet_cls_bbox, axis=2, begin=0, end=4)
                retinanet_cls_score = mx.sym.slice_axis(data=retinanet_cls_bbox, axis=2, begin=4, end=4 + num_classes)
                retinanet_cls_score = mx.sym.transpose(data=retinanet_cls_score, axes=(0, 2, 1, 3, 4))
                retinanet_cls_score = mx.sym.reshape(data=retinanet_cls_score, shape=(0, num_classes, -1))
                retinanet_bbox_pred = mx.sym.reshape(data=retinanet_bbox_pred, shape=(0, -1))
                retinanet_cls_score_list.append(retinanet_cls_score)
                retinanet_bbox_pred_list.append(retinanet_bbox_pred)
            retinanet_cls_score_concat = mx.symbol.concat(*retinanet_cls_score_list, dim=2)
            retinanet_bbox_pred_concat = mx.symbol.concat(*retinanet_bbox_pred_list, dim=1)
            logging.info('retinanet_cls_score_concat{}: {}'.format(suffix, retinanet_cls_score_concat.infer_shape(**input_dict)[1]))
            logging.info('retinanet_bbox_pred_concat{}: {}'.format(suffix, retinanet_bbox_pred_concat.infer_shape(**input_dict)[1]))
            return [rois] + self.get_retinanet_train(retinanet_cls_score_concat, retinanet_bbox_pred_concat, self.config, suffix=suffix)
        else:
            return rois

    def get_retinanet_train(self, retinanet_cls_score, retinanet_bbox_pred, config, suffix=''):
        retinanet_label = mx.sym.Variable(name='retinanet_label%s' % suffix)
        retinanet_bbox_target = mx.sym.Variable(name='retinanet_bbox_target%s' % suffix)
        retinanet_bbox_weight = mx.sym.Variable(name='retinanet_bbox_weight%s' % suffix)
        grad_scales = get_retinanet_grad_scales(config)
        # cls
        group_list = []
        if config.TRAIN.retinanet_cls_loss_type == 'focal_loss_cross_entropy':
            retinanet_cls_prob, retinanet_cls_loss = mx.sym.SigmoidFocalLoss(data=retinanet_cls_score,
                                                                             label=retinanet_label,
                                                                             ignore_label=-1,
                                                                             alpha=0.25,
                                                                             gamma=2.0,
                                                                             grad_scale=grad_scales[0])
            group_list.extend([mx.sym.BlockGrad(retinanet_label), retinanet_cls_prob, retinanet_cls_loss])
        elif config.TRAIN.retinanet_cls_loss_type == 'softmax':
            retinanet_cls_prob = sym.softmax_out(data=retinanet_cls_score,
                                                 label=retinanet_label,
                                                 ignore_label=-1,
                                                 multi_output=True,
                                                 grad_scale=grad_scales[0])
            group_list.extend([mx.sym.BlockGrad(retinanet_label), retinanet_cls_prob])
        else:
            raise ValueError("unknown retinanet cls loss type {}".format(config.TRAIN.retinanet_cls_loss_type))
        # bbox
        retinanet_bbox_loss_t = retinanet_bbox_weight * mx.sym.smooth_l1(data=(retinanet_bbox_pred - retinanet_bbox_target), scalar=3.0)
        retinanet_bbox_loss = mx.sym.MakeLoss(data=retinanet_bbox_loss_t, grad_scale=grad_scales[1])
        group_list.append(retinanet_bbox_loss)
        return group_list

    def get_proposal_rois(self, data, im_info, num_classes, name, config, is_train, **kwargs):
        if is_train:
            nms_pre_output_bbox_num = config.TRAIN.retinanet_pre_nms_top_n
            nms_post_output_bbox_num = config.TRAIN.retinanet_post_nms_top_n
            nms_threshold = config.TRAIN.retinanet_nms_thresh
            score_threshold = -10000
            batch_size = config.TRAIN.image_batch_size
        else:
            nms_pre_output_bbox_num = config.TEST.retinanet_pre_nms_top_n
            nms_post_output_bbox_num = config.TEST.retinanet_post_nms_top_n
            nms_threshold = config.TEST.retinanet_nms_thresh
            score_threshold = -math.log(1 / config.TEST.retinanet_score_thresh - 1)
            batch_size = config.TRAIN.image_batch_size
        logging.info('nms_pre_output_bbox_num: %d' % nms_pre_output_bbox_num)
        logging.info('nms_post_output_bbox_num: %d' % nms_post_output_bbox_num)

        rois = dpp(data=data,
                   im_info=im_info,
                   name=name,
                   feature_strides=config.network.retinanet_feat_strides,
                   ratios=config.network.retinanet_anchor_ratios,
                   scales=config.network.retinanet_anchor_scales,
                   output_score=True,
                   output_class=True,
                   nms_pre_output_bbox_num=nms_pre_output_bbox_num,
                   nms_post_output_bbox_num=nms_post_output_bbox_num,
                   nms_threshold=nms_threshold,
                   score_threshold=score_threshold,
                   num_classes=num_classes,
                   class_offset=1,
                   batch_size=batch_size,
                   **kwargs)
        return rois