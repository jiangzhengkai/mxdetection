import mxnet as mx
import logging
from mxdetection.ops.operator_py.proposal_rois_fpn import proposal_rois_fpn
from mxdetection.ops.operator_py.proposal_rois import proposal_rois
from rpn_symbol_utils import get_rpn_grad_scales

class RpnHead(object):
    def __init__(self, config):
        self.config = config

    def get_symbol(self, conv_feat, is_train, **kwargs):
        if 'frcnn' in self.config.network.task_type:
            return self.get_frcnn_rpn_net(conv_feat, is_train=is_train, **kwargs)
        elif 'fpn' in self.config.network.task_type:
            return self.get_fpn_rpn_net(conv_feat, is_train=is_train, **kwargs)

    def get_frcnn_rpn_net(self, conv_feat, is_train, suffix='', **kwargs):
        assert self.config.TRAIN.rpn_cls_loss_type == 'softmax'
        input_dict = kwargs['input_dict']
        num_anchors = len(self.config.network.rpn_anchor_scales) * len(self.config.network.rpn_anchor_ratios)
        rpn_conv = mx.sym.Convolution(data=conv_feat, name='rpn_conv%s' % suffix, num_filter=self.config.network.rpn_num_filter,
                                      kernel=(3, 3), stride=(1, 1), pad=(0, 0))
        rpn_relu = mx.sym.relu(data=rpn_conv, name='rpn_relu%s' % suffix)
        rpn_cls_score = mx.sym.Convolution(data=rpn_relu, name='rpn_cls_score%s' % suffix, num_filter=2 * num_anchors,
                                           kernel=(1, 1), stride=(1, 1), pad=(0, 0),  )
        rpn_bbox_pred = mx.sym.Convolution(data=rpn_relu, name='rpn_bbox_pred%s' % suffix, num_filter=4 * num_anchors,
                                           kernel=(1, 1), stride=(1, 1), pad=(0, 0))
        logging.info('rpn_cls_score{}: {}'.format(suffix, rpn_cls_score.infer_shape(**input_dict)[1]))
        logging.info('rpn_bbox_pred{}: {}'.format(suffix, rpn_bbox_pred.infer_shape(**input_dict)[1]))

        rpn_cls_score = mx.sym.Reshape(data=rpn_cls_score, shape=(0, 2, -1, 0))
        rpn_cls_prob = mx.sym.SoftmaxActivation(data=rpn_cls_score, mode="channel")
        rpn_cls_prob_reshape = mx.sym.Reshape(data=rpn_cls_prob, shape=(0, 2 * num_anchors, -1, 0))

        im_info = kwargs['im_info'] if 'im_info' in kwargs else mx.symbol.Variable(name='im_info')
        rois = proposal_rois(cls_prob=rpn_cls_prob_reshape,
                             bbox_pred=rpn_bbox_pred,
                             im_info=im_info,
                             is_train=is_train,
                             config=self.config)
        logging.info('rois{}: {}'.format(suffix, rois.infer_shape(**input_dict)[1]))

        if is_train:
            return [rois] + self.get_rpn_train(rpn_cls_score, rpn_bbox_pred, suffix=suffix)
        else:
            return rois

    def get_fpn_rpn_net(self, conv_feat, is_train, suffix='', **kwargs):
        assert self.config.TRAIN.rpn_cls_loss_type == 'softmax'
        input_dict = kwargs['input_dict']

        rpn_cls_score_list = []
        rpn_bbox_pred_list = []
        rpn_cls_prob_dict = {}
        rpn_bbox_pred_dict = {}
        for i, stride in enumerate(self.config.network.rpn_feat_stride):
            num_anchors = len(self.config.network.rpn_anchor_ratios) * len(self.config.network.rpn_anchor_scales[i])

            rpn_conv = mx.sym.Convolution(data=conv_feat['stride%d' % stride],
                                          name='rpn_conv_stride%d%s' % (stride, suffix),
                                          num_filter=self.config.network.rpn_num_filter, kernel=(3, 3), stride=(1, 1),
                                          pad=(1, 1))
            rpn_relu = mx.sym.relu(data=rpn_conv, name='rpn_relu_stride%d%s' % (stride, suffix))
            rpn_cls_score = mx.sym.Convolution(data=rpn_relu, name='rpn_cls_score_stride%d%s' % (stride, suffix),
                                               num_filter=2 * num_anchors,
                                               kernel=(1, 1), pad=(0, 0), stride=(1, 1))
            rpn_bbox_pred = mx.sym.Convolution(data=rpn_relu, name='rpn_bbox_pred_stride%d%s' % (stride, suffix),
                                               num_filter=4 * num_anchors,
                                               kernel=(1, 1), stride=(1, 1), pad=(0, 0))

            logging.info('rpn_conv_stride{}{}: {}'.format(stride, suffix, rpn_conv.infer_shape(**input_dict)[1]))
            logging.info('rpn_cls_score_stride{}{}: {}'.format(stride, suffix, rpn_cls_score.infer_shape(**input_dict)[1]))
            logging.info('rpn_bbox_pred_stride{}{}: {}'.format(stride, suffix, rpn_bbox_pred.infer_shape(**input_dict)[1]))

            rpn_cls_score_list.append(mx.symbol.Reshape(data=rpn_cls_score, shape=(0, 2, -1)))
            rpn_bbox_pred_list.append(mx.symbol.Reshape(data=rpn_bbox_pred, shape=(0, -1)))

            # do softmax
            rpn_cls_score_reshape = mx.symbol.Reshape(data=rpn_cls_score, shape=(0, 2, -1, 0))
            rpn_cls_prob = mx.symbol.SoftmaxActivation(data=rpn_cls_score_reshape, mode="channel")
            rpn_cls_prob_reshape = mx.symbol.Reshape(data=rpn_cls_prob, shape=(0, 2 * num_anchors, -1, 0))
            rpn_cls_prob_dict.update({'cls_prob_stride%d' % stride: rpn_cls_prob_reshape})
            rpn_bbox_pred_dict.update({'bbox_pred_stride%d' % stride: rpn_bbox_pred})

        args_dict = dict(rpn_cls_prob_dict.items() + rpn_bbox_pred_dict.items())
        im_info = kwargs['im_info'] if 'im_info' in kwargs else mx.symbol.Variable(name='im_info')
        rois = proposal_rois_fpn(im_info=im_info,
                                 config=self.config,
                                 is_train=is_train,
                                 **args_dict)
        logging.info('rois{}: {}'.format(suffix, rois.infer_shape(**input_dict)[1]))

        if is_train:
            rpn_cls_score_concat = mx.symbol.concat(*rpn_cls_score_list, dim=2)
            rpn_bbox_pred_concat = mx.symbol.concat(*rpn_bbox_pred_list, dim=1)
            logging.info('rpn_cls_score_concat{}: {}'.format(suffix, rpn_cls_score_concat.infer_shape(**input_dict)[1]))
            logging.info('rpn_bbox_pred_concat{}: {}'.format(suffix, rpn_bbox_pred_concat.infer_shape(**input_dict)[1]))
            return [rois] + self.get_rpn_train(rpn_cls_score_concat, rpn_bbox_pred_concat, suffix=suffix)
        else:
            return rois

    def get_rpn_train(self, rpn_cls_score, rpn_bbox_pred, suffix=''):
        rpn_label = mx.sym.Variable(name='rpn_label%s' % suffix)
        rpn_bbox_target = mx.sym.Variable(name='rpn_bbox_target%s' % suffix)
        rpn_bbox_weight = mx.sym.Variable(name='rpn_bbox_weight%s' % suffix)
        grad_scales = get_rpn_grad_scales(self.config)
        # classification
        if self.config.TRAIN.rpn_cls_loss_type == 'softmax':
            rpn_cls_prob = mx.sym.SoftmaxOutput(data=rpn_cls_score, label=rpn_label, name=None,
                                                grad_scale=grad_scales[0],
                                                use_ignore=True,
                                                ignore_label=-1,
                                                multi_output=True,
                                                normalization='valid')
        elif self.config.TRAIN.rpn_cls_loss_type == 'cross_entropy':
            rpn_cls_prob = sigmoid_cross_entropy_loss(data=rpn_cls_score, label=rpn_label,
                                                      grad_scale=grad_scales[0],
                                                      use_ignore=True, ignore_label=-1)
        else:
            raise ValueError("unknown rpn cls loss type {}".format(self.config.TRAIN.rpn_cls_loss_type))
        # bounding box regression
        rpn_bbox_loss_t = rpn_bbox_weight * mx.sym.smooth_l1(data=(rpn_bbox_pred - rpn_bbox_target), scalar=3.0)
        rpn_bbox_loss = mx.sym.MakeLoss(data=rpn_bbox_loss_t, grad_scale=grad_scales[1])
        return [mx.sym.BlockGrad(rpn_label), rpn_cls_prob, rpn_bbox_loss]



