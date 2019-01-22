from mask_symbol_utils import get_mask_grad_scales
from mxdetection.ops.operator_py.sigmoid_cross_entropy_loss import sigmoid_cross_entropy_loss
import mxnet as mx
import logging


class MaskHead(object):
    def __init__(self, config):
        self.config = config

    def get_symbol(self, mask_rois_feat, is_train=True, **kwargs):
        return self.mask_net(mask_rois_feat, is_train=True, **kwargs)

    def mask_net(self, conv_feat, is_train=True, **kwargs):
        for i in range(4):
            conv_feat = mx.sym.Convolution(data=conv_feat, num_filter=256, kernel=(3, 3), stride=(1, 1), pad=(1, 1), name='mask_conv%d' % (i + 1))
            conv_feat = mx.sym.relu(data=conv_feat, name='mask_relu%d' % (i + 1))
        conv_feat = mx.sym.Deconvolution(data=conv_feat, name='mask_up_deconv', num_filter=256, kernel=(2, 2),
                                         stride=(2, 2), pad=(0, 0), adj=(0, 0), num_group=1, workspace=512, no_bias=False)
        conv_feat = mx.sym.relu(data=conv_feat, name='mask_up_relu')
        logging.info('mask_up_deconv: {}'.format(conv_feat.infer_shape(**kwargs['input_dict'])[1]))
        num_filter = 1 if self.config.dataset.num_classes == 2 else self.config.dataset.num_classes

        mask_label_pred = mx.sym.Convolution(data=conv_feat, num_filter=num_filter, kernel=(1, 1),
                                             pad=(0, 0), stride=(1, 1), no_bias=False, name='mask_label_pred')
        if 'input_dict' in kwargs:
            logging.info('mask_up_deconv: {}'.format(conv_feat.infer_shape(**kwargs['input_dict'])[1]))
            logging.info('mask_label_pred: {}'.format(mask_label_pred.infer_shape(**kwargs['input_dict'])[1]))

        return self.get_mask_train_test(mask_label_pred=mask_label_pred, is_train=is_train, **kwargs)

    def get_mask_train_test(self, mask_label_pred, is_train=True, mask_label=None, rcnn_label=None, **kwargs):
        num_classes = self.config.dataset.num_classes
        if is_train:
            if num_classes > 2:
                assert rcnn_label is not None
                rcnn_label_reshape = mx.sym.Reshape(data=rcnn_label, shape=(-1, 1, 1, 1))
                mask_label_pred = mx.contrib.sym.ChannelOperator(data=mask_label_pred, name='mask_label_pred_pick',
                                                                 pick_idx=rcnn_label_reshape,
                                                                 group=num_classes, op_type='Group_Pick',
                                                                 pick_type='Label_Pick')
                if 'input_dict' in kwargs:
                    input_dict = kwargs['input_dict']
                    logging.info('mask_label_pred: {}'.format(mask_label_pred.infer_shape(**input_dict)[1]))
            grad_scales = get_mask_grad_scales(self.config)
            mask_prob = sigmoid_cross_entropy_loss(data=mask_label_pred, label=mask_label,
                                                   use_ignore=True, ignore_label=-1,
                                                   grad_scale=grad_scales[0])
            mask_label = mx.sym.Reshape(data=mask_label, shape=(self.config.TRAIN.image_batch_size, -1, 0, 0))
            mask_prob = mx.sym.Reshape(data=mask_prob, shape=(self.config.TRAIN.image_batch_size, -1, 0, 0))
            return [mx.sym.BlockGrad(mask_label), mask_prob]
        else:
            mask_prob = mx.sym.sigmoid(data=mask_label_pred, name='mask_label_pred_sigmoid')
            return [mask_prob, ]
