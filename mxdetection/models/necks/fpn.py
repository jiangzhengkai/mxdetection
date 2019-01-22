import mxnet as mx
import logging

class FPN(object):
    def __init__(self, config):
        self.config = config

    def get_symbol(self, in_layer_list, **kwargs):
        if 'frcnn' in self.config.network.task_type:
            return self.get_frcnn_conv_feat(in_layer_list, **kwargs)
        elif 'fpn' in self.config.network.task_type:
            return self.get_fpn_conv_feat(in_layer_list, **kwargs)
        else:
            raise NotImplementedError

    def get_frcnn_conv_feat(self, in_layer_list, **kwargs):
        res4 = in_layer_list[2]
        res5 = in_layer_list[-1]

        conv_new = mx.sym.Convolution(data=res5, num_filter=256, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                                      name='rcnn_conv_new')
        conv_new_relu = mx.sym.relu(data=conv_new, name='rcnn_conv_new_relu')
        if 'input_dict' in kwargs:
            input_dict = kwargs['input_dict']
            logging.info('rpn_conv_feat: {}'.format(res4.infer_shape(**input_dict)[1]))
            logging.info('rcnn_conv_feat: {}'.format(conv_new_relu.infer_shape(**input_dict)[1]))
        return res4, conv_new_relu

    def get_fpn_conv_feat(self, in_layer_list, **kwargs):
        assert self.config.network.image_stride == 32

        conv_feat = {'stride4': in_layer_list[0], 'stride8': in_layer_list[1],
                     'stride16': in_layer_list[2], 'stride32': in_layer_list[3]}
        all_conv_feat = self._get_fpn_feature(conv_feat)

        if 'input_dict' in kwargs:
            input_dict = kwargs['input_dict']
            for stride in [4, 8, 16, 32]:
                logging.info('conv_feat_stride{}: {}'.format(stride,conv_feat['stride%d' % stride].infer_shape(**input_dict)[1]))
            if 'rpn' in self.config.network.task_type:
                for stride in self.config.network.rpn_feat_stride:
                    logging.info('rpn_conv_feat_stride{}: {}'.format(stride, all_conv_feat['stride%d' % stride].infer_shape(**input_dict)[1]))
                for stride in self.config.network.rcnn_feat_stride:
                    logging.info('rcnn_conv_feat_stride{}: {}'.format(stride, all_conv_feat['stride%d' % stride].infer_shape(**input_dict)[1]))
            else:
                for stride in self.config.network.retinanet_feat_strides:
                    logging.info('rcnn_conv_feat_stride{}: {}'.format(stride, all_conv_feat['stride%d' % stride].infer_shape(**input_dict)[1]))

        return all_conv_feat, all_conv_feat

    def _get_fpn_feature(self, conv_feat, feature_dim=256):
        res5 = conv_feat['stride32']
        res4 = conv_feat['stride16']
        res3 = conv_feat['stride8']
        res2 = conv_feat['stride4']
        # lateral connection
        fpn_p5_1x1 = mx.sym.Convolution(data=res5, name='fpn_p5_1x1', num_filter=feature_dim, kernel=(1, 1), pad=(0, 0), stride=(1, 1))
        fpn_p4_1x1 = mx.sym.Convolution(data=res4, name='fpn_p4_1x1', num_filter=feature_dim, kernel=(1, 1), pad=(0, 0), stride=(1, 1))
        fpn_p3_1x1 = mx.sym.Convolution(data=res3, name='fpn_p3_1x1', num_filter=feature_dim, kernel=(1, 1), pad=(0, 0), stride=(1, 1))
        fpn_p2_1x1 = mx.sym.Convolution(data=res2, name='fpn_p2_1x1', num_filter=feature_dim, kernel=(1, 1), pad=(0, 0), stride=(1, 1))
        # top-down connection
        fpn_p5_upsample = mx.symbol.UpSampling(fpn_p5_1x1, scale=2, sample_type='nearest', name='fpn_p5_upsample')
        fpn_p4_plus = fpn_p5_upsample + fpn_p4_1x1
        fpn_p4_upsample = mx.symbol.UpSampling(fpn_p4_plus, scale=2, sample_type='nearest', name='fpn_p4_upsample')
        fpn_p3_plus = fpn_p4_upsample + fpn_p3_1x1
        fpn_p3_upsample = mx.symbol.UpSampling(fpn_p3_plus, scale=2, sample_type='nearest', name='fpn_p3_upsample')
        fpn_p2_plus = fpn_p3_upsample + fpn_p2_1x1
        # FPN feature
        fpn_p5 = mx.sym.Convolution(data=fpn_p5_1x1, name='fpn_p5', num_filter=feature_dim, kernel=(3, 3), stride=(1, 1), pad=(1, 1))
        fpn_p4 = mx.sym.Convolution(data=fpn_p4_plus, name='fpn_p4', num_filter=feature_dim, kernel=(3, 3), stride=(1, 1), pad=(1, 1))
        fpn_p3 = mx.sym.Convolution(data=fpn_p3_plus, name='fpn_p3', num_filter=feature_dim, kernel=(3, 3), stride=(1, 1), pad=(1, 1))
        fpn_p2 = mx.sym.Convolution(data=fpn_p2_plus, name='fpn_p2', num_filter=feature_dim, kernel=(3, 3), stride=(1, 1), pad=(1, 1))
        # extra p6
        fpn_p6 = mx.sym.Convolution(data=res5, name='fpn_p6', num_filter=feature_dim, kernel=(3, 3), stride=(2, 2), pad=(1, 1))
        fpn_p6_relu = mx.sym.relu(data=fpn_p6, name='fpn_p6_relu')
        fpn_p7 = mx.sym.Convolution(data=fpn_p6_relu, name='fpn_p7', num_filter=feature_dim, kernel=(3, 3),
                                    stride=(2, 2), pad=(1, 1))
        fpn_conv_feat = {'stride128': fpn_p7, 'stride64': fpn_p6, 'stride32': fpn_p5, 'stride16': fpn_p4,
                         'stride8': fpn_p3, 'stride4': fpn_p2}
        return fpn_conv_feat






