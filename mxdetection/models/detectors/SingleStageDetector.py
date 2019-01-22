import mxnet as mx
import logging
from mxdetection.models.utils.symbol_common import get_symbol_function

class SingleStageDetector(object):
    def __init__(self, config):
        self.config = config
        self.backbone = get_symbol_function(self.config.network.backbone)(config)
        self.neck = get_symbol_function(self.config.network.neck)(config)
        self.head = get_symbol_function(self.config.network.bbox_head)(config)

    def extract_feat(self, data, is_train, **kwargs):
        in_layer_list = self.backbone.get_symbol(data, is_train, **kwargs)
        rpn_conv_feat, rcnn_conv_feat = self.neck.get_symbol(in_layer_list, **kwargs)
        return rpn_conv_feat, rcnn_conv_feat

    def get_symbol(self, is_train=True):
        input_dict = {'data': (self.config.TRAIN.image_batch_size, 3, 800, 1280)}
        data = mx.symbol.Variable(name='data')
        im_info = mx.symbol.Variable(name='im_info')

        _, rcnn_conv_feat = self.extract_feat(data, is_train=is_train, input_dict=input_dict)
        group = self.head.get_symbol(conv_feat=rcnn_conv_feat, im_info=im_info, is_train=is_train,
                                     input_dict=input_dict)
        if isinstance(group, list):
            group = mx.symbol.Group(group[1:])
        logging.info('group: {}'.format(group.infer_shape(**input_dict)[1]))

        return group

    def get_train_symbol(self):
        return self.get_symbol(is_train=True)

    def get_test_symbol(self):
        return self.get_symbol(is_train=False)