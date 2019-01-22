from mxdetection.models.utils.symbol_common import get_symbol_function
from mxdetection.models.utils.mask_labels import  get_mask_rcnn_labels
from mxdetection.ops.operator_py.rcnn_post_processing import rcnn_post_processing_standard
import mxnet as mx
import logging

class TwoStageDetector(object):
    def __init__(self, config):
        super(TwoStageDetector, self).__init__()
        # backbone
        self.backbone = get_symbol_function(config.network.backbone)(config)

        if config.network.neck is not None:
            self.neck = get_symbol_function(config.network.neck)(config)
        else:
            raise NotImplementedError

        if config.network.rpn_head is not None:
            self.rpn_head = get_symbol_function(config.network.rpn_head)(config)

        if config.network.bbox_head is not None:
            self.bbox_roi_extractor = get_symbol_function(config.network.bbox_roi_extractor)(config)
            self.bbox_head = get_symbol_function(config.network.bbox_head)(config)
        else:
            raise NotImplementedError

        if 'mask' in config.network.task_type:
            self.mask_roi_extractor = get_symbol_function(config.network.mask_roi_extractor)(config)
            self.mask_head = get_symbol_function(config.network.mask_head)(config)
        else:
            self.mask_head = None

        self.nms_threshold = config.TEST.rcnn_softnms if config.TEST.rcnn_nms_method == 'softnms' else config.TEST.rcnn_nms

        self.config = config

    def extract_feat(self, data, is_train, **kwargs):
        in_layer_list = self.backbone.get_symbol(data, is_train, **kwargs)
        rpn_conv_feat, rcnn_conv_feat = self.neck.get_symbol(in_layer_list, **kwargs)
        return rpn_conv_feat, rcnn_conv_feat


    def get_train_symbol(self):
        input_dict = {'data': (self.config.TRAIN.image_batch_size, 3, 800, 1280)}
        data = mx.symbol.Variable(name='data')
        im_info = mx.symbol.Variable(name='im_info')

        rpn_conv_feat, rcnn_conv_feat = self.extract_feat(data, is_train=True, input_dict=input_dict)

        logging.info('***************rpn subnet****************')
        rpn_group = self.rpn_head.get_symbol(conv_feat=rpn_conv_feat, is_train=True,
                                             im_info=im_info, input_dict=input_dict)

        logging.info('**************all labels*****************')
        multitask_labels = get_mask_rcnn_labels(rpn_group[0], self.config, is_train=True, input_dict=input_dict)

        logging.info('**************rcnn subnet****************')
        rcnn_labels = multitask_labels['rcnn_labels']
        rcnn_rois = rcnn_labels[0]
        logging.info('rcnn_rois: {}'.format(rcnn_rois.infer_shape(**input_dict)[1]))
        rcnn_rois_feat = self.bbox_roi_extractor.get_symbol(rcnn_conv_feat, rcnn_rois, name='bbox_extract_roi_feat',
                                                            input_dict=input_dict)
        logging.info('rcnn_rois_feat: {}'.format(rcnn_rois_feat.infer_shape(**input_dict)[1]))
        rcnn_group = self.bbox_head.get_symbol(roi_feats=rcnn_rois_feat, is_train=True, suffix='',
                                               rcnn_label=rcnn_labels[1], rcnn_bbox_target=rcnn_labels[2],
                                               rcnn_bbox_weight=rcnn_labels[3],
                                               input_dict=input_dict)
        mask_group = []
        if self.mask_head is not None:
            logging.info('**************mask subnet****************')
            mask_labels = multitask_labels['mask_labels']
            mask_rois = mask_labels[0]
            logging.info('mask_rois: {}'.format(mask_rois.infer_shape(**input_dict)[1]))
            mask_rois_feat = self.mask_roi_extractor.get_symbol(rcnn_conv_feat, mask_rois, name='mask_extract_roi_feat',
                                                                input_dict=input_dict)
            mask_group = self.mask_head.get_symbol(mask_rois_feat=mask_rois_feat, is_train=True,
                                                   mask_label=mask_labels[1], rcnn_label=rcnn_labels[1],
                                                   input_dict=input_dict)

        group = rpn_group[1:] + rcnn_group + mask_group
        group = mx.symbol.Group(group)
        logging.info('group: {}'.format(group.infer_shape(**input_dict)[1]))

        return group

    def get_test_symbol(self):
        """Test without augmentation."""
        input_dict = {'data': (1, 3, 800, 1280)}
        data = mx.symbol.Variable(name='data')
        im_info = mx.symbol.Variable(name='im_info')

        rpn_conv_feat, rcnn_conv_feat = self.extract_feat(data, is_train=False, input_dict=input_dict)

        logging.info('***************rpn subnet****************')
        rpn_group = self.rpn_head.get_symbol(conv_feat=rpn_conv_feat, is_train=False,
                                             im_info=im_info, input_dict=input_dict)

        logging.info('**************all labels*****************')
        multitask_labels = get_mask_rcnn_labels(rpn_group[0], self.config, is_train=False, input_dict=input_dict)

        logging.info('**************rcnn subnet****************')
        rcnn_labels = multitask_labels['rcnn_labels']
        rcnn_rois = rcnn_labels[0]
        logging.info('rcnn_rois: {}'.format(rcnn_rois.infer_shape(**input_dict)[1]))

        rcnn_rois_feat = self.bbox_roi_extractor.get_symbol(rcnn_conv_feat, rcnn_rois, name='bbox_extract_roi_feat',
                                                            input_dict=input_dict)
        rcnn_group = self.bbox_head.get_symbol(roi_feats=rcnn_rois_feat, is_train=False,
                                               rcnn_label=rcnn_labels[1], rcnn_bbox_target=rcnn_labels[2],
                                               rcnn_bbox_weight=rcnn_labels[3],
                                               input_dict=input_dict)
        rcnn_rois, rcnn_output = rcnn_post_processing_standard(rois=rcnn_rois,
                                                               bbox_score=rcnn_group[0],
                                                               bbox_deltas=rcnn_group[1],
                                                               im_info=im_info,
                                                               name='rpp',
                                                               nms_method=config.TEST.rcnn_nms_method,
                                                               nms_threshold=nms_threshold,
                                                               score_threshold=config.TEST.rcnn_score_thresh,
                                                               rcnn_post_nms_top_n=config.TEST.rpn_post_nms_top_n,
                                                               num_classes=config.dataset.num_classes,
                                                               bbox_delta_std=config.network.rcnn_bbox_stds,
                                                               bbox_delta_mean=config.network.rcnn_bbox_means,
                                                               batch_size=1)
        rcnn_group = [rcnn_output]

        mask_group = []
        if self.mask_head is not None:
            logging.info('**************mask subnet****************')
            mask_labels = multitask_labels['mask_labels']

            mask_rois = rcnn_rois

            logging.info('mask_rois: {}'.format(mask_rois.infer_shape(**input_dict)[1]))
            mask_rois_feat = self.mask_roi_extractor.get_symbol(rcnn_conv_feat, mask_rois, name='mask_extract_roi_feat',
                                                                input_dict=input_dict)
            mask_group = self.mask_head.get_symbol(mask_rois_feat=mask_rois_feat, is_train=False,
                                                   mask_label=mask_labels[1], rcnn_label=rcnn_labels[1],
                                                   input_dict=input_dict)
            mask_group = [mx.sym.identity(mask_rois)] + mask_group

        group = rcnn_group + mask_group
        group = mx.symbol.Group(group)
        logging.info('group: {}'.format(group.infer_shape(**input_dict)[1]))

        return group









