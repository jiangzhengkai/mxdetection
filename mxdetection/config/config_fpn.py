from easydict import EasyDict as edict

def add_fpn_params(config):
    if 'TRAIN' not in config:
        config.TRAIN = edict()
    if 'TEST' not in config:
        config.TEST = edict()
    if 'network' not in config:
        config.network = edict()
    if 'dataset' not in config:
        config.dataset = edict()

    config.network.roi_style = 'standard'
    config.network.roi_extract_method = 'roi_align_fpn'
    config.network.rcnn_post_processing_style = 'standard'
    config.network.image_stride = 32

    config.network.deformable_units = [0, 0, 0, 0]
    config.network.num_deformable_group = [0, 0, 0, 0]
    config.network.rpn_feat_stride = [4, 8, 16, 32, 64]
    config.network.rpn_anchor_scales = ((8,), (8,), (8,), (8,), (8,))
    config.network.rpn_anchor_ratios = (0.5, 1, 2)
    config.network.rpn_num_filter = 256
    config.network.rcnn_feat_stride = [4, 8, 16, 32]
    
    config.network.neck_fpn_feat_strides = config.network.rpn_feat_stride
    config.network.neck_fpn_feat_dim = config.network.rpn_num_filter
    config.network.neck_fpn_aug = False

    # RPN assign_anchor
    config.TRAIN.rpn_batch_size = 256
    config.TRAIN.rpn_fg_fraction = 0.5
    config.TRAIN.rpn_positive_overlap = 0.7
    config.TRAIN.rpn_negative_overlap = 0.3
    config.TRAIN.rpn_ignore_overlap = 0.5
    config.TRAIN.rpn_do_ignore = True
    config.TRAIN.rpn_cls_loss_type = 'softmax'

    # TRAIN RPN proposal
    config.TRAIN.rpn_nms_thresh = 0.7
    config.TRAIN.rpn_pre_nms_top_n = 12000
    config.TRAIN.rpn_post_nms_top_n = 2000
    config.TRAIN.rpn_min_size = [0, 0, 0, 0, 0]
    config.TRAIN.rpn_loss_weights = [1.0, 1.0]

    # TEST RPN proposal
    config.TEST.rpn_do_test = False
    config.TEST.rpn_nms_thresh = 0.7
    config.TEST.rpn_pre_nms_top_n = 6000
    config.TEST.rpn_post_nms_top_n = 1000
    config.TEST.rpn_min_size = [0, 0, 0, 0, 0]

    # R-CNN
    config.TRAIN.rcnn_batch_rois = 128 * len(config.network.rcnn_feat_stride)
    config.TRAIN.rcnn_loss_weights = [1.0, 2.0]
    config.TRAIN.rcnn_fg_fraction = 0.25
    config.TRAIN.rcnn_fg_thresh = 0.5
    config.TRAIN.rcnn_bg_thresh_hi = 0.5
    config.TRAIN.rcnn_bg_thresh_lo = 0.0
    config.TRAIN.rcnn_ignore_overlap = 0.5
    config.TRAIN.rcnn_do_ignore = True
    config.TRAIN.rcnn_enable_ohem = False

    config.network.rcnn_pooled_size = (7, 7)
    config.network.rcnn_bbox_normalization_precomputed = True
    config.network.rcnn_bbox_means = (0.0, 0.0, 0.0, 0.0)
    config.network.rcnn_bbox_stds = (0.1, 0.1, 0.2, 0.2)

    # Cascade R-CNN
    config.TRAIN.cascade_rcnn_loss_weights = [[1.0, 2.0], [0.5, 1.0], [0.25, 0.5]]
    config.TRAIN.cascade_rcnn_fg_thresh = [0.5, 0.6, 0.7]
    config.TRAIN.cascade_rcnn_bg_thresh_hi = [0.5, 0.6, 0.7]
    config.network.cascade_rcnn_bbox_means = [(0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0)]
    config.network.cascade_rcnn_bbox_stds = [(0.1, 0.1, 0.2, 0.2), (0.05, 0.05, 0.1, 0.1), (0.033, 0.033, 0.067, 0.067)]
    config.network.use_fpn_augmentation = False
    config.network.add_nonlocal = False
    config.network.rcnn_class_agnostic = False


    config.TEST.rcnn_nms_method = 'softnms'
    config.TEST.rcnn_nms = 0.3
    config.TEST.rcnn_softnms = 0.6
    config.TEST.rcnn_score_thresh = 1e-3

    return config
    
