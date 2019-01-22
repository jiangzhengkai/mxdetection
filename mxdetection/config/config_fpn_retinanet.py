from easydict import EasyDict as edict

def add_fpn_retinanet_params(config):
    if 'TRAIN' not in config:
        config.TRAIN = edict()
    if 'TEST' not in config:
        config.TEST = edict()
    if 'network' not in config:
        config.network = edict()
    if 'dataset' not in config:
        config.dataset = edict()
    config.network.dpp_style = 'python'
    config.network.roi_style = '3dim'
    config.network.image_stride = 32

    config.network.deformable_units = [0, 1, 1, 3]
    config.network.num_deformable_group = [0, 4, 4, 4]
    config.network.retinanet_feat_strides = [8, 16, 32, 64, 128]
    config.network.retinanet_anchor_ratios = (0.5, 1, 2)
    config.network.retinanet_anchor_scales = []
    anchor_scale_list = []
    for i in range(3):
        anchor_scale_list.append(4 * (2 ** (i / 3.0)))
    for _ in range(len(config.network.retinanet_feat_strides)):
        config.network.retinanet_anchor_scales.append(anchor_scale_list)
    config.network.retinanet_num_convs = 4
    config.network.retinanet_num_filter = 256
    config.network.retinanet_cls_bbox_kernel = 3
    config.network.retinanet_share_cls_bbox_tower = False
    config.network.neck_fpn_feat_strides = config.network.retinanet_feat_strides
    config.network.neck_fpn_feat_dim = config.network.retinanet_num_filter
    config.network.neck_fpn_aug = False

    config.network.retinanet_num_branch = 1

    config.TRAIN.retinanet_positive_overlap = 0.5
    config.TRAIN.retinanet_negative_overlap = 0.4
    config.TRAIN.retinanet_bg_fg_ratio = -1

    config.TRAIN.retinanet_ignore_overlap = 0.5
    config.TRAIN.retinanet_do_ignore = True

    config.TRAIN.retinanet_loss_weights = [1.0, 1.0]
    config.TRAIN.retinanet_cls_loss_type = 'focal_loss_cross_entropy'

    config.TRAIN.retinanet_pre_nms_top_n = 12000
    config.TRAIN.retinanet_post_nms_top_n = 2000
    config.TRAIN.retinanet_nms_thresh = 0.5

    config.TEST.retinanet_pre_nms_top_n = 6000
    config.TEST.retinanet_post_nms_top_n = 100
    config.TEST.retinanet_score_thresh = 0.05
    config.TEST.retinanet_nms_method = 'nms'
    config.TEST.retinanet_nms_thresh = 0.5
    config.TEST.retinanet_softnms_thresh = 0.6

    return config
