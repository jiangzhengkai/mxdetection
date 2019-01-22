import numpy.random as npr
import numpy as np
from generate_anchor import generate_anchors, expand_anchors
from mxdetection.core.bbox.bbox_transform import bbox_overlaps, bbox_inner_overlaps, bbox_transform


def _unmap(data, count, inds, fill=0):
    """" unmap a subset inds of data into original data of size count """
    if count == len(inds):
        return data

    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret
def assign_anchor_fpn(feat_shape, gt_boxes, config, ignore_regions=None, feat_strides=[64, 32, 16, 8, 4],
                      scales=(8, 16, 32), ratios=(0.5, 1, 2), suffix=''):
    num_stride = len(feat_strides)
    anchors_list = []
    feat_infos = []
    A_list = []
    anchors_counter = []
    total_anchors = 0
    anchors_counter.append(0)
    for i in range(num_stride):
        scales_i = np.array(scales[i], dtype=np.float32)
        base_anchors = generate_anchors(base_size=feat_strides[i], ratios=list(ratios), scales=scales_i)
        feat_height, feat_width = feat_shape[i][-2:]
        all_anchors = expand_anchors(base_anchors, feat_height, feat_width, feat_strides[i])
        anchors_list.append(all_anchors)
        feat_infos.append([feat_height, feat_width])
        A_list.append(base_anchors.shape[0])
        total_anchors += all_anchors.shape[0]
        anchors_counter.append(total_anchors)
    anchors = np.concatenate(anchors_list)
    # assert total_anchors == len(anchors)
    inds_inside = range(total_anchors)

    # label: 1 is positive, 0 is negative, -1 is dont care
    labels = np.empty((len(inds_inside),), dtype=np.float32)
    labels.fill(-1)

    if gt_boxes.size > 0:
        # overlap between the anchors and the gt boxes
        # overlaps (ex, gt)
        overlaps = bbox_overlaps(anchors.astype(np.float), gt_boxes.astype(np.float))
        argmax_overlaps = overlaps.argmax(axis=1)
        max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
        gt_argmax_overlaps = overlaps.argmax(axis=0)
        gt_max_overlaps = overlaps[gt_argmax_overlaps, np.arange(overlaps.shape[1])]
        gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]
        # assign bg labels first so that positive labels can clobber them
        labels[max_overlaps < config.TRAIN.rpn_negative_overlap] = 0
        # fg label: for each gt, anchor with highest overlap
        labels[gt_argmax_overlaps] = 1
        # fg label: above threshold IoU
        labels[max_overlaps >= config.TRAIN.rpn_positive_overlap] = 1
    else:
        labels[:] = 0

    # subsample positive labels if we have too many
    num_fg = int(config.TRAIN.rpn_fg_fraction * config.TRAIN.rpn_batch_size)
    fg_inds = np.where(labels == 1)[0]
    if len(fg_inds) > num_fg:
        disable_inds = npr.choice(fg_inds, size=(len(fg_inds) - num_fg), replace=False)
        labels[disable_inds] = -1

    if config.TRAIN.rpn_do_ignore and ignore_regions is not None and len(ignore_regions) > 0:
        ignore_overlaps = bbox_inner_overlaps(anchors.astype(np.float), ignore_regions.astype(np.float))
        ignore_max_overlaps = ignore_overlaps.max(axis=1)
        fg_inds = np.where(labels == 1)[0]
        labels[ignore_max_overlaps >= config.TRAIN.rpn_ignore_overlap] = -1
        labels[fg_inds] = 1

    # subsample negative labels if we have too many
    num_bg = config.TRAIN.rpn_batch_size - np.sum(labels == 1)
    bg_inds = np.where(labels == 0)[0]
    if len(bg_inds) > num_bg:
        disable_inds = npr.choice(bg_inds, size=(len(bg_inds) - num_bg), replace=False)
        labels[disable_inds] = -1

    bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32)
    if gt_boxes.size > 0:
        bbox_targets[:] = bbox_transform(anchors, gt_boxes[argmax_overlaps, :4])

    bbox_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    bbox_weights[labels == 1, :] = 1.0

    # map up to original set of anchors
    labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
    bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
    bbox_weights = _unmap(bbox_weights, total_anchors, inds_inside, fill=0)

    # resahpe
    label_list = list()
    bbox_target_list = list()
    bbox_weight_list = list()
    for i in range(num_stride):
        feat_height, feat_width = feat_infos[i]
        A = A_list[i]
        label = labels[anchors_counter[i]:anchors_counter[i + 1]]
        bbox_target = bbox_targets[anchors_counter[i]:anchors_counter[i + 1]]
        bbox_weight = bbox_weights[anchors_counter[i]:anchors_counter[i + 1]]

        label = label.reshape((1, feat_height * feat_width, A)).transpose(0, 2, 1)
        label = label.reshape((1, -1))
        bbox_target = bbox_target.reshape((1, feat_height * feat_width, A * 4)).transpose(0, 2, 1)
        bbox_target = bbox_target.reshape((1, -1))
        bbox_weight = bbox_weight.reshape((1, feat_height * feat_width, A * 4)).transpose((0, 2, 1))
        bbox_weight = bbox_weight.reshape((1, -1))

        label_list.append(label)
        bbox_target_list.append(bbox_target)
        bbox_weight_list.append(bbox_weight)

    label_concat = np.concatenate(label_list, axis=1)
    bbox_target_concat = np.concatenate(bbox_target_list, axis=1)
    bbox_weight_concat = np.concatenate(bbox_weight_list, axis=1)

    label = {'rpn_label%s' % suffix: label_concat,
             'rpn_bbox_target%s' % suffix: bbox_target_concat,
             'rpn_bbox_weight%s' % suffix: bbox_weight_concat}
    return label
                          
                          

def assign_anchor(feat_shape, gt_boxes, im_info, config, ignore_regions=None, feat_stride=16,
                  scales=(8, 16, 32), ratios=(0.5, 1, 2), allowed_border=0, suffix=''):
    im_info = im_info[0]
    scales = np.array(scales, dtype=np.float32)
    base_anchors = generate_anchors(base_size=feat_stride, ratios=list(ratios), scales=scales)
    A = base_anchors.shape[0]
    feat_height, feat_width = feat_shape[-2:]

    all_anchors = expand_anchors(base_anchors, feat_height, feat_width, feat_stride)
    total_anchors = all_anchors.shape[0]

    # only keep anchors inside the image
    inds_inside = np.where((all_anchors[:, 0] >= -allowed_border) &
                           (all_anchors[:, 1] >= -allowed_border) &
                           (all_anchors[:, 2] < im_info[1] + allowed_border) &
                           (all_anchors[:, 3] < im_info[0] + allowed_border))[0]

    # keep only inside anchors
    anchors = all_anchors[inds_inside, :]

    # label: 1 is positive, 0 is negative, -1 is dont care
    labels = np.empty((len(inds_inside),), dtype=np.float32)
    labels.fill(-1)

    if gt_boxes.size > 0:
        # overlap between the anchors and the gt boxes
        # overlaps (ex, gt)
        overlaps = bbox_overlaps(anchors.astype(np.float), gt_boxes.astype(np.float))
        argmax_overlaps = overlaps.argmax(axis=1)
        max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
        gt_argmax_overlaps = overlaps.argmax(axis=0)
        gt_max_overlaps = overlaps[gt_argmax_overlaps, np.arange(overlaps.shape[1])]
        gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]
        # assign bg labels first so that positive labels can clobber them
        labels[max_overlaps < config.TRAIN.rpn_negative_overlap] = 0
        # fg label: for each gt, anchor with highest overlap
        labels[gt_argmax_overlaps] = 1
        # fg label: above threshold IoU
        labels[max_overlaps >= config.TRAIN.rpn_positive_overlap] = 1
    else:
        labels[:] = 0

    # subsample positive labels if we have too many
    num_fg = int(config.TRAIN.rpn_fg_fraction * config.TRAIN.rpn_batch_size)
    fg_inds = np.where(labels == 1)[0]
    if len(fg_inds) > num_fg:
        disable_inds = npr.choice(fg_inds, size=(len(fg_inds) - num_fg), replace=False)
        labels[disable_inds] = -1

    if config.TRAIN.rpn_do_ignore and ignore_regions is not None and ignore_regions.shape[0] > 0:
        ignore_overlaps = bbox_inner_overlaps(anchors.astype(np.float), ignore_regions.astype(np.float))
        ignore_max_overlaps = ignore_overlaps.max(axis=1)
        fg_inds = np.where(labels == 1)[0]
        labels[ignore_max_overlaps >= config.TRAIN.rpn_ignore_overlap] = -1
        labels[fg_inds] = 1

    # subsample negative labels if we have too many
    num_bg = config.TRAIN.rpn_batch_size - np.sum(labels == 1)
    bg_inds = np.where(labels == 0)[0]
    if len(bg_inds) > num_bg:
        disable_inds = npr.choice(bg_inds, size=(len(bg_inds) - num_bg), replace=False)
        labels[disable_inds] = -1

    bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32)
    if gt_boxes.size > 0:
        bbox_targets[:] = bbox_transform(anchors, gt_boxes[argmax_overlaps, :4])

    bbox_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    bbox_weights[labels == 1, :] = 1.0

    # map up to original set of anchors
    labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
    bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
    bbox_weights = _unmap(bbox_weights, total_anchors, inds_inside, fill=0)

    labels = labels.reshape((1, feat_height, feat_width, A)).transpose(0, 3, 1, 2)
    labels = labels.reshape((1, A * feat_height * feat_width))
    bbox_targets = bbox_targets.reshape((1, feat_height, feat_width, A * 4)).transpose(0, 3, 1, 2)
    bbox_weights = bbox_weights.reshape((1, feat_height, feat_width, A * 4)).transpose((0, 3, 1, 2))

    label = {'rpn_label%s' % suffix: labels,
             'rpn_bbox_target%s' % suffix: bbox_targets,
             'rpn_bbox_weight%s' % suffix: bbox_weights}
    return label

def assign_anchor_fpn_retinanet(feat_shape, gt_boxes, gt_classes, config, ignore_regions=None,
                                feat_strides=[8, 16, 32, 64, 128], scales=(8, 16, 32), ratios=(0.5, 1, 2), suffix=''):
    num_stride = len(feat_strides)
    anchors_list = []
    feat_infos = []
    A_list = []
    anchors_counter = []
    total_anchors = 0
    anchors_counter.append(0)
    for i in range(num_stride):
        scales_i = np.array(scales[i], dtype=np.float32)
        base_anchors = generate_anchors(base_size=feat_strides[i], ratios=list(ratios), scales=scales_i)
        feat_height, feat_width = feat_shape[i][-2:]
        all_anchors = expand_anchors(base_anchors, feat_height, feat_width, feat_strides[i])
        anchors_list.append(all_anchors)
        feat_infos.append([feat_height, feat_width])
        A_list.append(base_anchors.shape[0])
        total_anchors += all_anchors.shape[0]
        anchors_counter.append(total_anchors)
    anchors = np.concatenate(anchors_list)
    # assert total_anchors == len(anchors)
    inds_inside = range(total_anchors)

    # label: [1, oo) is positive, 0 is negative, -1 is dont care
    labels = np.empty((len(inds_inside),), dtype=np.float32)
    labels.fill(-1)

    if gt_boxes.size > 0:
        # overlap between the anchors and the gt boxes
        # overlaps (ex, gt)
        overlaps = bbox_overlaps(anchors.astype(np.float), gt_boxes.astype(np.float))
        argmax_overlaps = overlaps.argmax(axis=1)
        max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
        gt_argmax_overlaps = overlaps.argmax(axis=0)
        gt_max_overlaps = overlaps[gt_argmax_overlaps, np.arange(overlaps.shape[1])]
        gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]
        # assign bg labels first so that positive labels can clobber them
        labels[max_overlaps < config.TRAIN.retinanet_negative_overlap] = 0
        # fg label: for each gt, anchor with highest overlap
        gt_inds = argmax_overlaps[gt_argmax_overlaps]
        labels[gt_argmax_overlaps] = gt_classes[gt_inds]
        # fg label: above threshold IoU
        inds = max_overlaps >= config.TRAIN.retinanet_positive_overlap
        gt_inds = argmax_overlaps[inds]
        labels[inds] = gt_classes[gt_inds]
    else:
        labels[:] = 0
    num_fg = np.sum(labels >= 1)

    if config.TRAIN.retinanet_do_ignore and ignore_regions is not None and len(ignore_regions) > 0:
        ignore_overlaps = bbox_inner_overlaps(anchors.astype(np.float), ignore_regions.astype(np.float))
        ignore_max_overlaps = ignore_overlaps.max(axis=1)
        fg_inds = np.where(labels >= 1)[0]
        fg_classes = labels[fg_inds]
        labels[ignore_max_overlaps >= config.TRAIN.retinanet_ignore_overlap] = -1
        labels[fg_inds] = fg_classes

    if config.TRAIN.retinanet_bg_fg_ratio > 0:
        max_num_bg = num_fg * config.TRAIN.retinanet_bg_fg_ratio
        bg_inds = np.where(labels == 0)[0]
        if len(bg_inds) > max_num_bg:
            disable_inds = npr.choice(bg_inds, size=(len(bg_inds) - max_num_bg), replace=False)
            labels[disable_inds] = -1

    bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32)
    if gt_boxes.size > 0:
        bbox_targets[:] = bbox_transform(anchors, gt_boxes[argmax_overlaps, :4])

    bbox_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    if num_fg > 0:
        bbox_weights[labels >= 1, :] = 1.0 / num_fg

    # map up to original set of anchors
    labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
    bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
    bbox_weights = _unmap(bbox_weights, total_anchors, inds_inside, fill=0)

    # resahpe
    label_list = list()
    bbox_target_list = list()
    bbox_weight_list = list()
    for i in range(num_stride):
        feat_height, feat_width = feat_infos[i]
        A = A_list[i]
        label = labels[anchors_counter[i]:anchors_counter[i + 1]]
        bbox_target = bbox_targets[anchors_counter[i]:anchors_counter[i + 1]]
        bbox_weight = bbox_weights[anchors_counter[i]:anchors_counter[i + 1]]

        label = label.reshape((1, feat_height * feat_width, A)).transpose(0, 2, 1)
        label = label.reshape((1, -1))
        bbox_target = bbox_target.reshape((1, feat_height * feat_width, A * 4)).transpose(0, 2, 1)
        bbox_target = bbox_target.reshape((1, -1))
        bbox_weight = bbox_weight.reshape((1, feat_height * feat_width, A * 4)).transpose((0, 2, 1))
        bbox_weight = bbox_weight.reshape((1, -1))

        label_list.append(label)
        bbox_target_list.append(bbox_target)
        bbox_weight_list.append(bbox_weight)

    label_concat = np.concatenate(label_list, axis=1)
    bbox_target_concat = np.concatenate(bbox_target_list, axis=1)
    bbox_weight_concat = np.concatenate(bbox_weight_list, axis=1)

    label = {'retinanet_label%s' % suffix: label_concat,
             'retinanet_bbox_target%s' % suffix: bbox_target_concat,
             'retinanet_bbox_weight%s' % suffix: bbox_weight_concat}
    return label    
