import numpy.random as npr
import numpy as np
from mxdetection.core.anchor.generate_anchor import generate_anchors, expand_anchors
from .bbox_transform import bbox_overlaps, bbox_inner_overlaps, bbox_transform, expand_bbox_regression_targets


def sample_rois(rois, fg_rois_per_image, rois_per_image, num_classes, config, gt_boxes, ignore_regions, need_extra_vars=False):
    """
    generate random sample of ROIs comprising foreground and background examples
    :param rois: all_rois [n, 4]; e2e: [n, 5] with batch_index
    :param fg_rois_per_image: foreground roi number
    :param rois_per_image: total roi number
    :param num_classes: number of classes
    :param gt_boxes: optional for e2e [n, 5] (x1, y1, x2, y2, cls)
    :return: (labels, rois, bbox_targets, bbox_weights)
    """
    if len(gt_boxes) > 0:
        overlaps = bbox_overlaps(rois[:, 1:].astype(np.float), gt_boxes[:, :4].astype(np.float))
        gt_assignment = overlaps.argmax(axis=1)
        overlaps = overlaps.max(axis=1)
        labels = gt_boxes[gt_assignment, 4]
    else:
        gt_assignment = np.zeros((rois.shape[0],), dtype=np.float32)
        overlaps = np.zeros((rois.shape[0],), dtype=np.float32)
        labels = np.zeros((rois.shape[0],), dtype=np.float32)
    
    # foreground RoI with FG_THRESH overlap
    rcnn_fg_thresh = config.TRAIN.rcnn_fg_thresh
    fg_indexes = np.where(overlaps >= rcnn_fg_thresh)[0]
    # guard against the case when an image has fewer than fg_rois_per_image foreground RoIs
    fg_rois_per_this_image = np.minimum(fg_rois_per_image, fg_indexes.size)
    # Sample foreground regions without replacement
    if len(fg_indexes) > fg_rois_per_this_image:
        fg_indexes = npr.choice(fg_indexes, size=fg_rois_per_this_image, replace=False)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    rcnn_bg_thresh_hi = config.TRAIN.rcnn_bg_thresh_hi
    bg_indexes = np.where((overlaps < rcnn_bg_thresh_hi) & (overlaps >= config.TRAIN.rcnn_bg_thresh_lo))[0]

    if config.TRAIN.rcnn_do_ignore and len(ignore_regions) > 0 and len(bg_indexes) > 0:
        ignore_overlaps = bbox_inner_overlaps(rois[bg_indexes, 1:].astype(np.float), ignore_regions[:, :4].astype(np.float))
        ignore_max_overlaps = ignore_overlaps.max(axis=1)
        bg_indexes = bg_indexes[ignore_max_overlaps < config.TRAIN.rcnn_ignore_overlap]

    # Compute number of background RoIs to take from this image (guarding against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = np.minimum(bg_rois_per_this_image, bg_indexes.size)
    # Sample foreground regions without replacement
    if len(bg_indexes) > bg_rois_per_this_image:
        bg_indexes = npr.choice(bg_indexes, size=bg_rois_per_this_image, replace=False)

    if len(bg_indexes) > 0:
        keep_indexes = np.append(fg_indexes, bg_indexes)
        while keep_indexes.shape[0] < rois_per_image:
            gap = np.minimum(len(bg_indexes), rois_per_image - keep_indexes.shape[0])
            gap_indexes = np.random.choice(bg_indexes, size=gap, replace=False)
            keep_indexes = np.append(keep_indexes, gap_indexes)
        labels = labels[keep_indexes]
        labels[fg_rois_per_this_image:] = 0
    else:
        keep_indexes = fg_indexes
        while keep_indexes.shape[0] < rois_per_image:
            ignore_indexes = list(range(len(rois)))
            gap = np.minimum(len(ignore_indexes), rois_per_image - keep_indexes.shape[0])
            gap_indexes = np.random.choice(ignore_indexes, size=gap, replace=False)
            keep_indexes = np.append(keep_indexes, gap_indexes)
        labels = labels[keep_indexes]
        labels[fg_rois_per_this_image:] = -1
    rois = rois[keep_indexes]
    overlaps = overlaps[keep_indexes]
    # load or compute bbox_target
    if len(gt_boxes) > 0:
        targets = bbox_transform(rois[:, 1:], gt_boxes[gt_assignment[keep_indexes], :4])
    else:
        targets = np.zeros((rois.shape[0], 4), dtype=np.float32)
    if config.network.rcnn_bbox_normalization_precomputed:
        targets = ((targets - np.array(config.network.rcnn_bbox_means)) / np.array(config.network.rcnn_bbox_stds))
 

    bbox_target_data = np.hstack((labels[:, np.newaxis], targets))
    bbox_targets, bbox_weights = expand_bbox_regression_targets(bbox_target_data, num_classes, config.network.rcnn_class_agnostic)

    if need_extra_vars:
        extra_vars = dict()
        extra_vars['fg_gt_indexes'] = gt_assignment[keep_indexes][:fg_rois_per_this_image]
        return rois, labels, bbox_targets, bbox_weights, extra_vars
    else:
        return rois, labels, bbox_targets, bbox_weights
