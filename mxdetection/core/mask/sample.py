from mxdetection.core.bbox.sample import sample_rois
from mxdetection.core.mask.mask_transform import polys_or_rles_to_masks
import numpy as np

def sample_mask_from_rpn(rois, labels, gt_polys_or_rles, config, extra_vars):
    fg_gt_indexes = extra_vars['fg_gt_indexes']

    # masks
    mask_height = config.network.mask_pooled_size[0]
    mask_width = config.network.mask_pooled_size[1]
    mask_roi_batch_size = config.TRAIN.mask_roi_batch_size
    mask_labels = -np.ones((mask_roi_batch_size, 1, mask_height, mask_width))
    mask_labels_pick = np.zeros((mask_roi_batch_size, ))
    mask_rois = np.array([[0, 0.0, 0.0, 15.0, 15.0]] * mask_roi_batch_size, dtype=np.float32)

    if len(gt_polys_or_rles) > 0 and len(fg_gt_indexes) > 0:
        keep = np.where(labels > 0)[0]
        if len(keep) > mask_roi_batch_size:
            keep = keep[:mask_roi_batch_size]
        num_fg_gt_indexes = len(keep)
        if num_fg_gt_indexes > 0:
            fg_gt_indexes = fg_gt_indexes[keep]
            fg_gt_polys_or_rles = [gt_polys_or_rles[_] for _ in fg_gt_indexes]
            fg_rois = rois[keep]
            mask_labels_pick[:num_fg_gt_indexes] = labels[keep]
            mask_labels[:num_fg_gt_indexes, 0, :, :] = polys_or_rles_to_masks(polys_or_rles=fg_gt_polys_or_rles,
                                                                              boxes=fg_rois[:, 1:],
                                                                              mask_height=mask_height,
                                                                              mask_width=mask_width)
            mask_rois[:num_fg_gt_indexes, :] = fg_rois

    return mask_rois, mask_labels


def sample_rois_mask(rois, fg_rois_per_image, rois_per_image, num_classes, config, gt_boxes, ignore_regions, gt_polys_or_rles=[]):
    bbox_rois, bbox_labels, bbox_targets, bbox_weights, extra_vars = sample_rois(rois=rois,
                                                                                  fg_rois_per_image=fg_rois_per_image,
                                                                                  rois_per_image=rois_per_image,
                                                                                  num_classes=num_classes,
                                                                                  config=config,
                                                                                  gt_boxes=gt_boxes,
                                                                                  ignore_regions=ignore_regions,
                                                                                  need_extra_vars=True)
    res_list = [bbox_rois, bbox_labels, bbox_targets, bbox_weights]
    if 'mask' in config.network.task_type:
        res_list.extend(sample_mask_from_rpn(bbox_rois, bbox_labels, gt_polys_or_rles, config, extra_vars))
    return res_list
