import numpy as np
import time
import mxnet as mx
import logging
import cv2
from mxdetection.core.bbox.bbox_transform import clip_boxes, bbox_pred

def e2e_mask_predict(outputs, local_vars, config):
    res_dict = dict()
    res_dict['det_results'] = []
    res_dict['mask_results'] = dict()
    if len(outputs) == 0:
        return res_dict

    pred = ['rcnn_output']
    rcnn_output = outputs[pred.index('rcnn_output')][0]
    keep = np.where(rcnn_output[:, -1] > -1)[0]
    rcnn_output = rcnn_output[keep, :]
    rcnn_output[:, :4] /= local_vars['im_scale']
    if not config.TEST.use_gt_rois:
        rcnn_output[:, :4] = clip_boxes(rcnn_output[:, :4], (local_vars['roi_rec']['height'], local_vars['roi_rec']['width']))
    else:
        rcnn_output[:, 4] = 1
    res_dict['det_results'] = rcnn_output
    local_vars['pred_boxes'] = rcnn_output[:, :4]
    local_vars['pred_scores'] = rcnn_output[:, 4, np.newaxis]
    local_vars['pred_cls'] = rcnn_output[:, 5].astype(np.int32)

    def _e2e_sptask_predict(begin_ind, end_ind, predict_func):
        if outputs[begin_ind].ndim == 3:
            rois = outputs[begin_ind][0, :, :4]
        else:
            rois = outputs[begin_ind][:, 1:5]
        rois = rois[keep, :]
        task_outputs = [rois, ]
        for i in range(begin_ind + 1, end_ind):
            task_outputs.append(outputs[i][keep, :])
        results = predict_func(task_outputs, local_vars, config)
        return results


    if 'mask' in config.network.task_type:
        mask_pred = ['mask_rois', 'masks']
        pred.extend(mask_pred)
        res_dict['mask_results'] = _e2e_sptask_predict(len(pred) - len(mask_pred), len(pred), mask_predict)


    return res_dict

def mask_predict(outputs, local_vars, config):
    pred = ['mask_rois', 'masks']
    mask_rois = outputs[pred.index('mask_rois')]
    masks = outputs[pred.index('masks')]

    mask_rois = mask_rois[:, 1:] if mask_rois.shape[1] == 5 else mask_rois
    mask_boxes = mask_rois / local_vars['im_scale']
    mask_boxes = clip_boxes(mask_boxes, (local_vars['roi_rec']['height'], local_vars['roi_rec']['width']))

    if masks.shape[1] == 1:
        masks = masks[:, 0, :, :]
    else:
        masks = masks[range(masks.shape[0]), local_vars['pred_cls'], :, :]
    mask_results = dict()
    mask_results['mask_boxes'] = np.hstack((mask_boxes, local_vars['pred_scores']))
    mask_results['masks'] = masks
    return mask_results