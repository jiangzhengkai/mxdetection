import numpy as np
import math
from mxdetection.core.bbox.bbox_transform import clip_boxes

def rpn_predict(outputs, local_vars, thresh=0):
    pred = ['rois', 'rois_score']
    pred_boxes = outputs[pred.index('rois')][:, 1:]
    scores = outputs[pred.index('rois_score')]
    pred_boxes = pred_boxes / local_vars['im_scale']
    proposals = np.hstack((pred_boxes, scores))
    keep = np.where(proposals[:, 4] > thresh)[0]
    proposals = proposals[keep, :]
    return proposals


def rcnn_predict(outputs, local_vars):
    pred = ['rcnn_output']
    rcnn_output = outputs[pred.index('rcnn_output')][0]
    keep = np.where(rcnn_output[:, -1] > -1)[0]
    rcnn_output = rcnn_output[keep, :]
    rcnn_output[:, :4] /= local_vars['im_scale']
    rcnn_output[:, :4] = clip_boxes(rcnn_output[:, :4], (local_vars['roi_rec']['height'], local_vars['roi_rec']['width']))
    return rcnn_output


def rpn_predict(outputs, local_vars, config):
    score_threshold = -math.log(1 / config.TEST.rpn_score_thresh - 1)
    return rpn_predict(outputs, local_vars, thresh=score_threshold)


def rcnn_predict(outputs, local_vars, config):
    return rcnn_predict(outputs, local_vars)

