import numpy as np
import math 
from mxdetection.core.bbox.bbox_transform import clip_boxes


def retinanet_predict(outputs, local_vars, config):
    pred = ['retinanet_output']
    retinanet_output = outputs[pred.index('retinanet_output')][0]
    keep = np.where(retinanet_output[:, -1] > -1)[0]
    retinanet_output = retinanet_output[keep, :]
    retinanet_output[:, :4] /= local_vars['im_scale']
    retinanet_output[:, :4] = clip_boxes(retinanet_output[:, :4], (local_vars['roi_rec']['height'], local_vars['roi_rec']['width']))
    return retinanet_output


