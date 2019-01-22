def get_rcnn_eval_info_list(config):
    eval_info_list = []
    grad_scales = get_rcnn_grad_scales(config)
    # score
    eval_info = dict()
    eval_info['metric_type'] = 'Softmax'
    eval_info['metric_name'] = 'RCNNLabel'
    eval_info['grad_scale'] = grad_scales[0]
    eval_info['axis'] = -1
    eval_info['eval_fg'] = True
    eval_info_list.append(eval_info)
    # loss
    eval_info = dict()
    eval_info['metric_type'] = 'Sum'
    eval_info['metric_name'] = 'RCNNBBox'
    eval_info['grad_scale'] = grad_scales[1]
    eval_info_list.append(eval_info)
    return eval_info_list


def get_rcnn_grad_scales(config):
    grad_scale_0 = float(config.TRAIN.rcnn_loss_weights[0])
    grad_scale_1 = float(config.TRAIN.rcnn_loss_weights[1]) / (config.TRAIN.rcnn_batch_rois * config.TRAIN.image_batch_size)
    return [grad_scale_0, grad_scale_1]