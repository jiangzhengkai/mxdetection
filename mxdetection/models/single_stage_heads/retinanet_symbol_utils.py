def get_retinanet_grad_scales(config):
    grad_scale_0 = float(config.TRAIN.retinanet_loss_weights[0])
    grad_scale_1 = float(config.TRAIN.retinanet_loss_weights[1]) / config.TRAIN.image_batch_size
    return [grad_scale_0, grad_scale_1]

def get_retinanet_eval_info_list(config):
    eval_info_list = []
    grad_scales = get_retinanet_grad_scales(config)
    suffix = ''
    # cls
    if config.TRAIN.retinanet_cls_loss_type == 'focal_loss_cross_entropy':
        eval_info = dict()
        eval_info['metric_type'] = 'FocalLoss'
        eval_info['metric_name'] = 'RetinaNetLabel%s' % suffix
        eval_info['grad_scale'] = grad_scales[0]
        eval_info['axis'] = 1
        eval_info['type'] = 'cross_entropy'
        eval_info['eval_fg'] = True
        eval_info_list.append(eval_info)
    elif config.TRAIN.retinanet_cls_loss_type == 'softmax':
        eval_info = dict()
        eval_info['metric_type'] = 'Softmax'
        eval_info['metric_name'] = 'RetinaNetLabel%s' % suffix
        eval_info['grad_scale'] = grad_scales[0]
        eval_info['axis'] = 1
        eval_info['eval_fg'] = True
        eval_info_list.append(eval_info)
    else:
        raise ValueError("unknown retinanet cls loss type {}".format(config.TRAIN.retinanet_cls_loss_type))
    # bbox
    eval_info = dict()
    eval_info['metric_type'] = 'Sum'
    eval_info['metric_name'] = 'RetinaNetBBox%s' % suffix
    eval_info['grad_scale'] = grad_scales[1]
    eval_info_list.append(eval_info)
    return eval_info_list