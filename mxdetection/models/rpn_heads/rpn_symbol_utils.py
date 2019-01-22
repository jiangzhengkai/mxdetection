def get_rpn_eval_info_list(config):
    eval_info_list = []
    grad_scales = get_rpn_grad_scales(config)
    # score
    if config.TRAIN.rpn_cls_loss_type == 'softmax':
        eval_info = dict()
        eval_info['metric_type'] = 'Softmax'
        eval_info['metric_name'] = 'RPNLabel'
        eval_info['grad_scale'] = grad_scales[0]
        eval_info['axis'] = 1
        eval_info['eval_fg'] = False
        eval_info_list.append(eval_info)
    elif config.TRAIN.rpn_cls_loss_type == 'cross_entropy':
        eval_info = dict()
        eval_info['metric_type'] = 'CrossEntropy'
        eval_info['metric_name'] = 'RPNLabel'
        eval_info['grad_scale'] = grad_scales[0]
        eval_info['eval_fg'] = False
        eval_info_list.append(eval_info)
    else:
        raise ValueError("unknown rpn cls loss type {}".format(config.TRAIN.rpn_cls_loss_type))
    # loss
    eval_info = dict()
    eval_info['metric_type'] = 'Sum'
    eval_info['metric_name'] = 'RPNBBox'
    eval_info['grad_scale'] = grad_scales[1]
    eval_info_list.append(eval_info)
    return eval_info_list


def get_rpn_grad_scales(config):
    grad_scale_0 = float(config.TRAIN.rpn_loss_weights[0])
    grad_scale_1 = float(config.TRAIN.rpn_loss_weights[1]) / (config.TRAIN.rpn_batch_size * config.TRAIN.image_batch_size)
    return [grad_scale_0, grad_scale_1]
