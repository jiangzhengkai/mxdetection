def get_mask_grad_scales(config):
    grad_scale_0 = float(config.TRAIN.mask_loss_weights[0])
    return [grad_scale_0, ]

def get_mask_eval_info_list(config):
    grad_scales = get_mask_grad_scales(config)
    eval_info_list = []
    eval_info = dict()
    eval_info['metric_type'] = 'CrossEntropy'
    eval_info['metric_name'] = 'MaskLabel'
    eval_info['grad_scale'] = grad_scales[0]
    eval_info_list.append(eval_info)
    return eval_info_list