from easydict import EasyDict as edict
from .metric_sum import *
from .metric_cross_entropy import *
from .metric_softmax import *
from .metric_focal_loss import *
import mxnet as mx


def get_eval_metrics(eval_info_list):
    config = edict()
    config.pred_names = []
    config.label_names = []
    config.count = 0
    eval_metrics = mx.metric.CompositeEvalMetric()
    for eval_info in eval_info_list:
        config.count += 1
        grad_scale = eval_info['grad_scale'] if 'grad_scale' in eval_info else 1.0
        if eval_info['metric_type'] == 'Sum':
            config = add_sum_name(config)
            eval_metrics.add(SumLossMetric(config, metric_name=eval_info['metric_name'] + 'Loss', grad_scale=grad_scale))

        elif eval_info['metric_type'] == 'SumWithWeight':
            config = add_sum_with_weight_name(config)
            eval_metrics.add(SumWithWeightLossMetric(config, metric_name=eval_info['metric_name'] + 'Loss', grad_scale=grad_scale))

        elif eval_info['metric_type'] == 'CrossEntropy':
            config = add_cross_entropy_name(config)
            binary_thresh = eval_info['binary_thresh'] if 'binary_thresh' in eval_info else 0.5
            eval_metrics.add(CrossEntropyAccMetric(config, metric_name=eval_info['metric_name'] + 'Acc', binary_thresh=binary_thresh))
            if 'eval_fg' in eval_info and eval_info['eval_fg']:
                eval_metrics.add(CrossEntropyAccMetric(config, metric_name=eval_info['metric_name'] + 'AccFG', binary_thresh=binary_thresh, eval_fg=True))
            eval_metrics.add(CrossEntropyLossMetric(config, metric_name=eval_info['metric_name'] + 'Loss', grad_scale=grad_scale))

        elif eval_info['metric_type'] == 'Softmax':
            config = add_softmax_name(config)
            axis = eval_info['axis'] if 'axis' in eval_info else 1
            eval_metrics.add(SoftmaxAccMetric(config, metric_name=eval_info['metric_name'] + 'Acc', axis=axis))
            if 'eval_fg' in eval_info and eval_info['eval_fg']:
                eval_metrics.add(SoftmaxAccMetric(config, metric_name=eval_info['metric_name'] + 'AccFG', axis=axis, eval_fg=True))
            eval_metrics.add(SoftmaxLossMetric(config, metric_name=eval_info['metric_name'] + 'Loss', axis=axis, grad_scale=grad_scale))

        elif eval_info['metric_type'] == 'FocalLoss':
            config = add_focal_loss_name(config)
            axis = eval_info['axis'] if 'axis' in eval_info else 1
            type = eval_info['type'] if 'type' in eval_info else 'cross_entropy'
            binary_thresh = eval_info['binary_thresh'] if 'binary_thresh' in eval_info else 0.5
            eval_metrics.add(FocalLossAccMetric(config, metric_name=eval_info['metric_name'] + 'Acc', axis=axis, type=type, binary_thresh=binary_thresh))
            if 'eval_fg' in eval_info and eval_info['eval_fg']:
                eval_metrics.add(FocalLossAccMetric(config, metric_name=eval_info['metric_name'] + 'AccFG', axis=axis, type=type, binary_thresh=binary_thresh, eval_fg=True))
            config = add_sum_name(config)
            eval_metrics.add(SumLossMetric(config, metric_name=eval_info['metric_name'] + 'Loss', grad_scale=grad_scale))
        else:
            raise ValueError("unknown eval name {}".format(eval_info[0]))

    return eval_metrics
