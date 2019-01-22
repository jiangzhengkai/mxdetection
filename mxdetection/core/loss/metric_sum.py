import mxnet as mx
import numpy as np


def add_sum_name(config):
    config.pred_names.append('sum_' + str(config.count))
    return config


def add_sum_with_weight_name(config):
    config.pred_names.append('sum_' + str(config.count))
    config.pred_names.append('sum_weight_' + str(config.count))
    return config


class SumLossMetric(mx.metric.EvalMetric):
    def __init__(self, config, metric_name='SumLoss', grad_scale=1.0):
        super(SumLossMetric, self).__init__(metric_name)
        self._grad_scale = float(grad_scale)
        self._pred_names = config.pred_names
        self._label_names = config.label_names
        self._suffix = str(config.count)

    def update(self, labels, preds):
        loss = preds[self._pred_names.index('sum_' + self._suffix)].asnumpy()
        self.sum_metric += self._grad_scale * np.sum(loss)
        self.num_inst += 1


class SumWithWeightLossMetric(mx.metric.EvalMetric):
    def __init__(self, config, metric_name='SumWithWeightLoss', grad_scale=1.0):
        super(SumWithWeightLossMetric, self).__init__(metric_name)
        self._grad_scale = float(grad_scale)
        self._pred_names = config.pred_names
        self._label_names = config.label_names
        self._suffix = str(config.count)

    def update(self, labels, preds):
        loss = preds[self._pred_names.index('sum_' + self._suffix)].asnumpy()
        weight = preds[self._pred_names.index('sum_weight_' + self._suffix)].asnumpy()
        weight_sum = np.sum(weight)
        self.sum_metric += self._grad_scale * np.sum(loss) / weight_sum if weight_sum > 0 else 0
        self.num_inst += 1


