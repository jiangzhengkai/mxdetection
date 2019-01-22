import mxnet as mx
import numpy as np


def add_cross_entropy_name(config):
    config.pred_names.append('cross_entropy_label_' + str(config.count))
    config.pred_names.append('cross_entropy_pred_' + str(config.count))
    return config


class CrossEntropyAccMetric(mx.metric.EvalMetric):
    def __init__(self, config, metric_name='CrossEntropyAcc', binary_thresh=0.5, eval_fg=False):
        super(CrossEntropyAccMetric, self).__init__(metric_name)
        self._binary_thresh = float(binary_thresh)
        self._eval_fg = eval_fg
        self._pred_names = config.pred_names
        self._label_names = config.label_names
        self._suffix = str(config.count)

    def update(self, labels, preds):
        label_name = 'cross_entropy_label_' + self._suffix
        if label_name in self._pred_names:
            label = preds[self._pred_names.index(label_name)].asnumpy()
        else:
            label = labels[self._label_names.index(label_name)].asnumpy()
        pred = preds[self._pred_names.index('cross_entropy_pred_' + self._suffix)].asnumpy()
        label = label.reshape((-1,)).astype(np.int32)
        pred = pred.reshape((-1,))
        keep_inds = np.where(label > 0)[0] if self._eval_fg else np.where(label != -1)[0]
        label = label[keep_inds]
        pred = pred[keep_inds]
        pred = (pred >= self._binary_thresh).astype(np.int32)
        self.sum_metric += np.sum(label == pred)
        self.num_inst += len(keep_inds)


class CrossEntropyLossMetric(mx.metric.EvalMetric):
    def __init__(self, config, metric_name='CrossEntropyLoss', grad_scale=1.0):
        super(CrossEntropyLossMetric, self).__init__(metric_name)
        self._grad_scale = float(grad_scale)
        self._pred_names = config.pred_names
        self._label_names = config.label_names
        self._suffix = str(config.count)

    def update(self, labels, preds):
        label_name = 'cross_entropy_label_' + self._suffix
        if label_name in self._pred_names:
            label = preds[self._pred_names.index(label_name)].asnumpy()
        else:
            label = labels[self._label_names.index(label_name)].asnumpy()
        pred = preds[self._pred_names.index('cross_entropy_pred_' + self._suffix)].asnumpy()
        label = label.reshape((-1,))
        pred = pred.reshape((-1,))
        keep_inds = np.where(label != -1)[0]
        label = label[keep_inds]
        pred = pred[keep_inds] + 1e-12
        loss = -(label * np.log(pred) + (1.0 - label) * np.log(1.0 - pred))
        self.sum_metric += self._grad_scale * np.sum(loss)
        self.num_inst += len(keep_inds)
