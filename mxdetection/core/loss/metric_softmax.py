import mxnet as mx
import numpy as np


def add_softmax_name(config):
    config.pred_names.append('softmax_label_' + str(config.count))
    config.pred_names.append('softmax_pred_' + str(config.count))
    return config


class SoftmaxAccMetric(mx.metric.EvalMetric):
    def __init__(self, config, metric_name='SoftmaxAcc', axis=1, eval_fg=False):
        super(SoftmaxAccMetric, self).__init__(metric_name)
        self._axis = int(axis)
        self._eval_fg = eval_fg
        self._pred_names = config.pred_names
        self._label_names = config.label_names
        self._suffix = str(config.count)

    def update(self, labels, preds):
        label_name = 'softmax_label_' + self._suffix
        pred_name = 'softmax_pred_' + self._suffix
        if label_name in self._pred_names:
            label = preds[self._pred_names.index(label_name)].asnumpy()
        else:
            label = labels[self._label_names.index(label_name)].asnumpy()
        pred = preds[self._pred_names.index(pred_name)].asnumpy()
        axis = self._axis if self._axis >= 0 else self._axis + pred.ndim

        label = label.reshape((-1,)).astype(np.int32)
        pred = pred.argmax(axis=axis).reshape((-1,)).astype(np.int32)
        keep_inds = np.where(label > 0)[0] if self._eval_fg else np.where(label != -1)[0]
        label = label[keep_inds]
        pred = pred[keep_inds]
        self.sum_metric += np.sum(label == pred)
        self.num_inst += len(keep_inds)


class SoftmaxLossMetric(mx.metric.EvalMetric):
    def __init__(self, config, metric_name='SoftmaxLoss', axis=1, grad_scale=1.0):
        super(SoftmaxLossMetric, self).__init__(metric_name)
        self._axis = int(axis)
        self._grad_scale = float(grad_scale)
        self._pred_names = config.pred_names
        self._label_names = config.label_names
        self._suffix = str(config.count)

    def update(self, labels, preds):
        label_name = 'softmax_label_' + self._suffix
        pred_name = 'softmax_pred_' + self._suffix
        if label_name in self._pred_names:
            label = preds[self._pred_names.index(label_name)].asnumpy()
        else:
            label = labels[self._label_names.index(label_name)].asnumpy()
        pred = preds[self._pred_names.index(pred_name)].asnumpy()
        axis = self._axis if self._axis >= 0 else self._axis + pred.ndim

        label = label.reshape((-1,)).astype(np.int32)
        assert axis < pred.ndim
        if axis < pred.ndim - 1:
            leading = 1
            trailing = 1
            for i in range(axis):
                leading *= pred.shape[i]
            for i in range(axis + 1, pred.ndim):
                trailing *= pred.shape[i]
            pred = pred.reshape((leading, pred.shape[axis], trailing)).transpose((0, 2, 1))
        pred = pred.reshape((label.shape[0], -1))
        keep_inds = np.where(label != -1)[0]
        label = label[keep_inds]
        cls = pred[keep_inds, label]
        cls_loss = -1 * np.log(cls + 1e-12)
        self.sum_metric += self._grad_scale * np.sum(cls_loss)
        self.num_inst += len(keep_inds)

