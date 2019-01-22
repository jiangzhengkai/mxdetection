import mxnet as mx
import numpy as np


def add_focal_loss_name(config):
    config.pred_names.append('focal_loss_label_' + str(config.count))
    config.pred_names.append('focal_loss_pred_' + str(config.count))
    return config


class FocalLossAccMetric(mx.metric.EvalMetric):
    def __init__(self, config, metric_name='FocalLossAcc', axis=1,
                 type='cross_entropy', binary_thresh=0.5, eval_fg=False):
        super(FocalLossAccMetric, self).__init__(metric_name)
        self._axis = int(axis)
        assert type in ['cross_entropy', 'softmax']
        self._type = type
        self._binary_thresh = binary_thresh
        self._eval_fg = eval_fg
        self._pred_names = config.pred_names
        self._label_names = config.label_names
        self._suffix = str(config.count)

    def update(self, labels, preds):
        label_name = 'focal_loss_label_' + self._suffix
        pred_name = 'focal_loss_pred_' + self._suffix
        if label_name in self._pred_names:
            label = preds[self._pred_names.index(label_name)].asnumpy()
        else:
            label = labels[self._label_names.index(label_name)].asnumpy()
        pred = preds[self._pred_names.index(pred_name)].asnumpy()
        axis = self._axis if self._axis >= 0 else self._axis + pred.ndim
        assert axis < pred.ndim

        label = label.reshape((-1,)).astype(np.int32)
        if self._type == 'cross_entropy':
            if axis < pred.ndim - 1:
                leading = 1
                trailing = 1
                for i in range(axis):
                    leading *= pred.shape[i]
                for i in range(axis + 1, pred.ndim):
                    trailing *= pred.shape[i]
                pred = pred.reshape((leading, pred.shape[axis], trailing)).transpose((0, 2, 1))
            pred = pred.reshape((label.shape[0], -1))
            keep_inds = np.where(label > 0)[0] if self._eval_fg else np.where(label != -1)[0]
            label = label[keep_inds]
            pred = pred[keep_inds]
            pred_label = pred.argmax(axis=1).reshape((-1,)).astype(np.int32)
            pred_score = pred[np.arange(len(pred)), pred_label]
            pred_label += 1
            pred_label[pred_score < self._binary_thresh] = 0
            self.sum_metric += np.sum(label == pred_label)
            self.num_inst += len(keep_inds)
        else:
            pred = pred.argmax(axis=axis).reshape((-1,)).astype(np.int32)
            keep_inds = np.where(label > 0)[0] if self._eval_fg else np.where(label != -1)[0]
            label = label[keep_inds]
            pred = pred[keep_inds]
            self.sum_metric += np.sum(label == pred)
            self.num_inst += len(keep_inds)

