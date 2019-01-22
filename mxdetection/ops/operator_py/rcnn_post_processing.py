import mxnet as mx
import numpy as np
from mxdetection.core.bbox.bbox_transform import bbox_pred, clip_boxes
from mxdetection.core.bbox.nms.nms import py_nms_wrapper, py_softnms_wrapper
from distutils.util import strtobool

def rcnn_post_processing_standard(rois, bbox_score, bbox_deltas, im_info, name,
                                  nms_method, nms_threshold, score_threshold,
                                  rcnn_post_nms_top_n, num_classes,
                                  bbox_delta_std, bbox_delta_mean,
                                  batch_size=1):
    assert batch_size == 1
    output = mx.sym.Custom(rois=rois,
                           bbox_score=bbox_score,
                           bbox_deltas=bbox_deltas,
                           im_info=im_info,
                           op_type='rcnn_post_processing',
                           name=name,
                           nms_method=nms_method,
                           nms_threshold=nms_threshold,
                           score_threshold=score_threshold,
                           rcnn_post_nms_top_n=rcnn_post_nms_top_n,
                           num_classes=num_classes,
                           batch_size=batch_size,
                           bbox_delta_std=bbox_delta_std,
                           bbox_delta_mean=bbox_delta_mean)

    return output


class RcnnPostProcessingOperator(mx.operator.CustomOp):
    def __init__(self, nms_method, nms_threshold, score_threshold,
                 rcnn_post_nms_top_n, num_classes, batch_size,
                 bbox_delta_std, bbox_delta_mean):
        super(RcnnPostProcessingOperator, self).__init__()
        if nms_method == 'softnms':
            self.nms_func = py_softnms_wrapper(nms_threshold)
        else:
            assert nms_method == 'nms'
            self.nms_func = py_nms_wrapper(nms_threshold)
        self._score_threshold = score_threshold
        self._rcnn_post_nms_top_n = rcnn_post_nms_top_n
        self._num_classes = num_classes
        self._batch_size = batch_size
        self._bbox_delta_std = bbox_delta_std
        self._bbox_delta_mean = bbox_delta_mean


    def forward(self, is_train, req, in_data, out_data, aux):
        rois = in_data[0].asnumpy()
        bbox_score = np.squeeze(in_data[1].asnumpy())
        bbox_deltas = np.squeeze(in_data[2].asnumpy())
        im_info = in_data[3].asnumpy()
        output_rois = np.full((self._batch_size, self._rcnn_post_nms_top_n, 5), fill_value=-1, dtype=np.float32)
        output_rcnn = np.full((self._batch_size, self._rcnn_post_nms_top_n, 6), fill_value=-1, dtype=np.float32)

        bbox_deltas = bbox_deltas * self._bbox_delta_std + self._bbox_delta_mean
        pred_boxes = bbox_pred(rois[:, 1:], bbox_deltas)
        pred_boxes = pred_boxes.reshape((self._batch_size, -1, pred_boxes.shape[1]))
        bbox_score = bbox_score.reshape((self._batch_size, -1, bbox_score.shape[1]))

        for n in range(self._batch_size):
            pred_boxes_n = clip_boxes(pred_boxes[n, :, :], im_info[n, :2])
            bbox_score_n = bbox_score[n, :, :]
            all_boxes = [[] for _ in range(self._num_classes)]
            for j in range(1, self._num_classes):
                cls_boxes = pred_boxes_n[:, j * 4: (j+1) * 4]
                cls_scores = bbox_score_n[:, j, np.newaxis]
                keep = np.where(cls_scores > self._score_threshold)[0]
                cls_dets = np.hstack((cls_boxes, cls_scores)).astype(np.float32)[keep, :]
                keep = self.nms_func(cls_dets)
                all_boxes[j] = cls_dets[keep, :]

            image_scores = np.hstack([all_boxes[j][:, -1] for j in range(1, self._num_classes)])
            if len(image_scores) > self._rcnn_post_nms_top_n:
                image_thresh = np.sort(image_scores)[-self._rcnn_post_nms_top_n]
                for j in range(1, self._num_classes):
                    keep = np.where(all_boxes[j][:, -1] > image_thresh)[0]
                    all_boxes[j] = all_boxes[j][keep, :]

            start = 0
            for j in range(1, self._num_classes):
                if len(all_boxes[j]) > 0:
                    end = start + len(all_boxes[j])
                    output_rois[n, start:end, 0] = n
                    output_rois[n, start:end, 1:5] = all_boxes[j][:, :4]
                    output_rcnn[n, start:end, :5] = all_boxes[j]
                    output_rcnn[n, start:end, 5] = j
                    start = end

        output_rois = output_rois.reshape((-1, output_rois.shape[2]))
        for ind, val in enumerate([output_rois, output_rcnn]):
            self.assign(out_data[ind], req[ind], val)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        for i in range(len(in_grad)):
            self.assign(in_grad[i], req[i], 0)

@mx.operator.register('rcnn_post_processing')
class RcnnPostProcessingProp(mx.operator.CustomOpProp):
    def __init__(self, nms_method, nms_threshold, score_threshold, rcnn_post_nms_top_n,
                 num_classes, batch_size, bbox_delta_std, bbox_delta_mean):
        super(RcnnPostProcessingProp, self).__init__(need_top_grad=False)
        self._nms_method = nms_method
        self._nms_threshold = float(nms_threshold)
        self._score_threshold = float(score_threshold)
        self._rcnn_post_nms_top_n = int(rcnn_post_nms_top_n)
        self._num_classes = int(num_classes)
        self._batch_size = int(batch_size)
        self._bbox_delta_std = np.fromstring(bbox_delta_std[1:-1], dtype=float, sep=',')
        self._bbox_delta_mean = np.fromstring(bbox_delta_mean[1:-1], dtype=float, sep=',')
        self._bbox_delta_std = np.tile(self._bbox_delta_std, self._num_classes)
        self._bbox_delta_mean = np.tile(self._bbox_delta_mean, self._num_classes)

    def list_arguments(self):
        return ['rois', 'bbox_score', 'bbox_deltas', 'im_info']

    def list_outputs(self):
        return ['output_rois', 'output_rcnn']

    def infer_shape(self, in_shape):
        output_rois_shape = (self._batch_size * self._rcnn_post_nms_top_n, 5)
        output_rcnn_shape = (self._batch_size, self._rcnn_post_nms_top_n, 6)
        return in_shape, [output_rois_shape, output_rcnn_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return RcnnPostProcessingOperator(nms_method=self._nms_method,
                                          nms_threshold=self._nms_threshold,
                                          score_threshold=self._score_threshold,
                                          rcnn_post_nms_top_n=self._rcnn_post_nms_top_n,
                                          num_classes=self._num_classes,
                                          batch_size=self._batch_size,
                                          bbox_delta_std=self._bbox_delta_std,
                                          bbox_delta_mean=self._bbox_delta_mean)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []
