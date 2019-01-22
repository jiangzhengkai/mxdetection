import mxnet as mx
import numpy as np


class BoxAnnotatorOHEMOperator(mx.operator.CustomOp):
    def __init__(self, roi_per_img, batch_size):
        super(BoxAnnotatorOHEMOperator, self).__init__()
        self._roi_per_img = roi_per_img
        self._batch_size = batch_size

    def forward(self, is_train, req, in_data, out_data, aux):
        rcnn_cls_prob = in_data[0].asnumpy()
        rcnn_bbox_loss = in_data[1].asnumpy()
        rcnn_label = in_data[2].asnumpy()
        rcnn_bbox_weight = in_data[3].asnumpy()

        rcnn_cls_loss = rcnn_cls_prob[np.arange(rcnn_cls_prob.shape[0], dtype='int'), rcnn_label.astype('int')]
        rcnn_cls_loss = -1 * np.log(rcnn_cls_loss)
        rcnn_bbox_loss = np.sum(rcnn_bbox_loss, axis=1)

        all_loss = rcnn_cls_loss + rcnn_bbox_loss
        all_loss = all_loss.reshape((self._batch_size, -1))

        top_k_loss = np.argsort(all_loss)
        top_k_loss = top_k_loss[:, :-self._roi_per_img]

        rcnn_label = rcnn_label.reshape((self._batch_size, -1))
        rcnn_bbox_weight = rcnn_bbox_weight.reshape((self._batch_size, -1, rcnn_bbox_weight.shape[1]))
        for i in range(self._batch_size):
            top_k_loss_i = top_k_loss[i, :]
            rcnn_label[i, top_k_loss_i] = -1
            rcnn_bbox_weight[i, top_k_loss_i] = 0

        for ind, val in enumerate([rcnn_label, rcnn_bbox_weight]):
            self.assign(out_data[ind], req[ind], val.reshape(out_data[ind].shape))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        for i in range(len(in_grad)):
            self.assign(in_grad[i], req[i], 0)

@mx.operator.register('BoxAnnotatorOHEM')
class BoxAnnotatorOHEMProp(mx.operator.CustomOpProp):
    def __init__(self, roi_per_img, batch_size):
        super(BoxAnnotatorOHEMProp, self).__init__(need_top_grad=False)
        self._roi_per_img = int(roi_per_img)
        self._batch_size = int(batch_size)

    def list_arguments(self):
        return ['rcnn_cls_prob', 'rcnn_bbox_loss', 'rcnn_label', 'rcnn_bbox_weight']

    def list_outputs(self):
        return ['rcnn_label_ohem', 'rcnn_bbox_weight_ohem']

    def infer_shape(self, in_shape):
        return in_shape, [in_shape[2], in_shape[3]]

    def create_operator(self, ctx, shapes, dtypes):
        return BoxAnnotatorOHEMOperator(self._roi_per_img, self._batch_size)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []

def box_annotator_ohem(rcnn_cls_prob, rcnn_bbox_loss, rcnn_label, rcnn_bbox_weight, roi_per_img, batch_size):
    labels_ohem, bbox_weights_ohem = mx.sym.Custom(rcnn_cls_prob=rcnn_cls_prob,
                                                   rcnn_bbox_loss=rcnn_bbox_loss,
                                                   rcnn_label=rcnn_label,
                                                   rcnn_bbox_weight=rcnn_bbox_weight,
                                                   op_type='BoxAnnotatorOHEM',
                                                   roi_per_img=roi_per_img,
                                                   batch_size=batch_size)
    return labels_ohem, bbox_weights_ohem

# for test
from mxnet.test_utils import default_context, set_default_context, assert_almost_equal, check_numeric_gradient
def test_box_annotator_ohem():
    ctx = default_context()

    batch_size = 4
    num_classes = 5
    pre_roi_per_img = 20
    post_roi_per_img = 12

    rcnn_cls_prob_shape = (batch_size * pre_roi_per_img, num_classes)
    rcnn_bbox_loss_shape = (batch_size * pre_roi_per_img, 4)
    rcnn_label_shape = (batch_size * pre_roi_per_img, )
    rcnn_bbox_weight_shape = (batch_size * pre_roi_per_img, 4)

    rcnn_cls_prob = mx.random.uniform(0, 1, rcnn_cls_prob_shape, ctx=mx.cpu()).copyto(ctx)
    rcnn_bbox_loss = mx.random.uniform(0, 1, rcnn_bbox_loss_shape, ctx=mx.cpu()).copyto(ctx)
    rcnn_label = mx.random.uniform(0, 1, rcnn_label_shape, ctx=mx.cpu()).copyto(ctx)
    rcnn_bbox_weight = mx.random.uniform(0, 1, rcnn_bbox_weight_shape, ctx=mx.cpu()).copyto(ctx)
    rcnn_label_np = np.random.uniform(0, 1, rcnn_label_shape) * (num_classes - 1)
    rcnn_label_np = rcnn_label_np.astype(np.int32)
    rcnn_label[:] = rcnn_label_np

    in_shape = [rcnn_cls_prob_shape, rcnn_bbox_loss_shape, rcnn_label_shape, rcnn_bbox_weight_shape]
    in_data = [rcnn_cls_prob, rcnn_bbox_loss, rcnn_label, rcnn_bbox_weight]

    ohem_prop = BoxAnnotatorOHEMProp(roi_per_img=post_roi_per_img, batch_size=batch_size)
    out_shape = ohem_prop.infer_shape(in_shape)[1]
    req = []
    out_data = []
    for i in range(len(out_shape)):
        req.append('write')
        out_data.append(mx.nd.zeros(out_shape[i], ctx=ctx))

    ohem_operator = BoxAnnotatorOHEMOperator(roi_per_img=post_roi_per_img, batch_size=batch_size)
    ohem_operator.forward(is_train=True, req=req, in_data=in_data, out_data=out_data, aux=[])


if __name__ == '__main__':
    ctx = mx.gpu(0)
    set_default_context(ctx)
    test_box_annotator_ohem()




