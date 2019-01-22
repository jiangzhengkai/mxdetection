import mxnet as mx
import numpy as np

class SigmoidCrossEntropyLossOperator(mx.operator.CustomOp):
    def __init__(self, grad_scale=1.0, use_ignore=False, ignore_label=-1):
        super(SigmoidCrossEntropyLossOperator, self).__init__()
        self._grad_scale = float(grad_scale)
        self._use_ignore = use_ignore
        self._ignore_label = int(ignore_label)

    def forward(self, is_train, req, in_data, out_data, aux):
        self.assign(out_data[0], req[0], mx.nd.divide(1.0, (1.0 + mx.nd.exp(-in_data[0]))))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        if self._use_ignore:
            keep = in_data[1] != self._ignore_label
            num_output = np.sum(keep.asnumpy())
            if num_output == 0:
                self.assign(in_grad[0], req[0], 0.)
            else:
                self.assign(in_grad[0], req[0], (self._grad_scale/float(num_output))*keep*(out_data[0] - in_data[1]))
        else:
            num_output = out_data[0].size/out_data[0].shape[0]
            if num_output == 0:
                self.assign(in_grad[0], req[0], 0.)
            else:
                self.assign(in_grad[0], req[0], (self._grad_scale/float(num_output))*(out_data[0] - in_data[1]))


@mx.operator.register('SigmoidCrossEntropyLoss')
class SigmoidCrossEntropyLossProp(mx.operator.CustomOpProp):
    def __init__(self, grad_scale=1.0, use_ignore=False, ignore_label=-1):
        super(SigmoidCrossEntropyLossProp, self).__init__(need_top_grad=False)
        self._grad_scale = grad_scale
        self._use_ignore = use_ignore
        self._ignore_label = int(ignore_label)

    def list_arguments(self):
        return ['data', 'label']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        return [in_shape[0], in_shape[0]], [in_shape[0]]

    def create_operator(self, ctx, shapes, dtypes):
        return SigmoidCrossEntropyLossOperator(self._grad_scale, self._use_ignore, self._ignore_label)


def sigmoid_cross_entropy_loss(data, label, grad_scale=1.0, use_ignore=False, ignore_label=-1):
    return mx.sym.Custom(data=data,
                         label=label,
                         op_type='SigmoidCrossEntropyLoss',
                         grad_scale=grad_scale,
                         use_ignore=use_ignore,
                         ignore_label=ignore_label)

