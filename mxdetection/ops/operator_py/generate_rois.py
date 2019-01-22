import mxnet as mx
import numpy as np
from distutils.util import strtobool
from mxdetection.core.bbox.bbox_transform import clip_boxes


class GenerateROISOperator(mx.operator.CustomOp):
    def __init__(self, rescale_factor, jitter_center, aspect_ratio, compute_area, do_clip=False):
        super(GenerateROISOperator, self).__init__()
        self._rescale_factor = rescale_factor
        self._jitter_center = jitter_center
        self._aspect_ratio = aspect_ratio
        self._compute_area = compute_area
        self._do_clip = do_clip

    def forward(self, is_train, req, in_data, out_data, aux):
        rois = in_data[0].asnumpy()
        if self._compute_area:
            new_rois = np.zeros((rois.shape[0], 6), dtype=np.float32)
            new_rois[:, 5] = (rois[:, 3] - rois[:, 1] + 1) * (rois[:, 4] - rois[:, 2] + 1)
            new_rois[:, 1:5], _ = generate_new_rois(roi_boxes=rois[:, 1:],
                                                    roi_batch_size=rois.shape[0],
                                                    rescale_factor=self._rescale_factor,
                                                    jitter_center=self._jitter_center,
                                                    aspect_ratio=self._aspect_ratio)
            if self._do_clip:
                im_info = in_data[1].asnumpy()
                new_rois[:, 1:5] = clip_boxes(new_rois[:, 1:5], im_info[0, :2])
            self.assign(out_data[0], req[0], new_rois)
        else:
            rois[:, 1:], _ = generate_new_rois(roi_boxes=rois[:, 1:],
                                               roi_batch_size=rois.shape[0],
                                               rescale_factor=self._rescale_factor,
                                               jitter_center=self._jitter_center,
                                               aspect_ratio=self._aspect_ratio)
            if self._do_clip:
                im_info = in_data[1].asnumpy()
                rois[:, 1:] = clip_boxes(rois[:, 1:], im_info[0, :2])
            self.assign(out_data[0], req[0], rois)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        for i in range(len(in_grad)):
            self.assign(in_grad[i], req[i], 0)


@mx.operator.register('generate_rois')
class GenerateROISProp(mx.operator.CustomOpProp):
    def __init__(self, rescale_factor, jitter_center, aspect_ratio, compute_area, do_clip=False):
        super(GenerateROISProp, self).__init__(need_top_grad=False)
        self._rescale_factor = float(rescale_factor)
        self._jitter_center = strtobool(jitter_center)
        self._aspect_ratio = float(aspect_ratio)
        self._compute_area = strtobool(compute_area)
        self._do_clip = strtobool(do_clip)

    def list_arguments(self):
        return ['rois', 'im_info']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        rois_shape = in_shape[0]
        assert rois_shape[1] == 5
        if self._compute_area:
            out_shape = (rois_shape[0], 6)
        else:
            out_shape = (rois_shape[0], 5)
        return in_shape, [out_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return GenerateROISOperator(self._rescale_factor, self._jitter_center, self._aspect_ratio,
                                    self._compute_area, self._do_clip)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []

def generate_rois(rois, im_info, rescale_factor, jitter_center, aspect_ratio, compute_area=False, do_clip=False):
    group = mx.sym.Custom(rois=rois,
                          im_info=im_info,
                          op_type='generate_rois',
                          rescale_factor=rescale_factor,
                          jitter_center=jitter_center,
                          aspect_ratio=aspect_ratio,
                          compute_area=compute_area,
                          do_clip=do_clip)
    return group
