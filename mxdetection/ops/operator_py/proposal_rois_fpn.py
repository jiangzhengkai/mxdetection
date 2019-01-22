"""
Generate proposals for feature pyramid networks.
"""

import mxnet as mx
import numpy as np
import logging
import numpy.random as npr
from distutils.util import strtobool

from mxdetection.core.bbox.bbox_transform import bbox_pred, clip_boxes
from mxdetection.core.anchor.generate_anchor import generate_anchors, expand_anchors
from mxdetection.core.bbox.nms.nms import gpu_nms_wrapper


def generate_anchors_fpn(base_size, ratios, scales):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, 15, 15) window.
    """
    anchors = []
    for bs in base_size:
        anchors.append(generate_anchors(bs, ratios, scales))

    return anchors


class ProposalROISFPNOperator(mx.operator.CustomOp):
    def __init__(self, feat_stride_fpn, scales, ratios, output_score,
                 rpn_pre_nms_top_n, rpn_post_nms_top_n, threshold, rpn_min_size_fpn):
        super(ProposalROISFPNOperator, self).__init__()
        self._feat_stride_fpn = np.fromstring(feat_stride_fpn[1:-1], dtype=int, sep=',')
        self.fpn_keys = []
        fpn_stride = []
        for s in self._feat_stride_fpn:
            self.fpn_keys.append('stride%s' % s)
            fpn_stride.append(int(s))

        self.scales = np.fromstring(scales[1:-1], dtype=float, sep=',')
        self.ratios = np.fromstring(ratios[1:-1], dtype=float, sep=',')
        self.anchors_fpn = dict(zip(self.fpn_keys, generate_anchors_fpn(base_size=fpn_stride, scales=self.scales, ratios=self.ratios)))
        self.num_anchors = dict(zip(self.fpn_keys, [anchors.shape[0] for anchors in self._anchors_fpn.values()]))
        self.output_score = output_score
        self.rpn_pre_nms_top_n = rpn_pre_nms_top_n
        self.rpn_post_nms_top_n = rpn_post_nms_top_n
        self.threshold = threshold
        self.rpn_min_size_fpn = dict(zip(self.fpn_keys, np.fromstring(rpn_min_size_fpn[1:-1], dtype=int, sep=',')))


    def forward(self, is_train, req, in_data, out_data, aux):
        nms = gpu_nms_wrapper(self.threshold, in_data[0][0].context.device_id)

        cls_prob_dict = dict(zip(self.fpn_keys, in_data[0:len(self.fpn_keys)]))
        bbox_pred_dict = dict(zip(self.fpn_keys, in_data[len(self.fpn_keys):2 * len(self.fpn_keys)]))
        batch_size = in_data[0].shape[0]
        if batch_size > 1:
            raise ValueError("Sorry, multiple images each device is not implemented")

        pre_nms_topN = self.rpn_pre_nms_top_n
        post_nms_topN = self.rpn_post_nms_top_n
        min_size_dict = self.rpn_min_size_fpn

        im_info = in_data[-1].asnumpy()[0, :]
        proposals_list = []
        scores_list = []
        for s in self._feat_stride_fpn:
            stride = int(s)
            scores = cls_prob_dict['stride%s' % s].asnumpy()[:, self.num_anchors['stride%s' % s]:, :, :]
            bbox_deltas = bbox_pred_dict['stride%s' % s].asnumpy()

            height, width = int(im_info[0] / stride), int(im_info[1] / stride)
            anchors = expand_anchors(self.anchors_fpn['stride%s' % s].astype(np.float32), height, width, stride)

            bbox_deltas = self.clip_pad(bbox_deltas, (height, width))
            bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1)).reshape((-1, 4))

            scores = self.clip_pad(scores, (height, width))
            scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))

            proposals = bbox_pred(anchors, bbox_deltas)
            proposals = clip_boxes(proposals, im_info[:2])

            keep = self.filter_boxes(proposals, min_size_dict['stride%s' % s] * im_info[2])

            proposals = proposals[keep, :]
            scores = scores[keep]
            proposals_list.append(proposals)
            scores_list.append(scores)
        proposals = np.vstack(proposals_list)
        scores = np.vstack(scores_list)
        if proposals.shape[0] == 0:
            proposals = np.array([[0.0, 0.0, 15.0, 15.0]] * post_nms_topN, dtype=np.float32)
            scores = np.array([[0.9]] * post_nms_topN, dtype=np.float32)
        else:
            order = scores.ravel().argsort()[::-1]
            if pre_nms_topN > 0:
                order = order[:pre_nms_topN]
            proposals = proposals[order, :]
            scores = scores[order]

            det = np.hstack((proposals, scores)).astype(np.float32)
            keep = nms(det)
            if post_nms_topN > 0:
                keep = keep[:post_nms_topN]

            if len(keep) < post_nms_topN:
                pad = npr.choice(keep, size=post_nms_topN - len(keep))
                keep = np.hstack((keep, pad))
            proposals = proposals[keep, :]
            scores = scores[keep]

        batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
        blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))
        self.assign(out_data[0], req[0], blob)

        if self._output_score:
            self.assign(out_data[1], req[1], scores.astype(np.float32, copy=False))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        for i in range(len(in_grad)):
            self.assign(in_grad[i], req[i], 0)

    @staticmethod
    def filter_boxes(boxes, min_size):
        """ Remove all boxes with any side smaller than min_size """
        ws = boxes[:, 2] - boxes[:, 0] + 1
        hs = boxes[:, 3] - boxes[:, 1] + 1
        keep = np.where((ws >= min_size) & (hs >= min_size))[0]
        return keep

    @staticmethod
    def clip_pad(tensor, pad_shape):
        """
        Clip boxes of the pad area.
        :param tensor: [n, c, H, W]
        :param pad_shape: [h, w]
        :return: [n, c, h, w]
        """
        H, W = tensor.shape[2:]
        h, w = pad_shape

        if h < H or w < W:
            tensor = tensor[:, :, :h, :w].copy()

        return tensor


@mx.operator.register("proposal_rois_fpn")
class ProposalROISFPNProp(mx.operator.CustomOpProp):
    def __init__(self, feat_stride='(64,32,16,8,4)', scales='(8)', ratios='(0.5, 1, 2)', output_score='False',
                 rpn_pre_nms_top_n='6000', rpn_post_nms_top_n='300', threshold='0.3', rpn_min_size='(64,32,16,8,4)'):
        super(ProposalROISFPNProp, self).__init__(need_top_grad=False)
        self.feat_stride_fpn = feat_stride
        self.scales = scales
        self.ratios = ratios
        self.output_score = strtobool(output_score)
        self.rpn_pre_nms_top_n = int(rpn_pre_nms_top_n)
        self.rpn_post_nms_top_n = int(rpn_post_nms_top_n)
        self.threshold = float(threshold)
        self.rpn_min_size_fpn = rpn_min_size

    def list_arguments(self):
        args_list = []
        for s in np.fromstring(self.feat_stride_fpn[1:-1], dtype=int, sep=','):
            args_list.append('cls_prob_stride%s' % s)
        for s in np.fromstring(self.feat_stride_fpn[1:-1], dtype=int, sep=','):
            args_list.append('bbox_pred_stride%s' % s)
        args_list.append('im_info')

        return args_list

    def list_outputs(self):
        if self._output_score:
            return ['rois_output', 'score']
        else:
            return ['rois_output']

    def infer_shape(self, in_shape):
        batch_size = in_shape[0][0]

        in_shape[-1] = (batch_size, 3)
        rois_shape = (self.rpn_post_nms_top_n, 5)
        score_shape = (self.rpn_post_nms_top_n, 1)

        if self.output_score:
            return in_shape, [rois_shape, score_shape]
        else:
            return in_shape, [rois_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return ProposalROISFPNOperator(self.feat_stride_fpn, self.scales, self.ratios, self.output_score,
                                       self.rpn_pre_nms_top_n, self.rpn_post_nms_top_n, self.threshold,
                                       self.rpn_min_size_fpn)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []


def proposal_rois_fpn(im_info, config, is_train, **kwargs):
    if is_train:
        output_score = False
        rpn_pre_nms_top_n = config.TRAIN.rpn_pre_nms_top_n
        rpn_post_nms_top_n = config.TRAIN.rpn_post_nms_top_n
        rpn_min_size = config.TRAIN.rpn_min_size
        threshold = config.TRAIN.rpn_nms_thresh
    else:
        output_score = True
        rpn_pre_nms_top_n = config.TEST.rpn_pre_nms_top_n
        rpn_post_nms_top_n = config.TEST.rpn_post_nms_top_n
        rpn_min_size = config.TEST.rpn_min_size
        threshold = config.TEST.rpn_nms_thresh
    logging.info('rpn_pre_nms_top_n: %d' % rpn_pre_nms_top_n)
    logging.info('rpn_post_nms_top_n: %d' % rpn_post_nms_top_n)

    rpn_anchor_scales = list()
    for rpn_anchor_scale_i in config.network.rpn_anchor_scales:
        for rpn_anchor_scale_i_j in rpn_anchor_scale_i:
            rpn_anchor_scales.append(rpn_anchor_scale_i_j)
        rpn_anchor_scales.append(-1)
    del rpn_anchor_scales[-1]
    rois = mx.contrib.sym.ProposalFPN(im_info=im_info,
                                      feature_strides=tuple(config.network.rpn_feat_stride),
                                      scales=tuple(rpn_anchor_scales),
                                      ratios=tuple(config.network.rpn_anchor_ratios),
                                      output_score=output_score,
                                      rpn_pre_nms_top_n=rpn_pre_nms_top_n,
                                      rpn_post_nms_top_n=rpn_post_nms_top_n,
                                      rpn_min_size=tuple(rpn_min_size),
                                      threshold=threshold,
                                      **kwargs)
    # rois = mx.symbol.Custom(im_info=im_info,
    #                         op_type='proposal_rois_fpn',
    #                         feat_stride=config.network.rpn_feat_stride,
    #                         scales=config.network.rpn_anchor_scales,
    #                         ratios=config.network.rpn_anchor_ratios,
    #                         output_score=output_score,
    #                         rpn_pre_nms_top_n=rpn_pre_nms_top_n,
    #                         rpn_post_nms_top_n=rpn_post_nms_top_n,
    #                         rpn_min_size=rpn_min_size,
    #                         threshold=threshold,
    #                         **kwargs)
    return rois


from mxnet.test_utils import default_context, set_default_context, assert_almost_equal, check_numeric_gradient
def test_proposal_rois_fpn():
    stride_shapes = {'stride64': (10, 16), 'stride32': (19, 32), 'stride16': (38, 64),
                     'stride8': (76, 128), 'stride4': (152, 256)}
    in_shapes = []
    for stride in (64, 32, 16, 8, 4):
        stride_shape = stride_shapes['stride%d' % stride]
        in_shapes.append((1, 6, stride_shape[0], stride_shape[1]))
    for stride in (64, 32, 16, 8, 4):
        stride_shape = stride_shapes['stride%d' % stride]
        in_shapes.append((1, 12, stride_shape[0], stride_shape[1]))
    in_shapes.append((1, 3))

    prop = ProposalROISFPNProp()

    _, output_shapes = prop.infer_shape(in_shapes)

    in_data = []
    for in_shape in in_shapes:
        in_data.append(mx.random.uniform(0, 1, in_shape, ctx=mx.cpu()).copyto(ctx))
    in_data[-1][0, 0] = 608
    in_data[-1][0, 1] = 1024
    in_data[-1][0, 2] = 1.0

    out_data = []
    req = []
    for output_shape in output_shapes:
        out_data.append(mx.nd.zeros(output_shape, ctx=ctx))
        req.append('write')

    op = prop.create_operator(None, None, None)
    op.forward(is_train=True, req=req, in_data=in_data, out_data=out_data, aux=[])

    for a_out_data in out_data:
        print(a_out_data.asnumpy())
        print('------------------------------------------------')

if __name__ == '__main__':
    ctx = mx.gpu(0)
    set_default_context(ctx)
    test_proposal_rois_fpn()
