import mxnet as mx
import numpy as np
import logging
from distutils.util import strtobool
from mxdetection.core.bbox.bbox_transform import bbox_pred, clip_boxes
from mxdetection.core.anchor.generate_anchor import generate_anchors
from mxdetection.core.bbox.nms.nms import gpu_nms_wrapper

class ProposalROISOperator(mx.operator.CustomOp):
    def __init__(self, feat_stride, scales, ratios, output_score,
                 rpn_pre_nms_top_n, rpn_post_nms_top_n, nms_threshold, rpn_min_size):
        super(ProposalROISOperator, self).__init__()
        self.feat_stride = feat_stride
        self.scales = np.fromstring(scales[1:-1], dtype=float, sep=',')
        self.ratios = np.fromstring(ratios[1:-1], dtype=float, sep=',')
        self.anchors = generate_anchors(base_size=self.feat_stride, scales=self.scales, ratios=self.ratios)
        self.num_anchors = self._anchors.shape[0]
        self.output_score = output_score
        self.rpn_pre_nms_top_n = rpn_pre_nms_top_n
        self.rpn_post_nms_top_n = rpn_post_nms_top_n
        self.nms_thresh = nms_threshold
        self.rpn_min_size = rpn_min_size

    def forward(self, is_train, req, in_data, out_data, aux):
        nms = gpu_nms_wrapper(self.nms_thresh, in_data[0].context.device_id)

        batch_size = in_data[0].shape[0]
        if batch_size > 1:
            raise ValueError('Sorry, multiple images each device is not implemented')

        # for each (H, W) location i
        #   generate A anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the A anchors
        # clip predicted boxes to image
        # remove predicted boxes with either height or width < threshold
        # sort all (proposal, score) pairs by score from highest to lowest
        # take top pre_nms_topN proposals before NMS
        # apply NMS with threshold 0.7 to remaining proposals
        # take after_nms_topN proposals after NMS
        # return the top proposals (-> RoIs top, scores top)

        pre_nms_top_n = self.rpn_pre_nms_top_n
        post_nms_top_n = self.rpn_post_nms_top_n
        min_size = self.rpn_min_size

        # the first set of anchors are background probabilities
        # keep the second part
        scores = in_data[0].asnumpy()[:, self.num_anchors:, :, :]
        bbox_deltas = in_data[1].asnumpy()
        im_info = in_data[2].asnumpy()[0, :]

        # 1. Generate proposals from bbox_deltas and shifted anchors
        # use real image size instead of padded feature map sizes
        height, width = int(im_info[0] / self.feat_stride), int(im_info[1] / self.feat_stride)

        # Enumerate all shifts
        shift_x = np.arange(0, width) * self._feat_stride
        shift_y = np.arange(0, height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()

        # Enumerate all shifted anchors:
        #
        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        A = self.num_anchors
        K = shifts.shape[0]
        anchors = self._anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))
        anchors = anchors.reshape((K * A, 4))

        # Transpose and reshape predicted bbox transformations to get them
        # into the same order as the anchors:
        #
        # bbox deltas will be (1, 4 * A, H, W) format
        # transpose to (1, H, W, 4 * A)
        # reshape to (1 * H * W * A, 4) where rows are ordered by (h, w, a)
        # in slowest to fastest order
        bbox_deltas = self.clip_pad(bbox_deltas, (height, width))
        bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1)).reshape((-1, 4))

        # Same story for the scores:
        #
        # scores are (1, A, H, W) format
        # transpose to (1, H, W, A)
        # reshape to (1 * H * W * A, 1) where rows are ordered by (h, w, a)
        scores = self._clip_pad(scores, (height, width))
        scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))

        # Convert anchors into proposals via bbox transformations
        proposals = bbox_pred(anchors, bbox_deltas)

        # 2. clip predicted boxes to image
        proposals = clip_boxes(proposals, im_info[:2])

        # 3. remove predicted boxes with either height or width < threshold
        # (NOTE: convert min_size to input image scale stored in im_info[2])
        keep = self.filter_boxes(proposals, min_size * im_info[2])
        proposals = proposals[keep, :]
        scores = scores[keep]

        # 4. sort all (proposal, score) pairs by score from highest to lowest
        # 5. take top pre_nms_topN (e.g. 6000)
        order = scores.ravel().argsort()[::-1]
        if pre_nms_top_n > 0:
            order = order[:pre_nms_top_n]
        proposals = proposals[order, :]
        scores = scores[order]

        # 6. apply nms (e.g. threshold = 0.7)
        # 7. take after_nms_topN (e.g. 300)
        # 8. return the top proposals (-> RoIs top)
        det = np.hstack((proposals, scores)).astype(np.float32)
        keep = nms(det)
        if post_nms_top_n > 0:
            keep = keep[:post_nms_top_n]
        # pad to ensure output size remains unchanged
        if len(keep) < post_nms_top_n:
            pad = np.random.choice(keep, size=post_nms_top_n - len(keep))
            keep = np.hstack((keep, pad))
        proposals = proposals[keep, :]
        scores = scores[keep]

        # Output rois array
        # Our RPN implementation only supports a single input image, so all
        # batch inds are 0
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


@mx.operator.register("proposal_rois")
class ProposalROISProp(mx.operator.CustomOpProp):
    def __init__(self, feat_stride='16', scales='(8, 16, 32)', ratios='(0.5, 1, 2)', output_score='False',
                 rpn_pre_nms_top_n='6000', rpn_post_nms_top_n='300', threshold='0.3', rpn_min_size='16'):
        super(ProposalROISProp, self).__init__(need_top_grad=False)
        self.feat_stride = int(feat_stride)
        self.scales = scales
        self.ratios = ratios
        self.output_score = strtobool(output_score)
        self.rpn_pre_nms_top_n = int(rpn_pre_nms_top_n)
        self.rpn_post_nms_top_n = int(rpn_post_nms_top_n)
        self.threshold = float(threshold)
        self.rpn_min_size = int(rpn_min_size)

    def list_arguments(self):
        return ['cls_prob', 'bbox_pred', 'im_info']

    def list_outputs(self):
        if self._output_score:
            return ['output', 'score']
        else:
            return ['output']

    def infer_shape(self, in_shape):
        cls_prob_shape = in_shape[0]
        bbox_pred_shape = in_shape[1]
        assert cls_prob_shape[0] == bbox_pred_shape[0], 'ROI number does not equal in cls and reg'

        batch_size = cls_prob_shape[0]
        im_info_shape = (batch_size, 3)
        output_shape = (self._rpn_post_nms_top_n, 5)
        score_shape = (self._rpn_post_nms_top_n, 1)

        if self._output_score:
            return [cls_prob_shape, bbox_pred_shape, im_info_shape], [output_shape, score_shape]
        else:
            return [cls_prob_shape, bbox_pred_shape, im_info_shape], [output_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return ProposalROISOperator(self._feat_stride, self._scales, self._ratios, self._output_score,
                                    self._rpn_pre_nms_top_n, self._rpn_post_nms_top_n, self._threshold, self._rpn_min_size)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []


def proposal_rois(cls_prob, bbox_pred, im_info, is_train, config):
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

    args_dict = dict()
    args_dict['cls_prob_stride%d' % config.network.rpn_feat_stride] = cls_prob
    args_dict['bbox_pred_stride%d' % config.network.rpn_feat_stride] = bbox_pred
    rois = mx.contrib.sym.ProposalFPN(im_info=im_info,
                                      feature_strides=tuple([config.network.rpn_feat_stride]),
                                      scales=tuple(config.network.rpn_anchor_scales),
                                      ratios=tuple(config.network.rpn_anchor_ratios),
                                      output_score=output_score,
                                      rpn_pre_nms_top_n=rpn_pre_nms_top_n,
                                      rpn_post_nms_top_n=rpn_post_nms_top_n,
                                      rpn_min_size=tuple([rpn_min_size]),
                                      threshold=threshold,
                                      **args_dict)
    return rois
