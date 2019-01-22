import mxnet as mx
import numpy as np
from mxdetection.core.bbox.bbox_transform import bbox_pred, clip_boxes, filter_boxes
from mxdetection.core.anchor.generate_anchor import generate_anchors, expand_anchors
from mxdetection.core.bbox.nms.nms import nms

class DetectionPostProcessingOperator(mx.operator.CustomOp):
    def __init__(self, nms_pre_output_bbox_num, nms_post_output_bbox_num,
                 nms_threshold, score_threshold, num_classes, class_offset,
                 anchor_start_addr, num_anchors,
                 feature_strides, min_size_list):
        super(DetectionPostProcessingOperator, self).__init__()
        self._nms_pre_output_bbox_num = nms_pre_output_bbox_num
        self._nms_post_output_bbox_num = nms_post_output_bbox_num
        self._nms_threshold = nms_threshold
        self._score_threshold = score_threshold
        self._num_classes = num_classes
        self._class_offset = class_offset

        self._anchor_start_addr = anchor_start_addr
        self._num_anchors = num_anchors
        self._feature_strides = feature_strides
        self._min_size_list = min_size_list

    def forward(self, is_train, req, in_data, out_data, aux):
        num_branch = len(self._num_anchors)
        anchor_table = in_data[num_branch].asnumpy()
        im_info = in_data[num_branch + 1].asnumpy()
        batch_size = im_info.shape[0]

        outputs = np.full((batch_size, self._nms_post_output_bbox_num, 6), fill_value=-1, dtype=np.float32)

        bbox_deltas_list = []
        bbox_scores_list = []
        feat_shape_list = []
        for i in range(num_branch):
            bbox_data = in_data[i].asnumpy()
            feat_shape_list.append((bbox_data.shape[2], bbox_data.shape[3]))
            bbox_data = bbox_data.reshape((batch_size, self._num_anchors[i], 4 + self._num_classes,
                                           bbox_data.shape[2], bbox_data.shape[3]))
            bbox_data = bbox_data.transpose((0, 3, 4, 1, 2))
            bbox_data = bbox_data.reshape((batch_size, -1, 4 + self._num_classes))
            bbox_deltas = bbox_data[:, :, :4]
            bbox_scores = bbox_data[:, :, 4:]
            bbox_deltas_list.append(bbox_deltas)
            bbox_scores_list.append(bbox_scores)

        for n in range(batch_size):
            im_height = im_info[n, 0]
            im_width = im_info[n, 1]
            detections = []
            for i in range(num_branch):
                bbox_scores = bbox_scores_list[i][n]
                bbox_deltas = bbox_deltas_list[i][n]

                classes = bbox_scores.argmax(axis=1)
                max_scores = bbox_scores[np.arange(len(bbox_scores)), classes]

                anchor_start = int(self._anchor_start_addr[i])
                num_anchor = int(self._num_anchors[i])
                base_anchors = anchor_table[anchor_start:anchor_start+num_anchor, :]
                all_anchors = expand_anchors(base_anchors=base_anchors,
                                             feat_height=feat_shape_list[i][0],
                                             feat_width=feat_shape_list[i][1],
                                             feat_stride=self._feature_strides[i])
                pred_boxes = bbox_pred(all_anchors, bbox_deltas)

                if self._score_threshold > -10000:
                    candidate_inds = np.where(max_scores > self._score_threshold)[0]
                    if len(candidate_inds) == 0:
                        continue
                    classes = classes[candidate_inds]
                    max_scores = max_scores[candidate_inds]
                    pred_boxes = pred_boxes[candidate_inds]

                pred_boxes = clip_boxes(pred_boxes, (im_height, im_width))
                keep = filter_boxes(pred_boxes, self._min_size_list[i])
                if len(keep) > 0:
                    dets = np.zeros((pred_boxes.shape[0], 6))
                    dets[:, :4] = pred_boxes
                    dets[:, 4] = max_scores
                    dets[:, 5] = classes
                    detections.append(dets[keep])

            if len(detections) == 0:
                return np.zeros((0, 6), dtype=np.float32)
            # befort nms
            detections = np.vstack(detections)
            inds = np.argsort(-detections[:, 4])[:self._nms_pre_output_bbox_num]
            detections = detections[inds, :]
            # do nms
            new_detections = []
            for cls in range(self._num_classes):
                inds = np.where(detections[:, 5] == cls)[0]
                if len(inds) > 0:
                    dets = detections[inds, :]
                    keep = nms(dets[:, :5], self._nms_threshold)
                    if len(keep) > 0:
                        new_detections.append(dets[keep, :])
            detections = new_detections
            # after nms
            detections = np.vstack(detections)
            inds = np.argsort(-detections[:, 4])[:self._nms_post_output_bbox_num]
            detections = detections[inds, :]
            detections[:, -1] += self._class_offset
            outputs[n, :len(detections), :] = detections

        for ind, val in enumerate([outputs, ]):
            self.assign(out_data[ind], req[ind], val)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        for i in range(len(in_grad)):
            self.assign(in_grad[i], req[i], 0)

@mx.operator.register('detection_post_processing')
class DetectionPostProcessingProp(mx.operator.CustomOpProp):
    def __init__(self, nms_pre_output_bbox_num, nms_post_output_bbox_num,
                 nms_threshold, score_threshold, num_classes, class_offset,
                 anchor_start_addr, num_anchors,
                 feature_strides, min_size_list):
        super(DetectionPostProcessingProp, self).__init__(need_top_grad=False)
        self._nms_pre_output_bbox_num = int(nms_pre_output_bbox_num)
        self._nms_post_output_bbox_num = int(nms_post_output_bbox_num)
        self._nms_threshold = float(nms_threshold)
        self._score_threshold = float(score_threshold)
        self._num_classes = int(num_classes)
        self._class_offset = int(class_offset)

        self._anchor_start_addr = np.fromstring(anchor_start_addr[1:-1], dtype=np.int32, sep=',')
        self._num_anchors = np.fromstring(num_anchors[1:-1], dtype=np.int32, sep=',')
        self._feature_strides = np.fromstring(feature_strides[1:-1], dtype=np.int32, sep=',')
        self._min_size_list = np.fromstring(min_size_list[1:-1], dtype=np.int32, sep=',')

    def list_arguments(self):
        args = []
        for i in range(len(self._num_anchors)):
            args.append('data%d' % (i + 1))
        args.append('anchor_table')
        args.append('im_info')
        return args

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        num_branch = len(self._num_anchors)
        assert len(in_shape) == num_branch + 2
        for i in range(num_branch):
            assert in_shape[i][1] == self._num_anchors[i] * (4 + self._num_classes)
        assert in_shape[num_branch][0] == np.sum(self._num_anchors)
        assert in_shape[num_branch][1] == 4
        batch_size = in_shape[0][0]
        in_shape[num_branch + 1] = (batch_size, 3)
        output_shape = (batch_size, self._nms_post_output_bbox_num, 6)
        return in_shape, [output_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return DetectionPostProcessingOperator(nms_pre_output_bbox_num=self._nms_pre_output_bbox_num,
                                               nms_post_output_bbox_num=self._nms_post_output_bbox_num,
                                               nms_threshold=self._nms_threshold,
                                               score_threshold=self._score_threshold,
                                               num_classes=self._num_classes,
                                               class_offset=self._class_offset,
                                               anchor_start_addr=self._anchor_start_addr,
                                               num_anchors=self._num_anchors,
                                               feature_strides=self._feature_strides,
                                               min_size_list=self._min_size_list)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []
        
@mx.init.register
class AnchorTableFloatInitializer(mx.init.Initializer):
    def __init__(self, feature_strides, ratios, scales):
        super(AnchorTableFloatInitializer, self).__init__(feature_strides=feature_strides,
                                                          scales=scales,
                                                          ratios=ratios)
        self.scales = scales
        self.ratios = ratios
        self.feature_strides = feature_strides
    def _init_weight(self, _, arr):
        arr[:] = self.generate_anchor_table()

    def _init_default(self, name, arr):
        self._init_weight(name, arr)
    def _legacy_init(self, name, arr):
        self._init_weight(name, arr)
    def generate_anchor_table(self):
        anchors = []
        for i in range(len(self.feature_strides)):
            anchors.append(generate_anchors(self.feature_strides[i], self.ratios[i], self.scales[i]))
        return np.vstack(anchors)
        
    
def dpp_python(data, im_info, name, feature_strides, ratios, scales,
               nms_pre_output_bbox_num, nms_post_output_bbox_num,
               nms_threshold, score_threshold, num_classes, class_offset):
    assert name[:3] == 'dpp'
    anchor_table_name = 'dpp_anchor_table' + name[3:]

    num_input = len(data)
    in_data = dict()
    for i in range(num_input):
        in_data['data%d' % (i + 1)] = data[i]
    num_anchors = [len(ratios[i]) * len(scales[i]) for i in range(num_input)]
    anchor_table = mx.sym.var(anchor_table_name, shape=(sum(num_anchors), 4), dtype=np.float32,
                              init=AnchorTableFloatInitializer(feature_strides, ratios, scales))
    in_data['anchor_table'] = anchor_table
    in_data['im_info'] = im_info

    output = mx.sym.Custom(op_type='detection_post_processing',
                           name=name,
                           nms_pre_output_bbox_num=nms_pre_output_bbox_num,
                           nms_post_output_bbox_num=nms_post_output_bbox_num,
                           nms_threshold=nms_threshold,
                           score_threshold=score_threshold,
                           num_classes=num_classes,
                           class_offset=class_offset,
                           anchor_start_addr=list_pre_sum(num_anchors),
                           num_anchors=num_anchors,
                           feature_strides=feature_strides,
                           min_size_list=tuple([1, ] * num_input),
                           **in_data)
    return output
    
def dpp(data, im_info, name, feature_strides, ratios, scales, output_score, output_class,
        nms_pre_output_bbox_num, nms_post_output_bbox_num, nms_threshold, score_threshold,
        num_classes, class_offset=0, batch_size=1, **kwargs):
    

    data = as_list(data)
    feature_strides = as_list(feature_strides)
    ratios = as_2d_list(ratios)
    scales = as_2d_list(scales)
    if len(ratios) != len(scales):
        if len(ratios) == 1:
            ratios *= len(scales)
        else:
            assert False, "[dpp_x2] number of scales[{0}] does not match with number of ratios[{1}].".format(len(scales), len(ratios))

    num_input = len(data)
    assert num_input == len(feature_strides)
    assert num_input == len(ratios)
    assert num_input == len(scales)
    rois = dpp_python(data=data,
                      im_info=im_info,
                      name=name,
                      feature_strides=feature_strides,
                      ratios=ratios,
                      scales=scales,
                      nms_pre_output_bbox_num=nms_pre_output_bbox_num,
                      nms_post_output_bbox_num=nms_post_output_bbox_num,
                      nms_threshold=nms_threshold,
                      score_threshold=score_threshold,
                      num_classes=num_classes,
                      class_offset=class_offset)

    input_shift = kwargs['input_shift'] if 'input_shift' in kwargs else 5
    rois = rois_3dim_to_2dim(rois_3dim=rois,
                             output_score=output_score,
                             output_class=output_class,
                             nms_post_output_bbox_num=nms_post_output_bbox_num,
                             batch_size=batch_size,
                             input_shift=input_shift)
    return rois

def rois_3dim_to_2dim(rois_3dim, output_score, output_class,
                      nms_post_output_bbox_num, batch_size, input_shift):
    
    if is_predict_mode():
        rois_3dim = dpp_dequantize(rois_3dim, score_shift=input_shift)
    rois_3dim = mx.sym.reshape(rois_3dim, shape=(batch_size * nms_post_output_bbox_num, 6))

    bbox = mx.sym.slice_axis(data=rois_3dim, axis=1, begin=0, end=4)
    image_inds = []
    for i in range(batch_size):
        image_inds.append(mx.sym.full(shape=(nms_post_output_bbox_num, 1), val=i))
    image_inds = mx.sym.concat(*image_inds, dim=0)
    rois_2dim = mx.sym.concat(image_inds, bbox, dim=1)

    if output_score:
        cls_score = mx.sym.slice_axis(data=rois_3dim, axis=1, begin=4, end=5)
        if output_class:
            cls_index = mx.sym.slice_axis(data=rois_3dim, axis=1, begin=5, end=6)
            return mx.symbol.Group([rois_2dim, cls_score, cls_index])
        else:
            return mx.symbol.Group([rois_2dim, cls_score])
    else:
        return rois_2dim

def as_list(x):
    return x if isinstance(x, list) else [x, ]
    
def as_2d_list(x):
    if len(x) == 0:
        return x

    if isinstance(x[0], list) or isinstance(x[0], tuple):
        return x
    else:
        return [x, ]
def is_predict_mode():
    return mx.AttrScope.current.get(None).get("__predict__", "False") == str(True)
def list_pre_sum(lst):
    return [0, ] + [sum(lst[:(i + 1)]) for i in range(len(lst))][:-1]


