import numpy as np
import mxnet as mx
from mxnet.executor_manager import _split_input_slice
from mxdetection.core.anchor.assign_anchor import assign_anchor_fpn_retinanet
from mxdetection.core.processing.image import tensor_vstack, transform, get_image
from mxdetection.core.processing.image_aug_function import image_aug_function
from .base_iter import BaseIter, BaseTestIter
from .fpn_iter import FPNTestIter

from mxdetection.utils.utils import makedivstride

def get_retinanet_feat_sym(sym, config):
    retinanet_feat_sym = []
    suffix = ''
    if isinstance(config.network.retinanet_feat_strides, list):
        for stride in config.network.retinanet_feat_strides:
            retinanet_feat_sym.append(sym.get_internals()['retinanet_output_stride%d%s_output' % (stride, suffix)])
    else:
        assert False
    return retinanet_feat_sym
class RetinaNetIter(BaseIter):
    def __init__(self, roidb, config, batch_size, ctx=None):
        super(RetinaNetIter, self).__init__(roidb, config, batch_size, ctx)
        self.retinanet_feat_sym = get_retinanet_feat_sym(config.network.symbol_network, config)

        self.data_name = ['data']
        self.label_name = []
        suffix = ''
        self.label_name.extend(['retinanet_label%s' % suffix, 'retinanet_bbox_target%s' % suffix, 'retinanet_bbox_weight%s' % suffix])
        self.max_data_shape, self.max_label_shape = self.infer_max_shape()

        self.kwargs = dict()
        self.kwargs['flip'] = self.config.TRAIN.aug_strategy.flip
        self.kwargs['rotated_angle_range'] = self.config.TRAIN.aug_strategy.rotated_angle_range
        self.kwargs['scales'] = self.config.TRAIN.aug_strategy.scales
        self.kwargs['image_stride'] = self.config.network.image_stride

    def infer_max_shape(self):
        data_channel = 3
        max_data_height = makedivstride(max([v[0] for v in self.config.TRAIN.aug_strategy.scales]), self.config.network.image_stride)
        max_data_width = makedivstride(max([v[1] for v in self.config.TRAIN.aug_strategy.scales]), self.config.network.image_stride)
        max_data_shape = [('data', (self.batch_size, data_channel, max_data_height, max_data_width))]
        max_label_shape = self.infer_max_label_shape(max_data_shape)
        return max_data_shape, max_label_shape

    def infer_max_label_shape(self, max_data_shape):
        label = dict()
        if 'fpn' in self.config.network.task_type:
            feat_shape_list = []
            suffix = ''
            for i in range(len(self.config.network.retinanet_feat_strides)):
                feat_shape = self.retinanet_feat_sym[i].infer_shape(**dict(max_data_shape))[1][0]
                feat_shape_list.append(feat_shape)
            label.update(assign_anchor_fpn_retinanet(feat_shape=feat_shape_list,
                                                     gt_boxes=np.zeros((0, 5)),
                                                     gt_classes=np.zeros((0,)),
                                                     config=self.config,
                                                     feat_strides=self.config.network.retinanet_feat_strides,
                                                     scales=self.config.network.retinanet_anchor_scales,
                                                     ratios=self.config.network.retinanet_anchor_ratios,
                                                     suffix=suffix))
        else:
            raise ValueError("unknown task type {}".format(self.config.network.task_type))
        label = [label[k] for k in self.label_name]
        max_label_shape = [(k, tuple([self.batch_size] + list(v.shape[1:]))) for k, v in zip(self.label_name, label)]
        return max_label_shape

    def get_label(self, **kwargs):
        data_tensor = kwargs['data_tensor']
        gt_roidb_list = kwargs['gt_roidb_list']
        label_list = []
        if 'fpn' in self.config.network.task_type:
            feat_shape_list = []
            for i in range(len(self.config.network.retinanet_feat_strides)):
                feat_shape = self.retinanet_feat_sym[i].infer_shape(data=data_tensor.shape)[1][0]
               	feat_shape_list.append(feat_shape)
            for i in range(self.batch_size):
                batch_labels = dict()
                suffix = ''
                label = assign_anchor_fpn_retinanet(feat_shape=feat_shape_list,
                                                     gt_boxes=gt_roidb_list[i]['gt_boxes'][:, :4],
                                                     gt_classes=gt_roidb_list[i]['gt_boxes'][:, 4],
                                                     ignore_regions=gt_roidb_list[i]['ignore_regions'],
                                                     config=self.config,
                                                     feat_strides=self.config.network.retinanet_feat_strides,
                                                     scales=self.config.network.retinanet_anchor_scales,
                                                     ratios=self.config.network.retinanet_anchor_ratios,
                                                     suffix=suffix)
                batch_labels.update(label)
                label_list.append(batch_labels)
        else:
            raise ValueError("unknown task type {}".format(self.config.network.task_type))
        return label_list

    def get_gt_roidb(self, gt_boxes, gt_classes, res_dict):
        gt_boxes = np.hstack((gt_boxes, gt_classes.reshape((-1, 1))))
        gt_roidb_i = dict()
        gt_roidb_i['gt_boxes'] = gt_boxes[gt_classes > 0, :]
        gt_roidb_i['ignore_regions'] = gt_boxes[gt_classes < 0, :]
        return gt_roidb_i

    def get_batch(self):
        if not self.has_load_data:
            self.load_data()
        index_start = self.cur
        index_end = min(index_start + self.batch_size, self.size)
        roidb = [self.roidb[self.index[i]] for i in range(index_start, index_end)]
        slices = _split_input_slice(index_end - index_start, self.work_load_list)

        data_list = []
        gt_roidb_list = []
        for i_slice in slices:
            i_roidb = [roidb[i] for i in range(i_slice.start, i_slice.stop)]
            for j in range(len(i_roidb)):
                roidb_j = i_roidb[j]
                im = get_image(roidb_j, self.imgrec)
                all_boxes = roidb_j['boxes'].copy()
                res_dict = image_aug_function(im, all_boxes=all_boxes, **self.kwargs)
                im_tensor = transform(res_dict['img'], pixel_means=self.config.network.input_mean, scale=self.config.network.input_scale)  # (1, 3, h, w)
                data_list.append(im_tensor)
                gt_roidb = self.get_gt_roidb(res_dict['all_boxes'], roidb_j['gt_classes'], res_dict)
                gt_roidb_list.append(gt_roidb)

        all_data = dict()
        all_data['data'] = tensor_vstack(data_list)

        label_list = self.get_label(data_tensor=all_data['data'],
                                    gt_roidb_list=gt_roidb_list)
        all_label = dict()
        for name in self.label_name:
            pad = -1 if 'label' in name else 0
            all_label[name] = tensor_vstack([batch[name] for batch in label_list], pad=pad)

        self.data = [mx.nd.array(all_data[name]) for name in self.data_name]
        self.label = [mx.nd.array(all_label[name]) for name in self.label_name]


RetinanetTestIter = FPNTestIter
