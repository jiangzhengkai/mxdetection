import numpy as np
import copy
import mxnet as mx
import logging
import cv2
from mxnet.executor_manager import _split_input_slice
from .fpn_iter import FPNIter, FPNTestIter
from mxdetection.core.processing.image import tensor_vstack, transform, get_image
from mxdetection.core.processing.image_aug_function import image_aug_function
from mxdetection.utils.utils import makedivstride


class FPNMaskIter(FPNIter):
    def __init__(self, roidb, config, batch_size, ctx=None):
        super(FPNMaskIter, self).__init__(roidb, config, batch_size, ctx)

    def load_data(self):
        super(FPNMaskIter, self).load_data()

    def infer_max_shape(self):
        max_data_shape, max_label_shape = super(FPNMaskIter, self).infer_max_shape()
        return max_data_shape, max_label_shape

    def get_gt_roidb(self, gt_boxes, gt_classes, res_dict):
        gt_boxes = np.hstack((gt_boxes, gt_classes.reshape((-1, 1))))
        gt_roidb = dict()
        gt_roidb['gt_boxes'] = gt_boxes[gt_classes > 0, :]
        gt_roidb['ignore_regions'] = gt_boxes[gt_classes == -1, :]
        if 'mask' in self.config.network.task_type:
            gt_roidb['gt_polys_or_rles'] = res_dict['all_polys_or_rles']
        return gt_roidb

    def get_label(self, **kwargs):
        label_list = super(FPNMaskIter, self).get_label(**kwargs)
        return label_list

    def _resize_seg_label(self, seg_label, img_shape):
        if seg_label is not None:
            seg_label_height = seg_label.shape[0] / self.config.network.seg_feat_stride
            seg_label_width = seg_label.shape[1] / self.config.network.seg_feat_stride
            seg_label = cv2.resize(seg_label, (seg_label_width, seg_label_height), interpolation=cv2.INTER_NEAREST)
            seg_label = seg_label.reshape((1, seg_label.shape[0], seg_label.shape[1]))
        else:
            seg_label_height = img_shape[0] / self.config.network.seg_feat_stride
            seg_label_width = img_shape[1] / self.config.network.seg_feat_stride
            seg_label = np.full((1, seg_label_height, seg_label_width), fill_value=-1, dtype=np.float32)
        return seg_label


    def get_batch(self):
        if not self.has_load_data:
            self.load_data()
        index_start = self.cur
        index_end = min(index_start + self.batch_size, self.size)
        roidb = [self.roidb[self.index[i]] for i in range(index_start, index_end)]
        slices = _split_input_slice(index_end - index_start, self.work_load_list)

        data_list = []
        im_info_list = []
        gt_roidb_list = []
        for i_slice in slices:
            i_roidb = [roidb[i] for i in range(i_slice.start, i_slice.stop)]
            for j in range(len(i_roidb)):
                roidb_j = i_roidb[j]
                im = get_image(roidb_j, self.imgrec)
                all_boxes = roidb_j['boxes'].copy()
                keep = np.where(roidb_j['gt_classes'] > 0)[0]
                all_polys_or_rles = None
                if 'mask' in self.config.network.task_type:
                    all_polys_or_rles = copy.deepcopy([roidb_j['gt_masks'][_] for _ in keep])
                res_dict = image_aug_function(img=im,
                                         all_boxes=all_boxes,
                                         all_polys_or_rles=all_polys_or_rles,
                                         **self.kwargs)
                im_tensor = transform(res_dict['img'], pixel_means=self.config.network.input_mean, scale=self.config.network.input_scale)  # (1, 3, h, w)
                data_list.append(im_tensor)
                im_info_list.append(np.array([[im_tensor.shape[2], im_tensor.shape[3], res_dict['img_scale']]], dtype=np.float32))
                gt_roidb = self.get_gt_roidb(res_dict['all_boxes'], roidb_j['gt_classes'], res_dict)
                gt_roidb_list.append(gt_roidb)

        all_data = dict()
        all_data['data'] = tensor_vstack(data_list)
        all_data['im_info'] = tensor_vstack(im_info_list)
        all_data.update(self.serialize_gt_roidb(gt_roidb_list))

        label_list = self.get_label(data_tensor=all_data['data'],
                                    im_info_list=im_info_list,
                                    gt_roidb_list=gt_roidb_list)
        all_label = dict()
        for name in self.label_name:
            pad = -1 if 'label' in name else 0
            all_label[name] = tensor_vstack([batch[name] for batch in label_list], pad=pad)
        self.data = [mx.nd.array(all_data[name]) for name in self.data_name]
        self.label = [mx.nd.array(all_label[name]) for name in self.label_name]


FPNMaskTestIter = FPNTestIter