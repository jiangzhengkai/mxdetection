import mxnet as mx
import numpy as np
import random
import cPickle as pickle
from mxdetection.core.bbox.sample import sample_rois
from mxdetection.utils.utils import deserialize


class ProposalTargetOperator(mx.operator.CustomOp):
    def __init__(self, config):
        super(ProposalTargetOperator, self).__init__()
        self.config = config
        self.fg_rois_per_image = np.round(config.TRAIN.rcnn_batch_rois * config.TRAIN.rcnn_fg_fraction).astype(int)
        self.rois_per_image = config.TRAIN.rcnn_batch_rois
        self.num_classes = 2 if config.network.rcnn_class_agnostic else config.dataset.num_classes
        
        self.num_images = config.TRAIN.image_batch_size
        
    def forward(self, is_train, req, in_data, out_data, aux):
        all_rois = in_data[0].asnumpy().reshape((self.num_images, -1, 5))
        gt_roidb = in_data[1].asnumpy()
        
        all_res_list = []
        for i in range(self.num_images):
            all_rois_i = all_rois[i, :]
            gt_roidb_i = gt_roidb[i, :]
            gt_roidb_i = gt_roidb_i[1:int(gt_roidb_i[0]) + 1]
            
            gt_roidb_i = deserialize(gt_roidb_i)
            
            gt_boxes_i = gt_roidb_i['gt_boxes']
            if len(gt_boxes_i) > 0:
                image_ids = np.full((gt_boxes_i.shape[0], 1), i, dtype=gt_boxes_i.dtype)
                all_rois_i = np.vstack((all_rois_i, np.hstack((image_ids, gt_boxes_i[:, :-1]))))
            assert np.all(all_rois_i[:, 0] == i)
        
            sample_rois_params = dict()
            sample_rois_params['rois'] = all_rois_i
            sample_rois_params['fg_rois_per_image'] = self.fg_rois_per_image
            sample_rois_params['rois_per_image'] = self.rois_per_image
            sample_rois_params['num_classes'] = self.num_classes
            sample_rois_params['config'] = self.config
            sample_rois_params['gt_boxes'] = gt_boxes_i
            sample_rois_params['ignore_regions'] = gt_roidb_i['ignore_regions']
            assert len(sample_rois_params) == 7
       
            if 'mask' in self.config.network.task_type:
                sample_rois_params['gt_polys_or_rles'] = gt_roidb_i['gt_polys_or_rles']
          
            if len(sample_rois_params) > 7:
                from mxdetection.core.mask.sample import sample_rois_mask
                res_list = sample_rois_mask(**sample_rois_params)
            else:
                res_list = sample_rois(**sample_rois_params)
            all_res_list.append(res_list)       
 
        for i in range(len(all_res_list[0])):
            res_i = [all_res_list[j][i] for j in range(len(all_res_list))]
            if len(res_i) == 1:
                res_i = res_i[0]
            elif res_i[0].ndim == 1:
                res_i = np.hstack(res_i)
            else:
                res_i = np.vstack(res_i)
            self.assign(out_data[i], req[i], res_i)
        
    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        for i in range(len(in_grad)):
            self.assign(in_grad[i], req[i], 0)
    
@mx.operator.register('proposal_target')
class ProposalTargetProp(mx.operator.CustomOpProp):
    def __init__(self, config):
        super(ProposalTargetProp, self).__init__(need_top_grad=False)
        self.config = pickle.loads(config)
    
    def list_arguments(self):
        args = ['rois', 'gt_roidb']
        return args
    
    def list_outputs(self):
        outputs = ['bbox_rois', 'bbox_label', 'bbox_target', 'bbox_weight']
        if 'mask' in self.config.network.task_type:
            outputs.extend(['mask_rois','mask_label'])
        return outputs
    
    def infer_shape(self, in_shape):
        num_classes = 2 if self.config.network.rcnn_class_agnostic else self.config.dataset.num_classes
        num_images = self.config.TRAIN.image_batch_size

        num_rcnn_rois = num_images * self.config.TRAIN.rcnn_batch_rois
        bbox_rois_shape = (num_rcnn_rois, 5)
        bbox_label_shape = (num_rcnn_rois, )
        bbox_target_shape = (num_rcnn_rois, num_classes * 4)
        bbox_weight_shape = (num_rcnn_rois, num_classes * 4)
        ovr_shape = (num_rcnn_rois, 1)
        output_shape = [bbox_rois_shape, bbox_label_shape, bbox_target_shape, bbox_weight_shape]
        
        if 'mask' in self.config.network.task_type:
            mask_height = self.config.network.mask_pooled_size[0]
            mask_width = self.config.network.mask_pooled_size[1]
            
            num_mask_rois = num_images * self.config.TRAIN.mask_roi_batch_size
            mask_rois_shape = (num_mask_rois, 5)
            mask_label_shape = (num_mask_rois, 1, mask_height, mask_width)
            output_shape.extend([mask_rois_shape, mask_label_shape])

        return in_shape, output_shape
        
    def create_operator(self, ctx, shapes, dtypes):
        return ProposalTargetOperator(self.config)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []
       

def proposal_target(rois, gt_roidb, config):
    group = mx.sym.Custom(rois=rois,
                          gt_roidb=gt_roidb,
                          op_type='proposal_target',
                          config=pickle.dumps(config))
    return group
    
        

    

