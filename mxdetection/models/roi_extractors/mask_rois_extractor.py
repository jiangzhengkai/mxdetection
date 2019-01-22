import logging
import mxnet as mx

class Mask_rois_extractor(object):
    def __init__(self, config):

        self.config = config

    def get_symbol(self, shared_feat, rois, name, **kwargs):
        pooled_size = (self.config.network.mask_pooled_size[0] / 2, self.config.network.mask_pooled_size[1] / 2)
        if self.config.network.roi_extract_method == 'roi_align_fpn':
            subnet_shared_feat = dict()
            for stride in self.config.network.rcnn_feat_stride:
                subnet_shared_feat['data_res%d' % stride] = shared_feat["stride%d" % stride]
            feat_rois = mx.sym.ROIAlignFPN(rois=rois, name=name, sample_per_part=2,
                                           pooled_size=pooled_size,
                                           feature_strides=tuple(self.config.network.rcnn_feat_stride),
                                           **subnet_shared_feat)
        elif self.config.network.roi_extract_method == 'roi_align':
            feat_rois = mx.sym.ROIAlign(data=shared_feat,
                                        rois=rois,
                                        name=name,
                                        sample_per_part=2,
                                        pooled_size=pooled_size,
                                        spatial_scale=1.0 / self.config.network.rcnn_feat_stride)
        elif roi_extract_style == 'roi_align_ada_fpn':
            feat_rois = 0
            print'adaptive pooling stride', feat_stride
            for idx, stride in enumerate(feat_stride):
                feat_rois = feat_rois + mx.sym.ROIAlign(data=shared_feat['stride%d' % stride], rois=rois,
                                                        sample_per_part=2, pooled_size=pooled_size,
                                                        spatial_scale=1.0 / feat_stride[idx])
        else:
            raise ValueError("unknown roi extract method {}".format(self.config.network.roi_extract_method))
        logging.info('{}: {}'.format(name, feat_rois.infer_shape(**kwargs['input_dict'])[1]))
        return feat_rois
