import os
import logging
import cv2
from PIL import Image
import numpy as np
import mxnet as mx
from mxdetection.apis.module import MutableModule
from mxdetection.utils.utils import load_param
from mxdetection.models.utils.symbol_common import get_symbol_function


class BasePredictor(object):
    def __init__(self, config, prefix, epoch, provide_data, max_data_shape=None, ctx=mx.cpu(), allow_missing=False):
        self.config = config
        logging.info('load model from %s-%04d.params' % (prefix, epoch))
        
        if self.config.TEST.load_sym_from_file:
            symbol, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
        else:
            symbol = config.network.symbol_network = get_symbol_function(config.network.sym)(config).get_test_symbol()
            arg_params, aux_params = load_param(prefix, epoch)

        if not isinstance(ctx, list):
            ctx = [ctx]
        data_names = [k[0] for k in provide_data]
        self.mod_list = []
        for ctx_i in ctx:
            mod = MutableModule(symbol, data_names, None, context=ctx_i, max_data_shapes=max_data_shapes)
            mod.bind(provide_data, for_training=False)
            mod.init_params(arg_params=arg_params, aux_params=aux_params, allow_missing=allow_missing)
            self.mod_list.append(mod)
        self.count = 0

    def predict(self, data_batch, need_forward):
        for i in range(len(data_batch)):
            if need_forward[i]:
                self.mod_list[i].forward(data_batch[i])
        outputs = []
        for i in range(len(data_batch)):
            if need_forward[i]:
                outputs.append(self.mod_list[i].get_outputs())
            else:
                outputs.append([])
        return outputs 

    def predict_data(self, test_data, eval_func=None, alg='alg',
                     save_roidb_path=None, vis=False, **vis_kwargs):
        pass
    def vis_results(self, local_vars,
                    im_save_dir=None, im_save_max_num=-1,
                    writer=None, show_camera=False,
                    det_results=None, box_color=None, do_draw_box=True,
                    seg_results=None, seg_color=None,
                    mask_results=None, mask_color=None,
                    kps_results=None, point_color=None, skeleton_color=None,
                    densepose_results=None, densepose_color=None):
        im_draw = local_vars['im']
        
        if det_results is not None and len(det_results) > 0:
            if box_color is None:
                box_color = [(0, 255, 255), (0, 0, 255), (255, 0, 0), (255, 255, 0), (255, 0, 255), (255, 0, 170), (255, 0, 85)]
            all_boxes = det_results[:, :4]
            all_classes = det_results[:, 5]
            
            im_draw = draw_all(im=im_draw,
                               all_boxes=all_boxes,
                               all_classes=all_classes,
                               box_color=box_color,
                               do_draw_box=do_draw_box)
        if seg_results is not None:
            if seg_color is None:
                seg_color = seg_colormap()
                im_draw = draw_all(im=im_draw,
                                   all_segs=seg_results[np.newaxis, :],
                                   seg_color=seg_color)

        if mask_results is not None and 'mask' in mask_results:
            if mask_color is None:
                mask_boxes = mask_results['mask_boxes']
            masks = mask_results['masks']
            im_draw = draw_all(im=im_draw,
                               all_mask_boxes=mask_boxes,
                               all_masks=masks,
                               mask_color=mask_color)
        
        if kps_results is not None and len(kps_results) > 0:
            if point_color is None:
                point_color = (0, 255, 255)
            if skeleton_color is None:
                skeleton_color = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
                                  [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
                                  [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85], [0, 0, 255]]
            all_kps = np.array(kps_results)[:, :-1]
            im_draw = draw_all(im=im_draw,
                               all_kps=all_kps,
                               skeleton=self.config.dataset.kps_skeleton,
                               point_color=point_color,
                               skeleton_color=skeleton_color,
                               kps_thresh=0.2,
                               kps_show_num=False)
        
        if densepose_results is not None and 'densepose_masks' in densepose_results:
            if densepose_color is None:
                densepose_color = (240, 240, 240)
            densepose_boxes = densepose_results['densepose_boxes']
            densepose_masks = densepose_results['densepose_masks']
            im_draw = draw_all(im=im_draw,
                               all_segs=densepose_masks,
                               all_seg_boxes=densepose_boxes,
                               seg_color=densepose_color)

        if im_save_dir is not None:
            if im_save_max_num == -1 or self.count < im_save_max_num:
                im_name = os.path.splitext(os.path.basename(local_vars['roi_rec']['image']))[0]
                im_save_path = os.path.join(im_save_dir, im_name + '_draw.jpg')
                cv2.imwrite(im_save_path, im_draw[:, :, ::-1])
                self.count += 1
                # if seg_results is not None:
                #     seg_color = np.array(seg_colormap()).reshape((-1,))
                #     seg_label_save_path = os.path.join(im_save_dir, im_name + '_draw_label.png')
                #     png = Image.fromarray(seg_results).convert('P')
                #     png.putpalette(seg_color)
                #     png.save(seg_label_save_path, format='PNG')
        if writer is not None:
            writer.append_data(im_draw)
        if show_camera:
            cv2.namedWindow('test', cv2.WINDOW_NORMAL)
            cv2.imshow('test', im_draw[:, :, ::-1])
            cv2.waitKey(1)
