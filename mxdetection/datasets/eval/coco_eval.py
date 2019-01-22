import os
import io
import cv2
import json
import numpy as np
import logging
import pickle
from mxdetection.ops.pycocotools.coco import COCO
from mxdetection.ops.pycocotools.cocoeval import COCOeval
from mxdetection.ops.pycocotools.mask import encode as encodeMask_c
from mxdetection.ops.pycocotools.cocostuffhelper import segmentationToCocoResult
from mxdetection.ops.pycocotools.cocostuffeval import COCOStuffeval
from mxdetection.ops.pycocotools.densepose_cocoeval import denseposeCOCOeval
from mxdetection.datasets.load_roidb import filter_roidb


def mask_voc2coco(voc_masks, voc_boxes, im_height, im_width, binary_thresh=0.5):
    if isinstance(voc_masks, list):
        return voc_masks
    num_pred = len(voc_masks)
    assert(num_pred == voc_boxes.shape[0])
    mask_img = np.zeros((im_height, im_width, num_pred), dtype=np.uint8, order='F')
    for i in xrange(num_pred):
        pred_box = np.round(voc_boxes[i, :4]).astype(int)
        pred_mask = voc_masks[i]
        pred_mask = cv2.resize(pred_mask.astype(np.float32), (pred_box[2] - pred_box[0] + 1, pred_box[3] - pred_box[1] + 1))
        mask_img[pred_box[1]:pred_box[3]+1, pred_box[0]:pred_box[2]+1, i] = pred_mask >= binary_thresh
        # correct mask score
        pos_index = np.where(pred_mask >= binary_thresh)
        score = np.sum(pred_mask[pos_index])
        area = pos_index[0].shape[0]
        area = area if area > 0 else 1
        voc_boxes[i, -1] *= float(score) / area
    coco_mask = encodeMask_c(mask_img)
    return coco_mask


class COCOEval(object):
    def __init__(self, annotation_path, categories=None):
        self.coco = COCO(annotation_path)
        self.categories = categories

        self.imageset_name = annotation_path[:-5].split('_')[-1]
        self.imageset_index = self.coco.getImgIds()
        self.num_images = len(self.imageset_index)

        # deal with class names
        if 'stuff' in annotation_path:
            cat_ids = self.coco.getCatIds()
            if 183 in cat_ids:
                self.stuff_order = False
                cat_ids.remove(183)
                cat_ids = [183] + cat_ids
            else:
                self.stuff_order = True
            self.stuffStartId = np.min(cat_ids)
            self.stuffEndId = np.max(cat_ids)
            self.classes = [cat['name'] for cat in self.coco.loadCats(cat_ids)]
            self.num_classes = len(self.classes)
            self._class_to_coco_ind = dict(zip(self.classes, self.coco.getCatIds()))
            if not self.stuff_order:
                self._class_ind_to_coco_ind = dict(zip(xrange(len(cat_ids)), cat_ids))
        else:
            cats = [cat['name'] for cat in self.coco.loadCats(self.coco.getCatIds())]
            self.classes = ['__background__'] + cats
            self.num_classes = len(self.classes)
            self._class_to_coco_ind = dict(zip(cats, self.coco.getCatIds()))

    def sample_on_imdb(self, roidb, filter_strategy):
        roidb, choose_inds = filter_roidb(roidb, filter_strategy, need_inds=True)
        self.imageset_index = [self.imageset_index[i] for i in choose_inds]
        self.num_images = len(self.imageset_index)
        return roidb

    # det
    def evaluate_detections(self, detections, alg='alg', res_folder=''):
        res_file = os.path.join(res_folder, 'detections_%s_%s_results.json' % (self.imageset_name, alg))
        self.write_coco_det_results(detections, res_file)
        if 'test' not in self.imageset_name:
            ann_type = 'bbox'
            coco_dt = self.coco.loadRes(res_file)
            coco_eval = COCOeval(self.coco, coco_dt)
            coco_eval.params.useSegm = (ann_type == 'segm')
            coco_eval.params.imgIds = self.imageset_index
            if self.categories is not None and 'det' in self.categories:
                if self.categories['det'] == 'each':
                    for cls_ind, cls in enumerate(self.classes):
                        if cls == '__background__':
                            continue
                        coco_eval.params.catIds = [self._class_to_coco_ind[cls]]
                        coco_eval.evaluate()
                        coco_eval.accumulate()
                        logging.info('%s detection result:' % cls)
                        coco_eval.summarize()
                    return
                elif self.categories['det'] == 'person':
                    coco_eval.params.catIds = [1]
                else:
                    assert self.categories['det'] == 'all'
            coco_eval.evaluate()
            coco_eval.accumulate()
            logging.info('detection result:')
            coco_eval.summarize()

    def write_coco_det_results(self, detections, res_file):
        """ example results
        [{"image_id": 42,
          "category_id": 18,
          "bbox": [258.15,41.29,348.26,243.78],
          "score": 0.236}, ...]
        """
        results = []
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            if self.categories is not None and 'det' in self.categories:
                if self.categories['det'] == 'person' and cls != 'person':
                    continue
            print 'Collecting %s results (%d/%d)' % (cls, cls_ind, self.num_classes - 1)
            coco_cat_id = self._class_to_coco_ind[cls]
            results.extend(self._coco_det_results_one_category(detections[cls_ind], coco_cat_id))
        print 'Writing results json to %s' % res_file
        with open(res_file, 'w') as f:
            json.dump(results, f, sort_keys=True, indent=4)

    def _coco_det_results_one_category(self, boxes, cat_id):
        results = []
        for im_ind, index in enumerate(self.imageset_index):
            if len(boxes[im_ind]) == 0:
                continue
            dets = boxes[im_ind].astype(np.float)
            scores = dets[:, -1]
            xs = dets[:, 0]
            ys = dets[:, 1]
            ws = dets[:, 2] - xs + 1
            hs = dets[:, 3] - ys + 1
            result = [{'image_id': index,
                       'category_id': cat_id,
                       'bbox': [xs[k], ys[k], ws[k], hs[k]],
                       'score': scores[k]} for k in xrange(dets.shape[0])]
            results.extend(result)
        return results

    # kps
    def evaluate_keypoints(self, keypoints, alg='alg', res_folder=''):
        res_file = os.path.join(res_folder, 'person_keypoints_%s_%s_result.json' % (self.imageset_name, alg))
        self.write_coco_kps_results(keypoints, res_file)
        if 'test' not in self.imageset_name:
            ann_type = 'keypoints'
            coco_kps = self.coco.loadRes(res_file)
            coco_eval = COCOeval(self.coco, coco_kps, ann_type)
            coco_eval.params.imgIds = self.imageset_index
            if self.categories is not None and 'kps' in self.categories:
                if self.categories['kps'] == 'person':
                    coco_eval.params.catIds = [1]
                else:
                    assert self.categories['kps'] == 'all'
            coco_eval.evaluate()
            coco_eval.accumulate()
            logging.info('keypoint result:')
            coco_eval.summarize()

    def write_coco_kps_results(self, keypoints, res_file):
        results = []
        for im_ind, index in enumerate(self.imageset_index):
            points = keypoints[im_ind]
            if len(points) == 0:
                continue
            result = [{'image_id': index,
                       'category_id': 1,
                       'keypoints': point[0:-1],
                       'score': point[-1]} for point in points]
            results.extend(result)
        print 'Writing results json to %s' % res_file
        with open(res_file, 'w') as f:
            json.dump(results, f, sort_keys=True, indent=4)

    # mask
    def evalute_mask(self, detections, masks, binary_thresh=0.5, alg='alg', res_folder=''):
        res_file = os.path.join(res_folder, 'detections_%s_%s_results.json' % (self.imageset_name, alg))
        self.write_coco_mask_results(detections, masks, binary_thresh, res_file)
        if 'test' not in self.imageset_name:
            ann_type = 'segm'
            coco_dt = self.coco.loadRes(res_file)
            coco_eval = COCOeval(self.coco, coco_dt)
            coco_eval.params.useSegm = (ann_type == 'segm')
            coco_eval.params.imgIds = self.imageset_index
            if self.categories is not None and 'mask' in self.categories:
                if self.categories['mask'] == 'each':
                    for cls_ind, cls in enumerate(self.classes):
                        if cls == '__background__':
                            continue
                        coco_eval.params.catIds = [self._class_to_coco_ind[cls]]
                        coco_eval.evaluate()
                        coco_eval.accumulate()
                        logging.info('%s mask result:' % cls)
                        coco_eval.summarize()
                    return
                elif self.categories['mask'] == 'person':
                    coco_eval.params.catIds = [1]
                else:
                    assert self.categories['mask'] == 'all'
            coco_eval.evaluate()
            coco_eval.accumulate()
            logging.info('mask result:')
            coco_eval.summarize()

    def write_coco_mask_results(self, detections, masks, binary_thresh, res_file):
        results = []
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            if self.categories is not None and 'mask' in self.categories:
                if self.categories['mask'] == 'person' and cls != 'person':
                    continue
            print 'Collecting %s results (%d/%d)' % (cls, cls_ind, self.num_classes - 1)
            coco_cat_id = self._class_to_coco_ind[cls]
            results.extend(self._coco_mask_results_one_category(detections[cls_ind], masks[cls_ind], binary_thresh, coco_cat_id))
        print 'Writing results json to %s' % res_file
        with open(res_file, 'w') as f:
            json.dump(results, f, sort_keys=True, indent=4)

    def _coco_mask_results_one_category(self, boxes, masks, binary_thresh, cat_id):
        results = []
        for im_ind, index in enumerate(self.imageset_index):
            if len(boxes[im_ind]) == 0:
                continue
            height = self.coco.loadImgs(index)[0]['height']
            width = self.coco.loadImgs(index)[0]['width']
            dets = boxes[im_ind].astype(np.float)
            mask_encode = mask_voc2coco(masks[im_ind], dets, height, width, binary_thresh)
            scores = dets[:, -1]
            result = [{'image_id': index,
                       'category_id': cat_id,
                       'segmentation': mask_encode[k],
                       'score': scores[k]} for k in xrange(dets.shape[0])]
            results.extend(result)
        return results

    # stuff
    def evaluate_stuff(self, stuff_results, alg='alg', res_folder=''):
        res_file = os.path.join(res_folder, 'stuff_%s_%s_result.json' % (self.imageset_name, alg))
        self.write_coco_stuff_results(stuff_results, res_file)
        if 'test' not in self.imageset_name:
            coco_res = self.coco.loadRes(res_file)
            coco_eval = COCOStuffeval(self.coco, coco_res, stuffStartId=self.stuffStartId,
                                      stuffEndId=self.stuffEndId, addOther=False)
            coco_eval.params.imgIds = self.imageset_index
            if self.categories is not None and 'seg' in self.categories:
                if self.categories['seg'] == 'each':
                    for cls_ind, cls in enumerate(self.classes):
                        coco_eval.params.catIds = [self._class_to_coco_ind[cls]]
                        coco_eval.evaluate()
                        logging.info('%s seg result:' % cls)
                        coco_eval.summarize()
                    return
                else:
                    assert self.categories['seg'] == 'all'
            coco_eval.evaluate()
            logging.info('seg result:')
            coco_eval.summarize()

    def write_coco_stuff_results(self, stuff_results, res_file):
        with io.open(res_file, 'w', encoding='utf8') as output:
            print 'Writing results json to %s' % res_file
            # Annotation start
            output.write(unicode('[\n'))
            for i, img_id in enumerate(self.imageset_index):
                stuff_result = stuff_results[i]
                assert stuff_result.shape[0] == self.coco.loadImgs(img_id)[0]['height']
                assert stuff_result.shape[1] == self.coco.loadImgs(img_id)[0]['width']
                if not self.stuff_order:
                    for j in range(self.num_classes):
                        stuff_result[stuff_result == j] = self._class_ind_to_coco_ind[j]
                    stuff_result[stuff_result == -1] = 183
                else:
                    stuff_result[stuff_result == -1] = 0
                anns = segmentationToCocoResult(stuff_result.astype(np.uint8), img_id, stuffStartId=self.stuffStartId)
                # Write JSON
                str_ = json.dumps(anns)
                str_ = str_[1:-1]
                if len(str_) > 0:
                    output.write(unicode(str_))
                # Add comma separator
                if i < len(self.imageset_index) - 1 and len(str_) > 0:
                    output.write(unicode(','))
                # Add line break
                output.write(unicode('\n'))
            # Annotation end
            output.write(unicode(']'))

    # densepose
    def evalute_densepose(self, detections, densepose_results, alg='alg', res_folder=''):
        res_file = os.path.join(res_folder, 'densepose_%s_%s_results.json' % (self.imageset_name, alg))
        self.write_coco_densepose_results(detections, densepose_results, res_file)
        if 'test' not in self.imageset_name:
            ann_type = 'uv'
            with open(res_file, 'rb') as f:
                res = pickle.load(f)
            coco_dt = self.coco.loadRes(res)
            # Non-standard params used by the modified COCO API version from the DensePose fork
            test_sigma = 0.255
            coco_eval = denseposeCOCOeval(self.coco, coco_dt, ann_type, test_sigma)
            coco_eval.params.imgIds = self.imageset_index
            if self.categories is not None and 'densepose' in self.categories:
                if self.categories['densepose'] == 'person':
                    coco_eval.params.catIds = [1]
                else:
                    assert self.categories['densepose'] == 'all'
            coco_eval.evaluate()
            coco_eval.accumulate()
            logging.info('densepose result:')
            coco_eval.summarize()

    def write_coco_densepose_results(self, detections, densepose_results, res_file):
        results = []
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            if self.categories is not None and 'densepose' in self.categories:
                if self.categories['densepose'] == 'person' and cls != 'person':
                    continue
            print 'Collecting %s results (%d/%d)' % (cls, cls_ind, self.num_classes - 1)
            coco_cat_id = self._class_to_coco_ind[cls]
            results.extend(self._coco_densepose_results_one_category(detections[cls_ind], densepose_results[cls_ind], coco_cat_id))
        print 'Writing results json to %s' % res_file
        with open(res_file, 'wb') as f:
            pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)

    def _coco_densepose_results_one_category(self, boxes, densepose_results, cat_id):
        results = []
        for im_ind, index in enumerate(self.imageset_index):
            if len(boxes[im_ind]) == 0 or len(densepose_results[im_ind]) == 0:
                continue
            dets = boxes[im_ind].astype(np.float)
            scores = dets[:, -1]
            xs = dets[:, 0]
            ys = dets[:, 1]
            ws = dets[:, 2] - xs + 1
            hs = dets[:, 3] - ys + 1
            uv_dets = densepose_results[im_ind]
            for uv in uv_dets:
                uv[1:3, :, :] *= 255
            result = [{'image_id': index,
                       'category_id': cat_id,
                       'uv': uv_dets[k].astype(np.uint8),
                       'bbox': [xs[k], ys[k], ws[k], hs[k]],
                       'score': scores[k]} for k in xrange(dets.shape[0])]
            results.extend(result)
        return results











