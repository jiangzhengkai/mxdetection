import logging
import copy
from .load_roidb import load_roidb, filter_roidb
from mxdetection.datasets.eval.det_eval import evaluate_recall, evaluate_ap


def load_coco_test_roidb_eval(config):
    from mxdetection.datasets.eval.coco_eval import COCOEval
    # get roidb
    roidb = load_roidb(roidb_path_list=config.dataset.test_roidb_path_list,
                       imglst_path_list=config.dataset.test_imglst_path_list)
    logging.info('total num images for test: {}'.format(len(roidb)))

    roidb, choose_inds = filter_roidb(roidb, config.TEST.filter_strategy, need_index=True)
    logging.info('total num images for test after sampling: {}'.format(len(roidb)))

    def _load_and_check_coco(anno_path, imageset_index):
        categories = config.dataset.categories if 'categories' in config.dataset else None
        imdb = COCOEval(anno_path, categories)
        imdb.imageset_index = [imdb.imageset_index[i] for i in choose_inds]
        imdb.num_images = len(imdb.imageset_index)
        if imageset_index is None:
            imageset_index = copy.deepcopy(imdb.imageset_index)
        else:
            for i, j in zip(imageset_index, imdb.imageset_index):
                assert i == j
        return imdb, imageset_index
    imdb = None
    imageset_index = None
    imdb, imageset_index = _load_and_check_coco(config.dataset.test_coco_anno_path['instances'], imageset_index)
    seg_imdb = None
    if 'seg' in config.network.task_type:
        seg_imdb, imageset_index = _load_and_check_coco(config.dataset.test_coco_anno_path['seg'], imageset_index)
    assert imageset_index is not None

    def eval_func(**kwargs):
        task_type = config.network.task_type
        if 'rpn' in task_type and config.TEST.rpn_do_test:
            all_proposals = kwargs['all_proposals']
            for j in range(1, len(all_proposals)):
                logging.info('***************class %d****************' % j)
                gt_class_ind = j if config.network.rpn_rcnn_num_branch > 1 else None
                evaluate_recall(roidb, all_proposals[j], gt_class_ind=gt_class_ind)
        if 'rpn_rcnn' in task_type or 'retinanet' in task_type:
            imdb.evaluate_detections(kwargs['all_boxes'], alg=kwargs['alg'] + '-det')
        if 'seg' in task_type:
            seg_imdb.evaluate_stuff(kwargs['all_seg_results'], alg=kwargs['alg'] + '-seg')
        if 'kps' in task_type:
            imdb.evaluate_keypoints(kwargs['all_kps_results'], alg=kwargs['alg'] + '-kps')
        if 'mask' in task_type:
            imdb.evalute_mask(kwargs['all_mask_boxes'], kwargs['all_masks'],
                              binary_thresh=config.TEST.mask_binary_thresh, alg=kwargs['alg'] + '-mask')
        if 'densepose' in task_type:
            imdb.evalute_densepose(kwargs['all_densepose_boxes'], kwargs['all_densepose'], alg=kwargs['alg'] + '-densepose')
    return roidb, eval_func

