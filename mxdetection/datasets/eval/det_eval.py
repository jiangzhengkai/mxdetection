import numpy as np
import logging
from mxdetection.core.bbox.bbox_transform import bbox_overlaps


def evaluate_recall(roidb, candidate_boxes, thresholds=None, gt_class_ind=None):
    """
    evaluate detection proposal recall metrics
    record max overlap value for each gt box; return vector of overlap values
    :param roidb: used to evaluate
    :param candidate_boxes: if not given, use roidb's non-gt boxes
    :param thresholds: array-like recall threshold
    :return: None
    ar: average recall, recalls: vector recalls at each IoU overlap threshold
    thresholds: vector of IoU overlap threshold, gt_overlaps: vector of all ground-truth overlaps
    """
    area_names = ['all', '0-25', '25-50', '50-100',
                  '100-200', '200-300', '300-inf']
    area_ranges = [[0**2, 1e5**2], [0**2, 25**2], [25**2, 50**2], [50**2, 100**2],
                   [100**2, 200**2], [200**2, 300**2], [300**2, 1e5**2]]
    area_counts = []
    num_images = len(roidb)
    for area_name, area_range in zip(area_names[1:], area_ranges[1:]):
        area_count = 0
        for i in range(num_images):
            boxes = candidate_boxes[i]
            boxes_areas = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
            valid_range_inds = np.where((boxes_areas >= area_range[0]) & (boxes_areas < area_range[1]))[0]
            area_count += len(valid_range_inds)
        area_counts.append(area_count)
    total_counts = float(sum(area_counts))
    for area_name, area_count in zip(area_names[1:], area_counts):
        logging.info('percentage of %s: %f' % (area_name, area_count / total_counts if total_counts > 0 else 0))
    logging.info('average number of proposal %f' % (total_counts / num_images))
    for area_name, area_range in zip(area_names, area_ranges):
        gt_overlaps = np.zeros(0)
        num_pos = 0
        for i in range(num_images):
            if gt_class_ind is not None:
                gt_inds = np.where(roidb[i]['gt_classes'] == gt_class_ind)[0]
            else:
                gt_inds = np.where(roidb[i]['gt_classes'] > 0)[0]
            gt_boxes = roidb[i]['boxes'][gt_inds, :]
            gt_areas = (gt_boxes[:, 2] - gt_boxes[:, 0] + 1) * (gt_boxes[:, 3] - gt_boxes[:, 1] + 1)
            valid_gt_inds = np.where((gt_areas >= area_range[0]) & (gt_areas < area_range[1]))[0]
            gt_boxes = gt_boxes[valid_gt_inds, :]
            num_pos += len(valid_gt_inds)

            boxes = candidate_boxes[i]
            if boxes.shape[0] == 0:
                continue

            overlaps = bbox_overlaps(boxes.astype(np.float), gt_boxes.astype(np.float))

            _gt_overlaps = np.zeros((gt_boxes.shape[0]))
            # choose whatever is smaller to iterate
            rounds = min(boxes.shape[0], gt_boxes.shape[0])
            for j in range(rounds):
                # find which proposal maximally covers each gt box
                argmax_overlaps = overlaps.argmax(axis=0)
                # get the IoU amount of coverage for each gt box
                max_overlaps = overlaps.max(axis=0)
                # find which gt box is covered by most IoU
                gt_ind = max_overlaps.argmax()
                gt_ovr = max_overlaps.max()
                assert (gt_ovr >= 0), '%s\n%s\n%s' % (boxes, gt_boxes, overlaps)
                # find the proposal box that covers the best covered gt box
                box_ind = argmax_overlaps[gt_ind]
                # record the IoU coverage of this gt box
                _gt_overlaps[j] = overlaps[box_ind, gt_ind]
                assert (_gt_overlaps[j] == gt_ovr)
                # mark the proposal box and the gt box as used
                overlaps[box_ind, :] = -1
                overlaps[:, gt_ind] = -1
            # append recorded IoU coverage level
            gt_overlaps = np.hstack((gt_overlaps, _gt_overlaps))

        gt_overlaps = np.sort(gt_overlaps)
        if thresholds is None:
            step = 0.05
            thresholds = np.arange(0.5, 0.95 + 1e-5, step)
        recalls = np.zeros_like(thresholds)

        # compute recall for each IoU threshold
        for i, t in enumerate(thresholds):
            recalls[i] = (gt_overlaps >= t).sum() / float(num_pos)
        ar = recalls.mean()

        # print results
        logging.info('average recall for {}: {:.3f}'.format(area_name, ar))
        # for threshold, recall in zip(thresholds, recalls):
        #     logging.info('recall @{:.2f}: {:.3f}'.format(threshold, recall))


def evaluate_ap(roidb, candidate_boxes, ivu_thre=0.5, save_res_prefix=None, draw_ap=False):
    num_cls = len(candidate_boxes)
    aps = []
    for cls in range(num_cls):
        if save_res_prefix is not None:
            generate_eval_txt_gt(roidb, cls, save_res_prefix)
        gt_boxes = []
        for roi_rec in roidb:
            boxes = []
            for box, gt_id in zip(roi_rec['boxes'], roi_rec['gt_classes']):
                if gt_id != (cls + 1) and gt_id != -(cls + 1):
                    continue
                l, t, r, b = box
                boxes.append([l, t, r, b, gt_id])
            boxes = np.array(boxes).reshape(-1, 5)
            boxes[:, 4] = np.array(boxes[:, 4] <= 0, dtype=np.float)
            gt_boxes.append(boxes)

        det_boxes = []
        for dataset_det in candidate_boxes[cls]:
            det_boxes.append(dataset_det)
        i = 0
        mean_ap = 0
        i += 1
        ivu_thre = float(ivu_thre)
        ap, rec, pre, fppi, fp, tp, conf, tp_boxes, gt_detections, img_idx = evaluation(det_boxes, gt_boxes, ivu_thre)
        eval_result = 'ap: {}    recall: {}    IoU: {}\n'.format(ap, rec[-1], ivu_thre)
        logging.info(eval_result)
        if draw_ap:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'fuchsia', 'aquamarine']
            num_color = len(colors)
            color_index = i % num_color
            color = colors[color_index]
            plt.plot(rec, pre, color=color, label='{},IoU:{},ap:{:.2f}'.format(cls, ivu_thre, ap))
        mean_ap += ap
        mean_ap /= i
        logging.info('mean AP:', mean_ap)
        aps.append(ap)


def calap(recall, prec):
    mrec = [0] + list(recall.flatten()) + [1]
    mpre = [0] + list(prec.flatten()) + [0]
    ap = 0
    for i in xrange(len(mpre) - 1):
        if mpre[i + 1] > 0:  # 0.9:
            ap += (mrec[i + 1] - mrec[i]) * mpre[i + 1]
    return ap, mrec[1:-1], mpre[1:-1]


def evaluation(det_boxes, gt_boxes, ivu_thres):
    num_gt = 0
    tp_all = []
    fp_all = []
    tp_boxes_all = []
    all_conf = []
    num_image = len(det_boxes)
    diff_x = []
    gt_detections = []
    img_idx = []
    for i, gt in enumerate(gt_boxes):
        num_gt += gt.shape[0] - gt[:, 4].sum()
        num_gt_i = gt.shape[0]
        idx = np.argsort(-det_boxes[i][:, 4])
        det_b = det_boxes[i][idx]
        num_obj = len(det_boxes[i])
        conf = det_boxes[i][idx, 4].reshape(-1, 1)
        gt_detected = np.zeros((num_gt_i, 1))
        tp = np.zeros((num_obj, 1))
        fp = np.zeros((num_obj, 1))
        tp_boxes = np.zeros((num_obj, 1))
        img_idx.extend([i, j] for j in xrange(num_obj))
        for j in xrange(num_obj):
            b = det_b[j]
            kmax = -1
            ov_max = -1000000
            iv_max = -1000000
            for k in xrange(gt.shape[0]):
                if gt_detected[k] == 1:
                    continue
                bbgt = gt[k]
                bi = [max(b[0], bbgt[0]), max(b[1], bbgt[1]), min(b[2], bbgt[2]), min(b[3], bbgt[3])]
                det_w = b[2] - b[0]
                det_h = b[3] - b[1]
                m = bbgt[2] - bbgt[0]
                n = bbgt[3] - bbgt[1]
                thr = min(ivu_thres, m * n / (m + 10.0) / (n + 10.0))
                iw = bi[2] - bi[0] + 1
                ih = bi[3] - bi[1] + 1
                if iw > 0 and ih > 0:
                    ua = (b[2] - b[0] + 1) * (b[3] - b[1] + 1) + \
                         (bbgt[2] - bbgt[0] + 1) * (bbgt[3] - bbgt[1] + 1) - \
                         iw * ih
                    ov = iw * ih / ua
                    iv = iw * ih / (max(det_w * det_h, 1E-10))
                    if bbgt[4] > 0:
                        iv_max = max(iv, iv_max)
                    if ov > ov_max and ov > thr:
                        diff_x.append((bbgt[2] + bbgt[1] - b[2] - b[1]) / 2.0)
                        ov_max = ov
                        kmax = k
                    if ov > thr:
                        tp_boxes[j] = 1
            if kmax >= 0:
                if gt[kmax, 4] < 1:
                    tp[j] = 1
                gt_detected[kmax] = 1
            elif iv_max < 0.5:
                fp[j] = 1

        gt_detections.append(gt_detected)
        tp_all.append(tp)
        fp_all.append(fp)
        tp_boxes_all.append(zip(det_b, tp_boxes))
        all_conf.append(conf)
    tp = np.vstack(tp_all)
    fp = np.vstack(fp_all)
    conf = np.vstack(all_conf)
    idx = np.argsort(-conf, axis=0)
    conf = conf[idx]
    img_idx = np.array(img_idx)[idx]
    img_idx = img_idx.reshape(-1, 2)
    tp = np.require(tp[idx], dtype=np.float)
    fp = fp[idx]
    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    recall = tp / (num_gt + 1E-10)
    prec = tp / (tp + fp + 1E-10)
    fppi = fp / float(num_image)
    ap, recall, prec = calap(recall, prec)

    return ap, recall, prec, list(fppi.flatten()), list(fp.flatten()), list(tp.flatten()), list(
        conf.flatten()), tp_boxes_all, gt_detections, img_idx


def generate_eval_txt_gt(roidb, cls, save_res_prefix):
    txt_gt_path = save_res_prefix + '{}_det_gt.txt'.format(cls+1)
    fid = open(txt_gt_path, 'w')
    for roi_rec in roidb:
        fid.write(roi_rec['image'])
        for box, gt_id in zip(roi_rec['boxes'], roi_rec['gt_classes']):
            if gt_id != (cls+1) and gt_id != -(cls+1):
                continue
            l, t, r, b = box
            fid.write(" {} {} {} {} {}".format(l, t, r, b, gt_id))
        fid.write('\n')
    fid.close()

