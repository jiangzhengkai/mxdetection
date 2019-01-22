import pickle
import logging
import os
import time
import numpy as np

def load_roidb(roidb_path_list, imglst_path_list=None, filter_strategy=None):
    t = time.time()
    roidb_list = []
    if roidb_path_list is not None:
        for roidb_path in roidb_path_list:
            with open(roidb_path, 'rb') as fin:
                roidb = pickle.load(fin)
            roidb_list.append(roidb)
    # filter roidb according to filter_strategy
    if filter_strategy is not None:
        roidb_list = [filter_roidb(roidb, filter_strategy) for roidb in roidb_list]
    # 
    if imglst_path_list is not None:
        add_roidb_imgrec_idx(roidb_list, imglst_path_list)
        
    roidb = merge_roidb(roidb_list)
    end = time.time()
    logging.info('total num images: %d using time: %d' % (len(roidb), end - t))
    return roidb

def add_roidb_imgrec_idx(roidb_list, imglst_path_list):
    assert len(roidb_list) == len(imglst_path_list)
    for i, roidb in enumerate(roidb_list):
        img_list = {}
        with open(imglst_path_list[i], 'r') as fin:
            for line in fin.readlines():
                # imglst file as index \t label \t img_name \n
                line = line.strip().split('\t')
                img_list[line[-1]] = int(line[0])
        for roi_rec in roidb:
            img_name = roi_rec['image']
            if img_name not in img_list:
                img_name = os.path.basename(roi_rec['image'])
                assert img_name in img_list
            # imgrec_id is the index of roidbs
            # imgrec_idx is the index of each roidb
            roi_rec['imgrec_id'] = i
            roi_rec['imgrec_idx'] = img_list[img_name]

def merge_roidb(roidb_list):
    roidb = roidb_list[0]
    for r in roidb_list[1:]:
        roidb.extend(r)
    return roidb


def filter_roidb(roidb, filter_strategy, need_index=False):
    index = range(len(roidb))
    # filter roidb function 
    def filter_roidb_function(choose_index, filter_name, filter_function):
        if filter_name in filter_strategy and filter_strategy[filter_name]:
            num = len(choose_index)
            choose_index = [i for i in index if not filter_function(roidb[i])]
            num_after = len(choose_index)
            logging.info('filter %d %s roidb entries: %d -> %d' %(num - num_after, filter_name[7:], num, num_after))
        return choose_index
    # whether it is empty boxes
    def is_empty_boxes(entry):
        num_valid_boxes = np.sum(entry['gt_classes'] > 0)
        return num_valid_boxes == 0
    all_choose_index = filter_roidb_function(index, 'remove_empty_boxes', is_empty_boxes)
    
    roidb = [roidb[i] for i in all_choose_index]
    
    if need_index:
        return roidb, index
    else:
        return roidb
    
    
