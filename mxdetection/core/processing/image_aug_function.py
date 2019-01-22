import numpy as np
import random
import cv2
from .image import resize
from mxdetection.ops.pycocotools.mask import encode, decode

def flip_polys(polys_or_rles, img_width):
    all_flipped_polys = []
    for i, ann in enumerate(polys_or_rles):
        # Polygon format
        flipped_polys = []
        for poly in ann:
            flipped_poly = np.array(poly, dtype=np.float32)
            flipped_poly[0::2] = img_width - flipped_poly[0::2] - 1
            flipped_polys.append(flipped_poly.tolist())
        all_flipped_polys.append(flipped_polys)
    return all_flipped_polys


def flip_boxes(src_boxes, img_width):
    # src_boxes: (num_boxes, 4)  [x1, y1, x2, y2]
    dst_boxes = src_boxes.copy()
    dst_boxes[:, 0] = img_width - src_boxes[:, 2] - 1.0
    dst_boxes[:, 2] = img_width - src_boxes[:, 0] - 1.0
    return dst_boxes

def rotate_image(src, angle, flags=cv2.INTER_LINEAR):
    w = src.shape[1]
    h = src.shape[0]
    radian = angle / 180.0 * math.pi
    radian_sin = math.sin(radian)
    radian_cos = math.cos(radian)
    new_w = int(abs(radian_cos * w) + abs(radian_sin * h))
    new_h = int(abs(radian_sin * w) + abs(radian_cos * h))
    rot_mat = cv2.getRotationMatrix2D((w/2.0, h/2.0), angle, 1.0)
    rot_mat[0, 2] += (new_w - w) / 2.0
    rot_mat[1, 2] += (new_h - h) / 2.0
    dst_img = cv2.warpAffine(src, rot_mat, (new_w, new_h), flags=flags)
    return dst_img


def image_aug_function(img, all_boxes=None, all_polys_or_rles=None, all_masks=None, flip=False, rotated_angle_range=0, scales=None, image_stride=0):
    
    res_dict = dict()
    all_polys = []
    if all_polys_or_rles is not None and len(all_polys_or_rles) > 0:
        if type(all_polys_or_rles[0]) == list:
            # Polygon format
            all_polys = all_polys_or_rles
        else:
            # RLE format
            assert type(all_polys_or_rles[0]) == dict
            assert all_masks is None
            all_masks = np.array(decode(all_polys_or_rles), dtype=np.float32)
            all_masks = all_masks.transpose((2, 0, 1))
    
    # flip
    if flip and random.randint(0, 1) == 1:
        img = img[:, ::-1, :]
        if all_boxes is not None and len(all_boxes) > 0:
            all_boxes = flip_boxes(all_boxes, img.shape[1])
        if all_polys is not None and len(all_polys) > 0:
            all_polys = flip_polys(all_polys, img.shape[1])
        if all_masks is not None and len(all_masks) > 0:
            all_masks = all_masks[:, :, ::-1]

        res_dict['flip'] = True
    else:
        res_dict['flip'] = False
    
    # rotated_angle_range
    if rotated_angle_range > 0:
        assert all_polys is None
        rotated_angle = random.randint(-rotated_angle_range, rotated_angle_range)
        origin_image_shape = img.shape
        img = rotate_image(img, rotated_angle)
        
        if all_boxes is not None and len(all_boxes) > 0:
            all_boxes = rotate_boxes(all_boxes, rotated_angle, ori_img_shape, img.shape)
        
        if all_masks is not None and len(all_masks) > 0:
            num_mask = all_masks.shape[0]
            new_all_masks = np.zeros((num_mask, img.shape[0], img.shape[1]))
            for j in range(num_mask):
                new_all_masks[j, :, :] = rotate_image(all_masks[j, :, :], rotated_angle)
            all_masks = new_all_masks
            res_dict['rotated_angle'] = rotated_angle
    else:
        res_dict['rotated_angle'] = 0
        
    # scale
    if scales is not None:
        scale_ind = random.randint(0, len(scales) - 1)
        target_size = scales[scale_ind][0]
        max_size = scales[scale_ind][1]
        img, img_scale = resize(img, target_size, max_size, stride=image_stride)
        if all_boxes is not None and len(all_boxes) > 0:
            all_boxes = all_boxes * img_scale

        if all_polys is not None and len(all_polys) > 0:
            for i, ann in enumerate(all_polys):
                for j, poly in enumerate(ann):
                    poly = np.array(poly, dtype=np.float32)
                    poly *= img_scale
                    all_polys[i][j] = poly.tolist()

        if all_masks is not None and len(all_masks) > 0:
            num_mask = all_masks.shape[0]
            new_all_masks = np.zeros((num_mask, img.shape[0], img.shape[1]))
            for j in range(num_mask):
                new_mask = cv2.resize(all_masks[j, :, :], None, None, fx=img_scale, fy=img_scale, interpolation=cv2.INTER_LINEAR)
                new_all_masks[j, :new_mask.shape[0], :new_mask.shape[1]] = new_mask
            all_masks = new_all_masks
        res_dict['img_scale'] = img_scale
    else:
        res_dict['img_scale'] = 1.0

    if all_polys_or_rles is not None and len(all_polys_or_rles) > 0 and len(all_polys) == 0:
        all_masks = all_masks.transpose((1, 2, 0))
        all_masks = np.array(all_masks >= 0.5, dtype=np.uint8, order='F')
        all_polys_or_rles = encode(all_masks)
        all_masks = None
    else:
        all_polys_or_rles = all_polys
    
    res_dict['img'] = img
    res_dict['all_boxes'] = all_boxes
    res_dict['all_masks'] = all_masks
    res_dict['all_polys_or_rles'] = all_polys_or_rles
    return res_dict
    
