import numpy as np
import cv2
import mxnet as mx

def get_image(roi_rec, imgrec, rgb=True):
    if imgrec is not None:
        _, img = mx.recordio.unpack_img(imgrec[roi_rec['imgrec_id']].read_idx(roi_rec['imgrec_idx']), cv2.IMREAD_COLOR)
        if rgb:
            img = img[:, :, ::-1]
    else:
        img = cv2.imread(roi_rec['image'], cv2.IMREAD_COLOR)
        if rgb:
            img = img[:, :, ::-1]
    if 'height' in roi_rec:
        assert roi_rec['height'] == img.shape[0]
    else:
        roi_rec['height'] = img.shape[0]
    if 'width' in roi_rec:
        assert roi_rec['width'] == img.shape[1]
    else:
        roi_rec['width'] = img.shape[1]
    return img  # RGB
    
def tensor_vstack(tensor_list, pad=0):
    """
    vertically stack tensors
    :param tensor_list: list of tensor to be stacked vertically
    :param pad: label to pad with
    :return: tensor with max shape
    """
    ndim = len(tensor_list[0].shape)
    dtype = tensor_list[0].dtype
    islice = tensor_list[0].shape[0]
    dimensions = []
    first_dim = sum([tensor.shape[0] for tensor in tensor_list])
    dimensions.append(first_dim)
    for dim in range(1, ndim):
        dimensions.append(max([tensor.shape[dim] for tensor in tensor_list]))
    if pad == 0:
        all_tensor = np.zeros(tuple(dimensions), dtype=dtype)
    elif pad == 1:
        all_tensor = np.ones(tuple(dimensions), dtype=dtype)
    else:
        all_tensor = np.full(tuple(dimensions), pad, dtype=dtype)
    if ndim == 1:
        for ind, tensor in enumerate(tensor_list):
            all_tensor[ind*islice:(ind+1)*islice] = tensor
    elif ndim == 2:
        for ind, tensor in enumerate(tensor_list):
            all_tensor[ind*islice:(ind+1)*islice, :tensor.shape[1]] = tensor
    elif ndim == 3:
        for ind, tensor in enumerate(tensor_list):
            all_tensor[ind*islice:(ind+1)*islice, :tensor.shape[1], :tensor.shape[2]] = tensor
    elif ndim == 4:
        for ind, tensor in enumerate(tensor_list):
            all_tensor[ind*islice:(ind+1)*islice, :tensor.shape[1], :tensor.shape[2], :tensor.shape[3]] = tensor
    else:
        raise Exception('Sorry, unimplemented.')
    return all_tensor

    
def transform(im, pixel_means, scale=1.0):
    """
    transform into mxnet tensor
    substract pixel size and transform to correct format
    :param im: [height, width, channel] in RGB
    :param pixel_means: [B, G, R pixel means]
    :return: [batch, channel, height, width]
    """
    if len(im.shape) == 3 and im.shape[2] == 3:
        im_tensor = np.zeros((1, 3, im.shape[0], im.shape[1]), dtype=np.float32)
        for i in range(3):
            im_tensor[0, i, :, :] = (im[:, :, i] - pixel_means[2 - i]) * scale
    elif len(im.shape) == 2:
        im_tensor = np.zeros((1, 1, im.shape[0], im.shape[1]), dtype=np.float32)
        im_tensor[0, 0, :, :] = (im[:, :] - pixel_means[0]) * scale
    else:
        raise ValueError("can not transform image successfully")
    return im_tensor
    
def resize(im, target_size, max_size, stride=0, interpolation=cv2.INTER_LINEAR):
    """
    only resize input image to target size and return scale
    :param im: BGR image input by opencv
    :param target_size: one dimensional size (the short side)
    :param max_size: one dimensional max size (the long side)
    :param stride: if given, pad the image to designated stride
    :param interpolation: if given, using given interpolation method to resize image
    :return:
    """
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=interpolation)

    if stride == 0:
        return im, im_scale
    else:
        # pad to product of stride
        im_height = int(np.ceil(im.shape[0] / float(stride)) * stride)
        im_width = int(np.ceil(im.shape[1] / float(stride)) * stride)
        im_channel = im.shape[2]
        padded_im = np.zeros((im_height, im_width, im_channel))
        padded_im[:im.shape[0], :im.shape[1], :] = im
        return padded_im, im_scale
