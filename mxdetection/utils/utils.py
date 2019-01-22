import logging
import os
import cPickle as pickle
import numpy as np
import mxnet as mx
from contextlib import contextmanager

def makedivstride(input, stride=0):
    if stride == 0:
        return input
    else:
        return int(np.ceil(input / float(stride)) * stride)

def create_logger(log_path=None, log_format='%(asctime)-15s %(message)s'):
    if log_path is not None:
        while os.path.exists(log_path):
            log_path = log_path[:-4] + '_.log'
        log_dir = os.path.dirname(log_path)
        if log_dir != '' and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        logger = logging.getLogger()
        logger.handlers = []
        formatter = logging.Formatter(log_format)
        handler = logging.FileHandler(log_path)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logging.basicConfig(level=logging.INFO, format=log_format)
    else:
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logging.basicConfig(level=logging.INFO, format=log_format)
    return logger

def copy_file(src_file, dst_file):
    mx.filestream.copy2(src_file, dst_file)
    
def get_kv(num_devices):
    if num_devices > 1:
        if num_devices >= 4:
            kv = mx.kvstore.create('device')
        else:
            kv = mx.kvstore.create('local')
    else:
        kv = None
    return kv

def load_param(prefix, epoch):
    save_dict = mx.nd.load('%s-%04d.params' % (prefix, epoch))
    arg_params = {}
    aux_params = {}
    for k, v in save_dict.items():
        tp, name = k.split(':', 1)
        if tp == 'arg':
            arg_params[name] = v
        if tp == 'aux':
            aux_params[name] = v
    return arg_params, aux_params

def serialize(obj):
    """Serialize a Python object using pickle and encode it as an array of
    float32 values so that it can be feed into the workspace. See deserialize().
    """
    return np.frombuffer(pickle.dumps(obj), dtype=np.uint8).astype(np.float32)

def load_param(prefix, epoch):
    save_dict = mx.nd.load('%s-%04d.params' % (prefix, epoch))
    arg_params = {}
    aux_params = {}
    for k, v in save_dict.items():
        tp, name = k.split(':', 1)
        if tp == 'arg':
            arg_params[name] = v
        if tp == 'aux':
            aux_params[name] = v
    return arg_params, aux_params


def deserialize(arr):
    """Unserialize a Python object from an array of float32 values fetched from
    a workspace. See serialize().
    """
    return pickle.loads(arr.astype(np.uint8).tobytes())

def save_roidb(data, save_roidb_path):
    with mx.filestream.writer(save_roidb_path) as fid:
        pickle.dump(data, fid, cPickle.HIGHEST_PROTOCOL)

@contextmanager
def open_file(path, mode=None):
    tmp_path = None
    if 'hdfs://' in path:
        tmp_path = os.path.basename(path)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        copy_command = 'hdfs dfs -get %s %s' % (path, tmp_path)
        logging.info(copy_command)
        os.system(copy_command)
        r = open(tmp_path, mode)
    else:
        r = open(path, mode)
    try:
        yield r
    finally:
        r.close()
        if tmp_path is not None:
            os.remove(tmp_path)
