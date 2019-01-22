import os
import sys
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'
root_dir = os.path.abspath(__file__).split('experiments')[0]
sys.path.insert(0, root_dir)
third_party_dir = root_dir + '/3rdparty'
sys.path.insert(0, third_party_dir)
