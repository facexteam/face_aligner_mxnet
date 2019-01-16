import sys
import os.path as osp

def add_path(path):
    path =  osp.abspath(path)
    if path not in sys.path:
        sys.path.insert(0, path)


# set mxnet path
# mxnet_path = '/opt/mxnet/python'
mxnet_path = '/usr/local/lib/python2.7/dist-packages'
add_path(mxnet_path)
