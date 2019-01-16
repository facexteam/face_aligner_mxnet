import sys
import os
import os.path as osp

# set caffe path
caffe_root = '/opt/caffe/'
# caffe_root = '/workspace/mnt/group/face/shiyongjie/project/caffe/'
sys.path.insert(0, osp.join(caffe_root, 'python'))

# suppress caffe log
os.environ['GLOG_minloglevel'] = '2'
