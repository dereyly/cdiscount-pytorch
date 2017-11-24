# --------------------------------------------------------
# Hello PyCaffe
# Data Layer for Scale Augmantation
# LGPL
# Written by Nikolay Sergievsky (dereyly)
# --------------------------------------------------------
import sys
# sys.path.append('/home/dereyly/progs/caffe_cudnn33/python')
# sys.path.append('/usr/lib/python2.7/dist-packages')
# sys.path.insert(0,'/home/dereyly/progs/caffe-master-triplet/python')
# sys.path.insert(0,'/home/dereyly/progs/caffe-elu/python')

import caffe
import numpy as np
import yaml
from google.protobuf import text_format
import pickle as pkl


'''
layer {
  name: "save"
  type: "Python"
  bottom: "feats"
  bottom: "label;
  python_param {
    module: 'save_layer'
    layer: 'Saver'
    }
}

'''

class Saver(caffe.Layer):

    def setup(self, bottom, top):
        self.feats=[]
        self.cls=0
        self.count=0
        self.dir_feats='/home/dereyly/data_raw/data/domain_feats/'
        pass

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        feats=bottom[0].data
        lbl=bottom[1].data
        self.count+=1

        for k in range(feats.shape[0]):
            if self.cls!=lbl[k]:
                print(self.cls,len(self.feats))
                pkl.dump(self.feats, open(self.dir_feats+str(self.cls)+'.pkl', 'wb'))
                self.feats=[]
                self.cls=lbl[k]

            else:
                self.feats.append(feats[k].copy())




    def backward(self, top, propagate_down, bottom):
        # print 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb'
        pass

