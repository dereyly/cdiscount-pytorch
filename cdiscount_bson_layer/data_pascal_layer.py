# --------------------------------------------------------
# Hello PyCaffe
# Data Layer for Scale Augmantation
# LGPL
# Written by Nikolay Sergievsky (dereyly)
# --------------------------------------------------------
import sys
sys.path.append('/home/dereyly/progs/caffe-dev_plateau/python/python')
# sys.path.append('/usr/lib/python2.7/dist-packages')
# sys.path.insert(0,'/home/dereyly/progs/caffe-master-triplet/python')
# sys.path.insert(0,'/home/dereyly/progs/caffe-elu/python')
print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++1')
import caffe
print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++2')
import numpy as np
import cv2
from os import listdir
from os.path import isfile, join
import yaml
sys.path.insert(0,'/home/dereyly/progs/orange_people/src/')
from load_pascal_xml import load_pascal_annotation
# from utils.blob import prep_im_for_blob, im_list_to_blob
# from fast_rcnn.config import cfg
# from fast_rcnn.train import get_training_roidb, train_net
# from datasets.factory import get_imdb
import os
import random
'''
layer {
  name: "FakeData"
  type: "Python"
  #bottom: "data"
  top: "data"
  top: 'rois_bbox'
  top: 'label'
  python_param {
    param_str: "{'image_xml_path': '/media/dereyly/data_one/ImageDB/t-shirt/Olya', 'batch_size':20}"
    module: 'data_pascal_layer'
    layer: 'FakeData'
    }
}

'''
def get_data(image_xml_path):
    data=[]
    for dir in os.listdir(image_xml_path):
        path = os.path.join(image_xml_path,dir)
        if os.path.isdir(path):
            for fname in os.listdir(path):
                path_im = os.path.join(path, fname)
                if os.path.isfile(path_im) and fname.split('.')[-1]=='jpg':
                    xml= os.path.join(path,fname.split('.')[0])+'.xml'
                    if not os.path.isfile(xml):
                        continue
                    data.append({'img':path_im,'xml':xml,'lbl':int(dir)})
                    # data['xml'].append(xml)
                    # data['lbl'].append(int(dir))
    return data
def disturb(bb,sz):
    w = bb[2] - bb[0]
    h = bb[3] - bb[1]
    mx = (bb[2] + bb[0])/2
    my = (bb[3] + bb[1])/2
    dx = w * (np.random.rand() * 0.5 - 0.25)
    dy = h * (np.random.rand() * 0.4 - 0.2)
    dw = w* (np.random.rand() * 0.5 - 0.25)
    dh = h * (np.random.rand() * 0.5 - 0.25)
    w+=dw
    h+=dh
    bb[0] = max(0, mx +dx- w/2)
    bb[1] = max(0, my +dy- h/2)
    bb[2] = min(sz[0], mx+dx + w/2)
    bb[3] = min(sz[1], my+dy + h/2)
    return bb

class FakeData(caffe.Layer):
    def get_batch(self):
        if self.iter==0 or self.iter+self.batch_size>=len(self.data):
            self.idxs=range(len(self.data))
            random.shuffle(self.idxs)
            self.iter=0
        idx=self.idxs[self.iter:self.iter+self.batch_size]
        self.iter+=self.batch_size
        data=[]
        for i in range(self.batch_size):
            data.append(self.data[idx[i]])
        return data
    def setup(self, bottom, top):


        #self.num_of_gt=5
        #self.sz=(420,320) #ToDo use config
        #self.widths = [150, 600]
        self.sizes=[(224,224),(320,224),(320,320),(420,320),(640,420)] #,(740,640),(740,420)]
        #self.dir='/media/dereyly/data_fast/ImageDB/VOCDevkit/VOC2007/JPEGImages/'
        layer_params = yaml.load(self.param_str)
        image_xml_path = layer_params['image_xml_path']
        self.batch_size = layer_params['batch_size']
        #Data is 2 lvl structure
        self.data=get_data(image_xml_path)
        self.mean=[104,117,123]
        #self.files = [f for f in listdir(self.dir) if isfile(join(self.dir, f))]
        self.count=0
        self.iter=0
        top[0].reshape(1,3,self.sizes[0][0],self.sizes[0][1])
        top[1].reshape(1,5)
        top[2].reshape(1)
        #top[2].reshape(3)
        pass

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        IX=self.count
        self.count+=1
        data=self.get_batch()
        ix = random.sample(range(len(self.sizes)),1)[0]
        rsz=self.sizes[ix]
        top[0].reshape(self.batch_size,3,rsz[1],rsz[0])
        top[1].reshape(self.batch_size, 5)
        top[2].reshape(self.batch_size)
        #imgs=np.zeros((self.batch_size,self.sz[0],self.sz[0],3),np.float32)
        # rois=np.zeros((self.batch_size,5))
        # lbls=np.zeros(self.batch_size)

        for i in range(self.batch_size):
            im=cv2.imread(data[i]['img']).astype(np.float32)
            im_sz=im.shape
            coeff = np.array([1.0 * im_sz[1] / rsz[0], 1.0 * im_sz[0] / rsz[1], 1.0 * im_sz[1] / rsz[0], 1.0 * im_sz[0] / rsz[1]])
            boxes = load_pascal_annotation(data[i]['xml'])
            bb = np.array(boxes[0], np.float32)
            bb /= coeff
            bb=disturb(bb,rsz)
            im=cv2.resize(im,rsz)
            # cv2.rectangle(im, (bb[0], bb[1]), (bb[2], bb[3]), (255, 0, 0))
            # cv2.imshow("im", im/255)
            # cv2.waitKey(0)
            im-=self.mean
            img = np.transpose(im, (2, 0, 1))

            top[0].data[i]=img
            top[1].data[i]=np.array([i, bb[0],bb[1],bb[2],bb[3]])
            top[2].data[i]=data[i]['lbl']

        #imgs=np.transpose(imgs,(0,3,1,2))
        # top[0].reshape(*imgs.shape)
        # top[0].data[...]=imgs
        # top[1].reshape(*rois.shape)
        # top[1].data[...]=rois
        # top[2].reshape(*lbls.shape)
        # top[2].data[...]=lbls
        '''
        for bb in gt_boxes:
            cv2.rectangle(im,(bb[0],bb[1]),(bb[2],bb[3]), (255,0,0))
        cv2.imshow("im",im)
        cv2.waitKey(0)
        '''
        pass

    def backward(self, top, propagate_down, bottom):
        # print 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb'
        pass

