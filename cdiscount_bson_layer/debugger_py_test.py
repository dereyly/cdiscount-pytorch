__author__ = 'dereyly'
import sys

sys.path.insert(0, '/home/dereyly/progs/caffe-nccl/python')
#sys.path.insert(0, '/home/dereyly/progs/protobuf-3.2.0/python')
#sys.path.insert(0, '/home/dereyly/progs/caffe-ssd/python')
sys.path.insert(0,'/home/dereyly/data_raw/cdiscount_bson_layer')
# sys.path.insert(0,'/media/dereyly/data_one/progs_other/pva-faster-rcnn/lib')
import caffe
import numpy as np
import cv2

# prototxt = '/home/dereyly/data_raw/cdiscount_bson_layer/models/SE-ResNet-50_img.prototxt'
# caffemodel = '/home/dereyly/data_raw/cdiscount_bson_layer/snapshots/senet50_iter_190000.caffemodel'

prototxt='/home/dereyly/data_raw/cdiscount_bson_layer/models/SE-ResNet-50_V1_feats.prototxt'
caffemodel='/home/dereyly/data_raw/cdiscount_bson_layer/snapshots/senet50_domain_iter_65000_nobn_all.caffemodel'
caffe.set_mode_gpu()
net = caffe.Net(prototxt, caffemodel,caffe.TEST)
niter = 100
test_interval = 25
# losses will also be stored in the log
train_loss = np.zeros(niter)
test_acc = np.zeros(int(np.ceil(niter / test_interval)))
output = np.zeros((niter, 8, 10))

# the main solver loop

for it in range(niter):
    net.forward()
    print(it)
    imgs=net.blobs['data'].data
    imgs=np.transpose(imgs,(0,2,3,1))
    imgs+=128
    lbl = net.blobs['label'].data
    # prob = net.blobs['prob'].data
    # cls = prob.argmax(axis=1)
    # for i in range(imgs.shape[0]):
    #     if lbl[i]!=cls[i]:
    #         cv2.imshow('img',imgs[i]/255)
    #         cv2.waitKey(0)
            # ctg2_gt = cls2ctg2[lbl]
            # ctg2_ans = cls2ctg2[cls]
            # ctg1_gt = cls2ctg1[lbl]
            # ctg1_ans = cls2ctg1[cls]
    zz=0
    #sz=feat.shape
    '''
    ax = plt.subplot(111)
    x = np.array(range(sz[1]/5))
    w = 0.3
    # ax.bar(x, soft_max[0],width=w,color='b',align='center')
    ax.bar(x + w, feat[0,:100], width=w, color='r', align='center')
    ax.autoscale(tight=True)
    plt.show()
    '''
