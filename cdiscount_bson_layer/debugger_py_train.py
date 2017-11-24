__author__ = 'dereyly'
import sys

sys.path.insert(0, '/home/dereyly/progs/caffe-nccl/python')
#sys.path.insert(0, '/home/dereyly/progs/protobuf-3.2.0/python')
#sys.path.insert(0, '/home/dereyly/progs/caffe-ssd/python')
sys.path.insert(0,'/home/dereyly/data_raw/cdiscount_bson_layer')
# sys.path.insert(0,'/media/dereyly/data_one/progs_other/pva-faster-rcnn/lib')
import caffe
import numpy as np
import os
import matplotlib.pyplot as plt

# os.chdir('/media/dereyly/data_one/progs/HistogramLoss-master')

caffe.set_mode_gpu()
#/home/dereyly/data_raw/cdiscount_bson_layer/models/solver.prototxt --weights=/home/dereyly/progs/pva-faster-rcnn-master/models/pvanet/imagenet/pva9.1_preAct_train_iter_1900000.caffemodel
solver_name='/home/dereyly/data_raw/cdiscount_bson_layer/models/solver.prototxt'

weights = '/home/dereyly/progs/pva-faster-rcnn-master/models/pvanet/imagenet/pva9.1_pretrained_no_fc6.caffemodel'
solver = caffe.SGDSolver(solver_name)
#net = solver.net
#solver.net=net

solver.net.copy_from(weights)

niter = 30000
test_interval = 25
# losses will also be stored in the log
train_loss = np.zeros(niter)
test_acc = np.zeros(int(np.ceil(niter / test_interval)))
output = np.zeros((niter, 8, 10))


# the main solver loop
for it in range(niter):

    #soft=solver.net.blobs['ip1/soft'].data
    #conv1_param=solver.net.params['ip1/soft'][0].data



    solver.step(1)  # SGD by Caffe

    # ip3_w=solver.net.params['ip3'][0].data
    # sz=ip3_w.shape
    # ax = plt.subplot(111)
    # x = np.array(range(sz[1]))
    # w = 0.3
    # ax.bar(x, ip3_w[0],width=w,color='b',align='center')
    # ax.bar(x + w, ip3_w[1], width=w, color='r', align='center')
    # ax.autoscale(tight=True)
    # plt.show()

    zz=0

