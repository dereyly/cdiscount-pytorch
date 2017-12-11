import sys
import os
import numpy as mp
import pickle as pkl

pkl_in='/home/dereyly/ImageDB/cdiscount/train_OLD.pkl'
pkl_out='/home/dereyly/ImageDB/cdiscount/train.pkl'
fname_imagenet='/home/dereyly/progs/caffe-nccl/data/ilsvrc12/train.txt'
dir_img='/home/dereyly/ImageDB/imagenet/train/'
dir_img_bson='/home/dereyly/ImageDB/cdiscount/train/'
data_tr=pkl.load(open(pkl_in,'rb'))
fake_name='1000000251/439973_0.png'
for i in range(len(data_tr)):
    # print(data_tr[i][0])
    if not os.path.exists(dir_img_bson+data_tr[i][0]):
        print(data_tr[i][0])
        data_tr[i] = (fake_name, data_tr[i][1])


print(len(data_tr))


pkl.dump(data_tr,open(pkl_out,'wb'))