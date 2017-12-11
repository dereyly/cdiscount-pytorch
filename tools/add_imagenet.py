import sys
import os
import numpy as mp
import pickle as pkl

pkl_in='/home/dereyly/ImageDB/cdiscount/train.pkl'
pkl_out='/home/dereyly/ImageDB/cdiscount/train_imagenet.pkl'
fname_imagenet='/home/dereyly/progs/caffe-nccl/data/ilsvrc12/train.txt'
dir_img='/home/dereyly/ImageDB/imagenet/train/'
dir_img_bson='/home/dereyly/ImageDB/cdiscount/train/'
data_tr=pkl.load(open(pkl_in,'rb'))
for i in range(len(data_tr)):
    # print(data_tr[i][0])
    data_tr[i]=(dir_img_bson+data_tr[i][0],[data_tr[i][1][0], 0])

print(len(data_tr))
with open(fname_imagenet,'r') as f:
    for line in f.readlines():
        line=line.strip()
        name=line.split(' ')[0]
        path=dir_img+name
        if os.path.exists(path):
            cls=[0, int(line.split(' ')[-1])]
            data_tr.append((path,cls))
        else:
            print('skip path     ',path)

pkl.dump(data_tr,open(pkl_out,'wb'))