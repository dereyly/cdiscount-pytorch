import sys
import os
import numpy as mp
import pickle as pkl

pkl_in='/home/dereyly/data_raw/train.pkl'
pkl_out='/home/dereyly/data_raw/train_idx.pkl'
# dir_img='/home/dereyly/ImageDB/imagenet/train/'
# dir_img_bson='/home/dereyly/ImageDB/cdiscount/train/'
data_tr=pkl.load(open(pkl_in,'rb'))
# fake_name='1000000251/439973_0.png'
# lbl_list=[[] for i in range(5500)]
# for i in range(len(data_tr)):
#     # print(data_tr[i][0])
#     data_tr[i]=(data_tr[i][0],[data_tr[i][1][0], 0])
#     lbl_list[data_tr[i][1][0]].append(data_tr[i][0])
#
# data_tr2=[]
# for i in range(len(lbl_list)):
#     for j in range(len(lbl_list[i])):
#         data_tr2.append((lbl_list[i][j],[i]))



for i in range(len(data_tr)):
    # print(data_tr[i][0])
    data_tr[i]=(data_tr[i][0],[data_tr[i][1][0], i])



pkl.dump(data_tr,open(pkl_out,'wb'))