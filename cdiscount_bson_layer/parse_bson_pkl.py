# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle as pkl

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
#print(check_output(["ls", "data/"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

import io
import bson                       # this is installed with the pymongo package
import matplotlib.pyplot as plt
from skimage.data import imread   # or, whatever image library you prefer
import multiprocessing as mp      # will come in handy due to the size of the data

# Simple data processing
is_plain_paths=False
input_train=''
data_bson = bson.decode_file_iter(open('/media/dereyly/data/data/train.bson', 'rb'))
cls2ctg_pkl=open('/home/dereyly/ImageDB/cdiscount/cls2ctg.pkl','rb')

pkl_tr=open('/home/dereyly/ImageDB/cdiscount/train.pkl','wb')
pkl_tst=open('/home/dereyly/ImageDB/cdiscount/val.pkl','wb')
# data={'train':[],'val':[]}
data_tr=[]
data_tst=[]
cls_map=pkl.load(cls2ctg_pkl)
print(next(data_bson))
is_val=False
for c, d in enumerate(data_bson):
    if c % 10000 == 0:
        print(c)
    if np.random.rand()<0.01:
        is_val=True
    else:
        is_val=False
    product_id = d['_id']
    category_id = d['category_id']


    for e, pic in enumerate(d['imgs']):
        #picture = imread(io.BytesIO(pic['picture']))
        # do something with the picture, etc
        data_line={}
        if is_plain_paths:
            path =  str(product_id) + "_" + str(e) + '.png'
        else:
            path=str(category_id)+'/'+str(product_id) + "_" + str(e)+'.png'
        cls=[cls_map['cls2ind'][category_id], cls_map['cls2ctg2'][category_id],cls_map['cls2ctg1'][category_id]]
        if not is_val:
            data_tr.append((path,cls))
        else:
            data_tst.append((path,cls))

print(len(data_tr),len(data_tst))
pkl.dump(data_tr,pkl_tr)
pkl.dump(data_tst,pkl_tst)