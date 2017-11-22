# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle

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
input_train=''
data = bson.decode_file_iter(open('/home/dereyly/data_raw/train.bson', 'rb'))
fname_tr='/home/dereyly/data_raw/train2.txt'
fname_tst='/home/dereyly/data_raw/val2.txt'
f_tr=open(fname_tr,'w')
f_tst=open(fname_tst,'w')
prod_to_category = dict()

category_statistic = dict()
with open('category_statistic.pkl', 'rb') as f:
    category_statistic = dict(pickle.load(f))
category_to_ind={}
for id_dict, (key, value) in enumerate(category_statistic.items()):
    category_to_ind[key] = np.int64(id_dict)
print(next(data))
is_val=False
for c, d in enumerate(data):
    if c % 10000 == 0:
        print(c)
    if np.random.rand()<0.01:
        is_val=True
    else:
        is_val=False
    product_id = d['_id']
    category_id = d['category_id'] # This won't be in Test data
    if category_id not in category_statistic:
        category_statistic[category_id] = []
    prod_to_category[product_id] = category_id
    for e, pic in enumerate(d['imgs']):
        #picture = imread(io.BytesIO(pic['picture']))
        # do something with the picture, etc
        category_statistic[category_id].append(product_id)
        str_out=str(product_id) + "_" + str(e) + ".png "+str(category_to_ind[category_id])+'\n'
        if is_val:
            f_tst.write(str_out)
        else:
            f_tr.write(str_out)

with open('category_statistic_example.pkl', 'wb') as f:
    pickle.dump(category_statistic, f)

prod_to_category = pd.DataFrame.from_dict(prod_to_category, orient='index')
prod_to_category.index.name = '_id'
prod_to_category.rename(columns={0: 'category_id'}, inplace=True)

counter = 0
#data = bson.decode_all
z= 0
