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

prod_to_category = dict()

category_statistic = dict()

print(next(data))

for c, d in enumerate(data):
    if c % 10000 == 0:
        print(c)
    product_id = d['_id']
    category_id = d['category_id'] # This won't be in Test data
    if category_id not in category_statistic:
        category_statistic[category_id] = []
    prod_to_category[product_id] = category_id
    for e, pic in enumerate(d['imgs']):
        #picture = imread(io.BytesIO(pic['picture']))
        # do something with the picture, etc
        category_statistic[category_id].append(product_id)

with open('category_statistic_example.pkl', 'wb') as f:
    pickle.dump(category_statistic, f)

prod_to_category = pd.DataFrame.from_dict(prod_to_category, orient='index')
prod_to_category.index.name = '_id'
prod_to_category.rename(columns={0: 'category_id'}, inplace=True)

counter = 0
#data = bson.decode_all
z= 0
