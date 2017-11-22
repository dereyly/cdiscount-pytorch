import os, sys, math, io
import numpy as np
import pandas as pd
import multiprocessing as mp
import bson
import struct
import io
from PIL import Image
import matplotlib.pyplot as plt
import pickle as pkl
import cv2
#import seaborn as sns

from collections import defaultdict
from tqdm import *

INPUT_PATH='/home/dereyly/ImageDB/cdiscount/'
INPUT_BSON='/media/dereyly/data/data/'

num_dicts = 7069896 # according to data page
length_size = 4
IDS_MAPPING = {}
CATEGORY_NAMES_DF = pd.read_csv(os.path.join(INPUT_PATH, 'category_names.csv'))
TRAIN_DB = bson.decode_file_iter(open(os.path.join(INPUT_BSON, 'train.bson'), 'rb'))
TEST_DB = bson.decode_file_iter(open(os.path.join(INPUT_BSON, 'test.bson'), 'rb'))

# level_tags = CATEGORY_NAMES_DF.columns[1:]
#CATEGORY_NAMES_DF[CATEGORY_NAMES_DF['category_id'] == item['category_id']][level_tags]






# lvl3=CATEGORY_NAMES_DF['category_level3'].unique()
cls2ind={}
# for k,ctg in enumerate(lvl3):
#     cls2ind[ctg]=k
    
lvl2=CATEGORY_NAMES_DF['category_level2'].unique()
ctg2ind={}
for k,ctg in enumerate(lvl2):
    ctg2ind[ctg]=k

lvl1=CATEGORY_NAMES_DF['category_level1'].unique()
ctg1ind={}
for k,ctg in enumerate(lvl1):
    ctg1ind[ctg]=k
cls2ctg1={}
cls2ctg2={}
#mozno bilo vprincipe sploshnikom napisati categorii tipa od id
#ili sdelat vivod po odnomu cls srazu 3 znacheniya
for row in CATEGORY_NAMES_DF.iterrows():
    id=row[0]
    ctg1 = row[1]['category_level1']
    ctg2 = row[1]['category_level2']
    cls = row[1]['category_id']
    cls2ctg1[cls] = ctg1ind[ctg1]
    cls2ctg2[cls] = ctg2ind[ctg2]
    cls2ind[cls]=id
    zz = 0

pkl.dump({'cls2ctg1':cls2ctg1,'cls2ctg2':cls2ctg2, 'cls2ind':cls2ind}, open(INPUT_PATH+'cls2ctg.pkl', 'wb'))
#
#
# num_dicts = 7069896 # according to data page
# prod_to_cls_list = [None] * num_dicts
# prod2cls={}
# cls2prod={}
#
# with tqdm_notebook(total=num_dicts) as bar:
#     TRAIN_DB = bson.decode_file_iter(open(os.path.join(INPUT_BSON, 'train.bson'), 'rb'))
#
#     for i, item in enumerate(TRAIN_DB):
#         bar.update()
#         prod_to_cls_list[i] = (item['_id'], item['category_id'])
#         prod2cls[item['_id']]=item['category_id']
#         if item['category_id'] in cls2prod:
#             cls2prod[item['category_id']].append(item['_id'])
#         else:
#             cls2prod[item['category_id']]=[item['_id']]
# pkl.dump(prod_to_cls_list, open(INPUT_PATH+'prod_to_cls_list.pkl', 'wb'))
# pkl.dump(prod2cls, open(INPUT_PATH+'prod2cls.pkl', 'wb'))
# pkl.dump(cls2prod, open(INPUT_PATH+'cls2prod.pkl', 'wb'))