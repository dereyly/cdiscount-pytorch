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
out_file='/home/dereyly/ImageDB/cdiscount/ctg_dict_train.pkl'
cache_file='/home/dereyly/ImageDB/cdiscount/mapping_bson_train.pkl'
cache_file2='/home/dereyly/ImageDB/cdiscount/stats_bson_train.pkl'
cache_file3='/home/dereyly/ImageDB/cdiscount/category2_bson_train.pkl'
num_dicts = 7069896 # according to data page
length_size = 4
IDS_MAPPING = {}
CATEGORY_NAMES_DF = pd.read_csv(os.path.join(INPUT_PATH, 'category_names.csv'))
TRAIN_DB = bson.decode_file_iter(open(os.path.join(INPUT_BSON, 'train.bson'), 'rb'))
TEST_DB = bson.decode_file_iter(open(os.path.join(INPUT_BSON, 'test.bson'), 'rb'))

level_tags = CATEGORY_NAMES_DF.columns[1:]
#CATEGORY_NAMES_DF[CATEGORY_NAMES_DF['category_id'] == item['category_id']][level_tags]

def decode(data):
    arr = np.asarray(bytearray(data), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)



def decode_pil(data):
    return Image.open(io.BytesIO(data))



def decode_images(item_imgs):
    nx = 2 if len(item_imgs) > 1 else 1
    ny = 2 if len(item_imgs) > 2 else 1
    composed_img = np.zeros((ny * 180, nx * 180, 3), dtype=np.uint8)
    for i, img_dict in enumerate(item_imgs):
        img = decode(img_dict['picture'])
        h, w, _ = img.shape
        xstart = (i % nx) * 180
        xend = xstart + w
        ystart = (i // nx) * 180
        yend = ystart + h
        composed_img[ystart:yend, xstart:xend] = img
    return composed_img

if not os.path.exists(cache_file):
    with open(os.path.join(INPUT_BSON, 'train.bson'), 'rb') as f, tqdm_notebook(total=num_dicts) as bar:
        item_data = []
        offset = 0
        while True:
            bar.update()
            f.seek(offset)

            item_length_bytes = f.read(length_size)
            if len(item_length_bytes) == 0:
                break
                # Decode item length:
            length = struct.unpack("<i", item_length_bytes)[0]

            f.seek(offset)
            item_data = f.read(length)
            assert len(item_data) == length, "%i vs %i" % (len(item_data), length)

            # Check if we can decode
            item = bson.BSON.decode(item_data)

            IDS_MAPPING[item['_id']] = (offset, length)
            offset += length
    pkl.dump(IDS_MAPPING,open(cache_file,'wb'))
else:
    IDS_MAPPING=pkl.load(open(cache_file,'rb'))
    zz=0
    # IDS_MAPPING=


def get_item(item_id):
    assert item_id in IDS_MAPPING
    with open(os.path.join(INPUT_PATH, 'train.bson'), 'rb') as f:
        offset, length = IDS_MAPPING[item_id]
        f.seek(offset)
        item_data = f.read(length)
        return bson.BSON.decode(item_data)

# item = get_item(1234)
#
# mask = CATEGORY_NAMES_DF['category_id'] == item['category_id']
# cat_levels = CATEGORY_NAMES_DF[mask][level_tags].values.tolist()[0]
# cat_levels = [c[:25] for c in cat_levels]
# title = str(item['category_id']) + '\n'
# title += '\n'.join(cat_levels)
# plt.title(title)
# plt.imshow(decode_images(item['imgs']))
# _ = plt.axis('off')
#
# zz=0

# for cls in CATEGORY_NAMES_DF['category_level3']:
#     mask=CATEGORY_NAMES_DF['category_id'] == cls
#     ctg= CATEGORY_NAMES_DF[mask]['category_level2']
#     ctgg=ctg.values.tolist()[0]
#     ctg_id=ctg.axes[0][0]
#     clsl2ctg2[ctg_id]=cls
#     zz=0
clsl2ctg2={}
lvl2=CATEGORY_NAMES_DF['category_level2'].unique()
ctg2ind={}
for k,ctg in enumerate(lvl2):
    ctg2ind[ctg]=k

for row in CATEGORY_NAMES_DF.iterrows():
    # mask = CATEGORY_NAMES_DF['category_id'] == cls
    id=row[0]
    ctg = row[1]['category_level2']
    #cls = row[1]['category_level3']
    cls = row[1]['category_id']
    # ctgg = ctg.values.tolist()[0]
    # ctg_id = ctg.axes[0][0]
    clsl2ctg2[cls] = ctg2ind[ctg]
    zz = 0
print("Unique categories: ", len(CATEGORY_NAMES_DF['category_id'].unique()))
print("Unique level 1 categories: ", len(CATEGORY_NAMES_DF['category_level1'].unique()))
print("Unique level 2 categories: ", len(CATEGORY_NAMES_DF['category_level2'].unique()))
print("Unique level 3 categories: ", len(CATEGORY_NAMES_DF['category_level3'].unique()))


# plt.figure(figsize=(12,12))
# _ = sns.countplot(y=CATEGORY_NAMES_DF['category_level1'])
# plt.show()

num_dicts = 7069896 # according to data page
prod_to_category = [None] * num_dicts
prod_to_category_lvl2 = [None] * num_dicts
if not os.path.exists(cache_file2):
    with tqdm_notebook(total=num_dicts) as bar:
        TRAIN_DB = bson.decode_file_iter(open(os.path.join(INPUT_BSON, 'train.bson'), 'rb'))

        for i, item in enumerate(TRAIN_DB):
            bar.update()
            prod_to_category[i] = (item['_id'], item['category_id'])
            prod_to_category_lvl2[i] = (item['_id'], clsl2ctg2[item['category_id']])
    pkl.dump(prod_to_category, open(cache_file2, 'wb'))
    pkl.dump(prod_to_category_lvl2, open(cache_file3, 'wb'))
else:
    prod_to_category = pkl.load(open(cache_file2, 'rb'))
    prod_to_category_lvl2 = pkl.load(open(cache_file3, 'rb'))
    zz = 0

#TRAIN_CATEGORIES_DF = pd.DataFrame(prod_to_category_lvl2, columns=['_id', 'category_id'])
TRAIN_CATEGORIES_DF = pd.DataFrame(prod_to_category, columns=['_id', 'category_id'])
TRAIN_CATEGORIES_DF.head()

train_categories_gb = TRAIN_CATEGORIES_DF.groupby('category_id')
train_categories_count = train_categories_gb['category_id'].count()
print(train_categories_count.describe())
#ctg2_stats=train_categories_count.values
cls_stats=train_categories_count.values
pkl.dump(cls_stats, open(INPUT_PATH+'cls_stats.pkl', 'wb'))
# count=0
# for stat in ctg2_stats:
#     if stat>10000:
#        count+=1
# print('count=', count)
# cls_ctg_filter={}
# for key, val in clsl2ctg2.items():
#     if ctg2_stats[val]>10000:
#         cls_ctg_filter[key]=val
#
# pkl.dump(cls_ctg_filter, open(out_file, 'wb'))
# zz=0