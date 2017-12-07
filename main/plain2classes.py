import os
import sys
import shutil
import pandas as pd
import time

from dataset.transform import *
from dataset.sampler import *
from utility.file import *
from utility.draw import *

import pickle as pkl

category_statistic=pkl.load(open('/home/dereyly/ImageDB/cdiscount/category_statistic.pkl','rb'))
category_to_ind={}
for id_dict, (key, value) in enumerate(category_statistic.items()):
    category_to_ind[key] = np.int64(id_dict)

CDISCOUNT_DIR = '/home/dereyly/ImageDB/cdiscount'
CDISCOUNT_NUM_CLASSES = 5270
CDISCOUNT_HEIGHT=180
CDISCOUNT_WIDTH =180



split= 'train_id_v0_7019896'
#label to name
category_names_df = pd.read_csv (CDISCOUNT_DIR + '/category_names.csv')
category_names_df['label'] = category_names_df.index

label_to_category_id = dict(zip(category_names_df['label'], category_names_df['category_id']))
category_id_to_label = dict(zip(category_names_df['category_id'], category_names_df['label']))


print('read img list')
t1=time.time()
ids = read_list_from_file(CDISCOUNT_DIR + '/split/' + split, comment='#', func=int)
num_ids = len(ids)
t2 = time.time()
print('t1=',t2-t1)
train_df = pd.read_csv (CDISCOUNT_DIR + '/train_by_product_id.csv')
t3= time.time()
print('t2=', t3 - t2)
#df.columns
#Index(['_id', 'category_id', 'num_imgs'], dtype='object')

start = timer()
df = train_df.reset_index()
df = df[ df['_id'].isin(ids)]
print(len(df['category_id']))
print(len(df['_id']))

# for ctg in df['category_id']:
#     dir_path=CDISCOUNT_DIR+'/train/'+ str(ctg)
#     if not os.path.exists(dir_path):
#         os.mkdir(dir_path)
df = df.reindex(np.repeat(df.index.values, df['num_imgs']), method='ffill')
df['cum_sum' ] = df.groupby(['_id']).cumcount()
print(len(df['category_id']))
print(len(df['_id']))
print(len(df['cum_sum']))
t4=time.time()
print('t3=', t4 - t3)
#df['img_file'] = CDISCOUNT_DIR + '/image/'+ folder + '/' + df['category_id'].astype(str) + '/' +df['_id'].astype(str) + '-' + df['cum_sum'].astype(str)  + '.jpg'
#img_file_src = CDISCOUNT_DIR + '/train/'  + df['_id'].astype(str) + '_' + df['cum_sum'].astype(str) + '.png'
#img_file_dst = CDISCOUNT_DIR + '/train/' + df['category_id'].astype(str) + '/' + df['_id'].astype(str) + '_' + df['cum_sum'].astype(str) + '.jpg'
t5=time.time()
print('t4=', t5 - t4)
for j in range(len(img_file_src)):
    try:
        id1=img_file_src[j].rfind('/')
        #id2 = img_file_dst[j].rfind('_')
        img_file_dst=img_file_src[j][:id1]+'/'+str(df['category_id'][j])+'/'+img_file_src[j][id1:]
        if j==0 or j % 10000==0:
            print(time.time() - t5,img_file_src[j],'--->',img_file_dst)
        os.rename(img_file_src[j],img_file_dst)
    except:
        try:
            id1 = img_file_src[j].rfind('/')
            img_file_dst = img_file_src[j][:id1] + '/' + str(df['category_id'][j]) + '/' + img_file_src[j][id1:]
            if not os.path.exists(img_file_dst):
                print('skip file ==========', img_file_src[j])
        except:
            print('skip file ==========',j)
print('t3=', time.time() - t5)
#print(df['img_file'][0])