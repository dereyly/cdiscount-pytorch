import numpy as np
import sys
import os
import pickle as pkl
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cdist

# def find_clusters(self, dd):
    # cluster=[]


data_dir='/home/dereyly/data_raw/data/domain_feats/'
data_dir_out='/home/dereyly/data_raw/data/domain_corr/'
if not os.path.exists(data_dir_out):
    os.makedirs(data_dir_out)


def find_clusters(dd, k_num):
    dd_max = dd.max()
    assign = -np.ones(dd.shape[0], int)
    # max_iter=40
    # while dd.shape[0]>self.k_num:
    # row_ids=range(dd.shape[0])
    clust_list = []
    row_ids = np.array(range(dd.shape[0]), int)
    lim = 3 * int(dd.shape[0] / k_num)

    for i in range(lim):
        if (row_ids > 0).sum() < 1.8 * k_num:
            break
        # id = np.random.randint(dd.shape[0])
        #id = np.random.sample(row_ids[row_ids > 0], 1)[0]
        row_tmp=row_ids[row_ids > 0]
        idd=np.random.randint(0,len(row_tmp))
        id = row_tmp[idd]

        dd_loc = dd[id]
        ids = np.argsort(dd_loc)
        ids = ids[:k_num]
        assign[ids] = i
        ids_del = ids
        dd[:, ids_del] = dd_max + 1
        row_ids[ids_del] = -1
        row_ids[id] = -1
        clust_list.append(i)

    clust_list.append(i)
    assign[row_ids[row_ids > 0]] = i
    # print (assign==-1).sum()
    return assign, np.array(clust_list, int)

count=0
is_mean=True
for fname in os.listdir(data_dir):
    path=data_dir + fname
    count+=1
    if count<0:
        continue

    if os.path.isfile(path):
        data=pkl.load(open(path,'rb'))
        dim=len(data)
        data = np.array(data)
        data = np.squeeze(data, axis=(2, 3))
        if dim<300:
            idx=np.random.choice(dim, 1)
            #data = data[idx]
            feat_out = data[idx].copy()
        if dim>3000:
            idx = np.random.choice(dim, 3000)
            data = data[idx]

        #data=normalize(data)
        #n_clusters=int(np.log(dim/300)**2+1.5)
        #n_clusters = int(3*np.log(dim / 200)**2 + 1.2)
        n_clusters = int((np.log(dim / 300.0)+0.5) ** 2 + 1)
        print(count,n_clusters)
        if dim>=300 and n_clusters>1:
            dd = cdist(data, data, metric='correlation')
            assign_loc, clust_list = find_clusters(dd,int(dim/n_clusters))
            zz=0


        pkl.dump(feat_out,open(data_dir_out+fname,'wb'))
        zz=0

