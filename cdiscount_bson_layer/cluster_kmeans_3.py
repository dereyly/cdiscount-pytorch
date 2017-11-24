from sklearn.cluster import KMeans
import numpy as np
import sys
import os
import pickle as pkl
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cdist

data_dir='/home/dereyly/data_raw/data/domain_feats/'
data_dir_out='/home/dereyly/data_raw/data/domain_corr/'
if not os.path.exists(data_dir_out):
    os.makedirs(data_dir_out)
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
        if dim>3000:
            idx = np.random.choice(dim, 3000)
            data = data[idx]

        data0=data.copy()
        data=normalize(data)
        #n_clusters=int(np.log(dim/300)**2+1.5)
        #n_clusters = int(3*np.log(dim / 200)**2 + 1.2)
        n_clusters = int((np.log(dim / 300.0)+0.5) ** 2 + 1.4)
        print(count,n_clusters)
        if dim>=300:
            kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data)
            d=cdist(data,kmeans.cluster_centers_)
            idx=d.argmax(axis=0)


        feat_out=data0[idx].copy()
        pkl.dump(feat_out,open(data_dir_out+fname,'wb'))
        zz=0