from sklearn.cluster import KMeans
import numpy as np
import sys
import os
import pickle as pkl
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cdist

data_dir='/home/dereyly/data_raw/data/domain_feats/'

for fname in os.listdir(data_dir):
    path=data_dir + fname
    if os.path.isfile(path):
        data=pkl.load(open(path,'rb'))
        dim=len(data)
        if dim<300:
            idx=np.random.choice(dim, 1)
            data = np.array(data[idx[0]])
            data = np.squeeze(data, axis=(1, 2))
            data /= np.sqrt((data**2).sum())
            feat_out=data.copy()
            continue
        if dim>3000:
            idx = np.random.choice(dim, 3000)
            data = np.array(data)
            data = np.squeeze(data[idx], axis=(2, 3))
        else:
            data=np.array(data)
            data=np.squeeze(data,axis=(2,3))
        data=normalize(data)
        kmeans = KMeans(n_clusters=2, random_state=0).fit(data)
        d=cdist(data,kmeans.cluster_centers_)
        idx=d.argmax(axis=0)
        feat_out=data[idx].copy()
        zz=0