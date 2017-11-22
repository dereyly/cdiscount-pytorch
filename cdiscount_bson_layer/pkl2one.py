import numpy as np
import sys
import os
import pickle as pkl


pkl_dir='/home/dereyly/data_raw/data/domain_corr/'
pkl_out=pkl_dir+'../domain_corr.pkl'
try:
    feats=pkl.load(open(pkl_out,'rb'))
    print(feats.shape[0])
except:
    pass
count=0
feats_out=[]
for fname in os.listdir(pkl_dir):
    path = pkl_dir + fname
    count += 1
    print(count)
    if os.path.isfile(path):
        data=pkl.load(open(path,'rb'))
        if count==1:
            feats_out=data
        else:
            feats_out=np.vstack((feats_out,data))
pkl.dump(feats_out,open(pkl_out,'wb'))
print(feats_out.shape[0])
zz=0