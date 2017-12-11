import pickle as pkl
import numpy as np

pkl_in1='/media/dereyly/data/tmp/result/stats/cls_stats_train_tr_39.pkl'
pkl_in2='/media/dereyly/data/tmp/result/stats/cls_stats_train_tr_40.pkl'

stats1=pkl.load(open(pkl_in1,'rb'))
stats2=pkl.load(open(pkl_in2,'rb'))
dstats=stats2-stats1
# dstats.sort()
# dstats[:] = dstats[::-1]
dstats=dstats[:5500]
idx=np.argsort(-dstats)
sort_dstats = dstats[idx]
# pkl.dump(dstats[:5500],open('/home/dereyly/ImageDB/cdiscount/cls_stats_train.pkl','wb'))
idx2cls=np.zeros((8,5500),int)
idx2cls[0,idx[:300]]=np.array(range(1,301))
idx2cls[1,idx[100:600]]=np.array(range(1,501))
idx2cls[2,idx[300:1000]]=np.array(range(1,701))
idx2cls[3,idx[600:1600]]=np.array(range(1,1001))
idx2cls[4,idx[1000:2500]]=np.array(range(1,1501))
idx2cls[5,idx[1600:3100]]=np.array(range(1,1501))
idx2cls[6,idx[2500:4500]]=np.array(range(1,2001))
idx2cls[7,idx[3100:5500]]=np.array(range(1,2401))
print(dstats[idx[1000:2500]].sum()/dstats[idx[:1000]].sum())
# for k in range(idx2cls.shape[0]):
#     print(idx2cls[k].max())
print(idx2cls.max(axis=1))
pkl.dump(idx2cls,open('/home/dereyly/ImageDB/cdiscount/multi_idx2cls2.pkl','wb'))
zz=0