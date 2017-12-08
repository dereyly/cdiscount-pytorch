import pickle as pkl
import numpy as np

pkl_in='/media/dereyly/data/tmp/result/stats/cls_stats_train_tr_20.pkl'
pkl_in1='/media/dereyly/data/tmp/result/stats/cls_stats_train_tr_39.pkl'
pkl_in2='/media/dereyly/data/tmp/result/stats/cls_stats_train_tr_40.pkl'
stats=pkl.load(open(pkl_in,'rb'))/(20/6)
stats_im=stats[5500:]
print(stats[:5500].sum(),stats[5500:].sum(),stats.sum()/stats[5500:].sum())
stats1=pkl.load(open(pkl_in1,'rb'))
stats2=pkl.load(open(pkl_in2,'rb'))
dstats=stats2-stats1
pkl.dump(dstats[:5500],open('/home/dereyly/ImageDB/cdiscount/cls_stats_train.pkl','wb'))
zz=0