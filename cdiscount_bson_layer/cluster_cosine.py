import numpy as np
import sys
import os
import pickle as pkl
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cdist

def find_clusters(self, dd):
    # cluster=[]
    dd_max = dd.max()
    assign = -np.ones(dd.shape[0], int)
    # max_iter=40
    # while dd.shape[0]>self.k_num:
    # row_ids=range(dd.shape[0])
    clust_list = []
    row_ids = np.array(range(dd.shape[0]), int)
    lim = 3 * int(dd.shape[0] / self.k_num)

    for i in xrange(lim):
        if (row_ids > 0).sum() < 1.8 * self.k_num:
            break
        # id = np.random.randint(dd.shape[0])
        id = random.sample(row_ids[row_ids > 0], 1)[0]

        dd_loc = dd[id]
        ids = np.argsort(dd_loc)
        ids = ids[:self.k_num]
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