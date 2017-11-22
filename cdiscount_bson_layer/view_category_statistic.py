import pickle
import numpy as np
import matplotlib.pyplot as plt
from skimage.data import imread

print('load pkl')
with open('category_statistic.pkl', 'rb') as f:
    category_statistic = dict(pickle.load(f))
print('pkl loaded')
train_products = []
val_products = []

val_ratio = 0.02
train_ratio = 1 - val_ratio
assert (
train_ratio + val_ratio == 1 and train_ratio >= 0 and train_ratio <= 1 and val_ratio >= 0 and val_ratio <= 1)

for key, value in category_statistic.items():
    products_list_cur = list(set(value))
    train_num = min(max(int(round(len(products_list_cur) * train_ratio)), 1), len(products_list_cur)-1)
    test_num = len(products_list_cur) - train_num
    train_products_cur = products_list_cur[:train_num]
    val_products_cur = products_list_cur[train_num:]
    #print len(products_list_cur), train_num, test_num
    if np.random.rand()<0.001:
        print(key, value)
    #print len(products_list_cur), len(train_products_cur), len(val_products_cur)
    assert(len(products_list_cur) == (len(train_products_cur) + len(val_products_cur)))
    train_products += train_products_cur
    val_products += val_products_cur

train_products = np.array(train_products)
val_products = np.array(val_products)

train_and_val_products = {}
train_and_val_products['train'] = train_products
train_and_val_products['val'] = val_products

with open('train_and_val_products.pkl', 'wb') as f:
    pickle.dump(train_and_val_products, f)