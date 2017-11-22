import sys
import os
train_dir='/home/dereyly/ImageDB/cdiscount/train/'
val_dir='/home/dereyly/ImageDB/cdiscount/val/'

f_train=open('/home/dereyly/ImageDB/cdiscount/train_sync.txt','w')
f_test=open('/home/dereyly/ImageDB/cdiscount/val_sync.txt','w')

lbl_map={}
lbl=0
for dir_name in os.listdir(train_dir):
    lbl_map[dir_name]=lbl
    path=train_dir+dir_name
    for fname in os.listdir(path):
        f_train.write(dir_name+'/'+fname +' ' +str(lbl)+'\n')
    lbl+=1

for dir_name in os.listdir(val_dir):
    lbl= lbl_map[dir_name]
    path=val_dir+dir_name
    for fname in os.listdir(path):
        f_test.write(dir_name+'/'+fname +' ' +str(lbl)+'\n')