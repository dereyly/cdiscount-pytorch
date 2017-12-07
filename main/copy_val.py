import os
import sys
import shutil
import pickle as pkl

prod2cls=pkl.load(open('/home/dereyly/ImageDB/cdiscount/prod2cls.pkl','rb'))
fname_val='/home/dereyly/ImageDB/cdiscount/split/valid_id_v0_5000'
img_dir='/home/dereyly/ImageDB/cdiscount/'
with open(fname_val,'r') as f:
    for line in f.readlines():
        line=line.strip()
        id=int(line)
        cls=prod2cls[id]
        for i in range(5):
            path_src=img_dir+'/train/'+ str(cls) + '/' + str(id) +'_' + str(i)+'.png'
            path_dst=img_dir+'/val/'+ str(cls) + '/' + str(id) +'_' + str(i)+'.png'
            dir_dst=img_dir+'/val/'+ str(cls)
            if os.path.exists(path_src):
                if not os.path.exists(dir_dst):
                    os.makedirs(dir_dst)
                shutil.copy(path_src,path_dst)
