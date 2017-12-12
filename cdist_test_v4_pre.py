import argparse
import os
import shutil
import time
import numpy as np
import torch
print(torch.__version__)
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from model.resnet_mod import *
#from cdist_loader_pkl import CDiscountDatasetMy
from cdist_loader_bson import *
import sys
sys.path.insert(0,'/home/dereyly/progs/pytorch_examples/imagenet/main')
from dataset.transform import *
from net.model.cdiscount.resnet101  import ResNet101 as Net
import cv2
import pickle as pkl
# import io
# from skimage.io import imread, imsave
# import PIL
import bson
#CUDA_VISIBLE_DEVICES
#CUDA_DEVICE_ORDER
train_head=True

dir_im = '/home/dereyly/data_raw/images/'

model_path = '/home/dereyly/progs/pytorch_examples/imagenet/checkpoints/resnet101/00243000_model.pth'



is_val =False
test_out = '/media/dereyly/data_one/tmp/resault/12dec/'
out_dir='/media/dereyly/data/tmp/result/'
batch_size =256
workers=4
dir_im = '/home/dereyly/ImageDB/cdiscount/'
if is_val:
    bson_fname = '/home/dereyly/data_raw/train.bson'
else:
    bson_fname = '/home/dereyly/data_raw/test.bson'

cls2ctg_pkl='/home/dereyly/data_raw/cls2ctg.pkl'
cls_map=pkl.load(open(cls2ctg_pkl,'rb'))
re_cls_map={}
for key,val in cls_map['cls2ind'].items():
    re_cls_map[val]=key


#--arch=resnet18 /home/dereyly/data_raw/images/train /home/dereyly/data_raw/train2.txt --resume=/home/dereyly/progs/pytorch_examples/imagenet/model_best.pth.tar
# --start-epoch=2
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


# def prepare_batch(img_batch):
#     for
def valid_augment(image):
    #image = fix_resize(image, 224, 224)

    height,width=image.shape[0:2]
    w,h=160,160

    x0 = (height-h)//2
    y0 = (width -w)//2
    x1 = x0+w
    y1 = y0+h

    roi = (x0,y0,x1,y1) #crop center
    image = fix_crop(image, roi)

    return image

def single_test_augment(image):
    image = np.asarray(image, np.float32)
    #image = cv2.resize(image,(160,160))
    image = fix_center_crop(image)
    tensor = pytorch_image_to_tensor_transform(image)
    return tensor


def multi_test_augment(image):
    images = fix_multi_crop(image, roi_size=(224,224))
    tensors=[]
    for image in images:
        tensor = pytorch_image_to_tensor_transform(image)
        tensors.append(tensor)

    return tensors

def rand_test_augment(image):
    image = np.asarray(image,np.float32)
    sz=image.shape
    if random.random() < 0.55:
        image = random_shift_scale_rotate(image,
                  # shift_limit  = [0, 0],
                  shift_limit=[-0.07, 0.07],
                  scale_limit=[0.85, 1.2],
                  rotate_limit=[-15, 15],
                  aspect_limit=[1, 1],
                  # size=[1,299],
                  borderMode=cv2.BORDER_REFLECT_101, u=1)


    image = random_horizontal_flip(image, u=0.5)
    if random.random() < 0.33:
        image = cv2.resize(image, (160,160))
    else:
        image = random_crop(image, size=(160, 160), u=0.5)
    # if random.random()<0.35:
    #     image = random_brightness(image,u=0.5)
    #     image = random_contrast(image, u=0.5)
    tensor = pytorch_image_to_tensor_transform(image)
    return tensor

def train_augment(image):
    image = np.asarray(image,np.float32)
    sz=image.shape
    skip_aug=False
    if min(sz[0],sz[1])<160:
        rsz=180
        skip_aug = True
        if sz[0]>sz[1]:
            new_sz = (rsz, rsz * sz[0] / sz[1])
        else:
            new_sz = (rsz * sz[1] / sz[0], rsz)
        image=cv2.resize(image,new_sz)
    #im = PIL.Image.fromarray(numpy.uint8(I))
    if not skip_aug:
        if random.random() < 0.55:
            image = random_shift_scale_rotate(image,
                      # shift_limit  = [0, 0],
                      shift_limit=[-0.07, 0.07],
                      scale_limit=[0.9, 1.2],
                      rotate_limit=[-10, 10],
                      aspect_limit=[1, 1],
                      # size=[1,299],
                      borderMode=cv2.BORDER_REFLECT_101, u=1)
        elif random.random() < 0.30:
            image = random_shift_scale_rotate(image,
                      # shift_limit  = [0, 0],
                      shift_limit=[-0.1, 0.1],
                      scale_limit=[0.75, 1.3],
                      rotate_limit=[-90, 90],
                      aspect_limit=[1, 1],
                      # size=[1,299],
                      borderMode=cv2.BORDER_REFLECT_101, u=1)
            # cv2.imshow('img', image)
            # cv2.waitKey(0)
        else:
            pass
    # flip  random ---------
    image = random_horizontal_flip(image, u=0.5)
    image = random_crop(image, size=(160, 160), u=0.8)
    # if random.random()<0.35:
    #     image = random_brightness(image,u=0.5)
    #     image = random_contrast(image, u=0.5)
    #cv2.imshow('img',image/255)
    #cv2.waitKey(0)
    #cv2.imwrite('/home/dereyly/ImageDB/res.jpg',image)
    tensor = pytorch_image_to_tensor_transform(image)
    return tensor



#data_tst=[]
# data_dbg=pkl.load(open('/home/dereyly/ImageDB/cdiscount/dbg_val.pkl','rb'))
def read_batch_data(data_batch, test = False):
    img_batch = []
    targets = []
    indexes = []
    count = 0
    for c, d in enumerate(data):
        count += 1
        product_id = d['_id']
        if not test:
            category_id = d['category_id']  # This won't be in Test data
        if count > data_batch - len(d['imgs']):
            break
        for e, pic in enumerate(d['imgs']):
            #img = imread(io.BytesIO(pic['picture']))
            #
            img = cv2.imdecode(np.fromstring(pic['picture'], dtype=np.uint8), cv2.IMREAD_COLOR)
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            img_batch.append(img.copy())
            if not test:
                targets.append(cls_map['cls2ind'][category_id])
            indexes.append(int(product_id))
    if test:
        return img_batch, indexes
    return img_batch, targets, indexes

if __name__ == "__main__":

    #model = resnet101_fc(pretrained=True, num_classes=[5500, 5500])
    model = Net(in_shape=(3, 160, 160), num_classes=[5270], extend=False)
    if len(model_path) > 0:
        # checkpoint = torch.load(model_path) #,map_location=lambda storage, loc: storage)
        # model.load_state_dict(checkpoint)

        pretrained_state = torch.load(model_path)
        model_state = model.state_dict()
        print('loading params from', model_path)
        for k, v in pretrained_state.items():
            if not k in model_state:
                k = k[k.find('.') + 1:]
            if k in model_state and v.size() == model_state[k].size():
                model_state[k] = v
                print(k)
            else:
                print('skip -------------> ', k)

        model.load_state_dict(model_state)
    model = torch.nn.DataParallel(model).cuda()
    model.eval()
    criterion = nn.CrossEntropyLoss().cuda()


    # optionally resume from a checkpoint
    ### if args.resume:
    # if os.path.isfile(resume):
    #     print("=> loading checkpoint '{}'".format(resume))
    #     checkpoint = torch.load(resume)
    #     start_epoch = checkpoint['epoch']
    #     best_prec1 = checkpoint['best_prec1']
    #     model.load_state_dict(checkpoint['state_dict'])
    #     #optimizer.load_state_dict(checkpoint['optimizer'])
    #     print("=> loaded checkpoint '{}' (epoch {})"
    #           .format(resume, checkpoint['epoch']))
    # else:
    #     print("=> no checkpoint found at '{}'".format(resume))

    cudnn.benchmark = True


    #normalize = transforms.Normalize(mean=[1, 1, 1],
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    global data
    data = bson.decode_file_iter(open(bson_fname, 'rb'))
    softmax=torch.nn.Softmax()
    tmp_target_batch=-np.ones(batch_size)
    index_batch=-np.ones(batch_size)
    acc=0.0
    count =0
    batch_n=0
    acc2=0
    for z in range(1200000):
        batch_n=0
        if is_val:
            img_batch, targets, indexes = read_batch_data(512, not is_val)
        else:
            img_batch, indexes = read_batch_data(512, not is_val)
        if is_val:
            target_dict = {}
        out_dict={}
        if len(img_batch)==0:
            break
        input_batch=torch.FloatTensor(batch_size,3,160,160)
        k = 0
        for i in range(len(img_batch)):
            for r in range(1):
                #tensor = rand_test_augment(img_batch[i].copy())
                tensor = single_test_augment(img_batch[i])
                input_batch[k]=tensor
                index_batch[k]=indexes[i]
                # tmp_target_batch[k]=targets[i]
                if is_val:
                    target_dict[indexes[i]] = targets[i]
                k+=1
                if k>=batch_size or i==len(img_batch)-1:
                    input_var = torch.autograd.Variable(input_batch.cuda(), volatile=True)
                    output = model(input_var)
                    output=softmax(output)
                    out = output.data.cpu()
                    out=out.numpy()
                    for j in range(k):
                        #print(out[j].argmax(),tmp_target_batch[j])
                        idx=index_batch[j]
                        if idx in out_dict:
                            #out_dict[idx] = np.vstack((out_dict[idx],out[j])) # np.max(out_dict[idx],out[j])
                            out_dict[idx] += out[j]
                            #out_dict[idx] += np.maximum(out_dict[idx],out[j])
                        else:
                            out_dict[idx] = out[j]

                        zz=0
                    k = 0
                    input_batch = torch.FloatTensor(batch_size, 3, 160, 160)
                    break
            zz=0
        if is_val:
            for key, val in target_dict.items():
                count+=1
                # if len(out_dict[key].shape)>1:
                #     res=out_dict[key].argmax(axis=1)
                #     if (res==val).any():
                #         acc+=1
                #     zz=0
                # else:
                res=out_dict[key].argmax()
                if res==val:
                    acc+=1
            print('batch_n =%d accuracy=%f acc2=%f',batch_n,acc/count,acc2/len(img_batch))
        #print(indexes)
        else:
            print(z)
            with open(test_out+str(z)+'.txt','w') as f_out:
                for key, val in out_dict.items():
                    res = val.argmax()
                    prd_ind=re_cls_map[res]
                    f_out.write(str(int(key))+','+str(prd_ind)+'\n')







