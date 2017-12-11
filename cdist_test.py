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
sys.path.insert(0,'/home/dereyly/progs/pytorch_cdiscount/main/')
from dataset.transform import *
import cv2
import pickle as pkl
#CUDA_VISIBLE_DEVICES
#CUDA_DEVICE_ORDER
weighted=True
out_dir='/media/dereyly/data/tmp/result/'
batch_size =256
workers=4
dir_im = '/home/dereyly/ImageDB/cdiscount/'
TRAIN_BSON_FILE = '/media/dereyly/data/data/train.bson'



resume='/home/dereyly/progs/cdiscount-pytorch/checkpoint.pth.tar'
#--arch=resnet18 /home/dereyly/data_raw/images/train /home/dereyly/data_raw/train2.txt --resume=/home/dereyly/progs/pytorch_examples/imagenet/model_best.pth.tar
# --start-epoch=2
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))



# best_prec1 = 0
# def get_examples_weights():
#     stats = pkl.load(open(dir_im + 'cls_stats_train.pkl', 'rb'))
#     # stats2 = pkl.load(open(dir_im + 'cls_stats_train_re.pkl', 'rb'))
#     #weigths_cls=1/(np.log(stats/ 15.0+np.exp(1)) ** 2+0.5)
#     weigths_cls = 1 / (np.log(stats / 100.0 + np.exp(1)) ** 2)+0.025
#     weigths_cls/=weigths_cls.max()
#     data_tr=pkl.load(open(path_tr,'rb'))
#     cls_w=np.zeros(len(data_tr),np.float64)
#     for i,data in enumerate(data_tr):
#         cls_w[i]=weigths_cls[data[1][0]]
#
#     return torch.from_numpy(cls_w), weigths_cls

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
    # cv2.imshow('img',image/255)
    # cv2.waitKey(0)
    tensor = pytorch_image_to_tensor_transform(image)
    return tensor

def validate(val_loader, model, criterion, weigths_cls=None, batch_size=256):
    print_freq=20
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    log = open(out_dir + '/log.val.txt', mode='a')
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)[1]

        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            str_out ='Test: [{0}/{1}]\t' \
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t' \
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5)
            print(str_out)
        log.write(str_out + '\n')
    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg





class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count




def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res






if __name__ == "__main__":

    # Chenge this weh running on your machine
    N_TRAIN = 7069896
    BS = 32
    N_THREADS = workers

    # mapping the catigores into 0-5269 range
    meta_dump=dir_im+'/meta_bson.pkl'
    if not os.path.isfile(meta_dump):
        cat2idx, idx2cat = make_category_tables(CATEGS)
        # Scanning the metadata
        meta_data = read_bson(TRAIN_BSON_FILE, N_TRAIN, with_categories=True)
        meta_data.category_id = np.array([cat2idx[ind] for ind in meta_data.category_id])
        #meta_data = meta_data.iloc[np.arange(500)]  # Remove this!!!
        pkl.dump(meta_data,open(meta_dump,'wb'))
    else:
        meta_data = pkl.load(open(meta_dump,'rb'))
        # Dataset and loader

    #train_loader = data_utils.DataLoader(train_dataset, batch_size=BS, num_workers=N_THREADS, shuffle=False)

    # # Let's go fetch some data!
    #pbar = tqdm(total=len(train_loader))
    # for i, (batch, target) in enumerate(train_loader):
    #     pass
    #     pbar.update()
    # pbar.close()



    #model = resnet18_multi(num_classes=[5500, 500, 50])

    model = resnet101_fc(pretrained=True, num_classes=[5500, 5500])


    model = torch.nn.DataParallel(model).cuda()



    criterion = nn.CrossEntropyLoss().cuda()


    # optimizer = torch.optim.Adam(model.parameters(), eps=0.1)

    # optionally resume from a checkpoint
    ### if args.resume:
    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        #optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(resume))

    cudnn.benchmark = True


    #normalize = transforms.Normalize(mean=[1, 1, 1],
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])





    # val_dataset=CDiscountDatasetMy(dir_im+'/train/', path_val,
    #         transform=transforms.Compose([  #transforms.Scale(256),
    #         transforms.CenterCrop(160),
    #         transforms.ToTensor(),
    #         normalize,
    #     ]))
    # val_dataset= CDiscountDatasetMy(
    #     dir_im + '/train/', path_val,
    #     transform=lambda x: train_augment(x))
    train_dataset = CdiscountDatasetBSON(TRAIN_BSON_FILE, meta_data,
                                         transform=lambda x: train_augment(x))
    val_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)


    validate(val_loader, model, criterion)






