import argparse
import os
import shutil
import time

import torch
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
from cdist_loader_txt import CDiscountDataset
import sys
sys.path.insert(0,'/home/dereyly/progs/pytorch_examples/LSUV-pytorch/')
from LSUV import LSUVinit

out_dir='/media/dereyly/data_one/tmp/resault/'
#--arch=resnet18 /home/dereyly/data_raw/images/train /home/dereyly/data_raw/train2.txt --resume=/home/dereyly/progs/pytorch_examples/imagenet/checkpoints/checkpoint.pth.tar
# --start-epoch=2



root_data='/home/dereyly/data_raw/images/train'
#pretrained_file='/media/dereyly/data_one/tmp/resault/checkpoint/simple_0.01/00000000_model.pth'
pretrained_file='/home/dereyly/progs/pytorch_examples/imagenet/checkpoints/checkpoint.pth.tar'






def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


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





model=resnet18(num_classes=6000)


#model = torch.nn.parallel.DistributedDataParallel(model) #,device_ids=[0])
checkpoint = torch.load(pretrained_file)
model = torch.nn.DataParallel(model,device_ids=[0]).cuda()
#model.cuda()
#model.load_state_dict(checkpoint)
model.load_state_dict(checkpoint['state_dict'])
# model = torch.load(pretrained_file)

# new_params = model.state_dict()
# new_params.update(updated_params)
# model.load_state_dict(new_params)

for param in model.parameters():
      param.requires_grad = False
# define loss function (criterion) and optimizer
#optionally resume from a checkpoint
cudnn.benchmark = True
# Data loading code
#traindir = os.path.join(args.data, 'train')
#valdir = os.path.join(args.data, 'val')
#
#normalize = transforms.Normalize(mean=[1, 1, 1],
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])



list_test='/home/dereyly/data_raw/val2.txt'
val_loader = torch.utils.data.DataLoader(
        CDiscountDataset(root_data, list_test,6000,
        transform=transforms.Compose([  #transforms.Scale(256),
        transforms.CenterCrop(160),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=256, shuffle=False,
    num_workers=4, pin_memory=True)
# define loss function (criterion) and optimizer
criterion = nn.CrossEntropyLoss().cuda()
validate(val_loader, model, criterion)


