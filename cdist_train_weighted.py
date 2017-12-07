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
from cdist_loader_pkl import CDiscountDatasetMy
import pickle as pkl
import sys
sys.path.insert(0,'/home/dereyly/progs/pytorch_cdiscount/main/')
from dataset.transform import *
import cv2

#sys.path.insert(0,'/home/dereyly/progs/pytorch_examples/LSUV-pytorch/')
#from LSUV import LSUVinit
#CUDA_VISIBLE_DEVICES
#CUDA_DEVICE_ORDER
out_dir='/media/dereyly/data/tmp/result/'
schedule=np.array([30,50,260])
dir_im = '/home/dereyly/ImageDB/cdiscount/'
# data_tr_val=open('/home/dereyly/ImageDB/cdiscount/train.pkl','rb')
path_tr='/home/dereyly/ImageDB/cdiscount/train.pkl'
path_val='/home/dereyly/ImageDB/cdiscount/val.pkl'
#--arch=resnet18 /home/dereyly/data_raw/images/train /home/dereyly/data_raw/train2.txt --resume=/home/dereyly/progs/pytorch_examples/imagenet/model_best.pth.tar
# --start-epoch=2
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# parser.add_argument('data', metavar='DIR',
#                     help='path to dataset')
# parser.add_argument('fname_list', metavar='fname',
#                     help='path to list with dataset names and labels')

parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=6, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')

best_prec1 = 0

def train_augment(image):
    image = np.asarray(image,np.float32)
    #im = PIL.Image.fromarray(numpy.uint8(I))
    if random.random() < 0.55:
        image = random_shift_scale_rotate(image,
                  # shift_limit  = [0, 0],
                  shift_limit=[-0.07, 0.07],
                  scale_limit=[0.9, 1.2],
                  rotate_limit=[-10, 10],
                  aspect_limit=[1, 1],
                  # size=[1,299],
                  borderMode=cv2.BORDER_REFLECT_101, u=1)
    elif random.random() < 0.44:
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

def main():
    global args, best_prec1
    args = parser.parse_args()

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    # create model
    # if args.pretrained:
    #     print("=> using pre-trained model '{}'".format(args.arch))
    #     model = models.__dict__[args.arch](pretrained=True)
    # else:
    #     print("=> creating model '{}'".format(args.arch))
    #     model = models.__dict__[args.arch]()
    #model=resnet_mod18(num_classes=[5263,483,49])
    #model = resnet18_multi(num_classes=[5500, 500, 50])
    #model = resnet_mod18(num_classes=6000)
    model = resnet101_fc(pretrained=True, num_classes=6000)

    if not args.distributed:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features) #,device_ids=[0,1])
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
    else:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model) #,device_ids=[0,1])

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # optimizer = torch.optim.Adam(model.parameters(), eps=0.1)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    #traindir = os.path.join(args.data, 'train')
    #valdir = os.path.join(args.data, 'val')
    #
    #normalize = transforms.Normalize(mean=[1, 1, 1],
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = CDiscountDatasetMy(
        dir_im + '/train/', path_tr,
        transform=lambda x: train_augment(x))
        # transform=transforms.Compose([
        #     transforms.RandomSizedCrop(160),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     normalize,
        # ]))

    # if args.distributed:
    #     train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    # else:
    #     train_sampler = None
    # batch_size = 20
    stats = pkl.load(open(dir_im + 'cls_stats_train.pkl', 'rb'))
    # stats2 = pkl.load(open(dir_im + 'cls_stats_train_re.pkl', 'rb'))
    #weigths_cls=1/(np.log(stats/ 15.0+np.exp(1)) ** 2+0.5)
    weigths_cls = 1 / (np.log(stats / 100.0 + np.exp(1)) ** 2)
    weigths_cls/=weigths_cls.max()
    data_tr=pkl.load(open(path_tr,'rb'))
    cls_w=np.zeros(len(data_tr),np.float64)
    for i,data in enumerate(data_tr):
        cls_w[i]=weigths_cls[data[1][0]]

    weights = torch.from_numpy(cls_w)
    train_sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, 2000000)
    #trainloader = data_utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, sampler=sampler)
    #train_sam pler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)


    val_loader = torch.utils.data.DataLoader(
            CDiscountDatasetMy(dir_im+'/train/', path_val,
            transform=transforms.Compose([  #transforms.Scale(256),
            transforms.CenterCrop(160),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch)

        #test_loader(train_loader, 6000)
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    # switch to train mode
    model.train()

    end = time.time()


    log=open(out_dir+'/log.train.txt',mode='a')
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target = target[0].cuda(async=True)
        input_var = torch.autograd.Variable(input)
        # print('calc output')
        output = model(input_var)
        # print('+++++++++++calc output')
        target_var = torch.autograd.Variable(target)

        # if i == 0 and epoch == 0:
        #     model = LSUVinit(model, input_var, needed_std=1.0, std_tol=0.1, max_attempts=10, do_orthonorm=True, cuda=True)
        # compute output

        loss = criterion(output, target_var)
        # measure accuracy and record loss


        prec1, prec5 = accuracy(output.data, target.cuda(), topk=(1, 5))
            #prec1 = accuracy(output[0].data, target) #ToDO WTF not working with top1
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))

        #top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # 'LR: {.3f} \t' \
        lr=optimizer.param_groups[0]['lr']
        if i % args.print_freq == 0:
            str_out='Epoch: [{0}][{1}/{2}]\t' \
                  'LR {lr:f} \t' \
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                  'Multi Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                  'Prec_1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, i,len(train_loader),lr=lr, batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1)
            print(str_out)
            log.write(str_out+'\n')

        if i % 10000 ==0:
            torch.save(model.state_dict(), out_dir + '/checkpoint/%d_%08d_model.pth' % (epoch,i))
            # torch.save(model, 'filename.pt')
            # model = torch.load('filename.pt')

def test_loader(train_loader,num_classes):
    stats=np.zeros(num_classes,np.float32)
    for i, (input, target) in enumerate(train_loader):
        target=target[0].numpy()
        stats[target]+=1
        if i%100==0:
            print(i)
        if i%4000==4000-1:
            break
    pkl.dump(stats,open(dir_im+'cls_stats_train_re.pkl','wb'))
    zz=0
def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    log = open(out_dir + '/log.val.txt', mode='a')
    for i, (input, target) in enumerate(val_loader):
        target = target[0].cuda(async=True)
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

        if i % args.print_freq == 0:
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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    id_pow=np.where(schedule>epoch)[0][0]
    lr = args.lr*(0.1**id_pow)
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = lr
    optimizer.param_groups[0]['lr']=lr

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


if __name__ == '__main__':
    main()
