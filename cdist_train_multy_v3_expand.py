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
from cdist_loader_pkl_multi import CDiscountDatasetMy
import sys
sys.path.insert(0,'/home/dereyly/progs/pytorch_examples/imagenet/main')
from dataset.transform import *
from net.model.cdiscount.resnet101_v2  import ResNet101 as Net
import cv2
import pickle as pkl
#CUDA_VISIBLE_DEVICES
#CUDA_DEVICE_ORDER
train_head=True

out_dir='/media/dereyly/data_one/tmp/resault/'
#schedule=np.array([6,10,16,26],np.float32)
schedule=np.array([1,2,16,26],np.float32)

dir_im = '/home/dereyly/data_raw/images/'
path_tr='/home/dereyly/data_raw/train.pkl'
path_val='/home/dereyly/data_raw/val.pkl'
model_path = '/home/dereyly/progs/pytorch_examples/imagenet/checkpoints/resnet101/00243000_model.pth'
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
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
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
def get_examples_weights():
    stats = pkl.load(open(dir_im + 'cls_stats_train.pkl', 'rb'))
    # stats2 = pkl.load(open(dir_im + 'cls_stats_train_re.pkl', 'rb'))
    #weigths_cls=1/(np.log(stats/ 15.0+np.exp(1)) ** 2+0.5)
    weigths_cls = 1 / (np.log(stats / 100.0 + np.exp(1)) ** 2)+0.025
    weigths_cls/=weigths_cls.max()
    data_tr=pkl.load(open(path_tr,'rb'))
    cls_w=np.zeros(len(data_tr),np.float64)
    for i,data in enumerate(data_tr):
        cls_w[i]=weigths_cls[data[1][0]]

    return torch.from_numpy(cls_w), weigths_cls

def train_augment(image):
    image = np.asarray(image, np.float32)
    sz = image.shape
    if random.random() < 0.4:
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
        image = cv2.resize(image, (160, 160))
    else:
        image = random_crop(image, size=(160, 160), u=0.5)
    # if random.random()<0.35:
    #     image = random_brightness(image,u=0.5)
    #     image = random_contrast(image, u=0.5)
    tensor = pytorch_image_to_tensor_transform(image)
    return tensor

def main():
    global args, best_prec1
    args = parser.parse_args()

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)


    #model = resnet18_multi(num_classes=[5500, 500, 50])
    num_classes = [5270, 5270, 301, 501, 701, 1001, 1501, 1501, 2001, 2401]
    cls_weights =[1, 1, 1, 0.6, 0.4, 0.25, 0.15, 0.1, 0.06, 0.04]
    #model = resnet101_multi(pretrained=False, num_classes=num_classes )
    model = Net(in_shape=(3, 160, 160), num_classes=num_classes, extend=True)
    if len(model_path)>0:
        #checkpoint = torch.load(model_path) #,map_location=lambda storage, loc: storage)
        #model.load_state_dict(checkpoint)

        pretrained_state =  torch.load(model_path)
        model_state = model.state_dict()
        print('loading params from',model_path)
        for k, v in pretrained_state.items():
            if not k in model_state:
                k=k[k.find('.')+1:]
            if  k in model_state and v.size() == model_state[k].size():
                model_state[k]=v
                print(k)
            else:
                print('skip -------------> ', k)

        model.load_state_dict(model_state)
        if train_head:
            for param in model.parameters():
                # if len(param.data.shape)==2:
                #     break
                sz= param.data.shape
                if len(sz)==4 and sz[0]==1024 and sz[1]==2048:
                #if len(sz) == 4 and sz[0] == 2048 and sz[1] == 1024:
                    break
                param.requires_grad = False
                print(param.data.shape)

    if not args.distributed:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features,device_ids=[0])
            model.cuda()
        else:
            model = torch.nn.DataParallel(model,device_ids=[0]).cuda()
    else:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[0])

    # define loss function (criterion) and optimizer
    criterions=[]
    for k in range(len(num_classes)):
        weights_down = torch.ones(num_classes[k])
        weights_down[torch.LongTensor([0])] = cls_weights[k]
        criterions.append(nn.CrossEntropyLoss(weights_down).cuda())

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), args.lr,
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


    # if args.distributed:
    #     train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    # else:
    #     train_sampler = None

    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)



    val_dataset= CDiscountDatasetMy(
        dir_im + '/train/', path_val,
        transform=lambda x: train_augment(x))

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterions[0])
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterions, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterions[0])

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


def train(train_loader, model, criterions, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()
    top3 = AverageMeter()
    top4 = AverageMeter()
    # switch to train mode
    model.train()

    end = time.time()


    log=open(out_dir+'/log.train.txt',mode='a')
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)


        input_var = torch.autograd.Variable(input)
        # print('calc output')
        output = model(input_var)
        # print('+++++++++++calc output')
        #target_var=[]
        #loss_milti=[]

        for k in range(len(output)):
            target[k] = target[k].cuda(async=True)
            target_var = torch.autograd.Variable(target[k])
            #loss_milti.append(criterion(output[k], target_var))
            if k==0:
                loss=criterions[k](output[k], target_var)
            else:
                loss+=criterions[k](output[k], target_var)



        # if i == 0 and epoch == 0:
        #     model = LSUVinit(model, input_var, needed_std=1.0, std_tol=0.1, max_attempts=10, do_orthonorm=True, cuda=True)
        # compute output

        #loss = loss_milti[0]+loss_milti[1]+loss_milti[2]
        # measure accuracy and record loss
        prec=[[],[],[],[]]
        # for k in range(len(target)):
        #     prec1, prec5 = accuracy(output[k].data, target[k].cuda(), topk=(1, 5))
        #     prec.append(prec1)

        prec[0], prec5 = accuracy(output[0].data, target[0].cuda(), topk=(1, 5))
        prec[1], prec5 = accuracy(output[1].data, target[1].cuda(), topk=(1, 5))
        prec[2], prec5 = accuracy(output[2].data, target[2].cuda(), topk=(1, 5))
        prec[3], prec5 = accuracy(output[7].data, target[7].cuda(), topk=(1, 5))
            #prec1 = accuracy(output[0].data, target) #ToDO WTF not working with top1
        losses.update(loss.data[0], input.size(0))
        top1.update(prec[0][0], input.size(0))
        top2.update(prec[1][0], input.size(0))
        top3.update(prec[2][0], input.size(0))
        top4.update(prec[3][0], input.size(0))
        #top3.update(prec5[0], input.size(0))
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
                  'Prec_1 {top1.val:.3f} ({top1.avg:.3f})\t' \
                  'Prec_2 {top2.val:.3f} ({top2.avg:.3f})\t' \
                  'Prec_3 {top3.val:.3f} ({top3.avg:.3f})\t' \
                  'Prec_4 {top4.val:.3f} ({top4.avg:.3f})'.format(
                   epoch, i,len(train_loader),lr=lr, batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top2=top2, top3=top3, top4=top4)
            print(str_out)
            log.write(str_out+'\n')

        if i % 10000 ==0:
            torch.save(model.state_dict(), out_dir + '/checkpoint/%d_%08d_model.pth' % (epoch,i))
            # torch.save(model, 'filename.pt')
            # model = torch.load('filename.pt')


def validate(val_loader, model, criterion, weigths_cls=None, batch_size=256):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    if not weigths_cls is None:
        weigths_cls=np.tile(np.sqrt(np.sqrt(1/weigths_cls)), (batch_size, 1))
        # weigths_cls = 2*np.ones((batch_size, 5500),np.float32)
        weigths_cls=torch.from_numpy(weigths_cls)
        tc_weigths_cls=torch.autograd.Variable(weigths_cls.cuda())
    # switch to evaluate mode
    model.eval()

    end = time.time()
    log = open(out_dir + '/log.val.txt', mode='a')
    acc=0
    count=0
    for i, (input, target) in enumerate(val_loader):
        #tg=target[0].numpy()
        target = target[0].cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)[0]
        if not weigths_cls is None:
            output=output*tc_weigths_cls

        # out = output.data.cpu()
        # out=out.numpy()
        # res=out.argmax(axis=1)
        # print(tg[0:10])
        # print(res[0:10])
        # acc+=(tg==res).sum()
        # count+=args.batch_size
        # print('acc=',1.0*acc/count)
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
    lr = args.lr * (0.33 ** id_pow)
    #lr = args.lr*(0.1**id_pow)
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
