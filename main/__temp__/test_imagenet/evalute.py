from common import *
from utility.draw import *

from dataset.tool import *
from imagenetdataset import *

import torchvision.models as models
from PIL import Image


#from net.model.imagenet.resnet50 import *
#from net.model.excited_resnet50 import *
from net.model.dilated_resnet50 import *


# https://github.com/soeaver/caffe-model
# https://github.com/hujie-frank/SENet
#  SE-ResNet-50   22.37	   6.36  (ILSVRC 2017)

## https://github.com/fyu/drn/
#  DRN-D-54   21.2%	5.9%

# http://pytorch.org/docs/master/torchvision/models.html
#                 top1    top5 err
#   ResNet-34	  26.70	  8.58
#   ResNet-50	  23.85	  7.13

#################################################################################################

## resnet50 (0.248960, 0.076020) : '/root/share/data/models/pytorch/imagenet/resenet/resnet50-19c8e357.pth'
## dilated-resnet50 ( 0.220300, 0.062520) : '/root/share/data/models/reference/imagenet/dilated_resenet/drn_d_54-0e0534ff.pth'

def pytorch_image_to_tensor_transform(image):
    mean = [0.485, 0.456, 0.406 ]
    std  = [0.229, 0.224, 0.225 ]
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = image.transpose((2,0,1))
    tensor = torch.from_numpy(image).float().div(255)
    tensor[0] = (tensor[0] - mean[0]) / std[0]
    tensor[1] = (tensor[1] - mean[1]) / std[1]
    tensor[2] = (tensor[2] - mean[2]) / std[2]
    return tensor



# se_resnet
# def pytorch_image_to_tensor_transform(image):
#     #  https://github.com/hujie-frank/SENet/blob/master/models/SE-ResNet-50.prototxt
#     mean  = [104, 117, 123]
#     image = image.astype(np.float32) - np.array(mean, np.float32)
#     image = image.transpose((2,0,1))
#     tensor = torch.from_numpy(image)
#     return tensor



## imagenet
def set_smallest_size_transform(img, size):
    height, width = img.shape[0:2]
    if (width <= height and width == size) or (height <= width and height == size):
        pass
    if width > height:
       dsize = (int(round(size * width/height)), size)  #width/height
    else:
       dsize = (size,  int(round(size * height/width ))) #height/width
    img = cv2.resize(img, dsize, interpolation=cv2.INTER_LINEAR)
    return img

def crop_center_transform(img, size):
    height, width = img.shape[0:2]
    h, w = size

    y0 = int(round((height-h)/2.))
    y1 = y0 + h
    x0 = int(round((width-w)/2.))
    x1 = x0 + w

    return img[y0:y1, x0:x1, :]

#################################################################################################


# loss, accuracy
def top_accuracy(probs, labels, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = labels.size(0)

    values, indices = probs.topk(maxk, dim=1, largest=True,  sorted=True)
    indices  = indices.t()
    corrects = indices.eq(labels.view(1, -1).expand_as(indices))

    accuracy = []
    for k in topk:
        # https://stackoverflow.com/questions/509211/explain-slice-notation
        # a[:end]      # items from the beginning through end-1
        c = corrects[:k].view(-1).float().sum(0, keepdim=True)
        accuracy.append(c.mul_(1. / batch_size))
    return accuracy



def do_valid(net, valid_loader):

    valid_acc1 = 0
    valid_acc5 = 0
    valid_loss = 0
    valid_num  = 0

    for it, (images, labels, indices) in enumerate(valid_loader, 0):
        images = Variable(images.cuda(),volatile=True)#.half() ##
        labels = Variable(labels.cuda(),volatile=True)#.half() ##

        # img=pytorch_tensor_to_image(images.data.cpu()[0])
        # im_show('img',img,1)
        # cv2.waitKey(0)

        # forward
        logits = net(images)
        probs  = F.softmax(logits)

        #loss = criterion(logits, labels)
        accuracy = top_accuracy(probs, labels, topk=(1,5))

        batch_size = len(indices)
        valid_num += batch_size
        #test_loss += batch_size*loss.data[0]
        valid_acc1 += batch_size*accuracy[0].data[0]
        valid_acc5 += batch_size*accuracy[1].data[0]
        print('valid_num=%d  (%f, %f)  (%f, %f) '%(valid_num,valid_acc1/valid_num,valid_acc5/valid_num,1-valid_acc1/valid_num,1-valid_acc5/valid_num))

    assert(valid_num == len(valid_loader.sampler))
    valid_loss = 0 #valid_loss/valid_num
    valid_acc1 = valid_acc1/valid_num
    valid_acc5 = valid_acc5/valid_num

    return valid_loss, valid_acc1, valid_acc5




def run_validate():


    # net -----
    num_classes=1000
    C,H,W = 3,224,224

    # net = models.resnet50(pretrained=False)
    # net.load_state_dict(torch.load(pytorch_file))

    #net = SEResNet50(in_shape=(C,H,W), num_classes=num_classes)
    #load_pretrain_pytorch_file(net,'/root/share/data/models/pytorch/imagenet/resenet/resnet50-19c8e357.pth')

    # net = SEResNet50(in_shape=(C,H,W), num_classes=num_classes)
    # net.load_pretrain_pytorch_file('/root/share/data/models/reference/senet/ruotianluo/SE-ResNet-50.convert.pth')

    net = drn_d_54(in_shape=(C,H,W), num_classes=num_classes)
    net.load_pretrain_pytorch_file('/root/share/data/models/reference/imagenet/dilated_resenet/drn_d_54-0e0534ff.pth')



    net.cuda().eval()
    # data -----
    valid_dataset = Imagenet2012Dataset(
                       #'debug', 'val',
                       'valid_50000', 'val',
                       transform=[
                           lambda img: set_smallest_size_transform(img, 256),
                           lambda img: crop_center_transform(img, (H,W)),
                           lambda img: pytorch_image_to_tensor_transform(img),


                            # lambda img: Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)),
                            # transforms.Scale(256),
                            # transforms.CenterCrop(224),
                            # transforms.ToTensor(),
                            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

                            #lambda img: np.asarray(img),
                            #lambda img: pytorch_img_to_tensor_transform(img),
                       ],
                       mode='train' )

    valid_loader  = DataLoader(
                        valid_dataset,
                        sampler     = SequentialSampler(valid_dataset),
                        batch_size  = 16,
                        drop_last   = False,
                        num_workers = 8,
                        pin_memory  = True)
    valid_loss, valid_acc1, valid_acc5 = do_valid(net, valid_loader)

    print('')
    print('valid_loss = %f'%valid_loss)
    print('valid_acc1 (err)= %f (%f)'%(valid_acc1,1-valid_acc1))
    print('valid_acc5 (err)= %f (%f)'%(valid_acc5,1-valid_acc5))



########################################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_validate()