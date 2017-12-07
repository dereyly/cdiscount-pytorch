#convert from caffe
# https://github.com/hujie-frank/SENet
# https://github.com/ruotianluo/pytorch-resnet
# /ruotianluo/pytorch-resnet/master/convert.py
#
#
# install caffe python 3.6
#   https://yangcha.github.io/Caffe-Conda3/

'''

caffe_model.params.keys()
odict_keys(['conv1/7x7_s2', 'conv1/7x7_s2/bn', 'conv1/7x7_s2/bn/scale', 'conv2_1_1x1_reduce', 'conv2_1_1x1_reduce/bn', 'conv2_1_1x1_reduce/bn/scale', 'conv2_1_3x3', 'conv2_1_3x3/bn', 'conv2_1_3x3/bn/scale', 'conv2_1_1x1_increase', 'conv2_1_1x1_increase/bn', 'conv2_1_1x1_increase/bn/scale', 'conv2_1_1x1_down', 'conv2_1_1x1_up', 'conv2_1_1x1_proj', 'conv2_1_1x1_proj/bn', 'conv2_1_1x1_proj/bn/scale', 'conv2_2_1x1_reduce', 'conv2_2_1x1_reduce/bn', 'conv2_2_1x1_reduce/bn/scale', 'conv2_2_3x3', 'conv2_2_3x3/bn', 'conv2_2_3x3/bn/scale', 'conv2_2_1x1_increase', 'conv2_2_1x1_increase/bn', 'conv2_2_1x1_increase/bn/scale', 'conv2_2_1x1_down', 'conv2_2_1x1_up', 'conv2_3_1x1_reduce', 'conv2_3_1x1_reduce/bn', 'conv2_3_1x1_reduce/bn/scale', 'conv2_3_3x3', 'conv2_3_3x3/bn', 'conv2_3_3x3/bn/scale', 'conv2_3_1x1_increase', 'conv2_3_1x1_increase/bn', 'conv2_3_1x1_increase/bn/scale', 'conv2_3_1x1_down', 'conv2_3_1x1_up', 'conv3_1_1x1_reduce', 'conv3_1_1x1_reduce/bn', 'conv3_1_1x1_reduce/bn/scale', 'conv3_1_3x3', 'conv3_1_3x3/bn', 'conv3_1_3x3/bn/scale', 'conv3_1_1x1_increase', 'conv3_1_1x1_increase/bn', 'conv3_1_1x1_increase/bn/scale', 'conv3_1_1x1_down', 'conv3_1_1x1_up', 'conv3_1_1x1_proj', 'conv3_1_1x1_proj/bn', 'conv3_1_1x1_proj/bn/scale', 'conv3_2_1x1_reduce', 'conv3_2_1x1_reduce/bn', 'conv3_2_1x1_reduce/bn/scale', 'conv3_2_3x3', 'conv3_2_3x3/bn', 'conv3_2_3x3/bn/scale', 'conv3_2_1x1_increase', 'conv3_2_1x1_increase/bn', 'conv3_2_1x1_increase/bn/scale', 'conv3_2_1x1_down', 'conv3_2_1x1_up', 'conv3_3_1x1_reduce', 'conv3_3_1x1_reduce/bn', 'conv3_3_1x1_reduce/bn/scale', 'conv3_3_3x3', 'conv3_3_3x3/bn', 'conv3_3_3x3/bn/scale', 'conv3_3_1x1_increase', 'conv3_3_1x1_increase/bn', 'conv3_3_1x1_increase/bn/scale', 'conv3_3_1x1_down', 'conv3_3_1x1_up', 'conv3_4_1x1_reduce', 'conv3_4_1x1_reduce/bn', 'conv3_4_1x1_reduce/bn/scale', 'conv3_4_3x3', 'conv3_4_3x3/bn', 'conv3_4_3x3/bn/scale', 'conv3_4_1x1_increase', 'conv3_4_1x1_increase/bn', 'conv3_4_1x1_increase/bn/scale', 'conv3_4_1x1_down', 'conv3_4_1x1_up', 'conv4_1_1x1_reduce', 'conv4_1_1x1_reduce/bn', 'conv4_1_1x1_reduce/bn/scale', 'conv4_1_3x3', 'conv4_1_3x3/bn', 'conv4_1_3x3/bn/scale', 'conv4_1_1x1_increase', 'conv4_1_1x1_increase/bn', 'conv4_1_1x1_increase/bn/scale', 'conv4_1_1x1_down', 'conv4_1_1x1_up', 'conv4_1_1x1_proj', 'conv4_1_1x1_proj/bn', 'conv4_1_1x1_proj/bn/scale', 'conv4_2_1x1_reduce', 'conv4_2_1x1_reduce/bn', 'conv4_2_1x1_reduce/bn/scale', 'conv4_2_3x3', 'conv4_2_3x3/bn', 'conv4_2_3x3/bn/scale', 'conv4_2_1x1_increase', 'conv4_2_1x1_increase/bn', 'conv4_2_1x1_increase/bn/scale', 'conv4_2_1x1_down', 'conv4_2_1x1_up', 'conv4_3_1x1_reduce', 'conv4_3_1x1_reduce/bn', 'conv4_3_1x1_reduce/bn/scale', 'conv4_3_3x3', 'conv4_3_3x3/bn', 'conv4_3_3x3/bn/scale', 'conv4_3_1x1_increase', 'conv4_3_1x1_increase/bn', 'conv4_3_1x1_increase/bn/scale', 'conv4_3_1x1_down', 'conv4_3_1x1_up', 'conv4_4_1x1_reduce', 'conv4_4_1x1_reduce/bn', 'conv4_4_1x1_reduce/bn/scale', 'conv4_4_3x3', 'conv4_4_3x3/bn', 'conv4_4_3x3/bn/scale', 'conv4_4_1x1_increase', 'conv4_4_1x1_increase/bn', 'conv4_4_1x1_increase/bn/scale', 'conv4_4_1x1_down', 'conv4_4_1x1_up', 'conv4_5_1x1_reduce', 'conv4_5_1x1_reduce/bn', 'conv4_5_1x1_reduce/bn/scale', 'conv4_5_3x3', 'conv4_5_3x3/bn', 'conv4_5_3x3/bn/scale', 'conv4_5_1x1_increase', 'conv4_5_1x1_increase/bn', 'conv4_5_1x1_increase/bn/scale', 'conv4_5_1x1_down', 'conv4_5_1x1_up', 'conv4_6_1x1_reduce', 'conv4_6_1x1_reduce/bn', 'conv4_6_1x1_reduce/bn/scale', 'conv4_6_3x3', 'conv4_6_3x3/bn', 'conv4_6_3x3/bn/scale', 'conv4_6_1x1_increase', 'conv4_6_1x1_increase/bn', 'conv4_6_1x1_increase/bn/scale', 'conv4_6_1x1_down', 'conv4_6_1x1_up', 'conv5_1_1x1_reduce', 'conv5_1_1x1_reduce/bn', 'conv5_1_1x1_reduce/bn/scale', 'conv5_1_3x3', 'conv5_1_3x3/bn', 'conv5_1_3x3/bn/scale', 'conv5_1_1x1_increase', 'conv5_1_1x1_increase/bn', 'conv5_1_1x1_increase/bn/scale', 'conv5_1_1x1_down', 'conv5_1_1x1_up', 'conv5_1_1x1_proj', 'conv5_1_1x1_proj/bn', 'conv5_1_1x1_proj/bn/scale', 'conv5_2_1x1_reduce', 'conv5_2_1x1_reduce/bn', 'conv5_2_1x1_reduce/bn/scale', 'conv5_2_3x3', 'conv5_2_3x3/bn', 'conv5_2_3x3/bn/scale', 'conv5_2_1x1_increase', 'conv5_2_1x1_increase/bn', 'conv5_2_1x1_increase/bn/scale', 'conv5_2_1x1_down', 'conv5_2_1x1_up', 'conv5_3_1x1_reduce', 'conv5_3_1x1_reduce/bn', 'conv5_3_1x1_reduce/bn/scale', 'conv5_3_3x3', 'conv5_3_3x3/bn', 'conv5_3_3x3/bn/scale', 'conv5_3_1x1_increase', 'conv5_3_1x1_increase/bn', 'conv5_3_1x1_increase/bn/scale', 'conv5_3_1x1_down', 'conv5_3_1x1_up', 'classifier'])


'''

import sys
import os
sys.path.insert(0,'/opt/caffe/SENet-Caffe/SENet/python')
os.environ["GLOG_minloglevel"] = "2"

import caffe
from caffe.proto import caffe_pb2

import torch
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F

import re
import numpy as np
import skimage.io
from collections import OrderedDict

import cv2
from net.model.cdiscount.senet import SENet2048


##--------------------------------------------------------
## e.g.
##   pytorch_state_dict['conv.weight'] = caffe_net_params['conv'][0].data
##   pytorch_state_dict['conv.bias  '] = caffe_net_params['conv'][1].data

def check_and_assign(a,b):
    assert tuple(a.size())==b.shape, 'a.size()=%s\nb.shape=%s'%(str(a.size()),str(b.shape))
    a[...] = torch.from_numpy(b).float()

#---
def convert_bn(pytorch_state_dict, pytorch_key, caffe_net_params, caffe_key1, caffe_key2):
    check_and_assign(pytorch_state_dict[pytorch_key + '.weight'      ],caffe_net_params[caffe_key1][0].data)
    check_and_assign(pytorch_state_dict[pytorch_key + '.bias'        ],caffe_net_params[caffe_key1][1].data)
    check_and_assign(pytorch_state_dict[pytorch_key + '.running_mean'],caffe_net_params[caffe_key2][0].data)
    check_and_assign(pytorch_state_dict[pytorch_key + '.running_var' ],caffe_net_params[caffe_key2][1].data)
    assert(caffe_net_params[caffe_key2][2].data[0]==1)

#---
def convert_conv(pytorch_state_dict, pytorch_key, caffe_net_params, caffe_key):
    check_and_assign(pytorch_state_dict[pytorch_key + '.weight'],caffe_net_params[caffe_key][0].data)
    if len(caffe_net_params[caffe_key])==2:
        check_and_assign(pytorch_state_dict[pytorch_key + '.bias'],caffe_net_params[caffe_key][1].data)

#---
def convert_fc(pytorch_state_dict, pytorch_key, caffe_net_params, caffe_key):
    check_and_assign(pytorch_state_dict[pytorch_key + '.weight'],caffe_net_params[caffe_key][0].data)
    if len(caffe_net_params[caffe_key])==2:
        check_and_assign(pytorch_state_dict[pytorch_key + '.bias'],caffe_net_params[caffe_key][1].data)



## convert blocks ##----------------------------------------------

def convert_conv_bn(pytorch_state_dict, pytorch_key, caffe_net_params, caffe_key):
    convert_conv(pytorch_state_dict, pytorch_key+'.conv', caffe_net_params, caffe_key)
    convert_bn  (pytorch_state_dict, pytorch_key+'.bn',   caffe_net_params, caffe_key+'/bn/scale', caffe_key+'/bn')


def convert_excitation(pytorch_state_dict, pytorch_key, caffe_net_params, caffe_key):
    convert_conv(pytorch_state_dict, pytorch_key+'.fc1', caffe_net_params, caffe_key+'_down')
    convert_conv(pytorch_state_dict, pytorch_key+'.fc2', caffe_net_params, caffe_key+'_up')



## -----------------------------------------------------------
def load_caffe_net(prototxt_file,caffemodel_file):
    caffe.set_mode_cpu()
    net = caffe.Net(prototxt_file, caffemodel_file, caffe.TEST)
    return net

def run_caffe_net(net, img):

    net.blobs['data'].data[0] = img
    #assert net.blobs['data'].data[0].shape == (3, 224, 224)
    net.forward()
    prob = net.blobs['prob'].data[0]
    return prob
## -----------------------------------------------------------

def print_caffe_net():

    prototxt_file   = '/root/share/data/models/reference/senet/senet/SENet.prototxt.txt'
    caffemodel_file = '/root/share/data/models/reference/senet/senet/SENet.caffemodel'
    caffe_net       = load_caffe_net(prototxt_file, caffemodel_file)
    caffe_net_params = caffe_net.params


    for k in caffe_net_params:
        params = caffe_net_params[k]

        print ('%-32s  '%k, end = '', flush=True)
        for p in params:
            print ('%-18s  '%str(p.data.shape) + '   ', end = '', flush=True)
        print ('')

    xx=0
    pass

## -----------------------------------------------------------

def convert_from_caffe():

    # caffe model
    prototxt_file   = '/root/share/data/models/reference/senet/senet/SENet.prototxt.txt'
    caffemodel_file = '/root/share/data/models/reference/senet/senet/SENet.caffemodel'
    caffe_net       = load_caffe_net(prototxt_file, caffemodel_file)
    caffe_net_params = caffe_net.params

    # pytorch model
    save_pytorch_file = '/root/share/data/models/reference/senet/senet/SENet.convert.pth'
    pytorch_model = SENet2048(in_shape=(3,224,224),num_classes=1000)
    pytorch_model.eval()

    ##start of convert #############################################################################################
    if 1:
        caffe_net_params   = caffe_net.params
        pytorch_state_dict = pytorch_model.state_dict()

        #layer0
        convert_conv_bn(pytorch_state_dict, 'layer0.0', caffe_net_params, 'conv1_1/3x3_s2')
        convert_conv_bn(pytorch_state_dict, 'layer0.2', caffe_net_params, 'conv1_2/3x3')
        convert_conv_bn(pytorch_state_dict, 'layer0.4', caffe_net_params, 'conv1_3/3x3')

        #layer1
        for n in range(3):
            m=n+1
            convert_conv_bn   (pytorch_state_dict, 'layer1.%d.conv_bn1'%n, caffe_net_params, 'conv2_%d_1x1_reduce'%m)
            convert_conv_bn   (pytorch_state_dict, 'layer1.%d.conv_bn2'%n, caffe_net_params, 'conv2_%d_3x3'%m)
            convert_conv_bn   (pytorch_state_dict, 'layer1.%d.conv_bn3'%n, caffe_net_params, 'conv2_%d_1x1_increase'%m)
            convert_excitation(pytorch_state_dict, 'layer1.%d.scale'%n,    caffe_net_params, 'conv2_%d_1x1'%m)

            if n==0:
                 convert_conv_bn(pytorch_state_dict, 'layer1.%d.downsample'%n, caffe_net_params, 'conv2_%d_1x1_proj'%m)


        #layer2
        for n in range(8):
            m=n+1
            convert_conv_bn   (pytorch_state_dict, 'layer2.%d.conv_bn1'%n, caffe_net_params, 'conv3_%d_1x1_reduce'%m)
            convert_conv_bn   (pytorch_state_dict, 'layer2.%d.conv_bn2'%n, caffe_net_params, 'conv3_%d_3x3'%m)
            convert_conv_bn   (pytorch_state_dict, 'layer2.%d.conv_bn3'%n, caffe_net_params, 'conv3_%d_1x1_increase'%m)
            convert_excitation(pytorch_state_dict, 'layer2.%d.scale'%n,    caffe_net_params, 'conv3_%d_1x1'%m)

            if n==0:
                 convert_conv_bn(pytorch_state_dict, 'layer2.%d.downsample'%n, caffe_net_params, 'conv3_%d_1x1_proj'%m)


        #layer3
        for n in range(36):
            m=n+1
            convert_conv_bn   (pytorch_state_dict, 'layer3.%d.conv_bn1'%n, caffe_net_params, 'conv4_%d_1x1_reduce'%m)
            convert_conv_bn   (pytorch_state_dict, 'layer3.%d.conv_bn2'%n, caffe_net_params, 'conv4_%d_3x3'%m)
            convert_conv_bn   (pytorch_state_dict, 'layer3.%d.conv_bn3'%n, caffe_net_params, 'conv4_%d_1x1_increase'%m)
            convert_excitation(pytorch_state_dict, 'layer3.%d.scale'%n,    caffe_net_params, 'conv4_%d_1x1'%m)

            if n==0:
                 convert_conv_bn(pytorch_state_dict, 'layer3.%d.downsample'%n, caffe_net_params, 'conv4_%d_1x1_proj'%m)

        #layer4
        for n in range(3):
            m=n+1
            convert_conv_bn   (pytorch_state_dict, 'layer4.%d.conv_bn1'%n, caffe_net_params, 'conv5_%d_1x1_reduce'%m)
            convert_conv_bn   (pytorch_state_dict, 'layer4.%d.conv_bn2'%n, caffe_net_params, 'conv5_%d_3x3'%m)
            convert_conv_bn   (pytorch_state_dict, 'layer4.%d.conv_bn3'%n, caffe_net_params, 'conv5_%d_1x1_increase'%m)
            convert_excitation(pytorch_state_dict, 'layer4.%d.scale'%n,    caffe_net_params, 'conv5_%d_1x1'%m)

            if n==0:
                 convert_conv_bn(pytorch_state_dict, 'layer4.%d.downsample'%n, caffe_net_params, 'conv5_%d_1x1_proj'%m)

        #layer fc
        convert_fc(pytorch_state_dict, 'fc', caffe_net_params,  'classifier')

        ## end of conversion ####
        torch.save(pytorch_model.state_dict(), save_pytorch_file)

    ###############################################################################################
    #    (441)  810 n02823750 beer glass
    #    (  1)  449 n01443537 goldfish, Carassius auratus
    #    (  9)  384 n01518878 ostrich, Struthio camelus


    ## check
    #img  = np.random.uniform(0,255,size=(3,224,224))
    #img  = np.ones((3,224,224),np.float32)*128

    img = cv2.imread('/root/share/data/imagenet/dummy/256x256/beer_glass.jpg')
    #img = cv2.imread('/root/share/data/imagenet/dummy/256x256/bullet_train.jpg')
    #img = cv2.imread('/root/share/data/imagenet/dummy/256x256/goldfish.jpg')
    #img = cv2.imread('/root/share/data/imagenet/dummy/256x256/tabby_cat.jpg')
    #img = cv2.imread('/root/share/data/imagenet/dummy/256x256/blad_eagle.jpg')
    #img = cv2.imread('/root/share/data/imagenet/dummy/256x256/ostrich.jpg')

    img = cv2.resize(img,(224,224)).astype(np.float32) - np.array( [102.9801, 115.9465, 122.7717])
    img = img.transpose((2,0,1))


    #run caffe
    caffe_prob = run_caffe_net(caffe_net, img)
    #print('caffe_prob\n',caffe_prob)

    #run pytorch
    pytorch_prob = pytorch_model( Variable(torch.from_numpy(img).unsqueeze(0).float() ) )
    pytorch_prob = F.softmax(pytorch_prob).data.numpy().reshape(-1)
    #print('pytorch_prob\n',pytorch_prob)

    #check
    print('caffe   ', np.argmax(caffe_prob),   ' ', caffe_prob[np.argmax(caffe_prob)])
    print('pytorch ', np.argmax(pytorch_prob), ' ', pytorch_prob[np.argmax(pytorch_prob)])




# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))


    convert_from_caffe()
    #print_caffe_net()
    exit(0)









