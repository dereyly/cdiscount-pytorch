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
from net.model.se_resnet50 import SEResNet50

##--------------------------------------------------------
## e.g.
##   pytorch_state_dict['conv.weight'] = caffe_net_params['conv'][0].data
##   pytorch_state_dict['conv.bias  '] = caffe_net_params['conv'][1].data


def check_and_assign(a,b):
    assert tuple(a.size())==b.shape, 'a.size()=%s\nb.shape=%s'%(str(a.size()),str(b.shape))
    a[...] = torch.from_numpy(b).float()

#---
def convert_fc(pytorch_state_dict, pytorch_key, caffe_net_params, caffe_key):
    check_and_assign(pytorch_state_dict[pytorch_key + '.weight'],caffe_net_params[caffe_key][0].data)
    check_and_assign(pytorch_state_dict[pytorch_key + '.bias'  ],caffe_net_params[caffe_key][1].data)

def convert_scale(pytorch_state_dict, pytorch_key, caffe_net_params, caffe_key):
    convert_fc(pytorch_state_dict, pytorch_key+'.fc1', caffe_net_params, caffe_key+'_down')
    convert_fc(pytorch_state_dict, pytorch_key+'.fc2', caffe_net_params, caffe_key+'_up')

#---
def convert_conv(pytorch_state_dict, pytorch_key, caffe_net_params, caffe_key):
    check_and_assign(pytorch_state_dict[pytorch_key + '.weight'],caffe_net_params[caffe_key][0].data)
    #check_and_assign(pytorch_state_dict[pytorch_key + '.bias'  ],caffe_net_params[caffe_key][1].data)

def convert_bn(pytorch_state_dict, pytorch_key, caffe_net_params, caffe_key1, caffe_key2):
    check_and_assign(pytorch_state_dict[pytorch_key + '.weight'      ],caffe_net_params[caffe_key1][0].data)
    check_and_assign(pytorch_state_dict[pytorch_key + '.bias'        ],caffe_net_params[caffe_key1][1].data)
    check_and_assign(pytorch_state_dict[pytorch_key + '.running_mean'],caffe_net_params[caffe_key2][0].data)
    check_and_assign(pytorch_state_dict[pytorch_key + '.running_var' ],caffe_net_params[caffe_key2][1].data)
    assert(caffe_net_params[caffe_key2][2].data[0]==1)

def convert_conv_bn(pytorch_state_dict, pytorch_key, caffe_net_params, caffe_key):
    convert_conv(pytorch_state_dict, pytorch_key+'.conv', caffe_net_params, caffe_key)
    convert_bn  (pytorch_state_dict, pytorch_key+'.bn',   caffe_net_params, caffe_key+'/bn/scale', caffe_key+'/bn')



#------------------------------------------------------------------

def load_caffe_net(prototxt_file,caffemodel_file):
    caffe.set_mode_cpu()
    net = caffe.Net(prototxt_file, caffemodel_file, caffe.TEST)
    return net

def run_caffe_net(net, img):

    net.blobs['data'].data[0] = img
    assert net.blobs['data'].data[0].shape == (3, 224, 224)
    net.forward()
    prob = net.blobs['prob'].data[0]
    return prob

## -----------------------------------------------------------


# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

   #  Caffe uses a BGR
    img  = np.random.uniform(0,1,size=(3,224,224))
    #img  = np.ones((3,224,224),np.float32)*128

    #img = cv2.imread('/root/share/data/imagenet/dummy/256x256/beer_glass.jpg')
    #img = cv2.imread('/root/share/data/imagenet/dummy/256x256/bullet_train.jpg')
    img = cv2.imread('/root/share/data/imagenet/dummy/256x256/goldfish.jpg')
    img = cv2.resize(img,(224,224)).astype(np.float32) - np.array( [104, 117, 123])
    img = img.transpose((2,0,1))


    #caffe model
    prototxt_file   = '/root/share/data/models/reference/senet/ruotianluo/SE-ResNet-50.prototxt.txt'
    caffemodel_file = '/root/share/data/models/reference/senet/ruotianluo/SE-ResNet-50.caffemodel'
    caffe_net       = load_caffe_net(prototxt_file, caffemodel_file)


    #pytorch model
    save_pytorch_file = '/root/share/data/models/reference/senet/ruotianluo/SE-ResNet-50.convert.pth'
    model = SEResNet50(in_shape=(3,224,224),num_classes=1000)
    model.eval()

    ##start of convert #############################################################################################
    caffe_net_params   = caffe_net.params
    pytorch_state_dict = model.state_dict()

    #debug
    #convert_conv(pytorch_state_dict, 'layer0.0.conv', caffe_net_params, 'conv1/7x7_s2' )

    #layer 0
    convert_conv_bn(pytorch_state_dict, 'layer0.0', caffe_net_params,  'conv1/7x7_s2')

    #num_blocks=[3,4,6,3]
    #layer 1
    for n in range(3):
        m=n+1
        convert_conv_bn(pytorch_state_dict, 'layer1.%d.conv_bn1'%n, caffe_net_params,  'conv2_%d_1x1_reduce'%m   )
        convert_conv_bn(pytorch_state_dict, 'layer1.%d.conv_bn2'%n, caffe_net_params,  'conv2_%d_3x3'%m          )
        convert_conv_bn(pytorch_state_dict, 'layer1.%d.conv_bn3'%n, caffe_net_params,  'conv2_%d_1x1_increase'%m )
        convert_scale  (pytorch_state_dict, 'layer1.%d.scale'%n,    caffe_net_params,  'conv2_%d_1x1'%m          )
        if n==0: #residual projection
            convert_conv_bn(pytorch_state_dict, 'layer1.%d.downsample'%n, caffe_net_params,  'conv2_%d_1x1_proj'%m   )


    #layer 2
    for n in range(4):
        m=n+1
        convert_conv_bn(pytorch_state_dict, 'layer2.%d.conv_bn1'%n, caffe_net_params,  'conv3_%d_1x1_reduce'%m   )
        convert_conv_bn(pytorch_state_dict, 'layer2.%d.conv_bn2'%n, caffe_net_params,  'conv3_%d_3x3'%m          )
        convert_conv_bn(pytorch_state_dict, 'layer2.%d.conv_bn3'%n, caffe_net_params,  'conv3_%d_1x1_increase'%m )
        convert_scale  (pytorch_state_dict, 'layer2.%d.scale'%n,    caffe_net_params,  'conv3_%d_1x1'%m          )
        if n==0: #residual projection
            convert_conv_bn(pytorch_state_dict, 'layer2.%d.downsample'%n, caffe_net_params,  'conv3_%d_1x1_proj'%m   )

    #layer 3
    for n in range(6):
        m=n+1
        convert_conv_bn(pytorch_state_dict, 'layer3.%d.conv_bn1'%n, caffe_net_params,  'conv4_%d_1x1_reduce'%m   )
        convert_conv_bn(pytorch_state_dict, 'layer3.%d.conv_bn2'%n, caffe_net_params,  'conv4_%d_3x3'%m          )
        convert_conv_bn(pytorch_state_dict, 'layer3.%d.conv_bn3'%n, caffe_net_params,  'conv4_%d_1x1_increase'%m )
        convert_scale  (pytorch_state_dict, 'layer3.%d.scale'%n,    caffe_net_params,  'conv4_%d_1x1'%m          )
        if n==0: #residual projection
            convert_conv_bn(pytorch_state_dict, 'layer3.%d.downsample'%n, caffe_net_params,  'conv4_%d_1x1_proj'%m   )


    #layer 4
    for n in range(3):
        m=n+1
        convert_conv_bn(pytorch_state_dict, 'layer4.%d.conv_bn1'%n, caffe_net_params,  'conv5_%d_1x1_reduce'%m   )
        convert_conv_bn(pytorch_state_dict, 'layer4.%d.conv_bn2'%n, caffe_net_params,  'conv5_%d_3x3'%m          )
        convert_conv_bn(pytorch_state_dict, 'layer4.%d.conv_bn3'%n, caffe_net_params,  'conv5_%d_1x1_increase'%m )
        convert_scale  (pytorch_state_dict, 'layer4.%d.scale'%n,    caffe_net_params,  'conv5_%d_1x1'%m          )
        if n==0: #residual projection
            convert_conv_bn(pytorch_state_dict, 'layer4.%d.downsample'%n, caffe_net_params,  'conv5_%d_1x1_proj'%m   )

    #layer fc
    convert_fc(pytorch_state_dict, 'fc', caffe_net_params,  'classifier')

    ##end of convert #############################################################################################
    ##model.load_state_dict(pytorch_state_dict)
    torch.save(model.state_dict(), save_pytorch_file)


    #debug
    #put in some input and test
    caffe_prob = run_caffe_net(caffe_net, img)
    #caffe_prob = caffe_prob[0:30]
    print('caffe_prob\n',caffe_prob)


    hi=[]
    ho=[]
    def hook(module, input, output):
        #print module
        ho.append(output.data.numpy())
        hi.append(input[0].data.numpy())

    #model.layer0._modules['0'].conv.register_forward_hook(hook)
    #model.layer0._modules['2'].register_forward_hook(hook)
    #model.layer1._modules['0'].conv_bn3.register_forward_hook(hook)
    #model.layer1._modules['0'].scale.register_forward_hook(hook)
    #model.layer1._modules['0'].downsample.register_forward_hook(hook)
    #model.layer1._modules['0'].register_forward_hook(hook)  ##caffe_net.blobs['conv2_1'].data[0] OK
    #model.layer1._modules['2'].register_forward_hook(hook)  ##caffe_net.blobs['conv2_3'].data[0] OK
    #model.layer2._modules['0'].register_forward_hook(hook)  ##caffe_net.blobs['conv3_1'].data[0]
    model.layer2._modules['1'].conv_bn1.conv.register_forward_hook(hook)



    pytorch_prob = model( Variable(torch.from_numpy(img).unsqueeze(0).float() ) )
    pytorch_prob = F.softmax(pytorch_prob).data.numpy().reshape(-1)
    #pytorch_prob = pytorch_prob[0:30]
    print('pytorch_prob\n',pytorch_prob)

    #debug
    # http://davidstutz.de/pycaffe-tools-examples-and-resources/
    p=ho[0][0]
    c=caffe_net.blobs['xxx'].data[0]

    pp=np.squeeze(p)[0]
    cc=np.squeeze(c)[0]

    print(str(p.shape))
    print(str(c.shape))

    print(np.argmax(pytorch_prob))
    print(np.argmax(caffe_prob))
    print(pytorch_prob[np.argmax(pytorch_prob)])
    print(caffe_prob[np.argmax(caffe_prob)])

    xx=0







