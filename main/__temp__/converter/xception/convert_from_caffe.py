## https://github.com/BVLC/caffe/issues/3884

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
#from net.model.se_resnet50 import SEResNet50
from pytorch_xception import Xception


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
    if len(caffe_net_params[caffe_key])==2:
        check_and_assign(pytorch_state_dict[pytorch_key + '.bias'],caffe_net_params[caffe_key][1].data)

#---
def convert_conv(pytorch_state_dict, pytorch_key, caffe_net_params, caffe_key):
    check_and_assign(pytorch_state_dict[pytorch_key + '.weight'],caffe_net_params[caffe_key][0].data)
    if len(caffe_net_params[caffe_key])==2:
        check_and_assign(pytorch_state_dict[pytorch_key + '.bias'],caffe_net_params[caffe_key][1].data)

#---
def convert_bn(pytorch_state_dict, pytorch_key, caffe_net_params, caffe_key1, caffe_key2):
    check_and_assign(pytorch_state_dict[pytorch_key + '.weight'      ],caffe_net_params[caffe_key1][0].data)
    check_and_assign(pytorch_state_dict[pytorch_key + '.bias'        ],caffe_net_params[caffe_key1][1].data)
    check_and_assign(pytorch_state_dict[pytorch_key + '.running_mean'],caffe_net_params[caffe_key2][0].data)
    check_and_assign(pytorch_state_dict[pytorch_key + '.running_var' ],caffe_net_params[caffe_key2][1].data)
    assert(caffe_net_params[caffe_key2][2].data[0]==1)


def convert_conv_bn(pytorch_state_dict, pytorch_key, caffe_net_params, caffe_key):
    convert_conv(pytorch_state_dict, pytorch_key+'.conv', caffe_net_params, caffe_key)
    convert_bn  (pytorch_state_dict, pytorch_key+'.bn',   caffe_net_params, caffe_key+'_scale', caffe_key+'_bn')


def convert_sep_conv_bn(pytorch_state_dict, pytorch_key, caffe_net_params, caffe_key):
    convert_conv(pytorch_state_dict, pytorch_key+'.conv1', caffe_net_params, caffe_key+'_1')
    convert_conv(pytorch_state_dict, pytorch_key+'.conv2', caffe_net_params, caffe_key+'_2')
    convert_bn  (pytorch_state_dict, pytorch_key+'.bn',   caffe_net_params, caffe_key+'_scale', caffe_key+'_bn')

## convert blocks ##
def convert_e_block(pytorch_state_dict, pytorch_key, caffe_net_params, caffe_key):
    convert_sep_conv_bn(pytorch_state_dict, pytorch_key+'.conv1', caffe_net_params, caffe_key+'_conv1')
    convert_sep_conv_bn(pytorch_state_dict, pytorch_key+'.conv2', caffe_net_params, caffe_key+'_conv2')
    convert_conv_bn    (pytorch_state_dict, pytorch_key+'.downsample',   caffe_net_params, caffe_key+'_match_conv')



def convert_m_block(pytorch_state_dict, pytorch_key, caffe_net_params, caffe_key):
    convert_sep_conv_bn(pytorch_state_dict, pytorch_key+'.conv1', caffe_net_params, caffe_key+'_conv1')
    convert_sep_conv_bn(pytorch_state_dict, pytorch_key+'.conv2', caffe_net_params, caffe_key+'_conv2')
    convert_sep_conv_bn(pytorch_state_dict, pytorch_key+'.conv3', caffe_net_params, caffe_key+'_conv3')




#------------------------------------------------------------------

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


# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))


    if 0:##------------------------------------------------------------------------
        #prototxt_file   = '/root/share/data/models/reference/imagenet/xception/caffe-model/inception/xception/deploy_xception.prototxt'
        prototxt_file   = '/root/share/data/models/reference/imagenet/xception/caffe-model/inception/xception/deploy_xception_original_cudnn_engine.prototxt'
        caffemodel_file = '/root/share/data/models/reference/imagenet/xception/caffe-model/inception/xception/xception.caffemodel'
        caffe_net       = load_caffe_net(prototxt_file, caffemodel_file)


        caffe_keys = list(caffe_net.params.keys())
        print(caffe_keys)

        pytorch_model = Xception(in_shape=(3,224,224),num_classes=1000)
        pytorch_model.eval()
        pytorch_state_dict = pytorch_model.state_dict()
        pytorch_keys = list(pytorch_state_dict.keys())
        print(pytorch_keys)

        exit(0)
    ##----------------------------------------------------------------------------
    ## keras:  https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json


    ##  https://github.com/soeaver/caffe-model/blob/master/cls/inception/deploy_xception.prototxt
    ##  https://github.com/soeaver/caffe-model
    ##  https://github.com/soeaver/caffe-model/blob/master/cls/evaluation_cls.py
    #   https://github.com/soeaver/caffe-model/blob/master/cls/synset.txt
    #    (441)  810 n02823750 beer glass
    #    (  1)  449 n01443537 goldfish, Carassius auratus
    #    (  9)  384 n01518878 ostrich, Struthio camelus

    #  Caffe uses a BGR
    #img  = np.random.uniform(0,1,size=(3,299,299))
    #img  = np.ones((3,299,299),np.float32)*128

    #img = cv2.imread('/root/share/data/imagenet/dummy/256x256/beer_glass.jpg')
    #img = cv2.imread('/root/share/data/imagenet/dummy/256x256/bullet_train.jpg')
    img = cv2.imread('/root/share/data/imagenet/dummy/256x256/goldfish.jpg')
    #img = cv2.imread('/root/share/data/imagenet/dummy/256x256/tabby_cat.jpg')
    #img = cv2.imread('/root/share/data/imagenet/dummy/256x256/blad_eagle.jpg')
    #img = cv2.imread('/root/share/data/imagenet/dummy/256x256/ostrich.jpg')


    #----------------------------------------------------------------------------------------------
    img = cv2.resize(img,(299,299)).astype(np.float32) - np.array( [102.9801, 115.9465, 122.7717])
    #img = cv2.resize(img,(224,224)).astype(np.float32) - np.array( [104, 117, 123])
    img = img.transpose((2,0,1))


    #caffe model ##########################################################################################
    #prototxt_file   = '/root/share/data/models/reference/senet/ruotianluo/SE-ResNet-50.prototxt.txt' ##ok
    #caffemodel_file = '/root/share/data/models/reference/senet/ruotianluo/SE-ResNet-50.caffemodel'

    #prototxt_file   = '/root/share/data/models/reference/imagenet/xception/caffe-model/inception/xception/deploy_xception_original_cudnn_engine.prototxt'
    prototxt_file   = '/root/share/data/models/reference/imagenet/xception/caffe-model/inception/xception/deploy_xception_debug.prototxt'
    caffemodel_file = '/root/share/data/models/reference/imagenet/xception/caffe-model/inception/xception/xception.caffemodel'
    caffe_net       = load_caffe_net(prototxt_file, caffemodel_file)

    #pytorch model
    save_pytorch_file = '/root/share/data/models/reference/imagenet/xception/caffe-model/inception/xception/xception.convert.pth'
    pytorch_model = Xception(in_shape=(3,299,299),num_classes=1000)
    pytorch_model.eval()

    ##start of convert #############################################################################################
    if 1:
        caffe_net_params   = caffe_net.params
        pytorch_state_dict = pytorch_model.state_dict()

        #debug
        #convert_conv(pytorch_state_dict, 'layer0.0.conv', caffe_net_params, 'conv1/7x7_s2' )

        #entry 0
        convert_conv_bn(pytorch_state_dict, 'entry0.0', caffe_net_params, 'conv1')
        convert_conv_bn(pytorch_state_dict, 'entry0.2', caffe_net_params, 'conv2')

        #entry 1,2,3
        convert_e_block(pytorch_state_dict, 'entry1', caffe_net_params, 'xception1')
        convert_e_block(pytorch_state_dict, 'entry2', caffe_net_params, 'xception2')
        convert_e_block(pytorch_state_dict, 'entry3', caffe_net_params, 'xception3')

        #middle 1,2,3,3,4,5,6,7,8
        convert_m_block(pytorch_state_dict, 'middle1', caffe_net_params, 'xception4')
        convert_m_block(pytorch_state_dict, 'middle2', caffe_net_params, 'xception5')
        convert_m_block(pytorch_state_dict, 'middle3', caffe_net_params, 'xception6')
        convert_m_block(pytorch_state_dict, 'middle4', caffe_net_params, 'xception7')
        convert_m_block(pytorch_state_dict, 'middle5', caffe_net_params, 'xception8')
        convert_m_block(pytorch_state_dict, 'middle6', caffe_net_params, 'xception9')
        convert_m_block(pytorch_state_dict, 'middle7', caffe_net_params, 'xception10')
        convert_m_block(pytorch_state_dict, 'middle8', caffe_net_params, 'xception11')


        #exit 1
        convert_e_block(pytorch_state_dict, 'exit1', caffe_net_params, 'xception12')

        #exit 2
        convert_sep_conv_bn(pytorch_state_dict, 'exit2.conv1', caffe_net_params, 'conv3')
        convert_sep_conv_bn(pytorch_state_dict, 'exit2.conv2', caffe_net_params, 'conv4')

        #layer fc
        convert_fc(pytorch_state_dict, 'fc', caffe_net_params,  'classifier')

        ##end of convert #############################################################################################
        ##model.load_state_dict(pytorch_state_dict)
        torch.save(pytorch_model.state_dict(), save_pytorch_file)



    ################################################################################3
    ## check conversion results

    caffe_prob = run_caffe_net(caffe_net, img)
    #caffe_prob = caffe_prob[0:30]
    print('caffe_prob\n',caffe_prob)

    ##---------------------------------
    hi=[]
    ho=[]
    def hook(module, input, output):
        #print module
        ho.append(output.data.numpy())
        hi.append(input[0].data.numpy())

    pytorch_model.entry0._modules['0'].conv.register_forward_hook(hook)

    pytorch_prob = pytorch_model( Variable(torch.from_numpy(img).unsqueeze(0).float() ) )
    pytorch_prob = F.softmax(pytorch_prob).data.numpy().reshape(-1)
    print('pytorch_prob\n',pytorch_prob)

    print('caffe ', np.argmax(caffe_prob), ' ', caffe_prob[np.argmax(caffe_prob)])
    print('pytorch ', np.argmax(pytorch_prob), ' ', pytorch_prob[np.argmax(pytorch_prob)])


    ##---------
    p=ho[0][0]
    c=caffe_net.blobs['xxx'].data[0]

    xx=0
