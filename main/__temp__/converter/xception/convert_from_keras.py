## https://github.com/BVLC/caffe/issues/3884

import sys
import os
import cv2
import re
import numpy as np
import skimage.io
from collections import OrderedDict

from keras.models import Model as KModel
from keras.applications.xception import Xception as KXception, preprocess_input, decode_predictions
from keras.preprocessing import image



import torch
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F

from pytorch_xception import Xception


##--------------------------------------------------------
## e.g.
##   pytorch_state_dict['conv.weight'] = caffe_net_params['conv'][0].data
##   pytorch_state_dict['conv.bias  '] = caffe_net_params['conv'][1].data


def check_and_assign(a,b):
    assert tuple(a.size())==b.shape, 'a.size()=%s\nb.shape=%s'%(str(a.size()),str(b.shape))
    a[...] = torch.from_numpy(b).float()

#---
def convert_fc(pytorch_state_dict, pytorch_key, keras_weights, keras_key):
    check_and_assign(pytorch_state_dict[pytorch_key + '.weight'],keras_weights[keras_key+'/kernel:0'])
    try:
        check_and_assign(pytorch_state_dict[pytorch_key + '.bias'],keras_weights[keras_key+'/bias:0'])
    except Exception:
        pass



#---
#conv2d
def convert_conv(pytorch_state_dict, pytorch_key, keras_weights, keras_key):
    check_and_assign(pytorch_state_dict[pytorch_key + '.weight'],keras_weights[keras_key+'/kernel:0'])
    try:
        check_and_assign(pytorch_state_dict[pytorch_key + '.bias'],keras_weights[keras_key+'/bias:0'])
    except Exception:
        pass
#---
#bn
def convert_bn(pytorch_state_dict, pytorch_key, keras_weights, keras_key):
    check_and_assign(pytorch_state_dict[pytorch_key + '.weight'      ],keras_weights[keras_key+'/gamma:0'])
    check_and_assign(pytorch_state_dict[pytorch_key + '.bias'        ],keras_weights[keras_key+'/beta:0'])
    check_and_assign(pytorch_state_dict[pytorch_key + '.running_mean'],keras_weights[keras_key+'/moving_mean:0'])
    check_and_assign(pytorch_state_dict[pytorch_key + '.running_var' ],keras_weights[keras_key+'/moving_variance:0'])

#---
#sep_conv2d
def convert_sep_conv(pytorch_state_dict, pytorch_key1, pytorch_key2, keras_weights, keras_key):
    check_and_assign(pytorch_state_dict[pytorch_key1 + '.weight'],keras_weights[keras_key+'/depthwise_kernel:0'])
    check_and_assign(pytorch_state_dict[pytorch_key2 + '.weight'],keras_weights[keras_key+'/pointwise_kernel:0'])



#---
def convert_conv_bn(pytorch_state_dict, pytorch_key, keras_weights, keras_key):
    convert_conv(pytorch_state_dict, pytorch_key+'.conv', keras_weights, keras_key)
    convert_bn  (pytorch_state_dict, pytorch_key+'.bn',   keras_weights, keras_key+'_bn')


def convert_sep_conv_bn(pytorch_state_dict, pytorch_key, keras_weights, keras_key):
    convert_sep_conv(pytorch_state_dict, pytorch_key +'.conv1', pytorch_key +'.conv2', keras_weights, keras_key)
    convert_bn  (pytorch_state_dict, pytorch_key+'.bn',   keras_weights, keras_key+'_bn')



## convert blocks ##
def convert_e_block(pytorch_state_dict, pytorch_key, keras_weights, keras_key, i):
    convert_sep_conv_bn(pytorch_state_dict, pytorch_key+'.conv1', keras_weights, keras_key+'_sepconv1')
    convert_sep_conv_bn(pytorch_state_dict, pytorch_key+'.conv2', keras_weights, keras_key+'_sepconv2')
    convert_conv(pytorch_state_dict, pytorch_key+'.downsample.conv', keras_weights, 'conv2d_%d'%i)
    convert_bn  (pytorch_state_dict, pytorch_key+'.downsample.bn',   keras_weights, 'batch_normalization_%d'%i)


def convert_m_block(pytorch_state_dict, pytorch_key, keras_weights, keras_key):
    convert_sep_conv_bn(pytorch_state_dict, pytorch_key+'.conv1', keras_weights, keras_key+'_sepconv1')
    convert_sep_conv_bn(pytorch_state_dict, pytorch_key+'.conv2', keras_weights, keras_key+'_sepconv2')
    convert_sep_conv_bn(pytorch_state_dict, pytorch_key+'.conv3', keras_weights, keras_key+'_sepconv3')




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

    ##  https://github.com/soeaver/caffe-model/blob/master/cls/inception/deploy_xception.prototxt
    ##  https://github.com/soeaver/caffe-model
    ##  https://github.com/soeaver/caffe-model/blob/master/cls/evaluation_cls.py
    #   https://github.com/soeaver/caffe-model/blob/master/cls/synset.txt
    #    (441)  810 n02823750 beer glass
    #    (  1)  449 n01443537 goldfish, Carassius auratus
    #    (  9)  384 n01518878 ostrich, Struthio camelus
    img_file = '/root/share/data/imagenet/dummy/256x256/goldfish.jpg'
    img_file = '/root/share/data/imagenet/dummy/256x256/beer_glass.jpg'
    img_file = '/root/share/data/imagenet/dummy/256x256/ostrich.jpg'
    img = image.load_img(img_file, target_size=(299, 299))
    #img = np.ones((299,299,3),np.uint8)

    keras_model = KXception(include_top=True, weights='imagenet', input_tensor=None)
    xx = image.img_to_array(img)
    xx = np.expand_dims(xx, axis=0)
    xx = preprocess_input(xx)
    preds = keras_model.predict(xx)
    # decode the results into a list of tuples (class, description, probability)
    # (one such list for each sample in the batch)
    print('Predicted:', decode_predictions(preds, top=3)[0])




    ## dump keras weights ###############################################

    keras_weights = dict()
    for layer in keras_model.layers:   # dump all weights (trainable and not) to dict {layer_name: layer_weights}
        for layer, layer_weights in zip(layer.weights, layer.get_weights()):
            keras_weights[layer.name] = layer_weights


    for l in keras_weights:
        w = keras_weights[l]

        # BHWC layout to BCHW
        if len(w.shape) == 4:  # (in_c,out_c,y,x) -> (out_c,in_c,y,x)
            w = np.transpose(w, (3, 2, 0, 1))
        if len(w.shape) == 2:  # transpose fc layer
            w = np.transpose(w, (1, 0))

        if l.endswith('depthwise_kernel:0'):
            m, n, H, W = w.shape
            assert(m==1)
            w = w.reshape(n,1,H, W)


        keras_weights[l]=w

    zz=0

    ## assign to pytorch model  ###############################################
    pytorch_model = Xception(in_shape=(3,299,299),num_classes=1000)
    pytorch_model.eval()
    pytorch_state_dict = pytorch_model.state_dict()

    if 1:
        #entry 0
        convert_conv_bn(pytorch_state_dict, 'entry0.0', keras_weights, 'block1_conv1')
        convert_conv_bn(pytorch_state_dict, 'entry0.2', keras_weights, 'block1_conv2')

        #entry 1,2,3
        convert_e_block(pytorch_state_dict, 'entry1', keras_weights, 'block2', 1)
        convert_e_block(pytorch_state_dict, 'entry2', keras_weights, 'block3', 2)
        convert_e_block(pytorch_state_dict, 'entry3', keras_weights, 'block4', 3)

        #middle 1,2,3,3,4,5,6,7,8
        convert_m_block(pytorch_state_dict, 'middle1', keras_weights, 'block5')
        convert_m_block(pytorch_state_dict, 'middle2', keras_weights, 'block6')
        convert_m_block(pytorch_state_dict, 'middle3', keras_weights, 'block7')
        convert_m_block(pytorch_state_dict, 'middle4', keras_weights, 'block8')
        convert_m_block(pytorch_state_dict, 'middle5', keras_weights, 'block9')
        convert_m_block(pytorch_state_dict, 'middle6', keras_weights, 'block10')
        convert_m_block(pytorch_state_dict, 'middle7', keras_weights, 'block11')
        convert_m_block(pytorch_state_dict, 'middle8', keras_weights, 'block12')


        #exit 1
        convert_e_block(pytorch_state_dict, 'exit1', keras_weights, 'block13',4)

        #exit 2
        convert_sep_conv_bn(pytorch_state_dict, 'exit2.conv1', keras_weights, 'block14_sepconv1')
        convert_sep_conv_bn(pytorch_state_dict, 'exit2.conv2', keras_weights, 'block14_sepconv2')

        #layer fc
        convert_fc(pytorch_state_dict, 'fc', keras_weights,  'predictions')

        ##model.load_state_dict(pytorch_state_dict)
        save_pytorch_file = '/root/share/data/models/reference/imagenet/xception/caffe-model/inception/xception/xception.keras.convert.pth'
        torch.save(pytorch_model.state_dict(), save_pytorch_file)


    #exit(0)
    #run some image and test ###############
    if 1:
        img = cv2.imread(img_file)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        #img = np.ones((299,299,3),np.uint8)

        x = img.astype(np.float32)
        x = x.transpose((2,0,1))
        x /= 255.
        x -= 0.5
        x *= 2.

        save_pytorch_file = '/root/share/data/models/reference/imagenet/xception/caffe-model/inception/xception/xception.keras.convert.pth'
        pytorch_model.load_state_dict(torch.load(save_pytorch_file))

        pytorch_prob = pytorch_model( Variable(torch.from_numpy(x).unsqueeze(0).float() ) )
        pytorch_prob = F.softmax(pytorch_prob).data.numpy().reshape(-1)
        #print('pytorch_prob\n',pytorch_prob)
        print('pytorch ', np.argmax(pytorch_prob), ' ', pytorch_prob[np.argmax(pytorch_prob)])



        # hook and dump data ----------------- #-
        class Hook(object):
            def __init__(self, name):
                self.name = name

                # def hook(module, input, output, name):
            def __call__(self, module, input, output):
                name = self.name


                #print module
                output = output.data.numpy()
                input  = input[0].data.numpy()

                _, oc, oh, ow = output.shape
                _, ic, ih, iw = input.shape

                debug_dir ='/root/share/project/kaggle/cdiscount/results/__debug__/pytorch'
                np.savetxt(debug_dir+'/%s_input'%name,input[0,0], fmt='%0.5f')
                np.savetxt(debug_dir+'/%s_output'%name,output[0,0], fmt='%0.5f')

                print(name)
                print(module)
                print('input ', input.shape)
                print('output ', output.shape)
                print('')


        # for i, m in enumerate(pytorch_model.entry0.modules()):
        #     hook= Hook(str(i))
        #     m.register_forward_hook(hook)

        i0=10
        for i, m in enumerate(pytorch_model.entry1.modules()):
             hook= Hook(str(i+i0))
             m.register_forward_hook(hook)

        # hook and dump data ----------------- #-
        pytorch_prob = pytorch_model( Variable(torch.from_numpy(x).unsqueeze(0).float() ) )



    #run some image and test ####################################
    if 1:

        # keras_input_1         = KModel(input=keras_model.input, output=keras_model.get_layer('input_1').output).predict(xx)
        # keras_block1_conv1    = KModel(input=keras_model.input, output=keras_model.get_layer('block1_conv1').output).predict(xx)
        # keras_block1_conv1_bn = KModel(input=keras_model.input, output=keras_model.get_layer('block1_conv1_bn').output).predict(xx)
        # keras_block1_conv2_bn = KModel(input=keras_model.input, output=keras_model.get_layer('block1_conv2_bn').output).predict(xx)
        #
        # keras_batch_normalization_1 = KModel(input=keras_model.input, output=keras_model.get_layer('batch_normalization_1').output).predict(xx)
        # keras_block2_sepconv1       = KModel(input=keras_model.input, output=keras_model.get_layer('block2_sepconv1').output).predict(xx)
        # keras_block2_sepconv1_bn    = KModel(input=keras_model.input, output=keras_model.get_layer('block2_sepconv1_bn').output).predict(xx)
        # keras_block2_sepconv2_bn    = KModel(input=keras_model.input, output=keras_model.get_layer('block2_sepconv2_bn').output).predict(xx)

        keras_block3_sepconv2_bn    = KModel(input=keras_model.input, output=keras_model.get_layer('block3_sepconv2_bn').output).predict(xx)
        keras_batch_normalization_2 = KModel(input=keras_model.input, output=keras_model.get_layer('batch_normalization_2').output).predict(xx)


        keras_block14_sepconv2_act = KModel(input=keras_model.input, output=keras_model.get_layer('block14_sepconv2_act').output).predict(xx)

        keras_preds=preds.reshape(-1,1)
        # keras_input_1         = keras_input_1.transpose((0,3,1,2))
        # keras_block1_conv1    = keras_block1_conv1.transpose((0,3,1,2))
        # keras_block1_conv1_bn = keras_block1_conv1_bn.transpose((0,3,1,2))
        # keras_block1_conv2_bn = keras_block1_conv2_bn.transpose((0,3,1,2))
        #
        # keras_batch_normalization_1 = keras_batch_normalization_1.transpose((0,3,1,2))
        # keras_block2_sepconv1       = keras_block2_sepconv1.transpose((0,3,1,2))
        # keras_block2_sepconv1_bn    = keras_block2_sepconv1_bn.transpose((0,3,1,2))
        # keras_block2_sepconv2_bn    = keras_block2_sepconv2_bn.transpose((0,3,1,2))

        keras_block3_sepconv2_bn    = keras_block3_sepconv2_bn.transpose((0,3,1,2))
        keras_batch_normalization_2 = keras_batch_normalization_2.transpose((0,3,1,2))
        keras_block14_sepconv2_act  = keras_block14_sepconv2_act.transpose((0,3,1,2))


        # print(keras_input_1.shape)
        # print(keras_block1_conv1.shape)
        # print(keras_block1_conv1_bn.shape)

        debug_dir ='/root/share/project/kaggle/cdiscount/results/__debug__/keras'
        # np.savetxt(debug_dir+'/0_keras_input_1_input',keras_input_1[0,0], fmt='%0.5f')
        # np.savetxt(debug_dir+'/0_keras_input_1a_input',keras_input_1[0,1], fmt='%0.5f')
        # np.savetxt(debug_dir+'/0_keras_input_1b_input',keras_input_1[0,2], fmt='%0.5f')
        # np.savetxt(debug_dir+'/1_keras_block1_conv1_output',keras_block1_conv1[0,0], fmt='%0.5f')
        # np.savetxt(debug_dir+'/2_keras_block1_conv1_bn_output',keras_block1_conv1_bn[0,0], fmt='%0.5f')
        # np.savetxt(debug_dir+'/010_keras_block1_conv2_bn_output',keras_block1_conv2_bn[0,0], fmt='%0.5f')

        # np.savetxt(debug_dir+'/027_keras_batch_normalization_1_output',keras_batch_normalization_1[0,0], fmt='%0.5f')
        # np.savetxt(debug_dir+'/012_keras_block2_sepconv1_output',keras_block2_sepconv1[0,0], fmt='%0.5f')
        # np.savetxt(debug_dir+'/016_keras_block2_sepconv1_bn_output',keras_block2_sepconv1_bn[0,0], fmt='%0.5f')
        # np.savetxt(debug_dir+'/016_keras_block2_sepconv2_bn_output',keras_block2_sepconv2_bn[0,0], fmt='%0.5f')


        np.savetxt(debug_dir+'/039_keras_block3_sepconv2_bn_output',keras_block3_sepconv2_bn[0,0], fmt='%0.5f')
        np.savetxt(debug_dir+'/044_keras_batch_normalization_2_output',keras_batch_normalization_2[0,0], fmt='%0.5f')


        np.savetxt(debug_dir+'/0xx_keras_block14_sepconv2_act_output',keras_block14_sepconv2_act[0,9], fmt='%0.5f')
        np.savetxt(debug_dir+'/keras_preds',keras_preds, fmt='%0.5f')



        zz=0



###############################
# test grouped convolution

'''
k = nn.Conv2d(16, 4, kernel_size=3, padding=1, stride=1)
k.weight.size()
torch.Size([4, 16, 3, 3])

input = Variable(torch.randn(1, 16, 32,32))
output = k(input)

output.size()


k = nn.Conv2d(16, 4, kernel_size=3, padding=1, stride=1,groups=2)
k.weight.size()
##out_channels must be divisible by groups
torch.Size([4, 8, 3, 3])

k = nn.Conv2d(16, 4, kernel_size=3, padding=1, stride=1,groups=4)
k.weight.size()
torch.Size([4, 4, 3, 3])



k = nn.Conv2d(16, 16, kernel_size=3, padding=1, stride=1,groups=16)
k.weight.size()
torch.Size([16, 1, 3, 3])
'''
