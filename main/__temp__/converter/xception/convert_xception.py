#https://github.com/gzuidhof/nn-transfer/blob/master/example.ipynb

import keras
from keras import backend as K
from keras import layers
from keras.layers import Dense
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import SeparableConv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D


from collections import OrderedDict
import numpy as np
import h5py

import sys
import os

from keras.layers import *
from keras.optimizers import *
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as k

from keras.applications import *
#from xception import *

########################################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    model_dir='/root/share/data/models/reference/imagenet/xception'
    weights_path=model_dir + '/xception_weights_tf_dim_ordering_tf_kernels.h5'


    keras_model = Xception(include_top=True, weights='imagenet',
             input_tensor=None, input_shape=None,
             pooling=None,
             classes=1000)
    #model.load_weights(h5py_file)

    keras_model.save(model_dir + '/temp.h5')


    with h5py.File(model_dir + '/temp.h5') as f:
        model_weights = f['model_weights']
        layer_names = list(map(str, keras_model.keys()))

        print(layer_names)