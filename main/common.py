import os
#os.environ['PYTHONUNBUFFERED'] = '1'
#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
PROJECT_PATH = os.path.dirname(os.path.realpath(__file__))

#numerical libs
import math
import numpy as np
import random
import PIL
import cv2

import matplotlib
matplotlib.use('TkAgg')
#matplotlib.use('Qt4Agg')
#matplotlib.use('Qt5Agg')


# torch libs
import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import *

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

import torch.backends.cudnn as cudnn



# std libs
import collections
import numbers
import inspect
import shutil
#import pickle
#import dill
from timeit import default_timer as timer   #ubuntu:  default_timer = time.time,  seconds
from datetime import datetime
import csv
import pandas as pd
import pickle
import glob
import sys
#from time import sleep
from distutils.dir_util import copy_tree
import time


import matplotlib.pyplot as plt
import sklearn
import sklearn.metrics
from scipy import sparse

#from skimage import io
import io




'''
updating pytorch
    https://discuss.pytorch.org/t/updating-pytorch/309
    
    ./conda config --add channels soumith
    conda update pytorch torchvision
    conda install pytorch torchvision cuda80 -c soumith
    

'''


#---------------------------------------------------------------------------------
print('@%s:  ' % os.path.basename(__file__))
if 1:
    SEED=int(time.time()) #235202
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    print ('\tset random seed')
    print ('\t\tSEED=%d'%SEED)

if 1:
    cudnn.benchmark = True  ##uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms. -
    cudnn.enabled   = True
    print ('\tset cuda environment')
    try:
        print ('\t\tos[\'CUDA_VISIBLE_DEVICES\'] = ',os.environ['CUDA_VISIBLE_DEVICES'])
    except Exception:
        pass

    print ('\t\ttorch.cuda.device_count()   =', torch.cuda.device_count())
    print ('\t\ttorch.cuda.current_device() =', torch.cuda.current_device())


print('')

#---------------------------------------------------------------------------------