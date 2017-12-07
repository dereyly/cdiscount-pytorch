import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

from common import *
from dataset.transform import *
#from net.model.cdiscount.multi_crop_net import MultiCropNet_12 as MultiCropNet
#from trainer_excited_resnet50 import *
from trainer_excited_inception3_180 import *

#--------------------------------------------------------------
def run_training():


  pass

# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    do_submit()

    print('\nsucess!')