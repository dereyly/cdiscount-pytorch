import torch.utils.data as data

from PIL import Image
import os
import os.path

# from __future__ import print_function, division
# import os
# import torch
# import pandas as pd
# from skimage import io, transform
# import numpy as np
# import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pickle as pkl
import numpy as np
# from torchvision import transforms, utils

# # Ignore warnings
# import warnings
# warnings.filterwarnings("ignore")
def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


# def accimage_loader(path):
#     import accimage
#     try:
#         return accimage.Image(path)
#     except IOError:
#         # Potentially a decoding problem, fall back to PIL.Image
#         return pil_loader(path)

def default_loader(path):
    # from torchvision import get_image_backend
    # if get_image_backend() == 'accimage':
    #     return accimage_loader(path)
    # else:
    return pil_loader(path)

class CDiscountDatasetMy(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root, pkl_name, transform=None, loader=default_loader):
        with open(pkl_name,'rb') as f_pkl:
            self.imgs=pkl.load(f_pkl)
        self.root = root
        #self.classes = range(num_classes)
        self.transform = transform
        self.loader = loader
        self.idx2cls=pkl.load(open('/home/dereyly/ImageDB/cdiscount/multi_idx2cls.pkl','rb'))


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        fname, target = self.imgs[index]
        # target=int(target)
        path=self.root+'/' +fname
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        #maybe is better to not convert into numpy
        target_multi=[target[0]]
        for k in range(self.idx2cls.shape[0]):
            target_multi.append(self.idx2cls[k][target[0]])
        return img, target_multi #np.array(target,int)

    def __len__(self):
        return len(self.imgs)
