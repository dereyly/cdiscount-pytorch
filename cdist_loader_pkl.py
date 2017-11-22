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
# from torchvision import transforms, utils

# # Ignore warnings
# import warnings
# warnings.filterwarnings("ignore")
def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

class CDiscountDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root, pkl_name,state, transform=None, loader=default_loader):
        with open(pkl_name,'rb'):
            train_list=pkl.load(open(pkl_name,'rb'))[state]
        self.root = root
        self.imgs=[]
        for data_line in train_list:
            self.imgs.append((data_line['path'],data_line['cls']))
        #self.classes = range(num_classes)
        self.transform = transform
        self.loader = loader


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

        return img, target

    def __len__(self):
        return len(self.imgs)
