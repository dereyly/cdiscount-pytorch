"""
This code is based on:
-- https://www.kaggle.com/lamdang/fast-shuffle-bson-generator-for-keras
-- https://www.kaggle.com/humananalog/keras-generator-for-reading-directly-from-bson

(c) Aleksei Tiulpin, 2017

"""

import os
import pandas as pd
import numpy as np
import bson
import cv2
from tqdm import tqdm
import struct
from PIL import Image
from torchvision import transforms as transf
import torch.utils.data as data_utils
import bson




class CdiscountDatasetBSON(data_utils.Dataset):
    def __init__(self, path_bson, split, transform):
        self.data = bson.decode_file_iter(open('/media/dereyly/data/data/train.bson', 'rb'))

        self.transform = transform

    def __getitem__(self, index):
        entry = self.metadata.iloc[index]
        num_imgs, offset, length, target = entry
        obs = get_obs(self.dataset, offset, length)
        keep = np.random.choice(len(obs['imgs']))
        byte_str = obs['imgs'][keep]['picture']
        img = cv2.imdecode(np.fromstring(byte_str, dtype=np.uint8), cv2.IMREAD_COLOR)

        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img = self.transform(img)

        return img, target

    def __len__(self):
        return 7069896


