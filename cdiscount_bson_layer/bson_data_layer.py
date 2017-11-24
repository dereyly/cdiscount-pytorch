import sys
sys.path.insert(0, '/home/dereyly/progs/caffe-nccl/python')
# sys.path.insert(0, '/home/awin/progs/protobuf-3.2.0/python')

import bson
import pickle
import io
from skimage.data import imread
import numpy as np
import cv2
# print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++3')
print(sys.version_info)
import caffe
import time
# print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++4')
'''
layer {
  name: "Data"
  type: "Python"
  top: "data"
  top: "label"

  python_param {
    module: "bson_data_layer"
    layer: "BSON_Data_Layer"
    param_str: '{"batch_size": 16, "bson_file": "path_to_bson_file.txt", "train_and_val_pkl": "path_to_train_and_val_pkl", "category_statistic_pkl": "category_statistic.pkl", "dataset_type": "train"}'
  }
}
'''
def RandCrop(img,crop_sz):
    sz=img.shape
    dx = np.random.randint(0, sz[0] - crop_sz[0])
    dy = np.random.randint(0, sz[1] - crop_sz[1])
    img_out=img[dx:dx+crop_sz[0],dy:dy+crop_sz[1]]
    return img_out

class BSON_Data_Layer(caffe.Layer):
    def setup(self, bottom, top):
        # Read parameters

        print(1)
        params = eval(self.param_str)
        print(2)
        self.bson_file_path = params["bson_file"]
        train_and_val_pkl_file_path = params["train_and_val_pkl"]
        category_statistic_pkl_file_path = params["category_statistic_pkl"]
        self.batch_size = params["batch_size"]
        self.dataset_type = params["dataset_type"]
        print(3)
        # Read source file
        # I'm just assuming we have this method that reads the source file
        # and returns a list of tuples in the form of (img, label)
        self.bson_file_pointer = bson.decode_file_iter(open(self.bson_file_path, 'rb'))
        with open(train_and_val_pkl_file_path, 'rb') as f:
            train_and_val_pkl_file = pickle.load(f)
        if self.dataset_type == 'train':
            self.products = train_and_val_pkl_file['train']
        elif self.dataset_type == 'val':
            self.products = train_and_val_pkl_file['val']
        elif self.dataset_type == 'test':
            self.products = train_and_val_pkl_file['val']
        else:
            assert(False), 'dataset_type is not correct'
        del train_and_val_pkl_file
        print(4)
        with open(category_statistic_pkl_file_path, 'rb') as f:
            category_statistic = dict(pickle.load(f))
        self.category_to_ind = {}
        #for id_dict, (key, value) in enumerate(category_statistic.iteritems()):
        for id_dict, (key, value) in enumerate(category_statistic.items()):
            self.category_to_ind[key] = np.float32(id_dict)
        del category_statistic
        print(5)
        self.num_products = len(self.products)
        self.counter = 0  # use this to check if we need to restart the list of imgs
        self.out_sz=(160,160)
        ###### Reshape top ######
        # This could also be done in Reshape method, but since it is a one-time-only
        # adjustment, we decided to do it on Setup
        top[0].reshape(self.batch_size, 3, self.out_sz[0], self.out_sz[1])  # imgs
        top[1].reshape(self.batch_size)  # labels


    def forward(self, bottom, top):
        #t1=time.time()
        for itt in range(self.batch_size):
            # Use the batch loader to load the next image.
            im, label = self.load_next_image()

            # Add directly to the top blob
            top[0].data[itt, ...] = im
            top[1].data[itt, ...] = label
        #if self.counter<100:
        #print("time=",time.time()-t1)

    def load_next_image(self):
        # If we have finished forwarding all images, then an epoch has finished
        # and it is time to start a new one
        img_loaded = False
        while(not img_loaded):
            if self.counter >= self.num_products:
                self.bson_file_pointer = bson.decode_file_iter(open(self.bson_file_path, 'rb'))
                self.counter = 0
            else:
                cur_d = next(self.bson_file_pointer)
                self.counter += 1
                product_id = cur_d['_id']
                #print "product_id",product_id, "self.counter", self.counter
                if self.dataset_type=='test':
                    if not product_id in self.products:
                         continue
                else:
                    if product_id in self.products:
                         continue
                category_id = cur_d['category_id']
                pictures = []
                for e, pic in enumerate(cur_d['imgs']):
                    picture = imread(io.BytesIO(pic['picture']))
                    # convert RGB to BGR
                    picture_bgr = picture[:, :, ::-1]
                    pictures.append(picture_bgr)
                cur_pic_id = np.random.randint(0, len(pictures))
                img_out = pictures[cur_pic_id]
                #img_out = cv2.resize(img_out, (224, 224))
                img_out=RandCrop(img_out, self.out_sz)
                img_out-=np.array([104,117,127],np.uint8)
                img_out = np.transpose(img_out, (2, 0, 1))
                label_out = self.category_to_ind[category_id]
                return img_out, label_out

    def reshape(self, bottom, top):
        """
        There is no need to reshape the data, since the input is of fixed size
        (img shape and batch size)

        If we were processing a fixed-sized number of images (for example in Testing)
        and their number wasn't a  multiple of the batch size, we would need to
        reshape the top blob for a smaller batch.
        """
        pass

    def backward(self, bottom, top):
        """
        This layer does not back propagate
        """
        pass