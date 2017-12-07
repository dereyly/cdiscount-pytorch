## https://github.com/pytorch/examples/blob/master/imagenet/main.py
##  dataset transform:
##     train: transforms.RandomSizedCrop(224),
##     validate:  transforms.Scale(256),
##                transforms.CenterCrop(224),
##  http://pytorch.org/docs/master/torchvision/transforms.html

from dataset.tool import *
from utility.file import *

IMAGENET_DIR='/root/share/data/imagenet/ILSVRC2012'

#
# #data iterator ----------------------------------------------------------------

class Imagenet2012Dataset(Dataset):

    def __init__(self, split, folder, transform=[], mode='train'):
        super(Imagenet2012Dataset, self).__init__()
        self.split  = split
        self.folder = folder
        self.mode   = mode
        self.transform = transform

        #make dictionary to convert from name to label
        synset_file = IMAGENET_DIR + '/synset_words'
        lines = read_list_from_file(synset_file, comment='#')

        names        = [line[:9] for line in lines]
        descriptions = [line[9:] for line in lines]
        self.names = names
        self.descriptions = descriptions

        num_classes = len(self.names)
        self.name_to_label = dict(zip(self.names, range(num_classes)))

        #image index
        print ('load image index ...')
        split_file = IMAGENET_DIR +  '/splits/%s'%split
        ids = read_list_from_file(split_file)
        ids = [id[:-5] for id in ids]

        #annotations
        print ('load annotation ...')
        labels = [self.name_to_label[id[:9]] for id in ids]

        #save
        self.ids = ids
        self.labels = labels


    def get_img_file(self, id):
        img_file = IMAGENET_DIR + '/images/%s/%s.JPEG'%(self.folder,id)
        return img_file

    def get_train_item(self, index):
        id  = self.ids[index]
        img_file = self.get_img_file(id)
        img   = cv2.imread(img_file)
        label = self.labels[index]

        for t in self.transform:
            img = t(img)

        return img, label, index


    def get_eval_item(self, index):
        id  = self.ids[index]
        img_file = self.get_img_file(id)
        img = cv2.imread(img_file)
        return img, index

    def __getitem__(self, index):
        if self.mode=='train':
            return self.get_train_item(index)
        if self.mode=='eval':
            return self.get_eval_item (index)

    def __len__(self):
        return len(self.ids)



# ## ------------------------------------------------------------------------
# def run_check_dataset():
#
#     dataset = Imagenet2012Dataset( 'train_1281167', 'train',
#                                    transform=[
#                                        lambda img: set_smallest_size_transform(img, 256),
#                                        lambda img: crop_center_transform(img, (224,224))
#                                    ],
#                                    mode='train' )
#     sampler = RandomSamplerWithLength(dataset,100)
#
#     for n in iter(sampler):
#         img, label, index = dataset[n]
#
#         id = dataset.ids[index]
#         name = dataset.names[label] + ' ' + dataset.descriptions[label]
#
#         print('%09d  %s  %d, %s'%(index, id, label, str(img.shape)))
#         draw_label(img, label, name)
#         im_show('img',img)
#
#         cv2.waitKey(0)
#
#
# # etc ---------------------------------
# def run_make_split():
#
#     folder = 'val'
#     split_file = IMAGENET_DIR + '/splits/xxx'
#
#     # ---
#     synset_file = IMAGENET_DIR + '/synset_words'
#     lines = read_list_from_file(synset_file, comment='#')
#
#     num_img_files=0
#     with open(split_file, 'w') as f:
#
#         for i,line in enumerate(lines):
#             name = line[0:9]
#
#             img_files = sorted(glob.glob(IMAGENET_DIR + '/images/%s/%s/*.JPEG'%(folder,name)))
#             ids = [ img_file.replace(IMAGENET_DIR + '/images/%s/'%folder, '') for img_file in img_files ]
#
#             for id in ids:
#                 f.write('%s\n'%id)
#
#             num_img_files += len(ids)
#             print ('[%05d] %s: %d'%(i,name,len(ids)))
#             #break
#
#     print ('num_img_files=%d'%num_img_files)
#
#




# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    #run_check_dataset()
    #run_make_split()


    print('\nsucess!')







