##  https://www.kaggle.com/c/cdiscount-image-classification-challenge
##  Cdiscount’s Image Classification Challenge
##

from dataset.transform import *
from dataset.sampler import *
from utility.file import *
from utility.draw import *
import time



CDISCOUNT_DIR = '/home/dereyly/ImageDB/cdiscount'
CDISCOUNT_NUM_CLASSES = 5270
CDISCOUNT_HEIGHT=180
CDISCOUNT_WIDTH =180

#
# #data iterator ----------------------------------------------------------------

class CDiscountDataset(Dataset):

    def __init__(self, split, folder, transform=[], mode='train'):
        super(CDiscountDataset, self).__init__()
        self.split  = split
        self.folder = folder
        self.mode   = mode
        self.transform = transform

        #label to name
        category_names_df = pd.read_csv (CDISCOUNT_DIR + '/category_names.csv')
        category_names_df['label'] = category_names_df.index

        label_to_category_id = dict(zip(category_names_df['label'], category_names_df['category_id']))
        category_id_to_label = dict(zip(category_names_df['category_id'], category_names_df['label']))

        self.category_names_df    = category_names_df
        self.label_to_category_id = label_to_category_id
        self.category_id_to_label = category_id_to_label

        if mode=='train':
            print('read img list')
            t1=time.time()
            ids = read_list_from_file(CDISCOUNT_DIR + '/split/' + split, comment='#', func=int)
            num_ids = len(ids)
            t2 = time.time()
            print('t1=',t2-t1)
            train_df = pd.read_csv (CDISCOUNT_DIR + '/train_by_product_id.csv')
            t3= time.time()
            print('t2=', t3 - t2)
            #df.columns
            #Index(['_id', 'category_id', 'num_imgs'], dtype='object')

            start = timer()
            df = train_df.reset_index()
            df = df[ df['_id'].isin(ids)]
            df = df.reindex(np.repeat(df.index.values, df['num_imgs']), method='ffill')
            df['cum_sum' ] = df.groupby(['_id']).cumcount()
            t4=time.time()
            print('t3=', t4 - t3)
            #df['img_file'] = CDISCOUNT_DIR + '/image/'+ folder + '/' + df['category_id'].astype(str) + '/' +df['_id'].astype(str) + '-' + df['cum_sum'].astype(str)  + '.jpg'
            df['img_file'] = CDISCOUNT_DIR + '/' + folder + '/' + df['category_id'].astype(str) + '/' + df['_id'].astype(str) + '_' + df['cum_sum'].astype(str) + '.png'
            #print(df['img_file'][0])
            t5=time.time()
            print('t4=', t5 - t4)
            df['label'] = df['category_id'].map(category_id_to_label)

            img_files = list(df['img_file'])
            labels    = list(df['label'])
            num_img_files = len(img_files)

            print('\tnum_ids = %d'%(num_ids))
            print('\tnum_img_files = %d'%(num_img_files))
            print('\ttime = %0.2f min'%((timer() - start) / 60))
            print('')

        elif mode=='test':
            print('read img list')
            ids = read_list_from_file(CDISCOUNT_DIR + '/split/' + split, comment='#', func=int)
            num_ids = len(ids)

            test_df = pd.read_csv (CDISCOUNT_DIR + '/%s_by_product_id.csv'%folder)
            #df.columns
            #Index(['_id', 'num_imgs'], dtype='object')

            start = timer()
            df = test_df.reset_index()
            df = df[ df['_id'].isin(ids)]
            df = df.reindex(np.repeat(df.index.values, df['num_imgs']), method='ffill')
            df['cum_sum' ] = df.groupby(['_id']).cumcount()

            if folder=='test':
                df['img_file'] = CDISCOUNT_DIR + '/image/'+ folder + '/' + df['_id'].astype(str) + '-' + df['cum_sum'].astype(str)  + '.jpg'
            if folder=='train':
                df['img_file'] = CDISCOUNT_DIR + '/image/'+ folder + '/' + df['category_id'].astype(str) + '/' +df['_id'].astype(str) + '-' + df['cum_sum'].astype(str)  + '.jpg'

            img_files = list(df['img_file'])
            labels    = None
            num_img_files = len(img_files)

            print('\tnum_ids = %d'%(num_ids))
            print('\tnum_img_files = %d'%(num_img_files))
            print('\ttime = %0.2f min'%((timer() - start) / 60))
            print('')


        else:
            raise NotImplementedError

        assert( num_img_files!=0)
        if 0: #debug
            for i,img_file in enumerate(img_files):
                print('\r%10d %s'%(i,img_file),end='',flush=True)
                assert os.path.exists(img_file), 'file does not exist: %s'%img_file
            print('\n')

        #save
        self.transform = transform
        self.mode      = mode
        self.df        = df
        self.img_files = img_files
        self.labels    = labels


    def __getitem__(self, index):
        if self.mode=='train':
            image = cv2.imread(self.img_files[index], 1)
            if image is None:
                print('not read image -->',self.img_files[index])
            label = self.labels[index]
            for t in self.transform:
                image = t(image)

            #image = pytorch_image_to_tensor_transform(image)
            return image, label, index

        elif self.mode=='test':
            image = cv2.imread(self.img_files[index], 1)
            if image is None:
                print('not read image -->',self.img_files[index])
            for t in self.transform:
                image = t(image)

            #image = pytorch_image_to_tensor_transform(image)
            return image, index

        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.img_files)




## check ## ----------------------------------------------------------
## https://github.com/hujie-frank/SENet
##   Random Mirror	True
##   Random Crop	8% ~ 100%
##   Aspect Ratio	3/4 ~ 4/3
##   Random Rotation -10° ~ 10°
##   Pixel Jitter	 -20 ~ 20  (shift)
##


def run_check_dataset():

    dataset = CDiscountDataset( #'train_id_v0_5655916', 'train', mode='train',
                                #'test_id_1768182', 'test', mode='test',
                                'debug_train_id_v0_5000', 'train', mode='train',
                                   transform=[

                                       # lambda x: random_shift_scale_rotate(x,
                                       #          #shift_limit  = [0, 0],
                                       #          shift_limit  = [-0.04,  0.04],
                                       #          scale_limit  = [256/180, 1.3*256/180],
                                       #          rotate_limit = [-10,10],
                                       #          aspect_limit = [1,1],
                                       #          size=[224,224],
                                       #  borderMode=cv2.BORDER_REFLECT_101 , u=1),


                                       lambda x: random_crop_scale(x,
                                           scale_limit=[1/1.5,1.5],
                                           size=[-1,-1],
                                        u=1)
                                    ],
                                )
    #sampler = RandomSampler1(dataset,100)
    sampler = FixedSampler(dataset,[0]*1000)

    if dataset.mode=='train' :
        for n in iter(sampler):
            image, label, index = dataset[n]
            print('%09d %d, %s'%(index, label, str(image.shape)))
            im_show('image',image)
            cv2.waitKey(0)

    elif dataset.mode=='test' :
        for n in iter(sampler):
            image, index = dataset[n]
            print('%09d, %s'%(index, str(image.shape)))
            im_show('image',image)
            cv2.waitKey(0)





# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_check_dataset()
