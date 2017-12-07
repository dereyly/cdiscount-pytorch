from common import *
#from dataset.tool import *
from utility.file import *

import bson
CDISCOUNT_DIR = '/media/ssd/data/kaggle/cdiscount'

## https://qiita.com/wasnot/items/be649f289073fb96513b
## https://www.kaggle.com/inversion/processing-bson-files
## https://www.kaggle.com/carlossouza/extract-image-files-from-bson-and-save-to-disk/code
## test Processing BSON Files

def run_check_bson():
    bson_file='/media/ssd/data/kaggle-cdiscount/__download__/train_example.bson'
    data = bson.decode_file_iter(open(bson_file, 'rb'))

    prod_to_category = dict()
    for c, d in enumerate(data):
        product_id  = d['_id']
        category_id = d['category_id']
        prod_to_category[product_id] = category_id
        for e, pic in enumerate(d['imgs']):
            img = cv2.imdecode(np.asarray(bytearray(pic['picture']), dtype=np.uint8), -1)
            #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            im_show('img',img,1)
            cv2.waitKey(1)

    prod_to_category = pd.DataFrame.from_dict(prod_to_category, orient='index')
    prod_to_category.index.name = '_id'
    prod_to_category.rename(columns={0: 'category_id'}, inplace=True)
    prod_to_category.to_csv('/media/ssd/data/kaggle-cdiscount/__temp__/xxx.csv')

## https://www.kaggle.com/bguberfain/not-so-naive-way-to-convert-bson-to-files/
#  /train/[category]/[_id]-[index].jpg
def run_test_boson_to_image():

    bson_file = '/media/ssd/data/kaggle/cdiscount/__download__/test.bson'
    num_products= 1768182 # 7069896 for train and 1768182 for test
    out_dir = CDISCOUNT_DIR+ '/test'

    os.makedirs(out_dir,exist_ok=True)

    with open(bson_file, 'rb') as fbson:
        data = bson.decode_file_iter(fbson)
        #num_products = len(list(data))
        #print ('num_products=%d'%num_products)
        #exit(0)

        for n, d in enumerate(data):
            print('%08d/%08d'%(n,num_products))

            _id = d['_id']
            for i, pic in enumerate(d['imgs']):
                img_file = out_dir + '/' +'%s-%d.jpg'%(str(_id),i)
                #print(img_file)

                with open(img_file, 'wb') as f:
                    f.write(pic['picture'])



def run_train_boson_to_image():

    bson_file = '/media/ssd/data/kaggle/cdiscount/__download__/train.bson'
    num_products= 7069896   # 7069896 for train and 1768182 for test
    out_dir = CDISCOUNT_DIR+ '/train'

    os.makedirs(out_dir,exist_ok=True)
    categories = pd.read_csv(CDISCOUNT_DIR+'/category_names.csv', index_col='category_id')
    for category in categories.index:
        os.makedirs(out_dir+'/'+str(category),exist_ok=True)

    with open(bson_file, 'rb') as fbson:
        data = bson.decode_file_iter(fbson)
        #num_products = len(list(data))
        #print ('num_products=%d'%num_products)
        #exit(0)

        for n, d in enumerate(data):
            print('%08d/%08d'%(n,num_products))

            category = d['category_id']
            _id = d['_id']
            for i, pic in enumerate(d['imgs']):
                img_file = out_dir + '/' + str(category) + '/' +'%s-%d.jpg'%(str(_id), i)
                #print(img_file)

                with open(img_file, 'wb') as f:
                    f.write(pic['picture'])


def run_make_train_summary():

    bson_file = '/media/ssd/data/kaggle/cdiscount/__download__/train.bson'
    num_products= 7069896 # 7069896 for train and 1768182 for test
    out_dir = CDISCOUNT_DIR

    id = []
    num_imgs = []
    category_id = []

    with open(bson_file, 'rb') as fbson:
        data = bson.decode_file_iter(fbson)
        #num_products = len(list(data))
        #print ('num_products=%d'%num_products)
        #exit(0)

        for n, d in enumerate(data):
            print('\r%08d/%08d'%(n,num_products), flush=True,end='')

            category_id.append(d['category_id'])
            id.append(d['_id'])
            num_imgs.append(len(d['imgs']))
        print('')

    #by product id
    df = pd.DataFrame({ '_id' : id, 'num_imgs' : num_imgs, 'category_id' : category_id})
    df.to_csv('/media/ssd/data/kaggle/cdiscount/__temp__/train_by_product_id.csv', index=False)
    t = df['num_imgs'].sum()  #check :12371293
    print(t)

    #split by id --------------------------------------
    id_random = list(id)
    random.shuffle(id_random)

    #make train, valid
    num_train = int(0.8*(num_products))
    num_valid = num_products - num_train

    #by id
    file1 = CDISCOUNT_DIR +'/split/'+ 'train_id_v0_%d'%(num_train)
    file2 = CDISCOUNT_DIR +'/split/'+ 'valid_id_v0_%d'%(num_valid)
    id1 = id_random[0:num_train]
    id2 = id_random[num_train: ]
    write_list_to_file(id1, file1)
    write_list_to_file(id2, file2)


    #summary ------------------------------------
    g=(df.groupby('category_id')
       .agg({'_id':'count', 'num_imgs': 'sum'})
       .reset_index()
    )
    g.to_csv('/media/ssd/data/kaggle/cdiscount/__temp__/train_g.csv', index=False)




def run_make_split_file():
    split_id_file ='/media/ssd/data/kaggle/cdiscount/split/valid_id_v0_1413980'
    split_file ='/media/ssd/data/kaggle/cdiscount/split/valid_v0_1413980'

    id = read_list_from_file(split_id_file)
    df = pd.read_csv (CDISCOUNT_DIR + '/train_by_product_id.csv')
    df = df.set_index('_id')


    count=0
    if 1:
    #with open(split_file, 'w') as f:
        for i in id:
            category_id = df.loc[int(i)].values[0]
            num_imgs    = df.loc[int(i)].values[1]
            count += num_imgs
            # for n in range(num_imgs):
            #     f.write('%9d\t%d\t%s\n'%(int(i),n,category_id))
    ptint(count)






def run_make_test_summary():

    bson_file = '/media/ssd/data/kaggle/cdiscount/__download__/test.bson'
    num_products= 1768182 # 7069896 for train and 1768182 for test
    out_dir = CDISCOUNT_DIR

    if 0:
        id = []
        num_imgs = []
        with open(bson_file, 'rb') as fbson:
            data = bson.decode_file_iter(fbson)

            for n, d in enumerate(data):
                print('\r%08d/%08d'%(n,num_products), flush=True,end='')

                id.append(d['_id'])
                num_imgs.append(len(d['imgs']))
            print('')

        #by product id
        df = pd.DataFrame({ '_id' : id, 'num_imgs' : num_imgs })
        df.to_csv('/media/ssd/data/kaggle/cdiscount/__temp__/test_by_product_id.csv', index=False)
        t = df['num_imgs'].sum()  #check :12371293
        print('total num of images = %d'%t)


    if 1:
        df = pd.read_csv (CDISCOUNT_DIR + '/test_by_product_id.csv')

    #id list
    num_test = num_products
    file = CDISCOUNT_DIR +'/split/'+ 'test_id_%d'%(num_test)
    id   = list(df['_id'])

    write_list_to_file(id, file)


# --------------------------------------------------------------------
# http://pbpython.com/pandas_transform.html
# https://stackoverflow.com/questions/32038427/pandas-for-each-row-in-df-copy-row-n-times-with-slight-changes
# https://stackoverflow.com/questions/19377969/combine-two-columns-of-text-in-dataframe-in-pandas-python
def run_csv_to_img_file():

    # csv_file = '/media/ssd/data/kaggle/cdiscount/test_by_product_id.csv'
    # product_id_df = pd.read_csv (csv_file)
    # product_id_100_df = product_id_df[0:100]
    # product_id_100_df.to_csv('/media/ssd/data/kaggle/cdiscount/__temp__/test_by_product_100_id.csv', index=False)

    csv_file = '/media/ssd/data/kaggle/cdiscount/__temp__/test_by_product_100_id.csv'
    product_id_df = pd.read_csv (csv_file)

    df = product_id_df
    df = df.reindex(np.repeat(df.index.values, df['num_imgs']), method='ffill')
    df['cum_sum'] = df.groupby(['_id']).cumcount()
    df['img_file'] = df['_id'].astype(str)+ '-' + df['cum_sum'].astype(str)  + '.jpg'
    df = df['img_file'].reset_index(drop=True)

    xx=0

########################################################################

def run_make_new_split():
    valid_id_v0_5000    = np.loadtxt('/media/ssd/data/kaggle/cdiscount/split/valid_id_v0_5000',np.int32)
    train_id_v0_5655916 = np.loadtxt('/media/ssd/data/kaggle/cdiscount/split/train_id_v0_5655916',np.int32)
    valid_id_v0_1413980 = np.loadtxt('/media/ssd/data/kaggle/cdiscount/split/valid_id_v0_1413980',np.int32)

    valid = valid_id_v0_1413980[:50000]
    train = np.concatenate((train_id_v0_5655916,valid_id_v0_1413980[50000:]))
    xx=0


    np.savetxt('/media/ssd/data/kaggle/cdiscount/split/train_id_v0_7019896',train, fmt='%d')
    np.savetxt('/media/ssd/data/kaggle/cdiscount/split/valid_id_v0_50000',valid, fmt='%d')
    pass

# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_make_new_split()

