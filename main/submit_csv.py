import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

from common import *
from dataset.transform import *
#from net.model.cdiscount.multi_crop_net import MultiCropNet_12 as MultiCropNet
#from trainer_excited_resnet50 import *
#from trainer_excited_inception3_180 import *
#from trainer_resnext101_32x4d_180 import *
from trainer_xception_180 import *



## common functions ##

if 0:
    def single_test_augment(image):
         image = cv2.resize(image,(160,160))
         tensor = image_to_tensor_transform(image)
         return tensor


    def multi_test_augment(image):
        images = fix_multi_crop(image, roi_size=(224,224))
        tensors=[]
        for image in images:
            tensor = image_to_tensor_transform(image)
            tensors.append(tensor)

        return tensors

if 1:  # follow valid--------------------

    single_test_augment = valid_augment

    def multi_test_augment(image):
        raise NotImplementedError



#--------------------------------------------------------------
# predict as uint8
def scores_to_csv(scores, dataset, save_dir):

    df = dataset.df[['_id','num_imgs']]
    df = df.groupby(['_id']).agg({'num_imgs': 'mean'}).reset_index()
    df['cumsum'] = df['num_imgs'].cumsum()

    ids      = df['_id'].values
    num_imgs = df['num_imgs'].values
    cumsum   = df['cumsum'].values
    num_products  = len(ids)
    num_img_files = len(scores)

    assert(df['cumsum'].iloc[[-1]].values[0] == num_img_files)
    print('')
    print('making submission csv')
    print('\tnum_products=%d'%num_products)
    print('\tnum_img_files=%d'%num_img_files)
    print('')

    start  = timer()
    labels = []
    probs  = []
    for n in range(num_products):
        if n%10000==0:
            print('\r\t%10d/%d (%0.0f %%)  %0.2f min'%(n, num_products, 100*n/num_products,
                             (timer() - start) / 60), end='',flush=True)
        num = num_imgs[n]
        end = cumsum[n]
        if num_imgs[n]==1:
            s = scores[end-1]
        else:
            s = scores[end-num:end].mean(axis=0)

        l = s.argmax()
        labels.append(l)
        probs.append(s[l]/255)
    pass
    print('\n')

    # save results ---
    labels = np.array(labels)
    probs  = np.array(probs)
    df = pd.DataFrame({ '_id' : ids, 'category_id' : labels})
    df['category_id'] = df['category_id'].map(dataset.label_to_category_id)

    return df, labels, probs


def single_predict(net, test_loader):

    test_num  = len(test_loader.dataset)
    scores = np.zeros((test_num,CDISCOUNT_NUM_CLASSES), np.uint8)
    n = 0
    start = timer()
    for i, (images, indices) in enumerate(test_loader, 0):
        print('\r%5d:  %10d/%d (%0.0f %%)  %0.2f min'%(i, n, test_num, 100*n/test_num,
                         (timer() - start) / 60), end='',flush=True)

        # forward
        images = Variable(images,volatile=True).cuda(async=True)
        logits = net(images)
        #logits = net(images.half())  #
        probs = F.softmax(logits)
        probs = probs.data.cpu().numpy()*255

        batch_size = len(indices)
        scores[n:n+batch_size]=probs
        n += batch_size

    print('\n')
    assert(n == len(test_loader.sampler) and n == test_num)
    return scores




def multi_predict(net, test_loader):

    test_num  = len(test_loader.dataset)
    scores = np.zeros((test_num,CDISCOUNT_NUM_CLASSES), np.uint8)

    n = 0
    start = timer()
    for i, (all_images, indices) in enumerate(test_loader, 0):
        print('\r%5d:  %10d/%d (%0.0f %%)  %0.2f min'%(i, n, test_num, 100*n/test_num,
                         (timer() - start) / 60), end='',flush=True)

        # forward
        batch_size = len(indices)

        all_probs = torch.cuda.FloatTensor(batch_size,CDISCOUNT_NUM_CLASSES).fill_(0)
        num_arguments = len(all_images)
        for images in all_images:

            images = Variable(images,volatile=True).cuda(async=True)
            logits = net(images)
            probs = F.softmax(logits)
            all_probs += probs.data

        pass
        all_probs = all_probs/num_arguments *255
        probs = all_probs.cpu().numpy()

        scores[n:n+batch_size]=probs
        n += batch_size

    print('\n')
    assert(n == len(test_loader.sampler) and n == test_num)
    return scores



def do_submit():

    CROP='SINGLE' #'SINGLE' #
    if CROP=='SINGLE': ## single-icrop
        predict      = single_predict
        test_augment = single_test_augment
        #test_augment=valid_transform

    elif CROP=='MULTI': ## multi-crop
        predict      = multi_predict
        test_augment = multi_test_augment



    out_dir = '/root/share/project/kaggle/cdiscount/results/xception-180-01b' # s_xx1'
    checkpoint = out_dir +'/checkpoint/00158000_model.pth'  #final

    ## ------------------------------------
    os.makedirs(out_dir +'/submit', exist_ok=True)

    log = Logger()
    log.open(out_dir+'/log.submit.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 64))
    log.write('** some experiment setting **\n')
    log.write('\tPROJECT_PATH = %s\n' % PROJECT_PATH)
    log.write('\tcheckpoint   = %s\n'%checkpoint)
    log.write('\n')


    ## net ---------------------------------
    log.write('** net setting **\n')
    log.write('\n')

    net = Net(in_shape = (3,CDISCOUNT_HEIGHT,CDISCOUNT_WIDTH), num_classes=CDISCOUNT_NUM_CLASSES)
    net.load_state_dict(torch.load(checkpoint))
    #net.merge_bn()
    net.cuda()
    #net.half()  #use fp16 for fast experiments. use fp32 for final submission!
    net.eval()


    ## prediction  ----------------------------
    augment = '00117000_180' #CROP  #'default'
    splits=['test_id_v0_420000','test_id_v1_420000','test_id_v2_420000','test_id_v3_508182']
    num_splits = len(splits)

    ## if  'MULTI' in CROP: net = MultiCropNet(net)
    if 1:

        #split into several parts
        for i in range(num_splits):
            split = splits[i]
            #if i==0 or i==1:continue

            save_dir = out_dir +'/submit/%s/part-%d'%(augment,i)
            os.makedirs(save_dir, exist_ok=True)
            log.write('\n** save_dir = %s **\n'%save_dir)
            log.write('\tsplit = %s\n\n'%split)

            test_dataset = CDiscountDataset( split, 'test', mode='test',
                                         #'test_id_1768182', 'test', mode='test',
                                         #'test_id_6813', 'test', mode='test',
                                        transform =[ lambda x:test_augment(x),])
            test_loader  = DataLoader(
                                test_dataset,
                                sampler     = SequentialSampler(test_dataset), #FixedSampler(test_dataset,[0,1,2,]),  #
                                batch_size  = 512,  #880, #784,
                                drop_last   = False,
                                num_workers = 4,
                                pin_memory  = True)

            start = timer()
            scores = predict( net, test_loader )
            np.save(save_dir +'/scores.uint8.npy',scores)
            log.write('\tpredict : %0.2f min\n\n'%((timer() - start) / 60))

            start = timer()
            df, labels, probs = scores_to_csv(scores, test_dataset, save_dir)
            np.save(save_dir + '/labels.npy',labels)
            np.save(save_dir + '/probs.npy',probs)
            np.savetxt(save_dir + '/probs.txt',probs, fmt='%0.5f')
            df.to_csv(save_dir + '/submission_csv.gz', index=False, compression='gzip')
            log.write('\tscores_to_csv : %0.2f min\n\n'%((timer() - start) / 60))

        pass

    ## submission  ----------------------------
    if 1:
        start = timer()
        merge_csv_file = out_dir +'/submit/%s/merge_submission_csv.csv.gz'%augment
        csv_files = [ out_dir +'/submit/%s/part-%d'%(augment,i) + '/submission_csv.gz' for i in range(num_splits)]

        dfs=[]
        for csv_file in csv_files:
            df = pd.read_csv(csv_file, compression='gzip', error_bad_lines=False)
            dfs.append(df)

        merge_df = pd.concat(dfs)
        merge_df.to_csv(merge_csv_file, index=False, compression='gzip')
        log.write('\tmerge : %0.2f min\n\n'%((timer() - start) / 60))




# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    do_submit()

    print('\nsucess!')