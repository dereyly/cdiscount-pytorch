import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

from common import *
from net.model.cdiscount.multi_crop_net import MultiCropNet_12 as MultiCropNet
from trainer_excited_resnet50 import *



## common functions ##

def test_augment(image):
    tensor = image_to_tensor_transform(image)
    return tensor


#--------------------------------------------------------------
# predict as uint8

def predict(net, test_loader):
    test_num  = len(test_loader.dataset)
    scores = np.zeros((test_num,CDISCOUNT_NUM_CLASSES), np.uint8)
    n = 0
    start = timer()
    for i, (images, indices) in enumerate(test_loader, 0):
        print('\r%5d:  %10d/%d (%0.0f %%)  %0.2f min'%(i, n, test_num, 100*n/test_num,
                         (timer() - start) / 60), end='',flush=True)

        # forward
        images = Variable(images,volatile=True).cuda(async=True)
        #logits = net(images)
        logits = net(images.half())  #
        probs  = F.softmax(logits)
        probs  = probs.float().data.cpu().numpy()*255

        batch_size = len(indices)
        scores[n:n+batch_size]=probs
        n += batch_size

    print('\n')
    assert(n == len(test_loader.sampler) and n == test_num)
    return scores


def do_multi_submit():

    out_dir    = '/root/share/project/kaggle/cdiscount/results/excited-resnet50-00c' # s_xx1'
    checkpoint = out_dir +'/checkpoint/00078000_model.pth'  #final

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
    net.merge_bn()
    net.cuda()
    net.half()  #use fp16 for fast experiments. use fp32 for final submission!
    net.eval()


    ## prediction  ----------------------------
    augment ='multi'
    splits=['test_id_v0_420000','test_id_v1_420000','test_id_v2_420000','test_id_v3_508182']
    num_splits = len(splits)

    if 1:
        net = MultiCropNet(net)

        #split into several parts
        for i in range(num_splits):
            #if i!=3 : continue
            split = splits[i]

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
                                batch_size  = 8,  #512,
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

    do_multi_submit()

    print('\nsucess!')