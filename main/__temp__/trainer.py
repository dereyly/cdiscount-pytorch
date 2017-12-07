'''

## https://github.com/colesbury/examples/blob/8a19c609a43dc7a74d1d4c71efcda102eed59365/imagenet/main.py
## https://github.com/pytorch/examples/blob/master/imagenet/main.py

from common import *
from net.rate import *
from net.loss import *
from dataset.sampler import *
from dataset.transform import *


# --------------------------------------------
from dataset.cdimage import *

from net.model.resnet50 import ResNet50 as Net
#from net.model.se_resnet50 import SEResNet50 as Net



from dataset.tool import *

## csv tools ############################################################
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

        s = s.astype(np.float32)/255
        l = s.argmax()
        labels.append(l)
        probs.append(s[l])
    pass
    print('\n')

    # save results ---
    labels = np.array(labels)
    probs  = np.array(probs)
    df = pd.DataFrame({ '_id' : ids, 'category_id' : labels})
    df['category_id'] = df['category_id'].map(test_dataset.label_to_category_id)

    return df, labels, probs





## main functions ############################################################
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
        logits = net(images)
        probs  = F.softmax(logits)
        probs  = probs.data.cpu().numpy()*255

        batch_size = len(indices)
        scores[n:n+batch_size]=probs
        n += batch_size

    print('\n')
    assert(n == len(test_loader.sampler) and n == test_num)

    return scores



def evaluate( net, test_loader ):

    test_num  = 0
    test_loss = 0
    test_acc  = 0
    for iter, (images, labels, indices) in enumerate(test_loader, 0):
        images  = Variable(images,volatile=True).cuda()
        labels  = Variable(labels).cuda()

        logits = net(images)
        probs  = F.softmax(logits)

        loss = F.cross_entropy(logits, labels)
        acc  = top_accuracy(probs, labels, top_k=(1,))#1,5

        batch_size = len(indices)
        test_acc  += batch_size*acc[0][0]
        test_loss += batch_size*loss.data[0]
        test_num  += batch_size

    assert(test_num == len(test_loader.sampler))
    test_acc  = test_acc/test_num
    test_loss = test_loss/test_num
    return test_loss, test_acc




####################################################################################################



def train_augment(image):
    #image = fix_resize(image, 224, 224)

    x0 = np.random.choice(20)
    y0 = np.random.choice(20)
    roi = (x0,y0,x0+160,y0+160)
    image = fix_crop(image, roi)
    image = random_horizontal_flip(image, u=0.5)

    return image

def valid_augment(image):
    #image = fix_resize(image, 224, 224)

    roi = (10,10,170,170) #crop center
    image = fix_crop(image, roi)

    return image

# def test_augment(image):
#     #image = fix_resize(image, 224, 224)
#     #return image
test_augment=valid_augment



def do_training():

    out_dir  = '/root/share/project/kaggle/cdiscount/results/resnet50-freeze-03' # s_xx1'
    initial_checkpoint = \
        None #'/root/share/project/kaggle/cdiscount/results/xxx1/checkpoint/140_model.pth'
        #None


    ## ------------------------------------
    ## 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
    pretrained_file = '/root/share/data/models/pytorch/imagenet/resenet/resnet50-19c8e357.pth'
    skip = ['fc.weight', 'fc.bias']


    os.makedirs(out_dir +'/checkpoint', exist_ok=True)
    os.makedirs(out_dir +'/backup', exist_ok=True)
    backup_project_as_zip(PROJECT_PATH, out_dir +'/backup/code.train.zip')

    log = Logger()
    log.open(out_dir+'/log.train.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 64))
    log.write('** some experiment setting **\n')
    log.write('\tSEED         = %u\n' % SEED)
    log.write('\tPROJECT_PATH = %s\n' % PROJECT_PATH)
    log.write('\tout_dir      = %s\n' % out_dir)
    log.write('\n')


    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')
    batch_size  = 512  #96 #256

    train_dataset = CDiscountDataset('train_id_v0_5655916', 'train',  mode='train',
                                     #'train_id_v0_5000', 'train',  mode='train',  #'train_id_v0_100000',#

                                    transform =[ lambda x:train_augment(x),])
    train_loader  = DataLoader(
                        train_dataset,
                        #sampler = RandomSampler1(train_dataset,50000),
                        sampler = RandomSampler(train_dataset),
                        batch_size  = batch_size,
                        drop_last   = True,
                        num_workers = 4,
                        pin_memory  = True)

    valid_dataset = CDiscountDataset( 'valid_id_v0_5000', 'train',  mode='train',
                                    transform =[ lambda x:valid_augment(x),])
    valid_loader  = DataLoader(
                        valid_dataset,
                        sampler     = SequentialSampler(valid_dataset),  #None,
                        batch_size  = batch_size,
                        drop_last   = False,
                        num_workers = 4,
                        pin_memory  = True)


    log.write('\tbatch_size   = %d\n'%batch_size)
    log.write('\ttrain_dataset.split   = %s\n'%(train_dataset.split))
    log.write('\tvalid_dataset.split   = %s\n'%(valid_dataset.split))
    log.write('\tlen(train_loader)     = %d\n'%(len(train_loader)))
    log.write('\tlen(valid_loader)     = %d\n'%(len(valid_loader)))
    log.write('\n')

    #log.write(inspect.getsource(train_augment)+'\n')
    #log.write(inspect.getsource(valid_augment)+'\n')
    #log.write(inspect.getsource(total_loss)+'\n')
    log.write('\n')

    # if 0:  ## check data
    #     check_kgforest_dataset(train_dataset, train_loader)
    #     exit(0)



    ## net ----------------------------------------
    log.write('** net setting **\n')
    net = Net(in_shape = (3, CDISCOUNT_HEIGHT, CDISCOUNT_WIDTH), num_classes=CDISCOUNT_NUM_CLASSES)
    #net = torch.nn.DataParallel(net, device_ids=[0, 1, 2])

    if 1: #freeze early layers
        for p in net.layer0.parameters():
            p.requires_grad = False
        for p in net.layer1.parameters():
            p.requires_grad = False
        for p in net.layer2.parameters():
            p.requires_grad = False
        for p in net.layer3.parameters():
            p.requires_grad = False



    net.cuda()
    #log.write('\n%s\n'%(str(net)))
    log.write('%s\n\n'%(type(net)))
    #log.write(inspect.getsource(net.__init__)+'\n')
    #log.write(inspect.getsource(net.forward )+'\n')
    log.write('\n')


    ## optimiser ----------------------------------
    ## fine tunning
    ## optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)  ###0.0005
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=0.01, momentum=0.9, weight_decay=0.0005)


    #LR = StepLR([ (0, 0.01),  (200, 0.001),  (300, -1)])
    LR = StepLR([ (0, 0.01),])
    num_iters   = 1000  *1000
    iter_print  =  5
    iter_smooth = 20
    iter_test  = 1
    iter_valid = list(range(0,num_iters,1000))
    iter_save  = [0, num_iters-1]\
                   + list(range(0,num_iters,10*1000))

    #https://discuss.pytorch.org/t/problem-on-different-learning-rate-and-weight-decay-in-different-layers/3619
    #https://discuss.pytorch.org/t/how-to-perform-finetuning-in-pytorch/419/7


    ## resume from previous ----------------------------------
    start_iter=0
    if initial_checkpoint is not None:
        net.load_state_dict(torch.load(initial_checkpoint))

        checkpoint = torch.load(initial_checkpoint.replace('_model.pth','_optimizer.pth'))
        start_iter = checkpoint['iter']
        optimizer.load_state_dict(checkpoint['optimizer'])


    elif pretrained_file is not None:  #pretrain
        net.load_pretrain_pytorch_file(
            pretrained_file, skip=['fc.weight'	,'fc.bias']
        )


    ## start training here! ##############################################
    log.write('** start training here! **\n')

    log.write(' optimizer=%s\n'%str(optimizer) )
    log.write(' LR=%s\n\n'%str(LR) )
    log.write('   rate   iter   epoch  | valid_loss/acc | train_loss/acc | batch_loss/acc  | ... min\n')
    log.write('-------------------------------------------------------------------------------------\n')

    smooth_loss = 0.0
    smooth_acc  = 0.0
    train_loss  = 0.0
    train_acc   = 0.0
    valid_loss  = 0.0
    valid_acc   = 0.0
    batch_loss  = 0.0
    batch_acc   = 0.0
    rate = 0

    sum_train_loss = 0.0
    sum_train_acc  = 0.0
    sum = 0

    start = timer()
    i = start_iter
    while  i<num_iters:  # loop over the dataset multiple times
        for (images, labels, indices) in train_loader:
            epoch = (i*batch_size)/len(train_dataset)

            if i in iter_valid:
                net.eval()
                valid_loss, valid_acc = evaluate(net, valid_loader)

                print('\r',end='',flush=True)
                log.write('%0.4f  %5.1f k   %4.2f  | %0.4f  %0.4f | %0.4f  %0.4f | %0.4f  %0.4f  |  %5.1f min \n' % \
                        (rate, i/1000, epoch, valid_loss, valid_acc, train_loss, train_acc, batch_loss, batch_acc,
                          (timer() - start)/60))

            #if 1:
            if i in iter_save:
                torch.save(net.state_dict(),out_dir +'/checkpoint/%08d_model.pth'%(i))
                torch.save({
                    'optimizer' : optimizer.state_dict(),
                    'iter'      : i,
                }, out_dir +'/checkpoint/%08d_optimizer.pth'%(i))


            # learning rate schduler -------------
            lr = LR.get_rate(i)
            if lr<0 : break
            adjust_learning_rate(optimizer, lr)
            rate = get_learning_rate(optimizer)[0]


            # one iteration update  -------------
            net.train()
            images  = Variable(images).cuda()
            labels  = Variable(labels).cuda()
            logits = net(images)
            probs  = F.softmax(logits)

            loss = F.cross_entropy(logits, labels)
            acc  = top_accuracy(probs, labels, top_k=(1,))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print statistics  ------------
            batch_acc  = acc[0][0]
            batch_loss = loss.data[0]
            sum_train_loss += batch_loss
            sum_train_acc  += batch_acc
            sum += 1
            if i%iter_smooth == 0:
                train_loss = sum_train_loss/sum
                train_acc  = sum_train_acc /sum
                sum_train_loss = 0.
                sum_train_acc  = 0.
                sum = 0

            if i%iter_print == 0:
                print('\r%0.4f  %5.1f k   %4.2f  | ......  ...... | %0.4f  %0.4f | %0.4f  %0.4f ' % \
                        (rate, i/1000, epoch, train_loss, train_acc, batch_loss, batch_acc),\
                        end='',flush=True)


            i = i+1

        pass  #---- end of one data loader -----
    pass #---- end of all iterations -----



    ## check : load model and re-test
    if 1:
        torch.save(net.state_dict(),out_dir +'/checkpoint/%d_model.pth'%(i))
        torch.save({
            'optimizer': optimizer.state_dict(),
            'iter'     : i,
        }, out_dir +'/checkpoint/%d_optimizer.pth'%(i))

    log.write('\n')
    log.write('\tall time to train=%0.1f min\n'%((timer() - start) / 60))
    log.write('\n')




##to determine best threshold etc ... ## -----------------------------------------------------------





def do_single_submit():

    out_dir    = '/root/share/project/kaggle/cdiscount/results/xxx1' # s_xx1'
    checkpoint = out_dir +'/checkpoint/155_model.pth'  #final


    ## ------------------------------------
    os.makedirs(out_dir +'/submit', exist_ok=True)

    log = Logger()
    log.open(out_dir+'/log.submit.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 64))
    log.write('\n')



    ## net ----------------------------------------
    log.write('** net setting **\n')
    log.write('\tcheckpoint = %s\n'%checkpoint)
    log.write('\n')

    net = Net(in_shape = (3,CDISCOUNT_HEIGHT,CDISCOUNT_WIDTH), num_classes=CDISCOUNT_NUM_CLASSES)
    net.load_state_dict(torch.load(checkpoint))
    net.merge_bn()
    net.cuda()
    net.eval()


    ## prediction  ----------------------------
    splits=['test_id_v0_420000','test_id_v1_420000','test_id_v2_420000','test_id_v3_508182']
    num_splits = len(splits)

    if 1:
        #split into several parts
        for i in range(splits):
            split = splits[i]

            save_dir = out_dir +'/submit/part-%d'%i
            os.makedirs(save_dir, exist_ok=True)

            test_dataset = CDiscountDataset( split, 'test', mode='test',
                                         #'test_id_1768182', 'test', mode='test',
                                         #'test_id_6813', 'test', mode='test',
                                        transform =[ lambda x:test_augment(x),])
            test_loader  = DataLoader(
                                test_dataset,
                                sampler     = SequentialSampler(test_dataset), #FixedSampler(test_dataset,[0,1,2,]),  #
                                batch_size  = 512,
                                drop_last   = False,
                                num_workers = 4,
                                pin_memory  = True)


            scores = predict( net, test_loader )
            np.save(save_dir +'/scores.uint8.npy')

            df, labels, probs = scores_to_csv(scores, test_dataset, save_dir)
            np.save(save_dir + '/labels.npy',labels)
            np.save(save_dir + '/probs.npy',probs)
            np.savetxt(save_dir + '/probs.txt',probs, fmt='%0.5f')
            df.to_csv(save_dir + '/submission_csv.gz', index=False, compression='gzip')

        pass

    ## submission  ----------------------------
    if 1:
        merge_csv_file = out_dir +'/submit/merge_submission_csv.csv.gz'
        csv_files = [ out_dir +'/submit/part-%d'%i + '/submission_csv.gz' for i in range(splits)]

        dfs=[]
        for csv_file in csv_files:
            df = pd.read_csv(csv_file, compression='gzip', error_bad_lines=False)
            dfs.append(df)

        merge_df = pd.concat(dfs)
        merge_df.to_csv(merge_csv_file, index=False, compression='gzip')



            # log.write('\tbatch_size         = %d\n'%batch_size)
            # log.write('\ttest_dataset.split = %s\n'%(test_dataset.split))
            # log.write('\tlen(test_dataset)  = %d\n'%(len(test_dataset)))
            # log.write('\tlen(test_loader)   = %d\n'%(len(test_loader)))
            # log.write('\n')
            #
            # # do testing here ###########
            #
            #
            # start = timer()
            # print('np.save ... ', end='')
            # print('%0.2f min'%((timer() - start) / 60))


    ## sumission  ----------------------------
    if 1:


##-----------------------------------------
def do_submissions():

    out_dir  = '/root/share/project/kaggle/cdiscount/results/resnet50-freeze-00' # s_xx1'

    ## ------------------------------------
    log = Logger()
    log.open(out_dir+'/log.submissions.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 64))
    log.write('\n')

    scores = np.load(out_dir +'/submit/scores.uint8.npy')
    test_dataset = CDiscountDataset( 'test_id_1768182', 'test', mode='test', transform =[ ])

    df = test_dataset.df[['_id','num_imgs']]
    df = df.groupby(['_id']).agg({'num_imgs': 'mean'}).reset_index()
    df['cumsum']=df['num_imgs'].cumsum()

    ids      = df['_id'].values
    num_imgs = df['num_imgs'].values
    cumsum   = df['cumsum'].values
    num_products = len(ids)
    num_img_files = len(scores)

    assert(df['cumsum'].iloc[[-1]].values[0] == num_img_files)
    print('')
    print('making submission csv')
    print('\tnum_products=%d'%num_products)
    print('\tnum_img_files=%d'%num_img_files)
    print('')

    start = timer()
    #mean_scores = [[]for n in range(num_products)]
    labels=[]
    probs =[]
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

        s = s.astype(np.float32)/255
        l = s.argmax()
        labels.append(l)
        probs.append(s[l])
    pass
    print('\n')


    labels = np.array(labels)
    probs  = np.array(probs)
    np.save(out_dir +'/submit/submit_labels.npy',labels)
    np.save(out_dir +'/submit/submit_probs.npy',probs)
    np.savetxt(out_dir +'/submit/submit_probs.txt',probs, fmt='%0.5f')
    log.write('\tcompute labels = %f min\n'%((timer() - start) / 60)) #1 min


    start = timer()
    df = pd.DataFrame({ '_id' : ids, 'category_id' : labels})
    df['category_id'] = df['category_id'].map(test_dataset.label_to_category_id)

    gz_file  = out_dir + '/submit/submit_results-csv.gz'
    df.to_csv(gz_file, index=False, compression='gzip')
    log.write('\tdf.to_csv time = %f min\n'%((timer() - start) / 60)) #1 min
    log.write('\n')


    pass




# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    do_training()
    #do_predictions()
    #do_submissions()

    print('\nsucess!')

'''