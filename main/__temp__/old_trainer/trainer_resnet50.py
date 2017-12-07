import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

from common import *
from net.rate import *
from net.loss import *
from utility.file import *
from submit import *
from dataset.cdimage import *
from dataset.sampler import *
from dataset.transform import *

# --------------------------------------------

from net.model.cdiscount.resnet50  import ResNet50 as Net


## common functions ##

def train_augment(image):
    height,width=image.shape[0:2]

    # crop random ---------
    x0 = np.random.choice(width -160)
    y0 = np.random.choice(height-160)
    x1 = x0+160
    y1 = y0+160
    image = fix_crop(image, roi=(x0,y0,x1,y1))

    # flip  random ---------
    image = random_horizontal_flip(image, u=0.5)

    tensor = pytorch_image_to_tensor_transform(image)
    return tensor



def valid_augment(image):
    height,width=image.shape[0:2]


    # crop center ---------
    x0 = (width- 160)//2
    y0 = (height-160)//2
    x1 = x0+160
    y1 = y0+160
    image = fix_crop(image, roi=(x0,y0,x1,y1))

    tensor = pytorch_image_to_tensor_transform(image)
    return tensor


def test_augment(image):
    tensor = pytorch_image_to_tensor_transform(image)
    return tensor
# test_augment=valid_augment


#--------------------------------------------------------------
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



#--------------------------------------------------------------
def run_training():

    out_dir    = '/root/share/project/kaggle/cdiscount/results/resnet50-224-00' # s_xx1'
    initial_checkpoint = \
        None
        #'/root/share/project/kaggle/cdiscount/results/resnet50-freeze-03a/checkpoint/00034000_model.pth'
        #

    pretrained_file = '/root/share/data/models/pytorch/imagenet/resenet/resnet50-19c8e357.pth'
    skip = ['fc.weight', 'fc.bias']

    ## setup  ---------------------------
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


    ## net ------------------------------ -
    log.write('** net setting **\n')
    net = Net(in_shape = (3, CDISCOUNT_HEIGHT, CDISCOUNT_WIDTH), num_classes=CDISCOUNT_NUM_CLASSES)
    net.cuda()

    # if 0: #freeze early layers
    #     for p in net.layer0.parameters():
    #         p.requires_grad = False
    #     for p in net.layer1.parameters():
    #         p.requires_grad = False
    #     for p in net.layer2.parameters():
    #         p.requires_grad = False
    #     for p in net.layer3.parameters():
    #         p.requires_grad = False


    log.write('%s\n\n'%(type(net)))
    log.write('\n%s\n'%(str(net)), is_terminal=0)
    log.write(inspect.getsource(net.__init__)+'\n', is_terminal=0)
    log.write(inspect.getsource(net.forward )+'\n', is_terminal=0)
    log.write('\n')


    ## optimiser ----------------------------------
    #LR = StepLR([ (0, 0.01),  (200, 0.001),  (300, -1)])
    LR = StepLR([ (0, 0.01),])

    num_iters   = 1000  *1000
    iter_smooth = 20
    iter_test   = 1
    iter_valid  = list(range(0,num_iters,1000))
    iter_save   = [0, num_iters-1]\
                   + list(range(0,num_iters,2*1000))


    ## optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)  ###0.0005
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=0.01, momentum=0.9, weight_decay=0.0005)


    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')
    batch_size  = 96   #512  #96 #256
    iter_accum  = 4    #448//batch_size

    train_dataset = CDiscountDataset('train_id_v0_5655916', 'train',  mode='train',
                                     #'train_id_v0_5000', 'train',  mode='train',  #'train_id_v0_100000',#
                                    transform =[ lambda x:train_augment(x), ])
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
                        sampler     = SequentialSampler(valid_dataset),
                        batch_size  = batch_size,
                        drop_last   = False,
                        num_workers = 4,
                        pin_memory  = True)

    log.write('\ttrain_dataset.split = %s\n'%(train_dataset.split))
    log.write('\tvalid_dataset.split = %s\n'%(valid_dataset.split))
    log.write('\tlen(train_dataset)  = %d\n'%(len(train_dataset)))
    log.write('\tlen(valid_dataset)  = %d\n'%(len(valid_dataset)))
    log.write('\tlen(train_loader)   = %d\n'%(len(train_loader)))
    log.write('\tlen(valid_loader)   = %d\n'%(len(valid_loader)))
    log.write('\tbatch_size  = %d\n'%(batch_size))
    log.write('\titer_accum  = %d\n'%(iter_accum))
    log.write('\tbatch_size*iter_accum  = %d\n'%(batch_size*iter_accum))
    log.write('\n')

    #log.write(inspect.getsource(train_augment)+'\n')
    #log.write(inspect.getsource(valid_augment)+'\n')
    log.write('\n')

    # if 0:  ## check data
    #     check_dataset(train_dataset, train_loader)
    #     exit(0)


    ## resume from previous ----------------------------------
    start_iter = 0
    start_epoch= 0.
    if initial_checkpoint is not None:
        net.load_state_dict(torch.load(initial_checkpoint))
        checkpoint  = torch.load(initial_checkpoint.replace('_model.pth','_optimizer.pth'))
        start_iter  = checkpoint['iter' ]
        start_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])


    elif pretrained_file is not None:  #pretrain
        net.load_pretrain_pytorch_file(
            pretrained_file, skip=['fc.weight'	,'fc.bias']
        )


    ## start training here! ##############################################
    log.write('** start training here! **\n')

    log.write(' optimizer=%s\n'%str(optimizer) )
    log.write(' LR=%s\n\n'%str(LR) )
    log.write('   rate   iter   epoch  | valid_loss/acc | train_loss/acc | batch_loss/acc |  time   \n')
    log.write('-------------------------------------------------------------------------------------\n')


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
    j = 0

    #net = torch.nn.DataParallel(net, device_ids=[0, 1, 2])
    while  i<num_iters:  # loop over the dataset multiple times

        net.train()
        optimizer.zero_grad()
        for images, labels, indices in train_loader:
            i = j/iter_accum
            epoch = (i-start_iter)*batch_size*iter_accum/len(train_dataset) + start_epoch

            if i in iter_valid:
                net.eval()
                valid_loss, valid_acc = evaluate(net, valid_loader)
                net.train()

                print('\r',end='',flush=True)
                log.write('%0.4f  %5.1f k   %4.2f  | %0.4f  %0.4f | %0.4f  %0.4f | %0.4f  %0.4f | %5.0f min \n' % \
                        (rate, i/1000, epoch, valid_loss, valid_acc, train_loss, train_acc, batch_loss, batch_acc, (timer() - start)/60))

            #if 1:
            if i in iter_save:
                torch.save(net.state_dict(),out_dir +'/checkpoint/%08d_model.pth'%(i))
                torch.save({
                    'optimizer': optimizer.state_dict(),
                    'iter'     : i,
                    'epoch'    : epoch,
                }, out_dir +'/checkpoint/%08d_optimizer.pth'%(i))




            # learning rate schduler -------------
            lr = LR.get_rate(i)
            if lr<0 : break
            adjust_learning_rate(optimizer, lr/iter_accum)
            rate = get_learning_rate(optimizer)[0]*iter_accum


            # one iteration update  -------------
            images = Variable(images).cuda()
            labels = Variable(labels).cuda()
            logits = net(images)
            probs  = F.softmax(logits)

            loss = F.cross_entropy(logits, labels)
            acc  = top_accuracy(probs, labels, top_k=(1,))

            # single update
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            # accumulated update
            loss.backward()
            if j%iter_accum == 0:
                #torch.nn.utils.clip_grad_norm(net.parameters(), 1)
                optimizer.step()
                optimizer.zero_grad()

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

            print('\r%0.4f  %5.1f k   %4.2f  | ......  ...... | %0.4f  %0.4f | %0.4f  %0.4f | %5.0f min  %d,%d' % \
                    (rate, i/1000, epoch, train_loss, train_acc, batch_loss, batch_acc,(timer() - start)/60 ,i,j),\
                    end='',flush=True)
            j=j+1
        pass  #-- end of one data loader -
    pass #-- end of all iterations --

    ## check : load model and re-test
    if 1:
        torch.save(net.state_dict(),out_dir +'/checkpoint/%d_model.pth'%(i))
        torch.save({
            'optimizer': optimizer.state_dict(),
            'iter'     : i,
            'epoch'    : epoch,
        }, out_dir +'/checkpoint/%d_optimizer.pth'%(i))

    log.write('\n')




def do_single_submit():

    out_dir    = '/root/share/project/kaggle/cdiscount/results/resnet50-full-00' # s_xx1'
    checkpoint = out_dir +'/checkpoint/00142000_model.pth'  #final


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
    augment ='180'
    splits=['test_id_v0_420000','test_id_v1_420000','test_id_v2_420000','test_id_v3_508182']
    num_splits = len(splits)

    if 0:
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
                                batch_size  = 784,
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

    #run_training()
    do_single_submit()

    print('\nsucess!')