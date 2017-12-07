import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
NUM_CUDA_DEVICES = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))

from common import *
from net.rate import *
from net.loss import *
from utility.file import *

#from dataset.cdimage import *
from dataset.sampler import *
from dataset.transform import *

# --------------------------------------------------------

from net.model.cdiscount.excited_resnet50  import SEResNet50 as Net

####################################################################################################

from dataset.majority_classes import *
REDUCED_NUM_CLASSES=1000#2301


def run_check_reduce_class():
    keys   = MAJORITY_CLASSES
    values = np.arange(len(keys))
    values[REDUCED_NUM_CLASSES:]= REDUCED_NUM_CLASSES-1
    d = dict(zip(keys, values))
    pass




CDISCOUNT_DIR = '/media/ssd/data/kaggle/cdiscount'
CDISCOUNT_NUM_CLASSES = REDUCED_NUM_CLASSES
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
            ids = read_list_from_file(CDISCOUNT_DIR + '/split/' + split, comment='#', func=int)
            num_ids = len(ids)

            train_df = pd.read_csv (CDISCOUNT_DIR + '/train_by_product_id.csv')
            #df.columns
            #Index(['_id', 'category_id', 'num_imgs'], dtype='object')

            start = timer()
            df = train_df.reset_index()
            df = df[ df['_id'].isin(ids)]
            df = df.reindex(np.repeat(df.index.values, df['num_imgs']), method='ffill')
            df['cum_sum' ] = df.groupby(['_id']).cumcount()
            df['img_file'] = CDISCOUNT_DIR + '/image/'+ folder + '/' + df['category_id'].astype(str) + '/' +df['_id'].astype(str) + '-' + df['cum_sum'].astype(str)  + '.jpg'
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

        ##-------------------------
        keys   = MAJORITY_CLASSES
        values = np.arange(len(keys))
        values[REDUCED_NUM_CLASSES:]= REDUCED_NUM_CLASSES-1
        self.map_to_reduce_label = dict(zip(keys, values))



    def __getitem__(self, index):
        if self.mode=='train':
            image = cv2.imread(self.img_files[index], 1)
            label = self.labels[index]
            label = self.map_to_reduce_label[label]

            for t in self.transform:
                image = t(image)

            #image = pytorch_image_to_tensor_transform(image)
            return image, label, index

        elif self.mode=='test':
            image = cv2.imread(self.img_files[index], 1)
            for t in self.transform:
                image = t(image)

            #image = pytorch_image_to_tensor_transform(image)
            return image, index

        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.img_files)









####################################################################################################
## common functions ##

# se_resnet
def image_to_tensor_transform(image):
    #  https://github.com/hujie-frank/SENet/blob/master/models/SE-ResNet-50.prototxt
    mean  = [104, 117, 123]
    image = image.astype(np.float32) - np.array(mean, np.float32)
    image = image.transpose((2,0,1))
    tensor = torch.from_numpy(image)
    return tensor



def tensor_to_image_transform(tensor):
    mean  = [104, 117, 123]
    image = tensor.cpu().numpy()
    image = image.transpose((1,2,0))
    image = image + np.array(mean, np.float32)
    image = image.astype(np.uint8)
    return image



def train_augment(image):
    # image = random_brightness(image, limit=[-0.3,0.3], u=0.5)
    # image = random_contrast  (image, limit=[-0.3,0.3], u=0.5)
    # image = random_saturation(image, limit=[-0.3,0.3], u=0.5)
    tensor = image_to_tensor_transform(image)
    return tensor



def valid_augment(image):
    tensor = image_to_tensor_transform(image)
    return tensor


#--------------------------------------------------------------
def evaluate( net, test_loader ):

    test_num  = 0
    test_loss = 0
    test_acc  = 0
    for iter, (images, labels, indices) in enumerate(test_loader, 0):
        images  = Variable(images,volatile=True).cuda()
        labels  = Variable(labels).cuda()

        if NUM_CUDA_DEVICES!=0:
            logits = torch.nn.DataParallel(net)(images) #use default
        else:
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

#--------------------------------------------------------------
def run_training():

    out_dir  = '/root/share/project/kaggle/cdiscount/results/excited-resnet50-no-aug-00-x1' # s_xx1'
    initial_checkpoint = \
        None#'/root/share/project/kaggle/cdiscount/results/excited-resnet50-no-aug-00/checkpoint/00010000_model.pth'
        #

    pretrained_file = \
        '/root/share/project/kaggle/cdiscount/results/excited-resnet50-no-aug-00/checkpoint/00010000_model.pth'
        #'/root/share/project/kaggle/cdiscount/results/__submission__/stable-00/excited-resnet50-180-00a/checkpoint/00061000_model.pth'
        #'#'/home/ck/project/data/pretrain/SE-ResNet-50.convert.pth'
    skip = ['fc.'] #['fc.weight', 'fc.bias']

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

    if 1: #freeze early layers
        for p in net.layer0.parameters():
            p.requires_grad = False
        for p in net.layer1.parameters():
            p.requires_grad = False
        for p in net.layer2.parameters():
            p.requires_grad = False
        for p in net.layer3.parameters():
            p.requires_grad = False


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
    iter_log    = 1000
    iter_small_valid = 500
    iter_valid       = 5 *1000
    iter_save   = [0, num_iters-1]\
                   + list(range(0,num_iters,1*1000))
    images_per_epoch = 1000000. # len(train_dataset)  #plot loss,acc aginst number of images seen



    ## optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)  ###0.0005
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=0.01, momentum=0.25, weight_decay=0.0001)
    #optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, net.parameters()),lr=0.01, alpha=0.99, weight_decay=0)

    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')
    batch_size  = 64 #60   #512  #96 #256
    iter_accum  = 8 #2  #448//batch_size


    train_dataset = CDiscountDataset('train_id_v0_7019896', 'train',  mode='train', #'train_id_v0_5000',   #'train_id_v0_100000',#
                                    transform =[ lambda x:train_augment(x),])
    train_loader  = DataLoader(
                        train_dataset,
                        #sampler = RandomSampler1(train_dataset,50000),
                        sampler = RandomSampler(train_dataset),
                        batch_size  = batch_size,
                        drop_last   = True,
                        num_workers = 4,
                        pin_memory  = True)

    valid_dataset = CDiscountDataset( 'valid_id_v0_50000', 'train',  mode='train', ##'valid_id_v0_5000'
                                    transform =[ lambda x:valid_augment(x),])
    valid_loader  = DataLoader(
                        valid_dataset,
                        sampler     = SequentialSampler(valid_dataset),
                        batch_size  = batch_size,
                        drop_last   = False,
                        num_workers = 4,
                        pin_memory  = True)
    valid_small_dataset = CDiscountDataset( 'valid_id_v0_5000', 'train',  mode='train',
                                transform =[ lambda x:valid_augment(x),])
    valid_small_loader = DataLoader(
                        valid_small_dataset,
                        sampler     = SequentialSampler(valid_small_dataset), #sampler     = FixedSampler(valid_dataset,list(range(0,5000))),
                        batch_size  = batch_size,
                        drop_last   = False,
                        num_workers = 4,
                        pin_memory  = True)
    valid_loader = valid_small_loader

    log.write('\ttrain_dataset.split     = %s\n'%(train_dataset.split))
    log.write('\tvalid_dataset.split     = %s\n'%(valid_dataset.split))
    log.write('\tlen(train_dataset)      = %d\n'%(len(train_dataset)))
    log.write('\tlen(valid_dataset)      = %d\n'%(len(valid_dataset)))
    log.write('\tlen(train_loader)       = %d\n'%(len(train_loader)))
    log.write('\tlen(valid_loader)       = %d\n'%(len(valid_loader)))
    log.write('\tlen(valid_small_loader) = %d\n'%(len(valid_small_loader)))
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
        log.write('\tinitial_checkpoint = %s\n\n' % initial_checkpoint)
        net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))
 
        checkpoint  = torch.load(initial_checkpoint.replace('_model.pth','_optimizer.pth'),\
					  map_location=lambda storage, loc: storage)
        start_iter  = checkpoint['iter' ]
        start_epoch = checkpoint['epoch']
        #optimizer.load_state_dict(checkpoint['optimizer'])



    elif pretrained_file is not None:  #pretrain
        log.write('\tpretrained_file = %s\n\n' % pretrained_file)
        net.load_pretrain_pytorch_file( pretrained_file, skip )


    ## start training here! ##############################################
    log.write('** start training here! **\n')
    log.write(' optimizer=%s\n'%str(optimizer) )
    log.write(' momentum=%f\n'% optimizer.param_groups[0]['momentum'])
    log.write(' LR=%s\n\n'%str(LR) )

    log.write(' images_per_epoch= %d\n\n'%int(images_per_epoch))
    log.write('   rate   iter   epoch  | valid_loss/acc | train_loss/acc | batch_loss/acc |  time   \n')
    log.write('-------------------------------------------------------------------------------------\n')

    train_loss  = 0.0
    train_acc   = 0.0
    valid_loss  = 0.0
    valid_acc   = 0.0
    batch_loss  = 0.0
    batch_acc   = 0.0
    rate = 0



    start = timer()
    j = 0
    i = 0


    while  i<num_iters:  # loop over the dataset multiple times
        sum_train_loss = 0.0
        sum_train_acc  = 0.0
        sum = 0

        net.train()
        optimizer.zero_grad()
        for images, labels, indices in train_loader:
            i = j/iter_accum + start_iter
            epoch = (i-start_iter)*batch_size*iter_accum/images_per_epoch + start_epoch

            if i % iter_valid == 0:
                net.eval()
                valid_loss, valid_acc = evaluate(net, valid_loader)
                net.train()
            elif i % iter_small_valid == 0:
                net.eval()
                valid_loss, valid_acc = evaluate(net, valid_small_loader)
                net.train()

            if i % iter_log == 0:
                print('\r',end='',flush=True)
                log.write('%0.4f  %5.1f k   %4.2f  | %0.4f  %0.4f | %0.4f  %0.4f | %0.4f  %0.4f | %5.0f min \n' % \
                        (rate, i/1000, epoch, valid_loss, valid_acc, train_loss, train_acc, batch_loss, batch_acc, (timer() - start)/60))

            #if 1:
            if i in iter_save:
				#https://discuss.pytorch.org/t/dataparallel-optim-and-saving-correctness/4054
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

            if NUM_CUDA_DEVICES!=0:
                logits = torch.nn.DataParallel(net)(images) #use default
            else:
                logits = net(images)

            probs = F.softmax(logits)
            loss  = F.cross_entropy(logits, labels)
            acc   = top_accuracy(probs, labels, top_k=(1,))


            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            # accumulate gradients
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

            print('\r%0.4f  %5.1f k   %4.2f  | %0.4f  %0.4f | %0.4f  %0.4f | %0.4f  %0.4f | %5.0f min  %d,%d' % \
                    (rate, i/1000, epoch, valid_loss, valid_acc, train_loss, train_acc, batch_loss, batch_acc,(timer() - start)/60 ,i,j),\
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




##to determine best threshold etc ... ## ------------------------------ 
 

# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_training()
    #run_check_reduce_class()

    print('\nsucess!')
