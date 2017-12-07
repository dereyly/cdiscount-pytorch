import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

from common import *
from net.rate import *
from net.loss import *
from utility.file import *

from dataset.cdimage import *
from dataset.sampler import *
from dataset.transform import *

# --------------------------------------------

from net.model.cdiscount.dualpathnet  import DPN92 as Net
#from net.model.cdiscount.dualpathnet  import DPN107 as Net


## common functions ##
# https://github.com/rwightman/pytorch-dpn-pretrained/blob/fc2ca159630c8c190702763613883e34b2274956/model_factory.py
#      mean=[124 / 255, 117 / 255, 104 / 255],
#      std=[1 / (.0167 * 255)] * 3

## standard pytorch normalisation
def image_to_tensor_transform(image):

    mean=[124 / 255, 117 / 255, 104 / 255]
    std =[1 / (.0167 * 255)] * 3

    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = image.transpose((2,0,1))
    tensor = torch.from_numpy(image).float().div(255)

    tensor[0] = (tensor[0] - mean[0]) / std[0]
    tensor[1] = (tensor[1] - mean[1]) / std[1]
    tensor[2] = (tensor[2] - mean[2]) / std[2]

    return tensor

'''
def train_augment(image):
    image = cv2.resize(image,(256,256))

    # crop random ---------
    x0 = np.random.choice(256-224)
    y0 = np.random.choice(256-224)
    x1 = x0+224
    y1 = y0+224
    image = fix_crop(image, roi=(x0,y0,x1,y1))

    # flip  random ---------
    image = random_horizontal_flip(image, u=0.5)

    tensor = image_to_tensor_transform(image)
    return tensor



def valid_augment(image):

    image = cv2.resize(image,(256,256))

    # crop center ---------
    x0 = (256-224)//2
    y0 = (256-224)//2
    x1 = x0+224
    y1 = y0+224
    image = fix_crop(image, roi=(x0,y0,x1,y1))

    tensor = image_to_tensor_transform(image)
    return tensor
'''


def train_augment(image):

    if random.random() < 0.5:
        image = random_shift_scale_rotate(image,
                #shift_limit  = [0, 0],
                shift_limit  = [-0.06,  0.06],
                scale_limit  = [0.9*224/180, 1.2*224/180],
                rotate_limit = [-10,10],
                aspect_limit = [1,1],
                size=[224,224],
        borderMode=cv2.BORDER_REFLECT_101 , u=1)
    else:
        image = cv2.resize(image,(224,224))

    # flip  random ---------
    image = random_horizontal_flip(image, u=0.5)

    tensor = image_to_tensor_transform(image)
    return tensor



def valid_augment(image):
    image = cv2.resize(image,(224,224))
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



def run_training():

    out_dir    = '/root/share/project/kaggle/cdiscount/results/dual92-224-00' # s_xx1'
    initial_checkpoint = \
        '/root/share/project/kaggle/cdiscount/results/dual92-224-00/checkpoint/00012000_model.pth'

    #pretrained_file = '/root/share/data/models/reference/imagenet/dualpathnet/DPN-107_5k_to_1k/dpn107-extra.pth',
    pretrained_file = '/root/share/data/models/reference/imagenet/dualpathnet/DPN-92_5k_to_1k/dpn92-extra.pth'
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
    iter_valid  = list(range(0,num_iters,1000))
    iter_save   = [0, num_iters-1]\
                   + list(range(0,num_iters,1*1000))


    ## optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)  ###0.0005
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=0.01, momentum=0.9, weight_decay=0.0005)


    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')
    batch_size  = 32   #512  #96 #256
    iter_accum  = 16   #448//batch_size

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
    j = 0
    i = 0

    #net = torch.nn.DataParallel(net, device_ids=[0, 1, 2])
    while  i<num_iters:  # loop over the dataset multiple times

        net.train()
        optimizer.zero_grad()
        for images, labels, indices in train_loader:
            i = j/iter_accum + start_iter
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




# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_training()
    print('\nsucess!')