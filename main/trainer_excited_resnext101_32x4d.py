import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
NUM_CUDA_DEVICES = len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) 
 
from common import *
from net.rate import *
from net.loss import *
from utility.file import *

from dataset.cdimage import *
from dataset.sampler import *
from dataset.transform import *

# --------------------------------------------

from net.model.cdiscount.excited_resnext_101_32x4d import SEResnext101 as Net

 
####################################################################################################
## common functions ##

## standard pytorch normalisation
image_to_tensor_transform = pytorch_image_to_tensor_transform



def train_augment(image):

    if random.random() < 0.5:
        image = random_shift_scale_rotate(image,
                  # shift_limit  = [0, 0],
                  shift_limit=[-0.05, 0.05],
                  scale_limit=[0.9, 1.1],
                  #rotate_limit=[-10, 10],
                  aspect_limit=[1, 1],
                  # size=[ 224, 224],
                  borderMode=cv2.BORDER_REFLECT_101, u=1)
    else:
        pass

    # flip  random ---------
    image = random_horizontal_flip(image, u=0.5)

    #image  = fix_resize(image, 224, 224)
    tensor = image_to_tensor_transform(image)
    return tensor



def valid_augment(image): 
    #image  = fix_resize(image, 224, 224)
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



def run_training():
	
    out_dir    = '/home/ck/project/results/excited-resnext101-180-03a' # s_xx1'
    initial_checkpoint = \
      '/home/ck/project/results/excited-resnext101-180-03/checkpoint/00072000_model.pth'
      #None  #
        

    #pretrained_file = '/root/share/data/models/reference/imagenet/resnext/resnext_101_32x4d.convert.pth'
    pretrained_file = '/home/ck/project/results/excited-resnext101-180-02/checkpoint/00018000_model.pth'
    skip = [] #['.scale.',]

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
    #log.write(inspect.getsource(net.__init__)+'\n', is_terminal=0)
    #log.write(inspect.getsource(net.forward )+'\n', is_terminal=0)
    log.write('\n')


    ## optimiser ----------------------------------
    #LR = StepLR([ (0, 0.01),  (200, 0.001),  (300, -1)])
    LR = StepLR([ (0, 0.001),])

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




    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')
    batch_size  = 128 #64
    iter_accum  = 2#4

    train_dataset = CDiscountDataset('train_id_v0_7019896', 'train',  mode='train',
                                     #'train_id_v0_5000', 'train',  mode='train',  #'train_id_v0_100000',#
                                    transform =[ lambda x:train_augment(x),])
    train_loader  = DataLoader(
                        train_dataset,
                        sampler = RandomSampler(train_dataset),
                        batch_size  = batch_size,
                        drop_last   = True,
                        num_workers = 4,
                        pin_memory  = True)


    # valid_dataset = CDiscountDataset( 'valid_id_v0_50000', 'train',  mode='train', ##'valid_id_v0_5000'
    #                                 transform =[ lambda x:valid_augment(x),])
    # valid_loader  = DataLoader(
    #                     valid_dataset,
    #                     sampler     = SequentialSampler(valid_dataset),
    #                     batch_size  = batch_size,
    #                     drop_last   = False,
    #                     num_workers = 4,
    #                     pin_memory  = True)
    valid_small_dataset = CDiscountDataset( 'valid_id_v0_5000', 'train',  mode='train',
                                transform =[ lambda x:valid_augment(x),])
    valid_small_loader = DataLoader(
                        valid_small_dataset,
                        sampler     = SequentialSampler(valid_small_dataset), #sampler     = FixedSampler(valid_dataset,list(range(0,5000))),
                        batch_size  = batch_size,
                        drop_last   = False,
                        num_workers = 4,
                        pin_memory  = True)
    valid_dataset = valid_small_dataset
    valid_loader  = valid_small_loader


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


    #log.write(inspect.getsource(train_augment)+'\n',is_terminal=False)
    #log.write(inspect.getsource(valid_augment)+'\n',is_terminal=False)
    log.write('\n')

    # if 0:  ## check data
    #     check_dataset(train_dataset, train_loader)
    #     exit(0)


    ## resume from previous ----------------------------------
    start_iter = 0
    start_epoch= 0.
    if initial_checkpoint is not None:
        log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)
        net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))

        checkpoint  = torch.load(initial_checkpoint.replace('_model.pth','_optimizer.pth'))
        start_iter  = checkpoint['iter' ]
        start_epoch = checkpoint['epoch']
        #optimizer.load_state_dict(checkpoint['optimizer'])


    elif pretrained_file is not None:  #pretrain
        log.write('\tpretrained_file    = %s\n' % pretrained_file)
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




# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_training()
    print('\nsucess!')
