import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
NUM_CUDA_DEVICES = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))

from common import *
from dataset.transform import *
#from net.model.cdiscount.multi_crop_net import MultiCropNet_12 as MultiCropNet
#from trainer_excited_resnet50 import *
#from trainer_excited_inception3_180 import *
#from trainer_resnext101_32x4d_180 import *


from net.model.cdiscount.xception import Xception as Net

# Xception
def image_to_tensor_transform(image):

    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = image.transpose((2,0,1))
    tensor = torch.from_numpy(image).float().div(255)
    tensor[0] = (tensor[0] - 0.5) *2
    tensor[1] = (tensor[1] - 0.5) *2
    tensor[2] = (tensor[2] - 0.5) *2

    return tensor

def valid_augment(image):
    tensor = image_to_tensor_transform(image)
    return tensor


if 1:  # follow valid--------------------
    single_test_augment = valid_augment


def compute_loss(logits, labels):
    # batch_size, num_classes =  logits.size()
    # labels = labels.view(-1,1)
    # logits = logits.view(-1,num_classes)

    max_logits  = logits.max()
    log_sum_exp = torch.log(torch.sum(torch.exp(logits-max_logits), 1))
    loss = log_sum_exp - logits.gather(dim=1, index=labels.view(-1,1)).view(-1) + max_logits

    return loss


#--------------------------------------------------------------
def do_rank_samples():


    out_dir  = '/root/share/project/kaggle/cdiscount/results/xception-180-01b' # s_xx1'
    checkpoint = out_dir +'/checkpoint/00158000_model.pth'  #final

    ## ------------------------------------
    os.makedirs(out_dir +'/rank', exist_ok=True)

    log = Logger()
    log.open(out_dir+'/log.rank.txt',mode='a')
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
    net.eval()


    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')
    batch_size = 128 #60   #512  #96 #256


    train_dataset = CDiscountDataset('debug_train_id_v0_5000', 'train',  mode='train',
                                     #'train_id_v0_5000',  #'train_id_v0_100000',#'train_id_v0_7019896'
                                    transform =[ lambda x:valid_augment(x),])
    train_loader  = DataLoader(
                        train_dataset,
                        sampler = RandomSampler(train_dataset),
                        batch_size  = batch_size,
                        drop_last   = True,
                        num_workers = 4,
                        pin_memory  = True)


    log.write('\ttrain_dataset.split = %s\n'%(train_dataset.split))
    log.write('\tlen(train_dataset)  = %d\n'%(len(train_dataset)))
    log.write('\tlen(train_loader)   = %d\n'%(len(train_loader)))
    log.write('\n')

    num_batches = len(train_loader)
    num_train   = len(train_dataset)


    ## prediction  ----------------------------
    losses   = np.zeros(num_train,np.float32)
    scores   = np.zeros(num_train,np.float32)
    labels   = np.zeros(num_train,np.int32)
    corrects = np.zeros(num_train,np.int32)

    i=0
    for images, labels, indices in train_loader:
        batch_size = len(images)
        i=i+batch_size

        # one iteration -------------
        images = Variable(images,volatile=True).cuda(async=True)
        labels = Variable(labels,volatile=True).cuda(async=True)
        logits = net(images)
        loss   = compute_loss(logits, labels)


        #probs  = F.softmax(logits)
        #top_probs, top_labels = probs.topk(1, dim=1, largest=True,  sorted=True)
        losses[i-batch_size:i] = loss.data.cpu().numpy()

        #if i%i_save or i==num_train:
        if i==num_batches:

            np.savetxt(out_dir +'/rank/losses.txt', losses, fmt='%0.5f')
            pass


    np.savetxt(out_dir +'/rank/losses.txt', losses, fmt='%0.5f')
    pass



# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    do_rank_samples()

    print('\nsucess!')