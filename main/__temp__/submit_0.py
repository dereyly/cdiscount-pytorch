from common import *
from net.loss import *

from dataset.cdimage import *

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




## net tools  ############################################################
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






# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))



    print('\nsucess!')