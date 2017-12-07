## https://github.com/colesbury/examples/blob/8a19c609a43dc7a74d1d4c71efcda102eed59365/imagenet/main.py
## https://github.com/pytorch/examples/blob/master/imagenet/main.py

from common import *
from dataset.cdimage import *


## csv tools ############################################################
def ensemble_scores_to_csv(scores, dataset, save_dir, weights=None):
    num_augments =len(scores)
    if weights==None:
        weights=[1]*num_augments

    df = dataset.df[['_id','num_imgs']]
    df = df.groupby(['_id']).agg({'num_imgs': 'mean'}).reset_index()
    df['cumsum'] = df['num_imgs'].cumsum()

    ids      = df['_id'].values
    num_imgs = df['num_imgs'].values
    cumsum   = df['cumsum'].values
    num_products  = len(ids)
    num_img_files = len(scores[0])

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

        s = []
        for a in range(num_augments):
           for i in range(end-num,end):
                s.append(scores[a][i]*weights[a])

        s = np.array(s).mean(axis=0)
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



def run_ensemble():

    out_dir    = '/root/share/project/kaggle/cdiscount/results/__submission__/ensemble-00/all-04' # s_xx1'

    augments = [
        '__submission__/stable-00/excited-resnet50-180-00a/submit/00061000_single_180',
        '__submission__/stable-00/excited-inception3-180-02c/submit/00026000_single_180',
        '__submission__/stable-00/inception3-180-02c/submit/00075000_single_180',
        'excited-inception3_1-180-01a/submit/180',
        'resnext101-180-00c/submit/00117000_180',
        'xception-180-01b/submit/00158000_180',
        'xception-240-00/submit/00025000_220',
    ]
    augments = [ '/root/share/project/kaggle/cdiscount/results/'+ a for a in augments]
    num_augments = len(augments)
    weights = None


    ## setup  ---------------------------
    os.makedirs(out_dir, exist_ok=True)
    log = Logger()
    log.open(out_dir+'/log.ensemble.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 64))
    log.write('\n')
    for a in augments:
        log.write('\t%s\n'%a)
    log.write('\n')
    log.write('\tweights: %s\n'%str(weights))
    log.write('\n')



    ## ------------------------------------
    splits=['test_id_v0_420000','test_id_v1_420000','test_id_v2_420000','test_id_v3_508182']
    num_splits = len(splits)

    if 1:
        #split into several parts
        for i in range(num_splits):
            split = splits[i]
            print('---@ %s-----------'%split)


            save_dir = out_dir +'/submit/part-%d'%(i)
            os.makedirs(save_dir, exist_ok=True)

            test_dataset = CDiscountDataset( split, 'test', mode='test',
                                        transform =[ lambda x:test_augment(x),])


            print('np.load()')
            scores=[]
            for a in augments:
                npy_file  = '%s/part-%d/scores.uint8.npy'%(a,i)
                s = np.load(npy_file)
                scores.append(s)


            print('scores_to_csv()')
            df, labels, probs = ensemble_scores_to_csv(scores, test_dataset, save_dir, weights)
            np.save(save_dir + '/labels.npy',labels)
            np.save(save_dir + '/probs.npy',probs)
            np.savetxt(save_dir + '/probs.txt',probs, fmt='%0.5f')
            df.to_csv(save_dir + '/submission_csv.gz', index=False, compression='gzip')

        pass

    ## submission  ----------------------------
    if 1:

        merge_csv_file = out_dir +'/submit/merge_submission_csv.csv.gz'
        csv_files = [out_dir +'/submit/part-%d'%(i) + '/submission_csv.gz' for i in range(num_splits)]

        dfs=[]
        for csv_file in csv_files:
            df = pd.read_csv(csv_file, compression='gzip', error_bad_lines=False)
            dfs.append(df)

        merge_df = pd.concat(dfs)
        merge_df.to_csv(merge_csv_file, index=False, compression='gzip')


# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_ensemble()


    print('\nsucess!')