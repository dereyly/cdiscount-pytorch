"""

    #https://discuss.pytorch.org/t/problem-on-different-learning-rate-and-weight-decay-in-different-layers/3619
    #https://discuss.pytorch.org/t/how-to-perform-finetuning-in-pytorch/419/7



    top =  500  #5270
    num_products = 1768182
    num_imgs	 = 3095080


    top_prediction_scores = np.zeros((num_products,top),np.float32)
    top_prediction_idxs   = np.zeros((num_products,top),np.int32)
    start = timer()
    np.save('/root/share/project/kaggle/cdiscount/results/zzz/top_prediction_scores.npy',top_prediction_scores)
    np.save('/root/share/project/kaggle/cdiscount/results/zzz/top_prediction_idxs.npy',top_prediction_idxs)
    print('np.save time = %f min\n'%((timer() - start) / 60))




    top =  NUM_CLASSES  #5270
    num_products = 1768182
    num_imgs	 = 3095080


    top_prediction_scores = np.zeros((num_imgs,top),np.uint8)
    start = timer()
    np.save('/root/share/project/kaggle/cdiscount/results/zzz/top_prediction_scores.npy',top_prediction_scores)
    print('np.save time = %f min\n'%((timer() - start) / 60))



    start = timer()
    print('scores to np.array ... ', end='')
    N = len(scores)
    last = scores[N-1].copy()
    scores = np.array(scores[0:N-1]).reshape(-1,CDISCOUNT_NUM_CLASSES)
    scores = np.vstack((scores,last))
    print('%0.2f min'%((timer() - start) / 60))



     #debug
        # batch_size = 128 if i+128<3095080 else 3095080-i
        # probs = np.zeros((batch_size,CDISCOUNT_NUM_CLASSES),np.uint8)
        # scores.append(probs)
        # n += batch_size



"""