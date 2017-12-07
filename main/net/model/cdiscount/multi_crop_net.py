# https://github.com/sanghoon/pytorch_imagenet/blob/master/train_imagenet.py
# https://github.com/sanghoon/pytorch_imagenet/issues/2

from common import *

class MultiCropNet_12(nn.Module):
    def __init__(self, net):
        super(MultiCropNet_12, self).__init__()
        self.net=net

    # Naive code
    def forward(self, x):
        num_classes=self.net.num_classes
        batch_size,C,H,W = x.size()
        assert(C==3)
        assert(H==180)
        assert(W==180)

        rois = [
            (10,10,170,170),
            # ( 0, 0,160,160),
            # (20, 0,160,160),
            # ( 0,20,160,160),
            # (20,20,160,160),
            ( 0, 0,180,180), #upsize
        ]
        num_augments=4


        average = torch.cuda.FloatTensor(batch_size,num_classes).fill_(0)
        for is_flip in [False, True]:
        #for is_flip in [False]:
            if is_flip==True:
                #https://github.com/pytorch/pytorch/issues/229
                #https://discuss.pytorch.org/t/how-to-reverse-a-torch-tensor/382/4

                arr = (x.float().data).cpu().numpy()                          # Bring back to cpu
                arr = arr[:,:,:, ::-1]                                        # Flip
                x.data = torch.from_numpy(np.ascontiguousarray(arr)).cuda()   # Store


            for roi in rois:
                x0,y0,x1,y1 = roi
                xx = x[:, :, y0:y1, x0:x1]
                logits = self.net(xx)
                probs  = F.softmax(logits)
                average += probs.data

        pass
        average = average/num_augments
        average = Variable(average)

        return average