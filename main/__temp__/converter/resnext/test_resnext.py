from common import *

# --------------------------------------------
from net.model.cdiscount.resnext_101_32x4d  import Resnext101 as Net
#https://github.com/Cadene/pretrained-models.pytorch






def image_to_tensor_transform(image):
    mean = [0.485, 0.456, 0.406 ]
    std  = [0.229, 0.224, 0.225 ]
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = image.transpose((2,0,1))
    tensor = torch.from_numpy(image).float().div(255)
    tensor[0] = (tensor[0] - mean[0]) / std[0]
    tensor[1] = (tensor[1] - mean[1]) / std[1]
    tensor[2] = (tensor[2] - mean[2]) / std[2]

    return tensor



# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    ##  https://github.com/soeaver/caffe-model/blob/master/cls/inception/deploy_xception.prototxt
    ##  https://github.com/soeaver/caffe-model
    ##  https://github.com/soeaver/caffe-model/blob/master/cls/evaluation_cls.py
    #   https://github.com/soeaver/caffe-model/blob/master/cls/synset.txt
    #    (441)  810 n02823750 beer glass
    #    (  1)  449 n01443537 goldfish, Carassius auratus
    #    (  9)  384 n01518878 ostrich, Struthio camelus
    #img_file = '/root/share/data/imagenet/dummy/256x256/goldfish.jpg'
    #img_file = '/root/share/data/imagenet/dummy/256x256/beer_glass.jpg'
    img_file = '/root/share/data/imagenet/dummy/256x256/ostrich.jpg'
    img = cv2.imread(img_file)
    #img = np.ones((224,224,3),np.uint8)


    pytorch_model = Net(in_shape=(3,224,224),num_classes=1000)
    pytorch_model.load_pretrain_pytorch_file(
            '/root/share/data/models/reference/imagenet/resnext/resnext_101_32x4d.convert.pth',
            skip=[]
        )
    pytorch_model.eval()


    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    x = image_to_tensor_transform(img)


    pytorch_prob = pytorch_model( Variable((x).unsqueeze(0) ) )
    pytorch_prob = F.softmax(pytorch_prob).data.numpy().reshape(-1)
    #print('pytorch_prob\n',pytorch_prob)
    print('pytorch ', np.argmax(pytorch_prob), ' ', pytorch_prob[np.argmax(pytorch_prob)])


    print('\nsucess!')