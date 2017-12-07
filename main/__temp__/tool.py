from common import *
# common tool for dataset




































def tensor_to_img(img, mean=0, std=1, dtype=np.uint8):
    img = img.numpy()
    img = np.transpose(img, (1, 2, 0))
    img = img*std + mean
    img = img.astype(dtype)
    #img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
    return img

## transform (input is numpy array, read in by cv2)
def img_to_tensor(img, mean=0, std=1.):
    img = img.astype(np.float32)
    img = (img-mean)/std
    img = img.transpose((2,0,1))
    tensor = torch.from_numpy(img)   ##.float()
    return tensor




#http://enthusiaststudent.blogspot.jp/2015/01/horizontal-and-vertical-flip-using.html
#http://qiita.com/supersaiakujin/items/3a2ac4f2b05de584cb11
def randomVerticalFlip(img, u=0.5):
    if random.random() < u:
        img = cv2.flip(img,0)  #np.flipud(img)  #cv2.flip(img,0) ##up-down
    return img

def randomHorizontalFlip(img, u=0.5):
    shape=img.shape
    if random.random() < u:
        img = cv2.flip(img,1)  #np.fliplr(img)  #cv2.flip(img,1) ##left-right
    return img

def randomFlip(img, u=0.5):
    if random.random() < u:
        img = cv2.flip(img,random.randint(-1,1))
    return img

def randomTranspose(img, u=0.5):
    if random.random() < u:
        img = img.transpose(1,0,2)  #cv2.transpose(img)
    return img


def randomTransposeAndFlip(img, u=None):
    if u is None:
        u = random.randint(0,7)  #choose one of the 8 cases

    if u==1: #rotate90
        img = img.transpose(1,0,2)
        img = cv2.flip(img,1)
    if u==2: #rotate180
        img = cv2.flip(img,-1)
    if u==3: #rotate270
        img = img.transpose(1,0,2)
        img = cv2.flip(img,0)

    if u==4:
        img = cv2.flip(img,1)
    if u==5:
        img = cv2.flip(img,1)
        img = img.transpose(1,0,2)
        img = cv2.flip(img,1)

    if u==6:
        img = cv2.flip(img,0)
    if u==7:
        img = cv2.flip(img,0)
        img = img.transpose(1,0,2)
        img = cv2.flip(img,1)

    return img


def randomRotate(img, u=0.25, limit=(-90,90)):
    if random.random() < u:
        angle = random.uniform(limit[0],limit[1])  #degree
        height,width = img.shape[0:2]
        mat = cv2.getRotationMatrix2D((width/2,height/2),angle,1.0)
        img = cv2.warpAffine(img, mat, (height,width),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
        #img = cv2.warpAffine(img, mat, (height,width),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    return img




def randomShiftScaleRotate(img, shift_limit=(-0.0625,0.0625), scale_limit=(-0.1,0.1), rotate_limit=(-45,45), u=0.5):
    if random.random() < u:
        height,width,channel = img.shape

        angle = random.uniform(rotate_limit[0],rotate_limit[1])  #degree
        scale = random.uniform(1+scale_limit[0],1+scale_limit[1])
        dx    = round(random.uniform(shift_limit[0],shift_limit[1]))*width
        dy    = round(random.uniform(shift_limit[0],shift_limit[1]))*height

        cc = math.cos(angle/180*math.pi)*(scale)
        ss = math.sin(angle/180*math.pi)*(scale)
        rotate_matrix = np.array([ [cc,-ss], [ss,cc] ])

        box0 = np.array([ [0,0], [width,0],  [width,height], [0,height], ])
        box1 = box0 - np.array([width/2,height/2])
        box1 = np.dot(box1,rotate_matrix.T) + np.array([width/2+dx,height/2+dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0,box1)
        img = cv2.warpPerspective(img, mat, (width,height),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)  #cv2.BORDER_CONSTANT, borderValue = (0, 0, 0))  #cv2.BORDER_REFLECT_101

    return img

def centerCrop(img, height, width):

    h,w,c = img.shape
    dx = (h-height)//2
    dy = (w-width )//2

    y1 = dy
    y2 = y1 + height
    x1 = dx
    x2 = x1 + width
    img = img[y1:y2,x1:x2,:]

    return img

def randomCrop(img, height, width, u=0.5):
    h,w,c = img.shape
    dx = (h-height)//2
    dy = (w-width )//2
    if random.random() < u:
        dy = random.randint(0,(h-height-1))
        dx = random.randint(0,(w-width -1))

    y1 = dy
    y2 = y1 + height
    x1 = dx
    x2 = x1 + width
    img = img[y1:y2,x1:x2,:]

    return img


def resize(img, height, width):
    h,w = img.shape[0:2]
    if height!=h or width!=w:
        img = cv2.resize(img,(height,width))  #, interpolation=cv2.INTER_LINEAR
    return img

## unconverntional augmnet ################################################################################3
## https://stackoverflow.com/questions/6199636/formulas-for-barrel-pincushion-distortion

## https://stackoverflow.com/questions/10364201/image-transformation-in-opencv
## https://stackoverflow.com/questions/2477774/correcting-fisheye-distortion-programmatically
## http://www.coldvision.io/2017/03/02/advanced-lane-finding-using-opencv/

## barrel\pincushion distortion
def randomDistort1(img, distort_limit=0.35, shift_limit=0.25, u=0.5):

    if random.random() < u:
        height, width, channel = img.shape

        #debug
        # img = img.copy()
        # for x in range(0,width,10):
        #     cv2.line(img,(x,0),(x,height),(1,1,1),1)
        # for y in range(0,height,10):
        #     cv2.line(img,(0,y),(width,y),(1,1,1),1)

        k  = random.uniform(-distort_limit,distort_limit)  *0.00001
        dx = random.uniform(-shift_limit,shift_limit) * width
        dy = random.uniform(-shift_limit,shift_limit) * height

        #map_x, map_y = cv2.initUndistortRectifyMap(intrinsics, dist_coeffs, None, None, (width,height),cv2.CV_32FC1)
        #https://stackoverflow.com/questions/6199636/formulas-for-barrel-pincushion-distortion
        #https://stackoverflow.com/questions/10364201/image-transformation-in-opencv
        x, y = np.mgrid[0:width:1, 0:height:1]
        x = x.astype(np.float32) - width/2 -dx
        y = y.astype(np.float32) - height/2-dy
        theta = np.arctan2(y,x)
        d = (x*x + y*y)**0.5
        r = d*(1+k*d*d)
        map_x = r*np.cos(theta) + width/2 +dx
        map_y = r*np.sin(theta) + height/2+dy

        img = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
    return img


#http://pythology.blogspot.sg/2014/03/interpolation-on-regular-distorted-grid.html
## grid distortion
def randomDistort2(img, num_steps=10, distort_limit=0.2, u=0.5):

    if random.random() < u:
        height, width, channel = img.shape

        x_step = width//num_steps
        xx = np.zeros(width,np.float32)
        prev = 0
        for x in range(0, width, x_step):
            start = x
            end   = x + x_step
            if end > width:
                end = width
                cur = width
            else:
                cur = prev + x_step*(1+random.uniform(-distort_limit,distort_limit))

            xx[start:end] = np.linspace(prev,cur,end-start)
            prev=cur


        y_step = height//num_steps
        yy = np.zeros(height,np.float32)
        prev = 0
        for y in range(0, height, y_step):
            start = y
            end   = y + y_step
            if end > width:
                end = height
                cur = height
            else:
                cur = prev + y_step*(1+random.uniform(-distort_limit,distort_limit))

            yy[start:end] = np.linspace(prev,cur,end-start)
            prev=cur


        map_x,map_y =  np.meshgrid(xx, yy)
        map_x = map_x.astype(np.float32)
        map_y = map_y.astype(np.float32)
        img = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
    return img


## blur sharpen, etc
def randomFilter(img, limit=0.5, u=0.5):


    if random.random() < u:
        height, width, channel = img.shape

        alpha = limit*random.uniform(0, 1)

        ##kernel = np.ones((5,5),np.float32)/25
        kernel = np.ones((3,3),np.float32)/9*0.2

        # type = random.randint(0,1)
        # if type==0:
        #     kernel = np.ones((3,3),np.float32)/9*0.2
        # if type==1:
        #     kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])*0.5

        #kernel = alpha *sharp +(1-alpha)*blur
        #kernel = np.random.randn(5, 5)
        #kernel = kernel/np.sum(kernel*kernel)**0.5

        img = alpha*cv2.filter2D(img, -1, kernel) + (1-alpha)*img
        img = np.clip(img,0.,1.)

    return img


##https://github.com/pytorch/vision/pull/27/commits/659c854c6971ecc5b94dca3f4459ef2b7e42fb70
## color augmentation

#brightness, contrast, saturation-------------
#from mxnet code, see: https://github.com/dmlc/mxnet/blob/master/python/mxnet/image.py

# def to_grayscle(img):
#     blue  = img[:,:,0]
#     green = img[:,:,1]
#     red   = img[:,:,2]
#     grey = 0.299*red + 0.587*green + 0.114*blue
#     return grey


def randomBrightness(img, limit=0.2, u=0.5):
    if random.random() < u:
        alpha = 1.0 + limit*random.uniform(-1, 1)
        img = alpha*img
        img = np.clip(img, 0., 1.)
    return img


def randomContrast(img, limit=0.3, u=0.5):
    if random.random() < u:
        alpha = 1.0 + limit*random.uniform(-1, 1)

        coef = np.array([[[0.114, 0.587,  0.299]]]) #rgb to gray (YCbCr)
        gray = img * coef
        gray = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
        img = alpha*img  + gray
        img = np.clip(img,0.,1.)
    return img


def randomSaturation(img, limit=0.3, u=0.5):
    if random.random() < u:
        alpha = 1.0 + limit*random.uniform(-1, 1)

        coef = np.array([[[0.114, 0.587,  0.299]]])
        gray = img * coef
        gray = np.sum(gray,axis=2, keepdims=True)
        img  = alpha*img  + (1.0 - alpha)*gray
        img  = np.clip(img,0.,1.)

    return img


#return fix data for debug #####################################################3
class FixedSampler(Sampler):
    def __init__(self, data, list):
        self.num_samples = len(list)
        self.list = list

    def __iter__(self):
        #print ('\tcalling Sampler:__iter__')
        return iter(self.list)

    def __len__(self):
        #print ('\tcalling Sampler:__len__')
        return self.num_samples

class ProbSampler(Sampler):

    def sample_class_by_p(self, labels, p):

        num, num_classes = labels.shape
        idxs=[]
        for c in range(num_classes):
            N =  int(p[c]*num)
            idx = np.where(labels[:,c]>0.5)[0]
            idx = np.random.choice(idx,N,replace=len(idx)<N)
            idxs.append(idx)

        list = [i for idx in idxs for i in idx]
        random.shuffle(list)
        return list

    def __init__(self, data, p=None):

        labels = data.labels
        num, num_classes = labels.shape

        if p is None:
            p=[1./num_classes]*num_classes

        N = 0
        for c in range(num_classes):
            N +=  int(p[c]*num)

        self.p = p
        self.num_samples = num
        self.labels = labels

    def __iter__(self):
        #print ('\tcalling Sampler:__iter__')
        list = self.sample_class_by_p(self.labels, self.p)
        return iter(list)

    def __len__(self):
        #print ('\tcalling Sampler:__len__')
        return self.num_samples




def test_augment_1():

    #test transform
    def augment(x, u=0.5):
        if random.random()<u:
            if random.random()>0.5:
                x = randomDistort1(x, distort_limit=0.35, shift_limit=0.25, u=1)
            else:
                x = randomDistort2(x, num_steps=10, distort_limit=0.2, u=1)

            x = randomShiftScaleRotate(x, shift_limit=0.0625, scale_limit=0.10, rotate_limit=45, u=0.5)
            x = randomFlip(x, u=0.5)
            x = randomTranspose(x, u=0.5)
            x = randomContrast(x, limit=0.2, u=0.5)
            #x = randomSaturation(x, limit=0.2, u=0.5),

        return x


    height, width = 256,256
    for n in range(10000):
        img = np.zeros((height, width,3),np.uint8)
        img = cv2.imread('/root/share/data/kaggle-forest/classification/image/train-jpg/train_10059.jpg',1)##train_10052


        img = img.astype(np.float32)/255
        if 0:
            for x in range(0,width,10):
                cv2.line(img,(x,0),(x,height),(0,0.5,x/width),1)

            for y in range(0,height,10):
                cv2.line(img,(0,y),(width,y),(y/height,0.5,0),1)

        #img=randomShiftScaleRotate(img,u=1)
        #img=randomDistort1(img,u=1)
        #img=randomDistort2(img,u=1)
        img=augment(img,u=1)
        im_show('img',img*255)
        cv2.waitKey(500)  #12000

def test_augment_2():

    img = cv2.imread('/root/share/data/kaggle-forest/classification/image/test-jpg/file_23.jpg',1)##train_10052
    cv2.circle(img, (0,0), 50,(255,255,255),-1)
    cv2.circle(img, (0,256), 50,(0,0,255),-1)
    cv2.circle(img, (256,0), 50,(255,0,0),-1)
    im_show('img',img)
    cv2.waitKey(1)

    img = randomTransposeAndFlip(img)
    im_show('img1',img)
    cv2.waitKey(0)

# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    test_augment_2()

