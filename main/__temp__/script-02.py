


def random_crop_scale (image, scale_limit=[1/1.2,1.2], size=[-1,-1], u=0.5):
    if random.random() < u:
        image=image.copy()

        height,width,channel = image.shape
        if size[0]==-1: size[0]=width
        if size[1]==-1: size[1]=height
        box0 = np.array([ [0,0], [size[0],0],  [size[0],size[1]], [0,size[1]], ])


        scale = random.uniform(scale_limit[0],scale_limit[1])
        w = int(scale * width )
        h = int(scale * height)
        if scale>1:
            x0 = random.randint(0,w-size[0])
            y0 = random.randint(0,h-size[1])
            x1 = x0+size[0]
            y1 = y0+size[1]

        else:
            x0 = random.randint(0,size[0]-w)
            y0 = random.randint(0,size[1]-h)
            x1 = x0+w
            y1 = y0+h


        #<debug>
        cv2.rectangle(image,(0,0),(width-1,height-1),(0,0,255),1)
        cv2.rectangle(image,(x0,y0),(x1,y1),(0,255,0),1)



    return image




def train_augment(image):
    # height,width=image.shape[0:2]
    #
    # # crop random ---------
    # x0 = np.random.choice(width -160)
    # y0 = np.random.choice(height-160)
    # x1 = x0+160
    # y1 = y0+160
    # image = fix_crop(image, roi=(x0,y0,x1,y1))
    #
    # # flip  random ---------
    # image = random_horizontal_flip(image, u=0.5)


    tensor = image_to_tensor_transform(image)
    return tensor



def valid_augment(image):
    # height,width=image.shape[0:2]
    #
    #
    # # crop center ---------
    # x0 = (width- 160)//2
    # y0 = (height-160)//2
    # x1 = x0+160
    # y1 = y0+160
    # image = fix_crop(image, roi=(x0,y0,x1,y1))

    tensor = image_to_tensor_transform(image)
    return tensor
