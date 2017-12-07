from common import*
from net.model.imagenet.pretrain_convert_table import*

# squeeze-excite
# https://github.com/titu1994/keras-squeeze-excite-network

# https://github.com/moskomule/senet.pytorch
# https://github.com/hujie-frank/SENet
# https://github.com/ruotianluo/pytorch-resnet




#----- helper functions ------------------------------
BN_EPS = 1e-5  #1e-4  #1e-5

#https://github.com/pytorch/pytorch/issues/1959
class LayerNorm(nn.Module):

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))


    def forward(self, x):
        #mean = x.mean(-1, keepdim=True)
        std  = x.std(-1, keepdim=True)
        return self.gamma * (x) / (std + BN_EPS)





class ConvBn2d(nn.Module):

    def merge_bn(self):
        #raise NotImplementedError
        assert(self.conv.bias==None)
        conv_weight     = self.conv.weight.data
        bn_weight       = self.bn.weight.data
        bn_bias         = self.bn.bias.data
        bn_running_mean = self.bn.running_mean
        bn_running_var  = self.bn.running_var
        bn_eps          = self.bn.eps

        #https://github.com/sanghoon/pva-faster-rcnn/issues/5
        #https://github.com/sanghoon/pva-faster-rcnn/commit/39570aab8c6513f0e76e5ab5dba8dfbf63e9c68c

        N,C,KH,KW = conv_weight.size()
        std = 1/(torch.sqrt(bn_running_var+bn_eps))
        std_bn_weight =(std*bn_weight).repeat(C*KH*KW,1).t().contiguous().view(N,C,KH,KW )
        conv_weight_hat = std_bn_weight*conv_weight
        conv_bias_hat   = (bn_bias - bn_weight*std*bn_running_mean)

        self.bn   = None
        self.conv = nn.Conv2d(in_channels=self.conv.in_channels, out_channels=self.conv.out_channels, kernel_size=self.conv.kernel_size,
                              padding=self.conv.padding, stride=self.conv.stride, dilation=self.conv.dilation, groups=self.conv.groups,
                              bias=True)
        self.conv.weight.data = conv_weight_hat #fill in
        self.conv.bias.data   = conv_bias_hat


    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1, stride=1, groups=1, is_bn=True):
        super(ConvBn2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups, bias=False)
        self.bn   = nn.BatchNorm2d(out_channels, eps=BN_EPS)

        if is_bn is False:
            self.bn =None

    def forward(self,x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        return x


# class SEScale(nn.Module):
#     def __init__(self, channel, reduction=16):
#         super(SEScale, self).__init__()
#         self.fc1 = nn.Linear(channel, reduction)
#         self.fc2 = nn.Linear(reduction, channel)
#
#     def forward(self, x):
#         N, C, _, _ = x.size()
#         z = F.adaptive_avg_pool2d(x,1).view(N, C)
#         z = self.fc1(z)
#         z = F.relu(z, inplace=True)
#         z = self.fc2(z)
#         z = F.sigmoid(z).view(N, C, 1, 1)
#         x = x*z
#         return x

class SEScale(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEScale, self).__init__()
        self.fc1 = nn.Conv2d(channel, reduction, kernel_size=1, padding=0)
        self.fc2 = nn.Conv2d(reduction, channel, kernel_size=1, padding=0)

    def forward(self, x):
        x = F.adaptive_avg_pool2d(x,1)
        x = self.fc1(x)
        x = F.relu(x, inplace=True)
        x = self.fc2(x)
        x = F.sigmoid(x)
        return x


class SEBasicBlock(nn.Module):

    def __init__(self, in_planes, planes, out_planes, reduction, stride=1):
        super(SEBasicBlock, self).__init__()
        self.is_downsample = in_planes!=out_planes or stride != 1

        self.conv_bn1   = ConvBn2d(in_planes, planes,     kernel_size=3, padding=1, stride=stride)
        self.conv_bn2   = ConvBn2d(   planes, out_planes, kernel_size=3, padding=1, stride=1)
        self.scale      = SEScale(out_planes, reduction)

        if self.is_downsample:
            self.downsample = ConvBn2d(in_planes, out_planes, kernel_size=1, padding=0, stride=stride)

    def forward(self, x):
        if self.is_downsample:
            residual = self.downsample(x)
        else:
            residual = x

        x = self.conv_bn1(x)
        x = F.relu(x,inplace=True)
        x = self.conv_bn2(x)
        x = self.scale(x)*x + residual
        x = F.relu(x,inplace=True)
        return x



class SEBottleneck(nn.Module):
    def __init__(self, in_planes, planes, out_planes, reduction, stride=1):
        super(SEBottleneck, self).__init__()
        self.is_downsample = in_planes!=out_planes or stride != 1

        self.conv_bn1 = ConvBn2d(in_planes,     planes, kernel_size=1, padding=0, stride=1)
        self.conv_bn2 = ConvBn2d(   planes,     planes, kernel_size=3, padding=1, stride=stride)
        self.conv_bn3 = ConvBn2d(   planes, out_planes, kernel_size=1, padding=0, stride=1)
        self.scale    = SEScale(out_planes, reduction)

        if self.is_downsample:
            self.downsample = ConvBn2d(in_planes, out_planes, kernel_size=1, padding=0, stride=stride)


    def forward(self, x):
        if self.is_downsample:
            residual = self.downsample(x)
        else:
            residual = x

        x = self.conv_bn1(x)
        x = F.relu(x,inplace=True)
        x = self.conv_bn2(x)
        x = F.relu(x,inplace=True)
        x = self.conv_bn3(x)
        x = self.scale(x)*x + residual
        x = F.relu(x,inplace=True)
        return x

is_down_sample=True
#resnet
def make_layer(in_planes, planes, out_planes, reduction, num_blocks, stride):

    layers = []
    layers.append(SEBottleneck(in_planes, planes, out_planes, reduction, stride=stride))
    for i in range(1, num_blocks):
        layers.append(SEBottleneck(out_planes, planes, out_planes, reduction))

    return nn.Sequential(*layers)



# def make_flat(out):
#     flat = out.view(out.size(0), -1)
#     return flat




## resenet   ##
class SEResNet50_3x3(nn.Module):

    def load_pretrain_pytorch_file(self,pytorch_file, skip=[]):

        pytorch_state_dict = torch.load(pytorch_file,map_location=lambda storage, loc: storage)
        state_dict = self.state_dict()

        keys = list(state_dict.keys())
        for key in keys:
            # if key in skip:
            #     continue
            # # if '.scale.' in key:
            # #     continue
            # if 'layer0.' in key:
            #      continue
            # if 'fc_scale.' in key:
            #      continue
            if any(s in key for s in skip):
                continue

            #print(key)
            state_dict[key] = pytorch_state_dict[key]

        self.load_state_dict(state_dict)


    def merge_bn(self):
        print ('merging bn ....')

        for name, m in self.named_modules():
            if isinstance(m, (ConvBn2d,)):
                print('\t%s'%name)
                m.merge_bn()
        print('')

    #-----------------------------------------------------------------------
    def __init__(self, in_shape=(3,128,128), num_classes=5000 ):

        super(SEResNet50_3x3, self).__init__()
        in_channels, height, width = in_shape
        self.num_classes=num_classes

        # self.layer0 = nn.Sequential(
        #     ConvBn2d(in_channels, 64,kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(inplace=True),
        #     ConvBn2d(64, 64, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        #
        #     ConvBn2d(64, 64, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(inplace=True),
        #     ConvBn2d(64, 64, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        # )
        self.layer0 = nn.Sequential(
            ConvBn2d(in_channels, 64,kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            ConvBn2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            SEBasicBlock(64, 64, 64, reduction= 16, stride=2),
        )

        self.layer1 = make_layer(  64,  64,  256, reduction= 16, num_blocks=3, stride=1)  #out =  64*4 =  256
        self.layer2 = make_layer( 256, 128,  512, reduction= 32, num_blocks=4, stride=2)  #out = 128*4 =  512
        self.layer3 = make_layer( 512, 256, 1024, reduction= 64, num_blocks=6, stride=2)  #out = 256*4 = 1024
        self.layer4 = make_layer(1024, 512, 2048, reduction=128, num_blocks=3, stride=2)  #out = 512*4 = 2048
        self.fc  = nn.Linear(2048, num_classes)

        #self.fc_scale = LayerNorm(num_classes)
        self.fc_scale = SEScale(2048,256)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if 0:
            x = F.adaptive_avg_pool2d(x, output_size=1)
            x = x.view(x.size(0), -1)
            x = self.fc (x)
            x = self.fc_scale(x)

        if 1:
            x = F.adaptive_avg_pool2d(x, output_size=1)
            x = self.fc_scale(x)
            x = x.view(x.size(0), -1)
            x = self.fc (x)


        return x #logits



########################################################################################################

def run_check_convert():

    pytorch_file  = '/root/share/data/models/reference/senet/ruotianluo/SE-ResNet-50.convert.pth'
    pytorch_state_dict = torch.load(pytorch_file)

    net = SEResNet50()
    state_dict = net.state_dict()
    xx=0


def run_check_net():

    # https://discuss.pytorch.org/t/print-autograd-graph/692/8
    batch_size  = 1
    num_classes = 5000
    C,H,W = 3,128,128

    inputs = torch.randn(batch_size,C,H,W)
    labels = torch.randn(batch_size,num_classes)
    in_shape = inputs.size()[1:]


    net = SEResNet50_3x3(in_shape=in_shape, num_classes=num_classes)
    net.load_pretrain_pytorch_file(
            #'/root/share/project/kaggle/cdiscount/results/__submission__/stable-00/excited-resnet50-180-00a/checkpoint/00061000_model.pth',
            '/root/share/project/kaggle/cdiscount/results/excited-resnet50-3x3-01/checkpoint/00015000_model.pth',

            skip = ['layer0.', 'fc_scale', 'fc']
        )
    net.cuda().train()

    x = Variable(inputs).cuda()
    y = Variable(labels).cuda()
    logits = net.forward(x)
    probs  = F.softmax(logits)

    loss = F.binary_cross_entropy_with_logits(logits, y)
    loss.backward()

    print(type(net))
    print(net)

    print('probs')
    print(probs)

    #merging
    net.eval()
    net.merge_bn()


########################################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_check_net()

