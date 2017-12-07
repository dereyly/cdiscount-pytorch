from common import*
from net.model.imagenet.pretrain_convert_table import*


#----- helper functions ------------------------------
BN_EPS = 1e-4  #1e-4  #1e-5

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



# class BasicBlock(nn.Module):
#
#     def __init__(self, inplanes, planes, expansion = 1, stride=1, downsample=None):
#         super(BasicBlock, self).__init__()
#         self.stride = stride
#
#         self.conv_bn1   = ConvBn2d(inplanes, planes, kernel_size=3, padding=1, stride=stride)
#         self.conv_bn2   = ConvBn2d(planes, planes*expansion, kernel_size=3, padding=1, stride=1)
#         self.downsample = downsample
#
#     def forward(self, x):
#         residual = x
#         out = self.conv_bn1(x)
#         out = F.relu(out,inplace=True)
#         out = self.conv_bn2(out)
#
#         if self.downsample is not None:
#             residual = self.downsample(x)
#
#         out += residual
#         out = F.relu(out,inplace=True)
#         return out

class SEScale(nn.Module):
    def __init__(self, channel, reduction):
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

#-----------------------------------------------------------------------
class SENextBlock(nn.Module):
    def __init__(self, in_planes, planes, out_planes, reduction, groups=64, is_downsample=False, downsample_stride=1, downsample_kernel_size=1):
        super(SENextBlock, self).__init__()
        self.is_downsample = is_downsample

        self.conv_bn1 = ConvBn2d(in_planes,     planes, kernel_size=1, padding=0, stride=1)
        self.conv_bn2 = ConvBn2d(   planes, out_planes, kernel_size=3, padding=1, stride=downsample_stride, groups=groups)
        self.conv_bn3 = ConvBn2d(out_planes,out_planes, kernel_size=1, padding=0, stride=1)
        self.scale    = SEScale(out_planes, reduction)
        if is_downsample:
            self.downsample = ConvBn2d(in_planes, out_planes,
                                       kernel_size=downsample_kernel_size, padding=downsample_kernel_size//2,
                                       stride=downsample_stride)


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


### senet layers #############################################################
# def make_flat(out):
#     flat = out.view(out.size(0), -1)
#     return flat


def make_layer0(in_channels, channels, out_channels):

    layers=[
        ConvBn2d(in_channels, channels, kernel_size=3, stride=2, padding=1),
        nn.ReLU(inplace=True),

        ConvBn2d(channels, channels, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),

        ConvBn2d(channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),

        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
    ]
    return nn.Sequential(*layers)



def make_layer(in_planes, planes, out_planes, reduction, groups, num_blocks, downsample_stride, downsample_kernel_size):

    layers = []
    layers.append(SENextBlock(in_planes, planes, out_planes, reduction,
                              is_downsample=True, downsample_stride=downsample_stride,
                              downsample_kernel_size=downsample_kernel_size))
    for i in range(1, num_blocks):
        layers.append(SENextBlock(out_planes, planes, out_planes, reduction))

    return nn.Sequential(*layers)







## SeNet  #############################################################
class SENet4096(nn.Module):
    def __init__(self, in_shape=(3,128,128), num_classes=5000 ):
        super(SENet4096, self).__init__()
        pass
    def forward(self, x):
        pass




class SENet2048(nn.Module):

    def load_pretrain_pytorch_file(self,pytorch_file, skip=[]):
        pytorch_state_dict = torch.load(pytorch_file)
        state_dict = self.state_dict()
        keys = list(state_dict.keys())
        for key in keys:
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

        super(SENet2048, self).__init__()
        in_channels, height, width = in_shape
        self.num_classes=num_classes

        self.layer0 = make_layer0(in_channels, 64 ,128)
        self.layer1 = make_layer( 128,  128,  256, reduction= 16, groups=64, num_blocks= 3, downsample_stride=1, downsample_kernel_size=1)
        self.layer2 = make_layer( 256,  256,  512, reduction= 32, groups=64, num_blocks= 8, downsample_stride=2, downsample_kernel_size=3)
        self.layer3 = make_layer( 512,  512, 1024, reduction= 64, groups=64, num_blocks=36, downsample_stride=2, downsample_kernel_size=3)
        self.layer4 = make_layer(1024, 1024, 2048, reduction=128, groups=64, num_blocks= 3, downsample_stride=2, downsample_kernel_size=3)

        self.fc = nn.Linear(2048, num_classes)


    def forward(self, x):
                            ##; print('input ', x.size())
        x = self.layer0(x)  ##; print('layer0 ',x.size())
        x = self.layer1(x)  ##; print('layer1 ',x.size())
        x = self.layer2(x)  ##; print('layer2 ',x.size())
        x = self.layer3(x)  ##; print('layer3 ',x.size())
        x = self.layer4(x)  ##; print('layer4 ',x.size())

        x = F.adaptive_avg_pool2d(x, output_size=1)
        x = x.view(x.size(0), -1)
        x = self.fc (x)
        return x #logits



########################################################################################################


def run_check_net():

    # https://discuss.pytorch.org/t/print-autograd-graph/692/8
    batch_size  = 1
    num_classes = 5270
    C,H,W = 3,180,180

    inputs = torch.randn(batch_size,C,H,W)
    labels = torch.randn(batch_size,num_classes)
    in_shape = inputs.size()[1:]

    net = SENet2048(in_shape=in_shape, num_classes=num_classes)
    # net.load_pretrain_pytorch_file(
    #         '/root/share/data/models/pytorch/imagenet/resenet/resnet50-19c8e357.pth',
    #         skip=['fc.weight'	,'fc.bias']
    #     )
    net.cuda()
    net.train()

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
    #net.eval()
    #net.merge_bn()



########################################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_check_net()

