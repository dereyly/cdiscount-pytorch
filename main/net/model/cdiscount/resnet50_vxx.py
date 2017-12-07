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



class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes, expansion = 1, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.stride = stride

        self.conv_bn1   = ConvBn2d(inplanes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv_bn2   = ConvBn2d(planes, planes*expansion, kernel_size=3, padding=1, stride=1)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv_bn1(x)
        out = F.relu(out,inplace=True)
        out = self.conv_bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = F.relu(out,inplace=True)
        return out


class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, expansion=4, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.stride = stride

        self.conv_bn1 = ConvBn2d(inplanes, planes, kernel_size=1, padding=0, stride=1)
        self.conv_bn2 = ConvBn2d(planes, planes,   kernel_size=3, padding=1, stride=stride)
        self.conv_bn3 = ConvBn2d(planes, planes * expansion, kernel_size=1, padding=0, stride=1)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv_bn1(x)
        out = F.relu(out,inplace=True)
        out = self.conv_bn2(out)
        out = F.relu(out,inplace=True)
        out = self.conv_bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = F.relu(out,inplace=True)
        return out


#resnet
def make_layer(block, inplanes, planes, expansion, num_blocks, stride=1):
    downsample = None

    if stride != 1 or inplanes != planes*expansion:
        downsample = ConvBn2d(inplanes, planes*expansion, kernel_size=1, padding=0, stride=stride)

    layers = []
    layers.append(block(inplanes, planes, expansion, stride, downsample))
    inplanes = planes*expansion

    for i in range(1, num_blocks):
        layers.append(block(inplanes, planes, expansion))

    return nn.Sequential(*layers)



# def make_flat(out):
#     flat = out.view(out.size(0), -1)
#     return flat




## resenet   ##
class ResNet50(nn.Module):

    def load_pretrain_pytorch_file(self,pytorch_file, skip=[]):

        pytorch_state_dict = torch.load(pytorch_file)
        state_dict = self.state_dict()

        for key, pytorch_key  in RESNET50_CONVERT_TABLE:
            if key in skip: continue
            state_dict[key] = pytorch_state_dict[pytorch_key]

        self.load_state_dict(state_dict)
        #torch.save(state_dict,save_model_file)


    def merge_bn(self):
        print ('merging bn ....')

        for name, m in self.named_modules():
            if isinstance(m, (ConvBn2d,)):
                print('\t%s'%name)
                m.merge_bn()
        print('')

    #-----------------------------------------------------------------------
    def __init__(self, in_shape=(3,128,128), num_classes=5000 ):

        super(ResNet50, self).__init__()
        in_channels, height, width = in_shape

        self.layer0 = nn.Sequential(
            ConvBn2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = make_layer(Bottleneck,   64,  64, 4, num_blocks=3, stride=1)  #out =  64*4 =  256
        self.layer2 = make_layer(Bottleneck,  256, 128, 4, num_blocks=4, stride=2)  #out = 128*4 =  512
        self.layer3 = make_layer(Bottleneck,  512, 256, 4, num_blocks=6, stride=2)  #out = 256*4 = 1024
        self.layer4 = make_layer(Bottleneck, 1024, 512, 4, num_blocks=3, stride=2)  #out = 512*4 = 2048
        self.fc  = nn.Linear(2048, num_classes)


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

        x = F.adaptive_avg_pool2d(x, output_size=1)
        x = x.view(x.size(0), -1)
        x = self.fc (x)
        return x #logits



########################################################################################################

def run_check_convert():
    pytorch_file  = '/root/share/data/models/pytorch/imagenet/resenet/resnet50-19c8e357.pth'
    pytorch_state_dict = torch.load(pytorch_file)

    net = ResNet50()
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

    net = ResNet50(in_shape=in_shape, num_classes=num_classes)
    net.load_pretrain_pytorch_file(
            '/root/share/data/models/pytorch/imagenet/resenet/resnet50-19c8e357.pth',
            skip=['fc.weight'	,'fc.bias']
        )
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
    net.eval()
    net.merge_bn()



########################################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_check_net()

