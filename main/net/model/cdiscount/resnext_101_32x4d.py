from common import*
from net.model.imagenet.pretrain_convert_table import*



#  https://github.com/facebookresearch/ResNeXt
#  https://github.com/clcarwin/convert_torch_to_pytorch
#    mean = { 0.485, 0.456, 0.406 },
#     std = { 0.229, 0.224, 0.225 },


#----- helper functions ------------------------------
BN_EPS = 1e-5  #1e-4  #1e-5

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



#-----------------------------------------------------------
class NextBlock(nn.Module):
    def __init__(self, in_planes, planes, out_planes, groups=32, is_down_sample=False, stride=1):
        super(NextBlock, self).__init__()
        self.is_down_sample = is_down_sample

        self.conv_bn1 = ConvBn2d(in_planes,     planes, kernel_size=1, padding=0, stride=1)
        self.conv_bn2 = ConvBn2d(   planes,     planes, kernel_size=3, padding=1, stride=1, groups=groups)
        self.conv_bn3 = ConvBn2d(   planes, out_planes, kernel_size=1, padding=0, stride=1)

        if is_down_sample:
            self.downsample = ConvBn2d(in_planes, out_planes, kernel_size=1, padding=0, stride=stride)
            self.conv_bn2   = ConvBn2d(   planes,     planes, kernel_size=3, padding=1, stride=stride, groups=groups)


    def forward(self, x):

        if self.is_down_sample:
            residual = self.downsample(x)
        else:
            residual = x

        x = self.conv_bn1(x)
        x = F.relu(x,inplace=True)
        x = self.conv_bn2(x)
        x = F.relu(x,inplace=True)
        x = self.conv_bn3(x)
        x += residual
        x = F.relu(x,inplace=True)
        return x


#resnet
def make_layer(in_planes, planes, out_planes, num_blocks, stride):
    layers = []
    layers.append(NextBlock(in_planes, planes, out_planes, is_down_sample=True, stride=stride))
    for i in range(1, num_blocks):
        layers.append(NextBlock(out_planes, planes, out_planes))

    return nn.Sequential(*layers)


class Resnext101(nn.Module):

    def load_pretrain_pytorch_file(self,pytorch_file, skip=[]):
        #raise NotImplementedError

        pytorch_state_dict = torch.load(pytorch_file)
        state_dict = self.state_dict()
        for pytorch_key, key in RESNEXT101_CONVERT_TABLE:
            if key in skip: continue
            #print(key, pytorch_key)
            state_dict[key] = pytorch_state_dict[pytorch_key]

        self.load_state_dict(state_dict)


    def __init__(self, in_shape=(3,128,128), num_classes=5000 ):

        super(Resnext101, self).__init__()
        in_channels, height, width = in_shape
        self.num_classes=num_classes

        self.layer0 = nn.Sequential(
            ConvBn2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = make_layer(  64, 128,  256, num_blocks=3 ,stride=1) #4
        self.layer2 = make_layer( 256, 256,  512, num_blocks=4 ,stride=2) #5
        self.layer3 = make_layer( 512, 512, 1024, num_blocks=23,stride=2) #6
        self.layer4 = make_layer(1024,1024, 2048, num_blocks=3 ,stride=2) #7

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
        F.dropout(x, training=self.training, p=0.2)
        x = self.fc (x)
        return x #logits



#####################################################################################################3

def run_check_net():

    # https://discuss.pytorch.org/t/print-autograd-graph/692/8
    batch_size  = 1
    num_classes = 5270
    C,H,W = 3,180,180

    #inputs = torch.randn(batch_size,C,H,W)
    inputs = torch.ones(batch_size,C,H,W)
    labels = torch.randn(batch_size,num_classes)
    in_shape = inputs.size()[1:]

    net = Resnext101(in_shape=in_shape, num_classes=num_classes)
    net.load_pretrain_pytorch_file(
            '/root/share/project/kaggle/cdiscount/results/resnext101-180-00c/checkpoint1/00078000_model.pth',
            skip=[]#['fc.weight'	,'fc.bias']
        )
    torch.save(net.state_dict(),'/root/share/project/kaggle/cdiscount/results/resnext101-180-00c/checkpoint/00078000_model.pth')
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
    # net.eval()
    # net.merge_bn()


########################################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_check_net()












