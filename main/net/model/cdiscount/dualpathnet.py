## https://github.com/cypw/DPNs
## https://github.com/rwightman/pytorch-dpn-pretrained

from common import*
if __name__ == '__main__':
    from adaptive_avgmax_pool import adaptive_avgmax_pool2d
else:
    from .adaptive_avgmax_pool import adaptive_avgmax_pool2d


# https://github.com/rwightman/pytorch-dpn-pretrained/blob/fc2ca159630c8c190702763613883e34b2274956/model_factory.py
#      mean=[124 / 255, 117 / 255, 104 / 255],
#      std=[1 / (.0167 * 255)] * 3)


def dpn68(num_classes=1000, pretrained=False, test_time_pool=True):
    model = DPN(
        small=True, num_init_features=10, k_r=128, groups=32,
        k_sec=(3, 4, 12, 3), inc_sec=(16, 32, 32, 64),
        num_classes=num_classes, test_time_pool=test_time_pool)
    if pretrained:
        if model_urls['dpn68']:
            model.load_state_dict(model_zoo.load_url(model_urls['dpn68']))
        elif has_mxnet and os.path.exists('./pretrained/'):
            convert_from_mxnet(model, checkpoint_prefix='./pretrained/dpn68')
        else:
            assert False, "Unable to load a pretrained model"
    return model


def dpn68b(num_classes=1000, pretrained=False, test_time_pool=True):
    model = DPN(
        small=True, num_init_features=10, k_r=128, groups=32,
        b=True, k_sec=(3, 4, 12, 3), inc_sec=(16, 32, 32, 64),
        num_classes=num_classes, test_time_pool=test_time_pool)
    if pretrained:
        if model_urls['dpn68b-extra']:
            model.load_state_dict(model_zoo.load_url(model_urls['dpn68b-extra']))
        elif has_mxnet and os.path.exists('./pretrained/'):
            convert_from_mxnet(model, checkpoint_prefix='./pretrained/dpn68-extra')
        else:
            assert False, "Unable to load a pretrained model"
    return model


def dpn92(num_classes=1000, pretrained=False, test_time_pool=True, extra=True):
    model = DPN(
        num_init_features=64, k_r=96, groups=32,
        k_sec=(3, 4, 20, 3), inc_sec=(16, 32, 24, 128),
        num_classes=num_classes, test_time_pool=test_time_pool)
    if pretrained:
        # there are both imagenet 5k trained, 1k finetuned 'extra' weights
        # and normal imagenet 1k trained weights for dpn92
        key = 'dpn92'
        if extra:
            key += '-extra'
        if model_urls[key]:
            model.load_state_dict(model_zoo.load_url(model_urls[key]))
        elif has_mxnet and os.path.exists('./pretrained/'):
            convert_from_mxnet(model, checkpoint_prefix='./pretrained/' + key)
        else:
            assert False, "Unable to load a pretrained model"
    return model


def dpn98(num_classes=1000, pretrained=False, test_time_pool=True):
    model = DPN(
        num_init_features=96, k_r=160, groups=40,
        k_sec=(3, 6, 20, 3), inc_sec=(16, 32, 32, 128),
        num_classes=num_classes, test_time_pool=test_time_pool)
    if pretrained:
        if model_urls['dpn98']:
            model.load_state_dict(model_zoo.load_url(model_urls['dpn98']))
        elif has_mxnet and os.path.exists('./pretrained/'):
            convert_from_mxnet(model, checkpoint_prefix='./pretrained/dpn98')
        else:
            assert False, "Unable to load a pretrained model"
    return model


def dpn131(num_classes=1000, pretrained=False, test_time_pool=True):
    model = DPN(
        num_init_features=128, k_r=160, groups=40,
        k_sec=(4, 8, 28, 3), inc_sec=(16, 32, 32, 128),
        num_classes=num_classes, test_time_pool=test_time_pool)
    if pretrained:
        if model_urls['dpn131']:
            model.load_state_dict(model_zoo.load_url(model_urls['dpn131']))
        elif has_mxnet and os.path.exists('./pretrained/'):
            convert_from_mxnet(model, checkpoint_prefix='./pretrained/dpn131')
        else:
            assert False, "Unable to load a pretrained model"
    return model


def dpn107(num_classes=1000, pretrained=False, test_time_pool=True):
    model = DPN(
        num_init_features=128, k_r=200, groups=50,
        k_sec=(4, 8, 20, 3), inc_sec=(20, 64, 64, 128),
        num_classes=num_classes, test_time_pool=test_time_pool)
    if pretrained:
        if model_urls['dpn107-extra']:
            model.load_state_dict(model_zoo.load_url(model_urls['dpn107-extra']))
        elif has_mxnet and os.path.exists('./pretrained/'):
            convert_from_mxnet(model, checkpoint_prefix='./pretrained/dpn107-extra')
        else:
            assert False, "Unable to load a pretrained model"
    return model


class CatBnAct(nn.Module):
    def __init__(self, in_chs, activation_fn=nn.ReLU(inplace=True)):
        super(CatBnAct, self).__init__()
        self.bn = nn.BatchNorm2d(in_chs, eps=0.001)
        self.act = activation_fn

    def forward(self, x):
        x = torch.cat(x, dim=1) if isinstance(x, tuple) else x
        return self.act(self.bn(x))


class BnActConv2d(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size, stride,
                 padding=0, groups=1, activation_fn=nn.ReLU(inplace=True)):
        super(BnActConv2d, self).__init__()
        self.bn = nn.BatchNorm2d(in_chs, eps=0.001)
        self.act = activation_fn
        self.conv = nn.Conv2d(in_chs, out_chs, kernel_size, stride, padding, groups=groups, bias=False)

    def forward(self, x):
        return self.conv(self.act(self.bn(x)))


class InputBlock(nn.Module):
    def __init__(self, num_init_features, kernel_size=7,
                 padding=3, activation_fn=nn.ReLU(inplace=True)):
        super(InputBlock, self).__init__()
        self.conv = nn.Conv2d(
            3, num_init_features, kernel_size=kernel_size, stride=2, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(num_init_features, eps=0.001)
        self.act = activation_fn
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        return x


class DualPathBlock(nn.Module):
    def __init__(
            self, in_chs, num_1x1_a, num_3x3_b, num_1x1_c, inc, groups, block_type='normal', b=False):
        super(DualPathBlock, self).__init__()
        self.num_1x1_c = num_1x1_c
        self.inc = inc
        self.b = b
        if block_type is 'proj':
            self.key_stride = 1
            self.has_proj = True
        elif block_type is 'down':
            self.key_stride = 2
            self.has_proj = True
        else:
            assert block_type is 'normal'
            self.key_stride = 1
            self.has_proj = False

        if self.has_proj:
            # Using different member names here to allow easier parameter key matching for conversion
            if self.key_stride == 2:
                self.c1x1_w_s2 = BnActConv2d(
                    in_chs=in_chs, out_chs=num_1x1_c + 2 * inc, kernel_size=1, stride=2)
            else:
                self.c1x1_w_s1 = BnActConv2d(
                    in_chs=in_chs, out_chs=num_1x1_c + 2 * inc, kernel_size=1, stride=1)
        self.c1x1_a = BnActConv2d(in_chs=in_chs, out_chs=num_1x1_a, kernel_size=1, stride=1)
        self.c3x3_b = BnActConv2d(
            in_chs=num_1x1_a, out_chs=num_3x3_b, kernel_size=3,
            stride=self.key_stride, padding=1, groups=groups)
        if b:
            self.c1x1_c = CatBnAct(in_chs=num_3x3_b)
            self.c1x1_c1 = nn.Conv2d(num_3x3_b, num_1x1_c, kernel_size=1, bias=False)
            self.c1x1_c2 = nn.Conv2d(num_3x3_b, inc, kernel_size=1, bias=False)
        else:
            self.c1x1_c = BnActConv2d(in_chs=num_3x3_b, out_chs=num_1x1_c + inc, kernel_size=1, stride=1)

    def forward(self, x):
        x_in = torch.cat(x, dim=1) if isinstance(x, tuple) else x
        if self.has_proj:
            if self.key_stride == 2:
                x_s = self.c1x1_w_s2(x_in)
            else:
                x_s = self.c1x1_w_s1(x_in)
            x_s1 = x_s[:, :self.num_1x1_c, :, :]
            x_s2 = x_s[:, self.num_1x1_c:, :, :]
        else:
            x_s1 = x[0]
            x_s2 = x[1]
        x_in = self.c1x1_a(x_in)
        x_in = self.c3x3_b(x_in)
        if self.b:
            x_in = self.c1x1_c(x_in)
            out1 = self.c1x1_c1(x_in)
            out2 = self.c1x1_c2(x_in)
        else:
            x_in = self.c1x1_c(x_in)
            out1 = x_in[:, :self.num_1x1_c, :, :]
            out2 = x_in[:, self.num_1x1_c:, :, :]
        resid = x_s1 + out1
        dense = torch.cat([x_s2, out2], dim=1)
        return resid, dense


class DPN(nn.Module):
    def __init__(self, small=False, num_init_features=64, k_r=96, groups=32,
                 b=False, k_sec=(3, 4, 20, 3), inc_sec=(16, 32, 24, 128),
                 num_classes=1000, test_time_pool=False):
        super(DPN, self).__init__()
        self.test_time_pool = test_time_pool
        self.b = b
        bw_factor = 1 if small else 4

        blocks = collections.OrderedDict()

        # conv1
        if small:
            blocks['conv1_1'] = InputBlock(num_init_features, kernel_size=3, padding=1)
        else:
            blocks['conv1_1'] = InputBlock(num_init_features, kernel_size=7, padding=3)

        # conv2
        bw = 64 * bw_factor
        inc = inc_sec[0]
        r = (k_r * bw) // (64 * bw_factor)
        blocks['conv2_1'] = DualPathBlock(num_init_features, r, r, bw, inc, groups, 'proj', b)
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[0] + 1):
            blocks['conv2_' + str(i)] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'normal', b)
            in_chs += inc

        # conv3
        bw = 128 * bw_factor
        inc = inc_sec[1]
        r = (k_r * bw) // (64 * bw_factor)
        blocks['conv3_1'] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'down', b)
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[1] + 1):
            blocks['conv3_' + str(i)] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'normal', b)
            in_chs += inc

        # conv4
        bw = 256 * bw_factor
        inc = inc_sec[2]
        r = (k_r * bw) // (64 * bw_factor)
        blocks['conv4_1'] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'down', b)
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[2] + 1):
            blocks['conv4_' + str(i)] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'normal', b)
            in_chs += inc

        # conv5
        bw = 512 * bw_factor
        inc = inc_sec[3]
        r = (k_r * bw) // (64 * bw_factor)
        blocks['conv5_1'] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'down', b)
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[3] + 1):
            blocks['conv5_' + str(i)] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'normal', b)
            in_chs += inc
        blocks['conv5_bn_ac'] = CatBnAct(in_chs)

        self.features = nn.Sequential(blocks)

        # Using 1x1 conv for the FC layer to allow the extra pooling scheme
        self.fc = nn.Conv2d(in_chs, num_classes, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.features(x)
        if not self.training and self.test_time_pool:
            x = F.avg_pool2d(x, kernel_size=7, stride=1)
            out = self.fc(x)
            # The extra test time pool should be pooling an img_size//32 - 6 size patch
            out = adaptive_avgmax_pool2d(out, pool_type='avgmax')
        else:
            x = adaptive_avgmax_pool2d(x, pool_type='avg')
            out = self.fc(x)

        return out.view(out.size(0), -1)



class DPN92(nn.Module):

    def load_pretrain_pytorch_file(self,pytorch_file, skip=[]):
        pytorch_state_dict = torch.load(pytorch_file)
        state_dict = self.dpn.state_dict()

        keys = list(state_dict.keys())
        for key in keys:
            if key in skip: continue
            state_dict[key] = pytorch_state_dict[key]

        self.dpn.load_state_dict(state_dict)
        pass

    #-----------------------------------------------------------------------

    def __init__(self, in_shape=(3,128,128), num_classes=1000 ):
        super(DPN92, self).__init__()

        in_channels, height, width = in_shape
        self.num_classes=num_classes
        assert(in_channels==3)

        test_time_pool=True
        self.dpn = DPN(
            num_init_features=64, k_r=96, groups=32,
            k_sec=(3, 4, 20, 3), inc_sec=(16, 32, 24, 128),
            num_classes=num_classes, test_time_pool=test_time_pool)


    def forward(self, x):
       return self.dpn(x)



class DPN107(nn.Module):

    def load_pretrain_pytorch_file(self,pytorch_file, skip=[]):
        pytorch_state_dict = torch.load(pytorch_file)
        state_dict = self.dpn.state_dict()

        keys = list(state_dict.keys())
        for key in keys:
            if key in skip: continue
            state_dict[key] = pytorch_state_dict[key]

        self.dpn.load_state_dict(state_dict)
        pass

    #-----------------------------------------------------------------------

    def __init__(self, in_shape=(3,128,128), num_classes=1000 ):
        super(DPN107, self).__init__()

        in_channels, height, width = in_shape
        self.num_classes=num_classes
        assert(in_channels==3)

        test_time_pool=True
        self.dpn = DPN(
            num_init_features=128, k_r=200, groups=50,
            k_sec=(4, 8, 20, 3), inc_sec=(20, 64, 64, 128),
            num_classes=num_classes, test_time_pool=test_time_pool)


    def forward(self, x):
       return self.dpn(x)


########################################################################################
def run_check_net():

    # https://discuss.pytorch.org/t/print-autograd-graph/692/8
    batch_size  = 1
    num_classes = 5000
    C,H,W = 3,128,128

    inputs = torch.randn(batch_size,C,H,W)
    labels = torch.randn(batch_size,num_classes)
    in_shape = inputs.size()[1:]

    net = DPN107(in_shape=in_shape, num_classes=num_classes)
    net.load_pretrain_pytorch_file(
           '/root/share/data/models/reference/imagenet/dualpathnet/DPN-107_5k_to_1k/dpn107-extra.pth',
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
    # net.eval()
    # net.merge_bn()



########################################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_check_net()



