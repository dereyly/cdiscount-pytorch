import torch.nn as nn
import torch
import math
import torch.utils.model_zoo as model_zoo
# from se_module import SELayer
from .se_resnet import SEBasicBlock

__all__ = ['resnet_mod18', 'resnet18_multi', 'se_resnet_mod18', 'se_resnet_mod28', 'ResNet', 'resnet18', 'resnet34', 'resnet34_extract',
           'resnet50', 'resnet101', 'resnet101_fc', 'resnet101_multi',
           'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Delta(nn.Module):
    def __init__(self, in_channels):
        super(Delta, self).__init__()
        self.avgpool = nn.AvgPool2d(5)
        self.avgnopool = nn.AvgPool2d(kernel_size=5, stride=1, padding=3)
        self.shrink1x1 = nn.Conv2d(in_channels, 128, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, x):
        pool = self.avgpool(x)
        x = x - self.avgnopool(x)
        x = self.shrink1x1(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        pool = pool.view(pool.size(0), -1)
        outputs = [pool, x]
        return torch.cat(outputs, 1)





class ResNetFeature(nn.Module):
    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNetFeature, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(5)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class ResNetExtract(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNetExtract, self).__init__()
        self.features = ResNetFeature(block, layers)
        self.avgpool = nn.AvgPool2d(5)
        self.fc = nn.Linear(512 * block.expansion, num_classes)



        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)


        return [x,out]

    class ResNetMod(nn.Module):
        def __init__(self, block, layers, num_classes=1000):
            self.inplanes = 64
            super(ResNetMod, self).__init__()
            self.features = ResNetFeature(block, layers)
            self.conv_fin = nn.Conv2d(512 * block.expansion, 512, kernel_size=1, bias=False)
            self.FC = nn.Sequential(
                nn.Linear(512 * 5 * 5, 4096),
                nn.BatchNorm1d(4096),
                nn.ReLU(True),
                nn.Dropout(p=0.25),
                nn.Linear(4096, 4096),
                nn.BatchNorm1d(4096),
                nn.ReLU(True),
                nn.Dropout(p=0.25),
                nn.Linear(4096, num_classes)
            )

            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    n = m.weight.size(1)
                    m.weight.data.normal_(0, 0.01)
                    m.bias.data.zero_()

        def forward(self, x):
            x = self.features(x)
            x = self.conv_fin(x)
            x = x.view(x.size(0), -1)
            out = self.FC(x)

            return out

# ToDO create resnet as feature genreator class
class ResNetMulti(nn.Module):
    def __init__(self, block, layers, num_classes=[1000]):
        self.n_multi=len(num_classes)
        self.inplanes = 64
        super(ResNetMulti, self).__init__()
        self.features = ResNetFeature(block, layers)
        # self.avgpool = nn.AvgPool2d(5)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.FC = nn.Sequential(
            nn.Linear(512 * 5 * 5, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(True),
            nn.Dropout(p=0.25),
        )
        # out=[]
        # for num_cls in num_classes:
        #     out.append(nn.Linear(4096, num_cls))
        # self.out=torch.cat(out,1)
        # self.out = nn.Linear(4096, num_classes[0])
        self.out1 = nn.Sequential(nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(True),
            nn.Dropout(p=0.25),
            nn.Linear(4096, num_classes[0]) )
        if self.n_multi > 1:
            self.out2 = nn.Sequential(nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(p=0.25),
            nn.Linear(1024, num_classes[1]) )
        if self.n_multi > 2:
            self.out3 = nn.Sequential(nn.Linear(4096, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Dropout(p=0.25),
            nn.Linear(512, num_classes[2]) )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
                # elif isinstance(m, list):
                #     for m_i in m:
                #         n = m_i.weight.size(1)
                #         m_i.weight.data.normal_(0, 0.01)
                #         m_i.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.FC(x)
        out = []
        # for k in range(len(self.out)):
        #     out.append(self.out[k](x))
        out.append(self.out1(x))
        if self.n_multi>1:
            out.append(self.out2(x))
        if self.n_multi>2:
            out.append(self.out3(x))
        return out


class ResNetMulti_v2(nn.Module):
    def __init__(self, block, layers, num_classes=[1000]):
        self.n_multi=len(num_classes)
        self.inplanes = 64
        super(ResNetMulti_v2, self).__init__()
        self.features = ResNetFeature(block, layers)
        self.conv_fin = nn.Conv2d(512 * block.expansion, 128, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(5)
        self.fc_bson = nn.Linear(512 * block.expansion, num_classes[0])

        self.FC = nn.Sequential(
            nn.Linear(128 * 5 * 5, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(True),
            nn.Dropout(p=0.25),
        )

        self.out1 = nn.Sequential(nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(True),
            nn.Dropout(p=0.25),
            nn.Linear(4096, num_classes[0]) )


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


    def forward(self, x):
        out = []
        x = self.features(x)
        z = self.avgpool(x)
        z = z.view(z.size(0), -1)
        z = self.fc_bson(z)
        out.append(z)
        y = self.conv_fin(x)
        y = self.relu(y)
        y = y.view(y.size(0), -1)
        y = self.FC(y)
        out.append(self.out1(y))

        return out


class ResNetMulti_v3(nn.Module):
    def __init__(self, block, layers, num_classes=[1000]):
        self.n_multi = len(num_classes)
        self.inplanes = 64
        super(ResNetMulti_v3, self).__init__()
        self.features = ResNetFeature(block, layers)
        # self.conv_fin = nn.Conv2d(512 * block.expansion, 128, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(5)
        # self.drop_5=nn.Dropout(p=0.25)
        # self.fc_bson = nn.Linear(512 * block.expansion, num_classes[0])
        self.fc = nn.Linear(512 * block.expansion, num_classes[0])

        self.conv_6_0 = nn.Conv2d(512 * block.expansion, 1024, kernel_size=1, bias=False)
        self.bn_6_0 = nn.BatchNorm2d(1024)
        self.conv_6_1=nn.Conv2d(1024, 1024, kernel_size=3, bias=True, padding=(0,0))
        self.bn_6_1=nn.BatchNorm2d(1024)
        self.drop_6_1=nn.Dropout2d(p=0.25)
        self.conv_6_2 = nn.Conv2d(1024, 4096, kernel_size=1, bias=False)
        self.avgpool_6 = nn.AvgPool2d(3)
        self.drop_6_2 = nn.Dropout(p=0.25)
        self.fc_bson3 = nn.Linear(4096, num_classes[1])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        #out = []
        x = self.features(x)
        z = self.avgpool(x)
        z = z.view(z.size(0), -1)
        # z = self.drop_5(z)
        # z = self.fc_bson(z)
        z=self.fc(z)
        #out.append(z)

        y = self.conv_6_0(x)
        y = self.bn_6_0(y)
        y = self.relu(y)
        y = self.conv_6_1(y)
        y = self.bn_6_1(y)
        y = self.drop_6_1(y)
        y = self.relu(y)
        y = self.conv_6_2(y)
        y = self.avgpool_6(y)
        y = self.drop_6_2(y)
        y = y.view(y.size(0), -1)
        y = self.fc_bson3(y)
        #out.append(y)

        return [z,y]

class ResNetMulti_v4(nn.Module):
    def __init__(self, block, layers, num_classes=[1000]):
        self.n_multi = len(num_classes)
        self.inplanes = 64
        super(ResNetMulti_v4, self).__init__()
        self.features = ResNetFeature(block, layers)
        # self.conv_fin = nn.Conv2d(512 * block.expansion, 128, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(5)
        # self.drop_5=nn.Dropout(p=0.25)
        # self.fc_bson = nn.Linear(512 * block.expansion, num_classes[0])
        self.fc = nn.Linear(512 * block.expansion, num_classes[0])

        self.conv_6_0 = nn.Conv2d(512 * block.expansion, 1024, kernel_size=1, bias=False)
        self.bn_6_0 = nn.BatchNorm2d(1024)
        self.conv_6_1=nn.Conv2d(1024, 1024, kernel_size=3, bias=True, padding=(0,0))
        self.bn_6_1=nn.BatchNorm2d(1024)
        self.drop_6_1=nn.Dropout2d(p=0.25)
        self.conv_6_2 = nn.Conv2d(1024, 4096, kernel_size=1, bias=False)
        self.avgpool_6 = nn.AvgPool2d(3)
        self.drop_6_2 = nn.Dropout(p=0.25)
        self.fc_bson3 = nn.Linear(4096, num_classes[1])

        self.conv_7_0 = nn.Conv2d(1024, 512, kernel_size=3, bias=True, padding=(1, 1))
        self.bn_7_0 = nn.BatchNorm2d(512)
        self.conv_7_1 = nn.Conv2d(512, 4096, kernel_size=1, bias=False)
        self.bn_7_1 = nn.BatchNorm2d(4096)
        self.avgpool_7 = nn.AvgPool2d(3)
        self.drop_7 = nn.Dropout(p=0.25)
        self.fc_bson_add = nn.Linear(4096, num_classes[1])
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        #out = []
        x = self.features(x)
        z = self.avgpool(x)
        z = z.view(z.size(0), -1)
        # z = self.drop_5(z)
        # z = self.fc_bson(z)
        z=self.fc(z)
        #out.append(z)

        y = self.conv_6_0(x)
        y = self.bn_6_0(y)
        y = self.relu(y)
        y = self.conv_6_1(y)
        y = self.bn_6_1(y)
        y = self.drop_6_1(y)
        y = self.relu(y)
        y2 = self.conv_6_2(y)
        y2 = self.avgpool_6(y2)
        y2 = self.drop_6_2(y2)
        y2 = y2.view(y2.size(0), -1)
        y2 = self.fc_bson3(y2)
        #out.append(y)

        y3 = self.conv_7_0(y)
        y3 = self.bn_7_0(y3)
        y3 = self.relu(y3)
        y3 = self.conv_7_1(y3)
        y3 = self.bn_7_1(y3)
        y3 = self.avgpool_7(y3)
        y3 = self.drop_7(y3)
        y3 = y3.view(y3.size(0), -1)
        y3 = self.fc_bson_add(y3)
        y3 = 0.5*y3 + z.detach()
        return [z,y2,y3]

class ResNetMulti_v5(nn.Module):
    def __init__(self, block, layers, num_classes=[1000]):
        self.n_multi = len(num_classes)
        self.inplanes = 64
        super(ResNetMulti_v5, self).__init__()
        self.features = ResNetFeature(block, layers)
        # self.conv_fin = nn.Conv2d(512 * block.expansion, 128, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(5)
        # self.drop_5=nn.Dropout(p=0.25)
        # self.fc_bson = nn.Linear(512 * block.expansion, num_classes[0])
        self.fc = nn.Linear(512 * block.expansion, num_classes[0])

        self.conv_6_0 = nn.Conv2d(512 * block.expansion, 1024, kernel_size=1, bias=False)
        self.bn_6_0 = nn.BatchNorm2d(1024)
        self.conv_6_1=nn.Conv2d(1024, 2048, kernel_size=3, bias=True, padding=(0,0))
        self.bn_6_1=nn.BatchNorm2d(2048)
        self.drop_6_1=nn.Dropout2d(p=0.25)
        self.conv_6_2 = nn.Conv2d(2048, 4096, kernel_size=1, bias=False)
        self.avgpool_6 = nn.AvgPool2d(3)
        self.bn_6_2 = nn.BatchNorm1d(4096)
        self.drop_6_2 = nn.Dropout(p=0.25)
        self.fc_m1 = nn.Linear(4096, num_classes[1])
        self.fc_m2 = nn.Linear(4096, num_classes[2])
        self.fc_m3 = nn.Linear(4096, num_classes[3])
        self.fc_m4 = nn.Linear(4096, num_classes[4])
        self.fc_m5 = nn.Linear(4096, num_classes[5])
        self.fc_m6 = nn.Linear(4096, num_classes[6])
        self.fc_m7 = nn.Linear(4096, num_classes[7])
        self.fc_m8 = nn.Linear(4096, num_classes[8])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        #out = []
        x = self.features(x)
        z = self.avgpool(x)
        z = z.view(z.size(0), -1)
        # z = self.drop_5(z)
        # z = self.fc_bson(z)
        z=self.fc(z)
        #out.append(z)

        y = self.conv_6_0(x)
        y = self.bn_6_0(y)
        y = self.relu(y)
        y = self.conv_6_1(y)
        y = self.bn_6_1(y)
        y = self.drop_6_1(y)
        y = self.relu(y)
        y = self.conv_6_2(y)
        y = self.avgpool_6(y)
        y = self.bn_6_2(y)
        y = self.drop_6_2(y)
        y = y.view(y.size(0), -1)
        m1 = self.fc_m1(y)
        m2 = self.fc_m2(y)
        m3 = self.fc_m3(y)
        m4 = self.fc_m4(y)
        m5 = self.fc_m5(y)
        m6 = self.fc_m6(y)
        m7 = self.fc_m7(y)
        m8 = self.fc_m8(y)


        return [z,m1,m2,m3,m4,m5,m6,m7,m8]

# class ResNetDelta(nn.Module):
#
#     def __init__(self, block, layers, num_classes=1000):
#         self.inplanes = 64
#         super(ResNetDelta, self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
#                                bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(block, 64, layers[0])
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
#         #self.avgpool = nn.AvgPool2d(5)
#         self.delta = Delta(512)
#         #self.fc = nn.Linear(512 * block.expansion, num_classes)
#         self.classifier = nn.Sequential(
#             nn.Linear(512 * 5 * 5, 4096),
#             nn.BatchNorm1d(4096),
#             nn.ReLU(True),
#             nn.Dropout(p=0.25),
#             nn.Linear(4096, 4096),
#             nn.BatchNorm1d(4096),
#             nn.ReLU(True),
#             nn.Dropout(p=0.25),
#             nn.Linear(4096, num_classes),
#         )
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.Linear):
#                 n = m.weight.size(1)
#                 m.weight.data.normal_(0, 0.01)
#                 m.bias.data.zero_()
#
#     def _make_layer(self, block, planes, blocks, stride=1):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.inplanes, planes * block.expansion,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(planes * block.expansion),
#             )
#
#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample))
#         self.inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(self.inplanes, planes))
#
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#
#         #x = self.avgpool(x)
#         x = self.delta(x)
#         #x = x.view(x.size(0), -1)
#         x = self.classifier(x)
#
#         return x

def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet_mod18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetMod(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet18_multi(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetMulti(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model





def se_resnet_mod18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetMod(SEBasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def se_resnet_mod28(pretrained=False, **kwargs):  # se_resnet_28_fc
    """Constructs a ResNet-28 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetMod(SEBasicBlock, [3, 3, 4, 3], **kwargs)
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model

def resnet34_extract(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetExtract(BasicBlock, [3, 4, 6, 3], **kwargs)

    if pretrained:
        prefix = 'features.'
        print("=> using pre-trained model resnet101")
        print('use prefix ->' + prefix)
        pretrained_state = model_zoo.load_url(model_urls['resnet101'])
        model_state = model.state_dict()

        # pretrained_state = {k: v for k, v in pretrained_state.iteritems() if
        #                     k in model_state and v.size() == model_state[k].size()}
        for k, v in pretrained_state.items():
            key = prefix + k
            if key in model_state and v.size() == model_state[key].size():
                model_state[key] = v
                print(key)
            else:
                print('not copied --------> ', key)
        # model_state.update(pretrained_state)
        model.load_state_dict(model_state)
    return model

def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model

def resnet101_fc(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetMulti_v4(Bottleneck, [3, 4, 23, 3], **kwargs)

    if pretrained:
        prefix = 'features.'
        print("=> using pre-trained model resnet101")
        print('use prefix ->'+prefix)
        pretrained_state = model_zoo.load_url(model_urls['resnet101'])
        model_state = model.state_dict()

        # pretrained_state = {k: v for k, v in pretrained_state.iteritems() if
        #                     k in model_state and v.size() == model_state[k].size()}
        for k, v in pretrained_state.items():
            key=prefix+k
            if key in model_state and v.size() == model_state[key].size():
                model_state[key]=v
                print(key)
            else:
                print('not copied --------> ',key)
        #model_state.update(pretrained_state)
        model.load_state_dict(model_state)
    return model

def resnet101_multi(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetMulti_v5(Bottleneck, [3, 4, 23, 3], **kwargs)

    if pretrained:
        prefix = 'features.'
        print("=> using pre-trained model resnet101")
        print('use prefix ->'+prefix)
        pretrained_state = model_zoo.load_url(model_urls['resnet101'])
        model_state = model.state_dict()

        # pretrained_state = {k: v for k, v in pretrained_state.iteritems() if
        #                     k in model_state and v.size() == model_state[k].size()}
        for k, v in pretrained_state.items():
            key=prefix+k
            if key in model_state and v.size() == model_state[key].size():
                model_state[key]=v
                print(key)
            else:
                print('not copied --------> ',key)
        #model_state.update(pretrained_state)
        model.load_state_dict(model_state)
    return model

def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
