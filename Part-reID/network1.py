import os
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo

savepath='/home/wangminjie/Desktop/wmj/projects/Part-IBN/features'
if not os.path.exists(savepath):
    os.mkdir(savepath)

def draw_features(width,height,x,savename):
    tic=time.time()
    fig = plt.figure(figsize=(16, 16)) #图像分辨率
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
    for i in range(width * height):
        plt.subplot(height, width, i + 1)
        plt.axis('off')
        img = x[0, i, :, :]
        pmin = np.min(img)
        pmax = np.max(img)
        img = (img - pmin) / (pmax - pmin + 0.000001)
        plt.imshow(img, cmap='gray')
        print("{}/{}".format(i, width * height))
    fig.savefig(savename, dpi=100)
    fig.clf()
    plt.close()
    print("time:{}".format(time.time() - tic))


model_urls = {
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


class IBN(nn.Module):
    def __init__(self, planes):
        super(IBN, self).__init__()
        half1 = int(planes / 2)
        self.half = half1
        half2 = planes - half1
        self.IN = nn.InstanceNorm2d(half1, affine=True)
        self.BN = nn.BatchNorm2d(half2)

    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, ibn=False, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        if ibn:
            self.bn1 = IBN(planes)
        else:
            self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
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


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        scale = 64
        self.inplanes = scale
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, scale, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(scale)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, scale, layers[0])
        self.layer2 = self._make_layer(block, scale * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, scale * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, scale * 8, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=(12,4))
        self.fc = nn.Linear(scale * 8 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.InstanceNorm2d):
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
        ibn = True
        if planes == 512:
            ibn = False
        layers.append(block(self.inplanes, planes, ibn, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, ibn))

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

def init_pretrained_weights(model, model_url):

    pretrain_dict = model_zoo.load_url(model_url)
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)
    print("Initialized model with pretrained weights from {}".format(model_url))

def resnet50(pretrained=False,num_classes=1000):
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes)
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])
    return model


class Part_IBN_a(nn.Module):

    def __init__(self):
        super(Part_IBN_a, self).__init__()
        self.num=1
        self.num_classes=751
        self.feats=2048
        self.IBN_a=resnet50(pretrained=True,num_classes=self.num_classes)
        self.htri=nn.Sequential(
            nn.Linear(2048,2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5)
        )
        self.avgpool_g=nn.AvgPool2d(kernel_size=(12,4))
        self.avgpool_p=nn.AvgPool2d(kernel_size=(2,4))
        self.classifier=nn.Sequential(nn.Linear(self.feats, self.num_classes),nn.BatchNorm1d(self.num_classes),nn.ReLU())

    def forward(self, x):
        x=self.IBN_a.conv1(x)
        #draw_features(8, 8, x.cpu().detach().numpy(), "{}/Part_IBN_conv1-pic{}.png".format(savepath,self.num))

        x=self.IBN_a.bn1(x)
        x=self.IBN_a.relu(x)
        x=self.IBN_a.maxpool(x)

        x = self.IBN_a.layer1(x)
        x = self.IBN_a.layer2(x)
        x = self.IBN_a.layer3(x)
        x = self.IBN_a.layer4(x)

        g=self.avgpool_g(x)
        g=g.view(g.size(0),-1)
        g_end=self.classifier(g)

        p1 = x[:, :, 0:2, :]
        p2 = x[:, :, 2:4, :]
        p3 = x[:, :, 4:6, :]
        p4 = x[:, :, 6:8, :]
        p5 = x[:, :, 8:10, :]
        p6 = x[:, :, 10:12, :]

        p1 = self.avgpool_p(p1)
        p2 = self.avgpool_p(p2)
        p3 = self.avgpool_p(p3)
        p4 = self.avgpool_p(p4)
        p5 = self.avgpool_p(p5)
        p6 = self.avgpool_p(p6)

        p1 = p1.view(p1.size(0), -1)
        p2 = p2.view(p2.size(0), -1)
        p3 = p3.view(p3.size(0), -1)
        p4 = p4.view(p4.size(0), -1)
        p5 = p5.view(p5.size(0), -1)
        p6 = p6.view(p6.size(0), -1)

        p1 = self.classifier(p1)
        p2 = self.classifier(p2)
        p3 = self.classifier(p3)
        p4 = self.classifier(p4)
        p5 = self.classifier(p5)
        p6 = self.classifier(p6)

        p=torch.cat([p1,p2,p3,p4,p5,p6],dim=1)


        #x = self.IBN_a.avgpool(x)
        #plt.plot(np.linspace(1, 2048, 2048), x.cpu().detach().numpy()[0, :, 0, 0])
        #plt.savefig("{}/Part_IBN_avgpool-pic{}.png".format(savepath,self.num))
        #plt.clf()
        #plt.close()
        #x = x.view(x.size(0),-1)
        #plt.plot(np.linspace(1, 2048, 2048), x.cpu().detach().numpy()[0, :])
        #plt.savefig("{}/Part_IBN_fc-pic{}.png".format(savepath,self.num))
        #plt.clf()
        #plt.close()
        #y=self.htri(x)         #三元损失
        #v = self.IBN_a.fc(x)   #交叉熵
        #self.num=self.num+1

        return g_end, p, p1, p2, p3, p4, p5, p6




