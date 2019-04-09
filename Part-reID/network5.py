import copy
import torch
import torch.nn as nn
from opt import opt
from torchvision.models.resnet import resnet50, resnet101, Bottleneck

class CALayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class SpatialAttn(nn.Module):
    """Spatial Attention Layer"""
    def __init__(self):
        super(SpatialAttn, self).__init__()

    def forward(self, x):
        # global cross-channel averaging # e.g. 32,2048,24,8
        x = x.mean(1, keepdim=True)  # e.g. 32,1,24,8
        h = x.size(2)
        w = x.size(3)
        x = x.view(x.size(0),-1)     # e.g. 32,192
        z = x
        for b in range(x.size(0)):
            z[b] /= torch.sum(z[b])
        z = z.view(x.size(0),1,h,w)
        return z

class PALayer(nn.Module):
    """Spatial Attention Layer"""
    def __init__(self,h,w):
        super(PALayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(h * w, h * w // 4, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(h * w // 4, h * w, 1, padding=0, bias=True),
            nn.Sigmoid())
        self.h=h
        self.w=w

    def forward(self, x):
        # global cross-channel averaging # e.g. 32,2048,24,8
        y = x.mean(1, keepdim=True)  # e.g. 32,1,24,8
        y = y.view(y.size(0),-1)     # e.g. 32,192
        y=y.unsqueeze(dim=2).unsqueeze(dim=3)
        y=self.conv(y)
        y=y.view(y.size(0),1,self.h,self.w)
        return x*y

class MGN(nn.Module):
    def __init__(self):
        super(MGN, self).__init__()
        num_classes = 1500
        feats = 256
        resnet = resnet50(pretrained=True)

        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3[0]

        self.avgpool1 = nn.AvgPool2d(kernel_size=(96, 32))
        self.avgpool2 = nn.AvgPool2d(kernel_size=(48, 16))
        self.avgpool3 = nn.AvgPool2d(kernel_size=(24, 8))

        self.layer1_fc = nn.Sequential(
            nn.Linear(256, num_classes),
            nn.BatchNorm1d(num_classes),
            nn.ReLU(inplace=True),
        )

        self.layer2_fc = nn.Sequential(
            nn.Linear(512, num_classes),
            nn.BatchNorm1d(num_classes),
            nn.ReLU(inplace=True),
        )

        self.layer3_fc = nn.Sequential(
            nn.Linear(1024, num_classes),
            nn.BatchNorm1d(num_classes),
            nn.ReLU(inplace=True),
        )



        res_conv4 = nn.Sequential(*resnet.layer3[1:])

        res_g_conv5 = resnet.layer4

        res_p_conv5 = nn.Sequential(
            Bottleneck(1024, 512, downsample=nn.Sequential(nn.Conv2d(1024, 2048, 1, bias=False), nn.BatchNorm2d(2048))),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512))
        res_p_conv5.load_state_dict(resnet.layer4.state_dict())

        self.p1 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_g_conv5))
        self.p2 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))
        self.p3 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))

        self.maxpool_zg_p1 = nn.MaxPool2d(kernel_size=(12, 4))
        self.maxpool_zg_p2 = nn.MaxPool2d(kernel_size=(24, 8))
        self.maxpool_zg_p3 = nn.MaxPool2d(kernel_size=(24, 8))
        self.maxpool_zp2 = nn.MaxPool2d(kernel_size=(12, 8))
        self.maxpool_zp3 = nn.MaxPool2d(kernel_size=(8, 8))

        self.reduction = nn.Sequential(nn.Conv2d(2048, feats, 1, bias=False), nn.BatchNorm2d(feats), nn.ReLU())

        self._init_reduction(self.reduction)

        self.fc_id_2048_0 = nn.Linear(feats, num_classes)
        self.fc_id_2048_1 = nn.Linear(feats, num_classes)
        self.fc_id_2048_2 = nn.Linear(feats, num_classes)

        self.fc_id_256_1_0 = nn.Linear(feats, num_classes)
        self.fc_id_256_1_1 = nn.Linear(feats, num_classes)
        self.fc_id_256_2_0 = nn.Linear(feats, num_classes)
        self.fc_id_256_2_1 = nn.Linear(feats, num_classes)
        self.fc_id_256_2_2 = nn.Linear(feats, num_classes)

        self._init_fc(self.fc_id_2048_0)
        self._init_fc(self.fc_id_2048_1)
        self._init_fc(self.fc_id_2048_2)

        self._init_fc(self.fc_id_256_1_0)
        self._init_fc(self.fc_id_256_1_1)
        self._init_fc(self.fc_id_256_2_0)
        self._init_fc(self.fc_id_256_2_1)
        self._init_fc(self.fc_id_256_2_2)

        #self.fusion_conv = nn.Conv1d(3, 1, kernel_size=1, bias=False)
        self.attention = CALayer(2048)
        self.attention_pixel1 = PALayer(96, 32)
        self.attention_pixel2 = PALayer(48, 16)
        self.attention_pixel3 = PALayer(24, 8)
        # self.attention_pixel1 = CALayer(256)
        # self.attention_pixel2 = CALayer(512)
        # self.attention_pixel3 = CALayer(1024)
        # self.attention_c1 = PALayer(12,4)
        # self.attention_c2 = PALayer(24, 8)


    @staticmethod
    def _init_reduction(reduction):
        # conv
        nn.init.kaiming_normal_(reduction[0].weight, mode='fan_in')
        # nn.init.constant_(reduction[0].bias, 0.)

        # bn
        nn.init.normal_(reduction[1].weight, mean=1., std=0.02)
        nn.init.constant_(reduction[1].bias, 0.)

    @staticmethod
    def _init_fc(fc):
        nn.init.kaiming_normal_(fc.weight, mode='fan_out')
        # nn.init.normal_(fc.weight, std=0.001)
        nn.init.constant_(fc.bias, 0.)

    def forward(self, x):
        x = self.backbone(x)
        x = self.layer1(x)
        #print("layer1=", x.size())
        x1=self.attention_pixel1(x)
        #print("pA-1", x1.size())
        x1 = self.avgpool1(x1)
        x1 = x1.view(x1.size(0), -1)
        x1 = self.layer1_fc(x1)

        x = self.layer2(x)
        #print("layer2=", x.size())
        x2 = self.attention_pixel2(x)
        #print("pA-2", x2.size())
        x2 = self.avgpool2(x2)
        x2 = x2.view(x2.size(0), -1)
        x2 = self.layer2_fc(x2)

        x = self.layer3(x)
        #print("layer3=", x.size())
        x3 = self.attention_pixel3(x)
        #return x3
        #print("pA-3", x3.size())
        x3 = self.avgpool3(x3)
        x3 = x3.view(x3.size(0), -1)
        x3 = self.layer3_fc(x3)

        p1 = self.p1(x)
        p2 = self.p2(x)
        p3 = self.p3(x)
        # print("p1=", p1.size())
        # print("p2=", p2.size())
        # print("p3=", p3.size())

        #return p3

        ##channel attention
        p1 = self.attention(p1)
        p2 = self.attention(p2)
        p3 = self.attention(p3)
        #print("p1-c=", p1.size())
        #print("p2-c=", p2.size())
        #print("p3-c=", p3.size())
        #return p3


        zg_p1 = self.maxpool_zg_p1(p1)
        zg_p2 = self.maxpool_zg_p2(p2)
        zg_p3 = self.maxpool_zg_p3(p3)

        zp2 = self.maxpool_zp2(p2)
        z0_p2 = zp2[:, :, 0:1, :]
        z1_p2 = zp2[:, :, 1:2, :]

        zp3 = self.maxpool_zp3(p3)
        z0_p3 = zp3[:, :, 0:1, :]
        z1_p3 = zp3[:, :, 1:2, :]
        z2_p3 = zp3[:, :, 2:3, :]

        fg_p1 = self.reduction(zg_p1).squeeze(dim=3).squeeze(dim=2)
        fg_p2 = self.reduction(zg_p2).squeeze(dim=3).squeeze(dim=2)
        fg_p3 = self.reduction(zg_p3).squeeze(dim=3).squeeze(dim=2)
        f0_p2 = self.reduction(z0_p2).squeeze(dim=3).squeeze(dim=2)
        f1_p2 = self.reduction(z1_p2).squeeze(dim=3).squeeze(dim=2)
        f0_p3 = self.reduction(z0_p3).squeeze(dim=3).squeeze(dim=2)
        f1_p3 = self.reduction(z1_p3).squeeze(dim=3).squeeze(dim=2)
        f2_p3 = self.reduction(z2_p3).squeeze(dim=3).squeeze(dim=2)

        l_p1 = self.fc_id_2048_0(fg_p1)
        l_p2 = self.fc_id_2048_1(fg_p2)
        l_p3 = self.fc_id_2048_2(fg_p3)

        l0_p2 = self.fc_id_256_1_0(f0_p2)
        l1_p2 = self.fc_id_256_1_1(f1_p2)
        l0_p3 = self.fc_id_256_2_0(f0_p3)
        l1_p3 = self.fc_id_256_2_1(f1_p3)
        l2_p3 = self.fc_id_256_2_2(f2_p3)

        predict = torch.cat([fg_p1, fg_p2, fg_p3, f0_p2, f1_p2, f0_p3, f1_p3, f2_p3], dim=1)

        return predict, fg_p1, fg_p2, fg_p3, x1, x2, x3, l_p1, l_p2, l_p3, l0_p2, l1_p2, l0_p3, l1_p3, l2_p3
        #return l_p1


