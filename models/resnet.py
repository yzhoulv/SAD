# -*- coding: utf-8 -*-
"""
Created on 18-5-21 下午5:26

@author: ronghuaiyang
"""
import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.utils.weight_norm as weight_norm
import torch.nn.functional as F
from torch.nn import Parameter
from config.config import Config as opt

# __all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
#            'resnet152']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class AttentionDrop(nn.Module):

    def __init__(self, kernel_size=3):
        super(AttentionDrop, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.p = 0.8 # 越小遮的越少

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_con = torch.cat([avg_out, max_out], dim=1)
        x_con = self.conv1(x_con)
        x_con_sq = torch.squeeze(x_con)
        x_norm = x_con_sq
        x_norm = x_norm.reshape(x_norm.shape[0], x_norm.shape[1] * x_norm.shape[2])
        vals, _ = x_norm.topk(int((1 - self.p) * x_norm.shape[1]), dim=1, sorted=False)
        threholds, _ = torch.min(vals, dim=1, keepdim=True)
        threholds = threholds.expand(threholds.shape[0], x.shape[2]*x.shape[3])
        mask = torch.where(x_norm - threholds < 0, torch.ones(x_norm.shape, device="cuda"),
                           torch.zeros(x_norm.shape, device="cuda"))
        mask = mask.reshape(mask.shape[0], x_con_sq.shape[1], x_con_sq.shape[2])
        x_iptc = self.sigmoid(x_con_sq)
        x_rand = torch.rand(size=x_iptc.shape).cuda()
        x_rand = torch.where(x_rand - 0.5 < 0, torch.ones(x_rand.shape, device="cuda"),
                             torch.zeros(x_rand.shape, device="cuda"))
        final_map = mask * x_rand * x_iptc + x_iptc * (1.0 - x_rand)
        final_map = final_map.unsqueeze(1)

        if self.training:
            return x * final_map
        else:
            return x # * self.sigmoid(x_con)


class IRBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True):
        super(IRBlock, self).__init__()
        self.bn0 = nn.BatchNorm2d(inplanes)
        self.conv1 = conv3x3(inplanes, inplanes)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.prelu = nn.PReLU()
        self.conv2 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.use_se = use_se
        if self.use_se:
            self.se = SEBlock(planes)

    def forward(self, x):
        residual = x
        out = self.bn0(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.use_se:
            out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.prelu(out)

        return out


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.PReLU(),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y_avg = self.avg_pool(x).view(b, c)
        y_max = self.max_pool(x).view(b, c)
        y = y_max + y_avg
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ResNetFace(nn.Module):
    def __init__(self, block, layers, use_se=False):
        self.inplanes = 64
        self.use_se = use_se
        super(ResNetFace, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.prelu = nn.PReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.drop = AttentionDrop()
        self.bn2 = nn.BatchNorm2d(512)
        self.fc1 = nn.Linear(512 * 8 * 8, 512)
        self.bn3 = nn.BatchNorm1d(512)
        # self.fc = nn.Linear(512, opt.num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_se=self.use_se))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_se=self.use_se))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # x = self.pool(x)
        x = self.bn2(x)
        x = self.drop(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        feat = self.bn3(x)

        return feat


def resnet_face18(use_se=False, pretrained=False, **kwargs):
    model = ResNetFace(IRBlock, [2, 3, 2, 2], use_se=use_se, **kwargs)
    if pretrained:
        # model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load("./output/res20_data_new.pth"))
        # model.fc = nn.Linear(512, opt.num_classes)
    return model
