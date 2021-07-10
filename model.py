import torch
import torch.nn as nn
import torch.nn.functional as F

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()

        self.backbone_layer1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, padding=1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                    nn.ReLU(inplace=True))

        self.backbone_layer2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(128, 128, kernel_size=3, padding=1),
                                    nn.ReLU(inplace=True))

        self.backbone_layer3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, padding=1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(256, 256, kernel_size=3, padding=1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(256, 256, kernel_size=3, padding=1),
                                    nn.ReLU(inplace=True))

        self.backbone_layer4 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, padding=1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(512, 512, kernel_size=3, padding=1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(512, 512, kernel_size=3, padding=1),
                                    nn.ReLU(inplace=True))

        self.backbone_layer5 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(512, 512, kernel_size=3, padding=1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(512, 512, kernel_size=3, padding=1),
                                    nn.ReLU(inplace=True))
        
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        
        self.d2conv_ReLU = nn.Sequential(nn.Conv2d(512, 128, kernel_size=3, padding=2, dilation=2), nn.ReLU(inplace=True))
        self.d4conv_ReLU = nn.Sequential(nn.Conv2d(512, 128, kernel_size=3, padding=4, dilation=4), nn.ReLU(inplace=True))
        self.d8conv_ReLU = nn.Sequential(nn.Conv2d(512, 128, kernel_size=3, padding=8, dilation=8), nn.ReLU(inplace=True))
        self.d16conv_ReLU = nn.Sequential(nn.Conv2d(512, 128, kernel_size=3, padding=16, dilation=16), nn.ReLU(inplace=True))
        
        self.conv1 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=1), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=1), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=1), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1), nn.ReLU(inplace=True))

        self.predict_layer = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=1),
                                           nn.ReLU(inplace=True),
                                           nn.Conv2d(512, 512, kernel_size=1),
                                           nn.ReLU(inplace=True),
                                           nn.Conv2d(512, 2, kernel_size=1))
        
    def forward(self, x):
        
        input_size = x.size()[2:] #[1, 3, 375, 500]
        
        ###* Encoder, VGG
        stage1 = self.backbone_layer1(x) #[1, 64, 202, 500]
        stage1_maxpool = self.maxpool(stage1) #[1, 64, 101, 250]

        stage2 = self.backbone_layer2(stage1_maxpool) #[1, 128, 101, 250]
        stage2_maxpool = self.maxpool(stage2) #[1, 128, 51, 125]

        stage3 = self.backbone_layer3(stage2_maxpool) #[1, 256, 51, 125]
        stage3_maxpool = self.maxpool(stage3) #[1, 256, 26, 63]
        tmp_size = stage3.size()[2:]


        stage4 = self.backbone_layer4(stage3_maxpool) #[1, 512, 26, 63]
        stage4_maxpool = self.maxpool(stage4) #[1, 512, 13, 32]

        stage5 = self.backbone_layer5(stage4_maxpool) #[1, 512, 13, 32]
        
        
        ###* ASPP Module
        d2conv_ReLU = self.d2conv_ReLU(stage5)
        d4conv_ReLU = self.d4conv_ReLU(stage5)
        d8conv_ReLU = self.d8conv_ReLU(stage5)
        d16conv_ReLU = self.d16conv_ReLU(stage5)
        
        dilated_conv_concat = torch.cat((d2conv_ReLU, d4conv_ReLU, d8conv_ReLU, d16conv_ReLU), 1)
        
        ###* A layer of convolution on different encoder layers
        sconv1 = self.conv1(dilated_conv_concat)
        sconv1 = F.interpolate(sconv1, size=tmp_size, mode='bilinear', align_corners=True)

        sconv2 = self.conv2(stage5)
        sconv2 = F.interpolate(sconv2, size=tmp_size, mode='bilinear', align_corners=True)

        sconv3 = self.conv3(stage4)
        sconv3 = F.interpolate(sconv3, size=tmp_size, mode='bilinear', align_corners=True)

        sconv4 = self.conv4(stage3)
        sconv4 = F.interpolate(sconv4, size=tmp_size, mode='bilinear', align_corners=True)

        sconcat = torch.cat((sconv1, sconv2, sconv3, sconv4), 1)
        
        ###* 2-dim vector prediction
        pred_flux = self.predict_layer(sconcat)
        pred_flux = F.interpolate(pred_flux, size=input_size, mode='bilinear', align_corners=True)

        return pred_flux



'''***********************************'''


from helpers import maybe_download
from layer_factory import conv1x1, conv3x3, CRPBlock

data_info = {7: "Person", 21: "VOC", 40: "NYU", 60: "Context"}

models_urls = {
    "50_person": "https://cloudstor.aarnet.edu.au/plus/s/mLA7NxVSPjNL7Oo/download",
    "101_person": "https://cloudstor.aarnet.edu.au/plus/s/f1tGGpwdCnYS3xu/download",
    "152_person": "https://cloudstor.aarnet.edu.au/plus/s/Ql64rWqiTvWGAA0/download",
    "50_voc": "https://cloudstor.aarnet.edu.au/plus/s/xp7GcVKC0GbxhTv/download",
    "101_voc": "https://cloudstor.aarnet.edu.au/plus/s/CPRKWiaCIDRdOwF/download",
    "152_voc": "https://cloudstor.aarnet.edu.au/plus/s/2w8bFOd45JtPqbD/download",
    "50_nyu": "https://cloudstor.aarnet.edu.au/plus/s/gE8dnQmHr9svpfu/download",
    "101_nyu": "https://cloudstor.aarnet.edu.au/plus/s/VnsaSUHNZkuIqeB/download",
    "152_nyu": "https://cloudstor.aarnet.edu.au/plus/s/EkPQzB2KtrrDnKf/download",
    "101_context": "https://cloudstor.aarnet.edu.au/plus/s/hqmplxWOBbOYYjN/download",
    "152_context": "https://cloudstor.aarnet.edu.au/plus/s/O84NszlYlsu00fW/download",
    "50_imagenet": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "101_imagenet": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "152_imagenet": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
}

stages_suffixes = {0: "_conv", 1: "_conv_relu_varout_dimred"}


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
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
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

class ResNetLW(nn.Module):
    def __init__(self, block, layers, num_classes=21):
        self.inplanes = 64
        super(ResNetLW, self).__init__()
        self.do = nn.Dropout(p=0.5)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
    
        self.d2conv_ReLU = nn.Sequential(nn.Conv2d(1024, 128, kernel_size=3, padding=2, dilation=2), nn.ReLU(inplace=True))
        self.d4conv_ReLU = nn.Sequential(nn.Conv2d(1024, 128, kernel_size=3, padding=4, dilation=4), nn.ReLU(inplace=True))
        self.d8conv_ReLU = nn.Sequential(nn.Conv2d(1024, 128, kernel_size=3, padding=8, dilation=8), nn.ReLU(inplace=True))
        self.d16conv_ReLU = nn.Sequential(nn.Conv2d(1024, 128, kernel_size=3, padding=16, dilation=16), nn.ReLU(inplace=True))
      
        self.conv1a = nn.Sequential(nn.Conv2d(512, 256, kernel_size=1), nn.ReLU(inplace=True))
        self.conv2a = nn.Sequential(nn.Conv2d(1024, 256, kernel_size=1), nn.ReLU(inplace=True))
        self.conv3a = nn.Sequential(nn.Conv2d(512, 256, kernel_size=1), nn.ReLU(inplace=True))
        self.conv4a = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1), nn.ReLU(inplace=True))

        self.predict_layer = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=1),
                                           nn.ReLU(inplace=True),
                                           nn.Conv2d(512, 512, kernel_size=1),
                                           nn.ReLU(inplace=True),
                                           nn.Conv2d(512, 2, kernel_size=1))

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):

        input_size = x.size()[2:] #[1, 3, 333, 500]
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) #[1, 64, 94, 125]

        l1 = self.layer1(x) #[1, 256, 84, 125]
        l2 = self.layer2(l1) #[1, 512, 47, 63]
        tmp_size = l2.size()[2:]
        l3 = self.layer3(l2) #[1, 1024, 24, 32]
        # l4 = self.layer4(l3) #[1, 2048, 12, 16]
                
        ###* ASPP Module
        d2conv_ReLU = self.d2conv_ReLU(l3) #[1, 128, 24, 32]        
        d4conv_ReLU = self.d4conv_ReLU(l3) #[1, 128, 24, 32]
        d8conv_ReLU = self.d8conv_ReLU(l3)#[1, 128, 24, 32]
        d16conv_ReLU = self.d16conv_ReLU(l3)#[1, 128, 24, 32]

        dilated_conv_concat = torch.cat((d2conv_ReLU, d4conv_ReLU, d8conv_ReLU, d16conv_ReLU), 1) #[1, 512, 21, 32]

        ###* A layer of convolution on different encoder layers
        sconv1 = self.conv1a(dilated_conv_concat)
        sconv1 = F.interpolate(sconv1, size=tmp_size, mode='bilinear', align_corners=True)

        sconv2 = self.conv2a(l3)
        sconv2 = F.interpolate(sconv2, size=tmp_size, mode='bilinear', align_corners=True)

        sconv3 = self.conv3a(l2)
        sconv3 = F.interpolate(sconv3, size=tmp_size, mode='bilinear', align_corners=True)

        sconv4 = self.conv4a(l1)
        sconv4 = F.interpolate(sconv4, size=tmp_size, mode='bilinear', align_corners=True)

        sconcat = torch.cat((sconv1, sconv2, sconv3, sconv4), 1)
        
        ###* 2-dim vector prediction
        pred_flux = self.predict_layer(sconcat)
        pred_flux = F.interpolate(pred_flux, size=input_size, mode='bilinear', align_corners=True) #[1, 2, 375, 500]

        return pred_flux


