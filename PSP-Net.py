import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.resnet import Bottleneck


class PSPNet(nn.Module):
    def __init__(self, num_classes=7):
        super(PSPNet, self).__init__()

        base = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        self.layer0 = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool
        )
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4


        for block in self.layer3.children():
            if isinstance(block, Bottleneck):
                self._modify_block(block, dilation=2, first_stride=1)


        for block in self.layer4.children():
            if isinstance(block, Bottleneck):
                self._modify_block(block, dilation=4, first_stride=1)

        self.psp = PyramidPooling(2048, pool_sizes=[6, 3, 2, 1])
        self.decoder = DecodePSPFeature(num_classes)
        self.aux = AuxiliaryPSPlayers(1024, num_classes)

    def _modify_block(self, block, dilation, first_stride):

        if block.conv1.stride != (1, 1):
            block.conv1.stride = (first_stride, first_stride)


        block.conv2.stride = (1, 1)
        block.conv2.dilation = (dilation, dilation)
        block.conv2.padding = (dilation, dilation)


        if block.downsample is not None:
            block.downsample[0].stride = (1, 1)


        for m in block.modules():
            if isinstance(m, nn.Conv2d):
                if m.kernel_size[0] == 3:
                    padding = (m.dilation[0] * (m.kernel_size[0] - 1)) // 2
                    m.padding = (padding, padding)

    def forward(self, x):
        _, _, h, w = x.shape

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x_aux = self.layer3(x)

        output_aux = self.aux(x_aux)
        output_aux = F.interpolate(output_aux, size=(h, w), mode='bilinear', align_corners=True)

        x = self.layer4(x_aux)
        x = self.psp(x)
        output = self.decoder(x)
        output = F.interpolate(output, size=(h, w), mode='bilinear', align_corners=True)

        return output, output_aux


class PyramidPooling(nn.Module):
    def __init__(self, in_channels, pool_sizes):
        super(PyramidPooling, self).__init__()
        self.pool_sizes = pool_sizes
        out_channels = in_channels // len(pool_sizes)

        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(pool_size),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for pool_size in pool_sizes
        ])

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + out_channels * len(pool_sizes), 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def forward(self, x):
        h, w = x.size()[2:]
        features = [x]

        for block in self.blocks:
            pooled = block(x)
            upsampled = F.interpolate(pooled, size=(h, w), mode='bilinear', align_corners=True)
            features.append(upsampled)

        x = torch.cat(features, dim=1)
        return self.bottleneck(x)


class DecodePSPFeature(nn.Module):
    def __init__(self, num_classes):
        super(DecodePSPFeature, self).__init__()
        self.classifier = nn.Sequential(
            nn.Conv2d(512, num_classes, kernel_size=1),
        )

    def forward(self, x):
        return self.classifier(x)


class AuxiliaryPSPlayers(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(AuxiliaryPSPlayers, self).__init__()
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def forward(self, x):
        return self.classifier(x)

