import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet101
from torchvision.models.resnet import Bottleneck

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class CFF(nn.Module):
    def __init__(self, in_channel1, in_channel2, out_channel):
        super(CFF, self).__init__()
        act_fn = nn.ReLU(inplace=True)


        self.layer0 = BasicConv2d(in_channel1, out_channel // 2, 1)
        self.layer1 = BasicConv2d(in_channel2, out_channel // 2, 1)


        self.layer3_1 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel // 2),
            act_fn
        )
        self.layer5_1 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel // 2, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channel // 2),
            act_fn
        )
        self.layer3_2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel // 2),
            act_fn
        )
        self.layer5_2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel // 2, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channel // 2),
            act_fn
        )

        self.layer_out = nn.Sequential(
            nn.Conv2d(out_channel // 2, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            act_fn
        )

    def forward(self, x0, x1):

        x0_1 = self.layer0(x0)
        x1_1 = self.layer1(x1)


        x_3_1 = self.layer3_1(torch.cat([x0_1, x1_1], dim=1))
        x_5_1 = self.layer5_1(torch.cat([x1_1, x0_1], dim=1))


        x_3_2 = self.layer3_2(torch.cat([x_3_1, x_5_1], dim=1))
        x_5_2 = self.layer5_2(torch.cat([x_5_1, x_3_1], dim=1))


        out = self.layer_out(x0_1 + x1_1 + torch.mul(x_3_2, x_5_2))
        return out


class DeepLabResNet101(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(DeepLabResNet101, self).__init__()


        self.backbone = ResNet101Backbone(pretrained=pretrained)


        self.low_level_channels = 256
        self.mid_level_channels = 512
        self.high_level_channels = 2048


        self.CA = CoordAtt(self.high_level_channels, self.high_level_channels)


        self.dense_aspp = _DenseASPPBlock(
            in_channels=self.high_level_channels,
            inter_channels1=64,
            inter_channels2=64
        )
        dense_aspp_out = 2 * self.high_level_channels + 5 * 64
        self.dense_aspp_conv = nn.Sequential(
            nn.Conv2d(dense_aspp_out, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )


        self._build_cff_modules()


        self.cls_conv = nn.Conv2d(256, num_classes, 1)

    def _build_cff_modules(self):

        self.cff = CFF(
            in_channel1=self.low_level_channels,
            in_channel2=self.mid_level_channels,
            out_channel=256
        )


        self.high_up_conv = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        l1, l2, l3, l4 = self.backbone(x)
        high_feat = self.CA(l4)
        high_feat = self.dense_aspp(high_feat)
        high_feat = self.dense_aspp_conv(high_feat)  # [B,256,16,16]
        l2_up = F.interpolate(l2, scale_factor=2, mode='bilinear', align_corners=True)
        cff_out = self.cff(l1, l2_up)
        high_up = F.interpolate(high_feat, scale_factor=8, mode='bilinear', align_corners=True)
        high_up = self.high_up_conv(high_up)
        fused_feat = cff_out + high_up

        output = self.cls_conv(fused_feat)
        output = F.interpolate(output, scale_factor=4, mode='bilinear', align_corners=True)

        return output


class ResNet101Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet101Backbone, self).__init__()
        resnet = resnet101(pretrained=pretrained)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self._add_dilation()

    def _add_dilation(self):
        for layer in [self.layer3, self.layer4]:
            for i, block in enumerate(layer.children()):
                if isinstance(block, Bottleneck) and i != 0:
                    block.conv2.stride = (1, 1)
                    if layer == self.layer3:
                        block.conv2.dilation = (2, 2)
                        block.conv2.padding = (2, 2)
                    elif layer == self.layer4:
                        block.conv2.dilation = (4, 4)
                        block.conv2.padding = (4, 4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        return l1, l2, l3, l4


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        return identity * a_w * a_h


class _DenseASPPConv(nn.Sequential):
    def __init__(self, in_channels, inter_channels, out_channels, atrous_rate, drop_rate=0.1,
                 norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.add_module('conv1', nn.Conv2d(in_channels, inter_channels, 1))
        self.add_module('bn1', norm_layer(inter_channels))
        self.add_module('relu1', nn.ReLU(True))
        self.add_module('conv2', nn.Conv2d(inter_channels, out_channels, 3, padding=atrous_rate, dilation=atrous_rate))
        self.add_module('bn2', norm_layer(out_channels))
        self.add_module('relu2', nn.ReLU(True))
        self.drop_rate = drop_rate

    def forward(self, x):
        x = super().forward(x)
        if self.drop_rate > 0:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        return x


class _DenseASPPBlock(nn.Module):
    def __init__(self, in_channels, inter_channels1, inter_channels2, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.aspp_3 = _DenseASPPConv(in_channels, inter_channels1, inter_channels2, 3, 0.1, norm_layer)
        self.aspp_6 = _DenseASPPConv(in_channels + inter_channels2 * 1, inter_channels1, inter_channels2, 6, 0.1,
                                     norm_layer)
        self.aspp_12 = _DenseASPPConv(in_channels + inter_channels2 * 2, inter_channels1, inter_channels2, 12, 0.1,
                                      norm_layer)
        self.aspp_18 = _DenseASPPConv(in_channels + inter_channels2 * 3, inter_channels1, inter_channels2, 18, 0.1,
                                      norm_layer)
        self.aspp_24 = _DenseASPPConv(in_channels + inter_channels2 * 4, inter_channels1, inter_channels2, 24, 0.1,
                                      norm_layer)
        self.SP = StripPooling(in_channels)

    def forward(self, x):
        x1 = self.SP(x)
        aspp3 = self.aspp_3(x)
        x = torch.cat([aspp3, x], dim=1)
        aspp6 = self.aspp_6(x)
        x = torch.cat([aspp6, x], dim=1)
        aspp12 = self.aspp_12(x)
        x = torch.cat([aspp12, x], dim=1)
        aspp18 = self.aspp_18(x)
        x = torch.cat([aspp18, x], dim=1)
        aspp24 = self.aspp_24(x)
        x = torch.cat([aspp24, x], dim=1)
        x = torch.cat([x, x1], dim=1)
        return x


class StripPooling(nn.Module):
    def __init__(self, in_channels, up_kwargs={'mode': 'bilinear', 'align_corners': True}):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((1, None))
        self.pool_w = nn.AdaptiveAvgPool2d((None, 1))
        inter_channels = in_channels // 4
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, (1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(inter_channels)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, (3, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(inter_channels)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(inter_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels)
        )
        self._up_kwargs = up_kwargs

    def forward(self, x):
        _, _, h, w = x.size()
        x1 = self.conv1(x)
        x2 = F.interpolate(self.conv2(self.pool_h(x1)), (h, w), **self._up_kwargs)
        x3 = F.interpolate(self.conv3(self.pool_w(x1)), (h, w), **self._up_kwargs)
        x4 = self.conv4(F.relu(x2 + x3))
        out = self.conv5(x4)
        return F.relu(x + out)

