import torch
import torch.nn as nn
import torchvision.models as models

class FCN(nn.Module):
    def __init__(self, num_classes=7):
        super(FCN, self).__init__()


        self.backbone = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)


        self.stage1 = nn.Sequential(*list(self.backbone.children())[:5])
        self.stage2 = self.backbone.layer2
        self.stage3 = self.backbone.layer3
        self.stage4 = self.backbone.layer4


        self.conv2048_256 = nn.Conv2d(2048, 256, 1)
        self.conv1024_256 = nn.Conv2d(1024, 256, 1)
        self.conv512_256 = nn.Conv2d(512, 256, 1)


        self.upsample2x = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True)
        )


        self.final_upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)


        self.out_conv = nn.Conv2d(256, num_classes, kernel_size=3, padding=1)

    def forward(self, x):

        s1 = self.stage1(x)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)


        s4 = self.conv2048_256(s4)
        s3 = self.conv1024_256(s3)
        s2 = self.conv512_256(s2)


        x = self.upsample2x(s4)
        x = x + s3
        x = self.upsample2x(x)
        x = x + s2


        x = self.final_upsample(x)


        x = self.out_conv(x)
        return x
