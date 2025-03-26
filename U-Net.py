import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()


        self.enc1 = self.conv_block(3, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)


        self.bottom = self.conv_block(512, 1024)


        self.up4 = self.upconv(1024, 512)
        self.dec4 = self.conv_block(1024, 512)

        self.up3 = self.upconv(512, 256)
        self.dec3 = self.conv_block(512, 256)

        self.up2 = self.upconv(256, 128)
        self.dec2 = self.conv_block(256, 128)

        self.up1 = self.upconv(128, 64)
        self.dec1 = self.conv_block(128, 64)


        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def upconv(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, kernel_size=2))
        enc3 = self.enc3(F.max_pool2d(enc2, kernel_size=2))
        enc4 = self.enc4(F.max_pool2d(enc3, kernel_size=2))

        # Bottom
        bottom = self.bottom(F.max_pool2d(enc4, kernel_size=2))

        # Decoder
        up4 = self.up4(bottom)
        dec4 = self.dec4(torch.cat([up4, enc4], dim=1))

        up3 = self.up3(dec4)
        dec3 = self.dec3(torch.cat([up3, enc3], dim=1))

        up2 = self.up2(dec3)
        dec2 = self.dec2(torch.cat([up2, enc2], dim=1))

        up1 = self.up1(dec2)
        dec1 = self.dec1(torch.cat([up1, enc1], dim=1))

        # Output
        out = self.out_conv(dec1)
        return out
