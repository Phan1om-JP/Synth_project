import torch
import torch.nn as nn


class ConvBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet2D(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base_ch=64):
        super().__init__()
        b = base_ch
        self.enc1     = ConvBlock2D(in_ch, b)
        self.enc2     = ConvBlock2D(b,     b*2)
        self.enc3     = ConvBlock2D(b*2,   b*4)
        self.enc4     = ConvBlock2D(b*4,   b*8)
        self.bottleneck = ConvBlock2D(b*8, b*16)
        self.up4      = nn.ConvTranspose2d(b*16, b*8, 2, stride=2)
        self.dec4     = ConvBlock2D(b*16, b*8)
        self.up3      = nn.ConvTranspose2d(b*8,  b*4, 2, stride=2)
        self.dec3     = ConvBlock2D(b*8,  b*4)
        self.up2      = nn.ConvTranspose2d(b*4,  b*2, 2, stride=2)
        self.dec2     = ConvBlock2D(b*4,  b*2)
        self.up1      = nn.ConvTranspose2d(b*2,  b,   2, stride=2)
        self.dec1     = ConvBlock2D(b*2,  b)
        self.pool     = nn.MaxPool2d(2)
        self.head     = nn.Conv2d(b, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


class ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet3D(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base_ch=16):
        super().__init__()
        b = base_ch
        self.enc1     = ConvBlock3D(in_ch, b)
        self.enc2     = ConvBlock3D(b,     b*2)
        self.enc3     = ConvBlock3D(b*2,   b*4)
        self.bottleneck = ConvBlock3D(b*4, b*8)
        self.up3      = nn.ConvTranspose3d(b*8, b*4, 2, stride=2)
        self.dec3     = ConvBlock3D(b*8,  b*4)
        self.up2      = nn.ConvTranspose3d(b*4, b*2, 2, stride=2)
        self.dec2     = ConvBlock3D(b*4,  b*2)
        self.up1      = nn.ConvTranspose3d(b*2, b,   2, stride=2)
        self.dec1     = ConvBlock3D(b*2,  b)
        self.pool     = nn.MaxPool3d(2)
        self.head     = nn.Conv3d(b, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b  = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b),  e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


