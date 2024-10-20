import torch
import torch.nn as nn

class UNetGenerator(nn.Module):
    def __init__(self):
        super(UNetGenerator, self).__init__()
        self.down1 = self.conv_block(1, 64)     # 1024 -> 512
        self.down2 = self.conv_block(64, 128)   # 512 -> 256
        self.down3 = self.conv_block(128, 256)  # 256 -> 128
        self.down4 = self.conv_block(256, 512)  # 128 -> 64
        self.down5 = self.conv_block(512, 512)  # 64 -> 32
        self.down6 = self.conv_block(512, 512)  # 32 -> 16

        self.up1 = self.up_conv_block(512, 512) # 16 -> 32
        self.up2 = self.up_conv_block(1024, 512) # 32 -> 64
        self.up3 = self.up_conv_block(1024, 256) # 64 -> 128
        self.up4 = self.up_conv_block(512, 128)  # 128 -> 256
        self.up5 = self.up_conv_block(256, 64)   # 256 -> 512
        self.final = nn.Conv2d(128, 3, kernel_size=1)  # 512 -> 1024

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def up_conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)

        u1 = self.up1(d6)
        u2 = self.up2(torch.cat([u1, d5], dim=1))
        u3 = self.up3(torch.cat([u2, d4], dim=1))
        u4 = self.up4(torch.cat([u3, d3], dim=1))
        u5 = self.up5(torch.cat([u4, d2], dim=1))
        return self.final(torch.cat([u5, d1], dim=1))

class CNNDiscriminator(nn.Module):
    def __init__(self):
        super(CNNDiscriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=4, stride=2, padding=1),  # 1024 -> 512
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 512 -> 256
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 256 -> 128
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # 128 -> 64
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),  # 64 -> 32
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0),    # 32 -> 1
        )

    def forward(self, x):
        return self.net(x)
