"""UNet model"""

from torch import nn
from ._down import Down
from ._bottleneck import BottleNeck
from ._up import Up
from ._out_conv import OutConv


class UNet(nn.Module):
    """UNet model"""

    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.down_1 = Down(n_channels, 64)
        self.down_2 = Down(64, 128)
        self.down_3 = Down(128, 256)
        self.down_4 = Down(256, 512)

        self.bottle_neck = BottleNeck(512, 1024)

        self.up_1 = Up(1024, 512, 256)
        self.up_2 = Up(512, 256, 128)
        self.up_3 = Up(256, 128, 64)
        self.up_4 = Up(128, 64, 64)
        self.out_conv = OutConv(64, n_classes)

    def forward(self, x):
        """Forward pass"""
        x, x_1 = self.down_1(x)
        x, x_2 = self.down_2(x)
        x, x_3 = self.down_3(x)
        x, x_4 = self.down_4(x)

        x = self.bottle_neck(x)

        x = self.up_1(x, x_4)
        x = self.up_2(x, x_3)
        x = self.up_3(x, x_2)
        x = self.up_4(x, x_1)

        return self.out_conv(x)
