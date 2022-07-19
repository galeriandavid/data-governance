"""Unet Up layer"""

import torch
from torch import nn
import torch.nn.functional as F
from ._double_conv import DoubleConv


class Up(nn.Module):
    """Unet Up layer"""

    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.double_conv = DoubleConv(
            in_channels=in_channels,
            mid_channels=mid_channels,
            out_channels=out_channels,
        )

    def forward(self, x_1, x_2):
        """Forward pass"""
        x_1 = self.upsample(x_1)

        diff_h = x_2.size()[2] - x_1.size()[2]
        diff_w = x_2.size()[3] - x_1.size()[3]

        x_1 = F.pad(
            x_1, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2]
        )

        x = torch.cat([x_2, x_1], dim=1)
        x = self.double_conv(x)
        return x
