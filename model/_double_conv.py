"""Double convolution layer"""


from torch import nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Double convolution layer"""

    def __init__(self, in_channels, out_channels, mid_channels=None, kernel=(3, 3)):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels

        self.conv_1 = nn.Conv2d(
            in_channels, mid_channels, kernel_size=kernel, padding=1, bias=False
        )
        self.bn_1 = nn.BatchNorm2d(mid_channels)

        self.conv_2 = nn.Conv2d(
            mid_channels, out_channels, kernel_size=kernel, padding=1, bias=False
        )
        self.bn_2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        """Forward pass"""
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = F.relu(x)

        x = self.conv_2(x)
        x = self.bn_2(x)
        x = F.relu(x)

        return x
