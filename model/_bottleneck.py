"""
Bottleneck
"""

from torch import nn


class BottleNeck(nn.Module):
    """Bottleneck layer"""

    def __init__(self, in_channels, mid_channels, kernel=1):
        super().__init__()
        out_channels = in_channels
        self.conv_1 = nn.Conv2d(
            in_channels=in_channels, out_channels=mid_channels, kernel_size=kernel
        )
        self.conv_2 = nn.Conv2d(
            in_channels=mid_channels, out_channels=out_channels, kernel_size=kernel
        )

    def forward(self, x):
        """Forward pass"""
        x = self.conv_1(x)
        x = self.conv_2(x)
        return x
