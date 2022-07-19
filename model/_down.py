"""UNet down layer"""

from torch import nn
from ._double_conv import DoubleConv


class Down(nn.Module):
    """UNet down layer"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.double_conv = DoubleConv(
            in_channels=in_channels, out_channels=out_channels
        )
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        """Forward pass"""
        x_before_pool = self.double_conv(x)
        x_after_pool = self.pool(x_before_pool)
        return x_after_pool, x_before_pool
