"""Last convolutional layer with sigmoid"""

import torch
from torch import nn


class OutConv(nn.Module):
    """Last convolutional layer with sigmoid"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """Forward pass"""
        x = self.conv(x)
        return torch.sigmoid(x)
