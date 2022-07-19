"""Test double conv layer
"""

from model._double_conv import DoubleConv
import torch


def test_double_conv_output_shape():
    """Test double conv output shape"""
    conv = DoubleConv(in_channels=10, out_channels=20)
    dummy_batch = torch.rand((5, 10, 256, 256))
    output = conv.forward(dummy_batch)
    assert tuple(output.shape) == (5, 20, 256, 256)
