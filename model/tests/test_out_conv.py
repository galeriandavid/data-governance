"""Tets out conv layer
"""

from model._out_conv import OutConv
import torch


def test_out_conv_output_shape():
    """Test output tensor shape of out conv layer"""
    layer = OutConv(in_channels=32, out_channels=3)
    input_shape = (5, 32, 256, 256)
    dummy_batch = torch.rand(input_shape)
    output = layer.forward(dummy_batch)
    assert tuple(output.shape) == (5, 3, 256, 256)
