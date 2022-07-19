"""Test down layer
"""

from model._down import Down
import torch


def test_down_output_shape():
    """Test output shape of down layer"""
    layer = Down(in_channels=10, out_channels=20)
    input_shape = (5, 10, 256, 256)
    dummy_batch = torch.rand(input_shape)
    output_1, output_2 = layer.forward(dummy_batch)
    assert tuple(output_1.shape) == (5, 20, 128, 128)
    assert tuple(output_2.shape) == (5, 20, 256, 256)
