"""Test bottle neck layer
"""
from model._bottleneck import BottleNeck
import torch


def test_bottleneck_output_shape():
    """Test bottle neck output shape"""
    layer = BottleNeck(in_channels=10, mid_channels=20)
    dummy_batch = torch.rand((5, 10, 256, 256))
    output = layer.forward(dummy_batch)
    assert tuple(output.shape) == (5, 10, 256, 256)
