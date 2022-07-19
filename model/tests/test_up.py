"""Test up layer"""

from model._up import Up
import torch


def test_up_output_shape():
    """Tets output tensor shape of up layer"""
    layer = Up(in_channels=16, mid_channels=8, out_channels=4)
    input_shape_1 = (5, 8, 16, 16)
    input_shape_2 = (5, 8, 32, 32)
    dummy_batch_1 = torch.rand(input_shape_1)
    dummy_batch_2 = torch.rand(input_shape_2)
    output = layer.forward(dummy_batch_1, dummy_batch_2)
    assert tuple(output.shape) == (5, 4, 32, 32)
