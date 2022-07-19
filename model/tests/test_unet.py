"""Test UNet"""

from model.unet import UNet
import torch


def test_unet_output_shape():
    """Test output tensor shape of UNet model"""
    unet = UNet(n_channels=3, n_classes=4)
    dummy_batch = torch.rand((5, 3, 256, 256))
    output = unet.forward(dummy_batch)
    assert tuple(output.shape) == (5, 4, 256, 256)
