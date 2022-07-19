"""Test dice loss output values
"""

from loss import DiceLoss
import torch
import pytest


pred_1 = torch.Tensor(
    [
        [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ]
    ]
)

pred_2 = torch.Tensor(
    [
        [
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 1, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
        ]
    ]
)

true_labels = torch.Tensor(
    [
        [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ]
    ]
)


@pytest.mark.parametrize("predictions, expected_loss", [(pred_1, 0), (pred_2, 0.8888)])
def test_dice_loss_value(predictions, expected_loss):
    """Test dice loss output values"""
    loss_func = DiceLoss()
    loss = loss_func.forward(predictions, true_labels)
    print(loss)
    assert loss.item() - expected_loss < 0.001
