"""Dice loss"""

from torch import nn


class DiceLoss(nn.Module):
    """Dice loss class for image segmantation tasks"""

    def forward(self, predictions, target, smooth=1):
        """Forwars pass"""
        intersection = predictions * target
        dice = (2 * intersection.sum() + smooth) / (
            predictions.sum() + target.sum() + smooth
        )
        return 1 - dice
