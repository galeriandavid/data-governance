"""Test util module"""

import numpy as np
import torch
import pytest
from utils import rle2mask, mask2rle
from utils import split_image, concatenate_images

# Test rle encoding/decoding
mask_1 = torch.tensor(
    [
        [1, 1, 1, 0, 0],
        [1, 1, 0, 0, 1],
        [1, 0, 0, 1, 1],
        [0, 0, 1, 1, 1],
    ],
    dtype=torch.int8,
)

rle_1 = "0 3 5 2 9 2 13 2 17 3"

mask_2 = torch.tensor(
    [[0, 0, 0, 0], [0, 0, 0, 1], [1, 1, 1, 1], [1, 0, 0, 0], [0, 0, 0, 0]],
    dtype=torch.int8,
)

rle_2 = "7 6"


@pytest.mark.parametrize("rle, expected_mask", [(rle_1, mask_1), (rle_2, mask_2)])
def test_rle2mask(rle, expected_mask):
    """Test that rle2mask generate correct mask"""
    shape = expected_mask.shape
    mask = rle2mask(rle, shape)
    assert torch.equal(mask, expected_mask)


@pytest.mark.parametrize("mask, expected_rle", [(mask_1, rle_1), (mask_2, rle_2)])
def test_mask2rle(mask, expected_rle):
    """Test that mask2rle generate correct rle"""
    rle = mask2rle(mask)
    assert rle == expected_rle


# Test spliting image into tiles and concatenate tiles back to th epicture

image = torch.Tensor(
    [
        [[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 3, 3], [2, 2, 3, 3]],
        [[4, 4, 5, 5], [4, 4, 5, 5], [6, 6, 7, 7], [6, 6, 7, 7]],
    ]
)
tiles = torch.Tensor(
    [
        [[[0, 0], [0, 0]], [[4, 4], [4, 4]]],
        [[[1, 1], [1, 1]], [[5, 5], [5, 5]]],
        [[[2, 2], [2, 2]], [[6, 6], [6, 6]]],
        [[[3, 3], [3, 3]], [[7, 7], [7, 7]]],
    ]
)


@pytest.mark.parametrize("image, expected_tiles", [(image, tiles)])
def test_split_image(image, expected_tiles):
    """Test that split_image generate correct tiles"""
    tiles = split_image(image.permute((1, 2, 0)), tile_size=(2, 2), stride=(2, 2))
    assert torch.equal(tiles, expected_tiles)


@pytest.mark.parametrize("tiles, expected_image", [(tiles, image)])
def test_concatenate_image(tiles, expected_image):
    """Test that concatenate_image generate correct image from tiles"""
    image = concatenate_images(tiles, output_size=(4, 4), stride=(2, 2))
    assert torch.equal(image, expected_image)
