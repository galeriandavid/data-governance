"""Banch of methods for mask processing"""

import numpy as np
import torch


def rle2mask(rle: str, img_shape: tuple) -> torch.Tensor:
    """Convert Run-Length Encodings to binary mask

    Parameters
    ----------
    rle : str
        Run-Length Encodings
    img_shape : tuple
        shape of generated mask: (height, width)

    Returns
    -------
    torch.Tensor
        tensor with binary mask shape (height, width)
    """
    height, width = img_shape
    mask = torch.zeros(height * width, dtype=torch.int8)
    if rle != "":
        rle = list(map(int, rle.split(" ")))
        rle = [(rle[i], rle[i + 1]) for i in range(0, len(rle), 2)]
        for start, length in rle:
            mask[start : start + length] = 1
    return mask.reshape((height, width))


def mask2rle(mask: np.ndarray) -> str:
    """Convert binary mask to rle

    Parameters
    ----------
    mask : np.ndarray
        numpy array or torch tensor with shape(height, width)

    Returns
    -------
    str
        return rle string
    """
    if not isinstance(type(mask), np.ndarray):
        mask = mask.numpy()
    mask = mask.flatten()
    mask = np.concatenate([[0], mask, [0]])
    mask_diff = np.diff(mask)
    start_ind = np.where(mask_diff == 1)[0]
    end_ind = np.where(mask_diff == -1)[0]
    return " ".join(
        [f"{start} {length}" for start, length in zip(start_ind, end_ind - start_ind)]
    )


def split_image(img: torch.Tensor, tile_size: tuple, stride: tuple) -> torch.Tensor:
    """Split image into tiles

    Parameters
    ----------
    img : torch.tensor
        input image with shape: (height, width, n_channels)
    tile_size : tuple, optional
        size of output tiles
    stride : tuple, optional
        stride

    Returns
    -------
    torch.tensor
        return torch tensor with image tile, shape(n_tiles, n_channels, h, w)
    """
    img = img.unfold(0, tile_size[0], stride[0]).unfold(1, tile_size[1], stride[1])
    img = img.reshape((img.shape[0] * img.shape[1], *img.shape[2:]))
    return img


def concatenate_images(
    imgs: torch.Tensor, output_size: tuple, stride: tuple
) -> torch.Tensor:
    """Concatenate tiles in one img, calculate mean in case of overlaping tiles

    Parameters
    ----------
    imgs : torch.tensor
        torch tensor with shape: (n_imgs, n_channels, height, width)
    output_size : tuple
        Output image size: (height, width)
    stride : tuple
        with which stride tiles was generated

    Returns
    -------
    torch.tensor
        return image obtained by cocatenating separate image tiles, shape: (n_channels, height, width)
    """
    n_channels = imgs.shape[1]
    kernel_size = tuple(imgs.shape[2:])
    imgs = imgs.reshape(
        (-1, n_channels * kernel_size[0] * kernel_size[1], 1)
    ).transpose(0, 2)
    img = torch.nn.functional.fold(
        imgs, output_size, kernel_size=kernel_size, stride=stride
    )[0]
    counter = torch.nn.functional.fold(
        torch.ones_like(imgs), output_size, kernel_size=kernel_size, stride=stride
    )[0]
    return img / counter
