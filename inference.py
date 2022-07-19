"""Script for inference
usage:
from inference import inference
inference(img_path, experiment_path, batch_size, device)
or
python inference.py --experiment path/to/experiment/folder --img_path path/to/your/img.tiff
"""


import argparse
import numpy as np
from PIL import Image
import torch
import torchvision

from model import UNet
from utils import split_image, concatenate_images


def inference(img_path, experiment_path, batch_size, device=None):
    """get predicted segmentation maskfor input tiff image

    Parameters
    ----------
    img_path : str
        path to your .tiff image
    experiment_path : str
        Path to experiment folder with saved checkpoints
    batch_size : int
        batch size
    device : str, optional
        'cpu' or 'gpu', by default None, if None gpu will used if possible
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = UNet(n_channels=3, n_classes=1)
    checkpoint = torch.load(experiment_path + "/checkpoint/best.pt")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    img = Image.open(img_path)
    img = img.resize((1536, 1536))
    img = torch.tensor(np.asarray(img))
    classes = ["kidney"]  # Fix
    tiles = split_image(img, tile_size=((256, 256)), stride=(128, 128)) / 255

    predictions = []
    for i in range(0, tiles.shape[0], batch_size):
        output = model(tiles[i : i + batch_size].to(device))
        predictions.append(output.detach().cpu())
    predictions = torch.cat(predictions)

    predictions = concatenate_images(
        predictions, output_size=(1536, 1536), stride=(128, 128)
    )

    for i, cls in enumerate(classes):
        torchvision.utils.save_image(
            predictions[i], img_path.replace(".tiff", f"_{cls}.jpg")
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment", type=str, default="experiment/", help="experiment path"
    )
    parser.add_argument(
        "--img_path",
        type=str,
        default="train_data/raw_data/11497.tiff",
        help="path to train data",
    )
    parser.add_argument("--batch_size", type=int, default=5, help="batch size")
    parser.add_argument(
        "--device", type=str, help="cpu or gpu by default use gpu if available"
    )
    args = parser.parse_args()

    inference(args.img_path, args.experiment, args.batch_size, args.device)
