"""Script for validation segmentation model
usage:
from validation import validate
validate(experiment_path, data_path, batch_size, device)
or
python validation.py --experiment path/to/experiment/folder -- data_path path/to/your/val_data"""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import SegmentationDataset
from loss import DiceLoss
from model import UNet


def validate(experiment_path, data_path, batch_size, device):
    """Validate segmentation model

    Parameters
    ----------
    experiment_path : str
        Path to foldaer with experiment
    data_path : str
        Path to validation data
    batch_size : int
        batch size
    device : str, optional
        'cpu' or 'cuda', if None than cuda will used if it's possible
    """
    torch.manual_seed(42)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = SegmentationDataset(data_path)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    )

    model = UNet(n_channels=dataset.num_channels, n_classes=dataset.num_classes)
    checkpoint = torch.load(experiment_path + "/checkpoint/best.pt")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    loss_func = DiceLoss()

    loss = []
    for X_batch, y_batch in data_loader:
        output = model(X_batch.to(device))
        loss.append(loss_func(output, y_batch.to(device)).item())

    sorted_loss = np.argsort(loss)

    for i, batch in enumerate(data_loader):
        if i == sorted_loss[0]:
            X_batch, y_batch = batch
            predictions = model(X_batch.to(device))
            print_batch_predicitons(
                X_batch,
                y_batch,
                predictions,
                loss[sorted_loss[0]],
                dataset.classes,
                experiment_path + "best_batch.jpg",
            )
        if i == sorted_loss[-1]:
            X_batch, y_batch = batch
            predictions = model(X_batch.to(device))
            print_batch_predicitons(
                X_batch,
                y_batch,
                predictions,
                loss[sorted_loss[-1]],
                dataset.classes,
                experiment_path + "worst_batch.jpg",
            )
    loss_json = {"val loss": sum(loss) / len(loss)}
    with open(experiment_path + "loss.json", "w") as json_file:
        json.dump(loss_json, json_file)


def print_batch_predicitons(X_batch, y_batch, predictions, loss, classes, save_path):
    """Generate image with predictions example and save it as .jpg"""
    X_batch = X_batch.permute((0, 2, 3, 1)).numpy()
    y_batch = y_batch.numpy()
    predictions = predictions.detach().cpu().numpy()
    fig, axs = plt.subplots(
        X_batch.shape[0], 2 * y_batch.shape[1] + 1, figsize=(10, 4 * X_batch.shape[0])
    )
    fig.suptitle(f"batch loss: {loss}")
    for i, row in enumerate(axs):
        row[0].imshow(X_batch[i])
        row[0].set_title("image")
        for j, class_name in enumerate(classes):
            row[2 * j + 1].imshow(y_batch[i][j])
            row[2 * j + 1].set_title(class_name + " true")
            row[2 * j + 2].imshow(predictions[i][j])
            row[2 * j + 2].set_title(class_name + " pred")
    fig.savefig(save_path)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment", type=str, default="experiment/", help="experiment path"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="train_data/processed_data/val/",
        help="path to train data",
    )
    parser.add_argument("--batch_size", type=int, default=5, help="batch size")
    parser.add_argument(
        "--device", type=str, help="cpu or gpu by default use gpu if available"
    )
    args = parser.parse_args()

    validate(args.experiment, args.data_path, args.batch_size, args.device)
