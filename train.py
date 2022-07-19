"""Script for training segmentation model
usage:
python train.py --experiment path/to/experiment/folder -- data_path path/to/your/data"""

from model import UNet
from loss import DiceLoss
from trainer import Trainer
from dataset import SegmentationDataset
import matplotlib.pyplot as plt
import torch
import argparse
from validation import validate


def main(
    experiment_path,
    data_path,
    batch_size,
    learning_rate,
    num_epochs,
    continue_training=False,
    device=None,
):
    """Train segmentation model"""

    torch.manual_seed(42)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = SegmentationDataset(data_path + "train/")
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    val_dataset = SegmentationDataset(data_path + "val/")
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )

    model = UNet(
        n_channels=train_dataset.num_channels, n_classes=train_dataset.num_classes
    )
    loss_func = DiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        steps_per_epoch=len(train_loader),
        epochs=num_epochs,
        anneal_strategy="linear",
    )

    trainer = Trainer(
        model,
        loss_func,
        optimizer,
        scheduler,
        experiment_path=experiment_path,
        device=device,
    )
    if continue_training:
        trainer.continue_training(train_loader=train_loader, val_loader=val_loader)
    else:
        trainer.train(
            train_loader=train_loader, val_loader=val_loader, num_epochs=num_epochs
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment", type=str, default="experiment/", help="experiment path"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="train_data/processed_data/",
        help="path to train data",
    )
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="learning rate")
    parser.add_argument("--num_epochs", type=int, default=1, help="number of epoch")
    parser.add_argument(
        "--resume",
        type=bool,
        default=False,
        help="continue training. If True, provide a path to an existing experiment",
    )
    parser.add_argument(
        "--device", type=str, help="cpu or gpu by default use gpu if available"
    )
    args = parser.parse_args()

    main(
        args.experiment,
        args.data_path,
        args.batch_size,
        args.lr,
        args.num_epochs,
        args.resume,
        args.device,
    )
