"""Test trainer
"""

from model import UNet
from loss import DiceLoss
from trainer import Trainer
from dataset import SegmentationDataset
import torch


def test_trainer():
    """Testing that trainer work by overfitting base UNet model on one test image"""
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = SegmentationDataset("dataset/tests/test_data/")
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=2)

    learning_rate = 1e-3
    num_epochs = 12
    model = UNet(n_channels=3, n_classes=1)
    loss_func = DiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        steps_per_epoch=int(len(data_loader)),
        epochs=num_epochs,
        anneal_strategy="linear",
    )

    trainer = Trainer(
        model,
        loss_func,
        optimizer,
        scheduler,
        experiment_path="trainer/tests/test_experiment/",
        device=device,
    )

    trainer.train(
        train_loader=data_loader, val_loader=data_loader, num_epochs=num_epochs
    )
    with open("trainer/tests/test_experiment/train_log.csv", "r") as f:
        last_epoch = f.readlines()[-1].split(",")[1]
    assert float(last_epoch) < 0.2
