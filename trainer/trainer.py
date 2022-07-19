"""Pytorch nn trainer class
"""

import os
from tqdm import tqdm
import torch


class Trainer:
    """Class for training pytorch nn model"""

    def __init__(
        self,
        model,
        loss_func,
        optimizer,
        scheduler=None,
        experiment_path="experiment/",
        device="cpu",
    ):
        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.experiment_path = experiment_path
        self.__setup_experiment()

    def continue_training(self, train_loader, val_loader):
        """Continue training from last checkpoint"""
        start_epoch, num_epochs = self.__load_model_state(
            self.experiment_path + "/checkpoint/last.pt"
        )
        self.train(train_loader, val_loader, num_epochs, start_epoch)

    def train(self, train_loader, val_loader, num_epochs, start_epoch=0):
        """Train nn model

        Parameters
        ----------
        train_loader : pytorch dataloader
            train data loader
        val_loader : pytorch dataloader
            val data loader
        num_epochs : int
            number of epoch
        start_epoch : int, optional
            start epoch, by default 0
        """
        self.model = self.model.to(self.device)
        self.loss_func = self.loss_func.to(self.device)

        train_log = open(self.experiment_path + "train_log.csv", "w")
        train_log.write("epoch,train_loss,val_loss\n")

        train_data_len = len(train_loader)
        val_data_len = len(val_loader)

        val_loss = float("inf")
        best_val_loss = float("inf")

        pbar = tqdm(range(start_epoch, num_epochs))
        for epoch in pbar:

            self.model.train()
            train_loss = 0
            for j, batch in enumerate(tqdm(train_loader, desc="TRAIN: ", leave=False)):

                batch_loss = self.__train_loop(batch)

                train_loss += batch_loss

                if j % 1 == 0:
                    pbar.set_postfix_str(
                        "TRAIN LOSS: {:.6f} | VAL LOSS: {:.6f} | BEST VAL LOSS {:.6f}".format(
                            batch_loss, val_loss, best_val_loss
                        )
                    )
            train_loss /= train_data_len

            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="VAL:", leave=False):
                    loss = self.__val_loop(batch)
                    val_loss += loss

            val_loss /= val_data_len
            train_log.write(f"{epoch},{train_loss},{val_loss}\n")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.__save_model_state(epoch, num_epochs, "best")
            self.__save_model_state(epoch, num_epochs, "last")

            pbar.set_postfix_str(
                "TRAIN LOSS: {:.6f} | VAL LOSS: {:.6f} | BEST VAL LOSS {:.6f}".format(
                    train_loss, val_loss, best_val_loss
                )
            )
        train_log.close()

    def __train_loop(self, batch):
        """Train loop

        Parameters
        ----------
        batch : tuple
            batch (img_batch, mask_batch)

        Returns
        -------
        float
            loss
        """
        self.optimizer.zero_grad()
        loss = self.__forward_batch(batch)
        loss.backward()
        self.optimizer.step()
        if not self.scheduler is None:
            self.scheduler.step()
        return loss.item()

    def __val_loop(self, batch):
        """Val loop

        Parameters
        ----------
        batch : tuple
            batch (img_batch, mask_batch)

        Returns
        -------
        float
            loss
        """
        loss = self.__forward_batch(batch)
        return loss.item()

    def __forward_batch(self, batch):
        """Forward pass and loss calculation

        Parameters
        ----------
        batch : tuple
            batch (img_batch, mask_batch)

        Returns
        -------
        torch.Tensor
            loss
        """
        X_batch, y_batch = batch
        X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
        output = self.model(X_batch)
        loss = self.loss_func(output, y_batch)
        return loss

    def __setup_experiment(self):
        """Create folders for new experiment
        """
        if not os.path.isdir(self.experiment_path + "checkpoint"):
            print("Create new experiment")
            os.makedirs(self.experiment_path + "checkpoint")
        elif os.path.exists(self.experiment_path + "checkpoint/last.pt"):
            print(
                """WARNING: experiment directory contain saved checkpoint
            use .continue_training() method to continue training,
            if you use .train() method existing checkpoint will be overwritten"""
            )

    def __save_model_state(self, epoch, num_epoch, state_type):
        """Save checkpoint"""
        state_dict = {
            "current_epoch": epoch,
            "num_epoch": num_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        if not self.scheduler is None:
            state_dict["scheduler"] = self.scheduler.state_dict()

        torch.save(state_dict, f"{self.experiment_path}checkpoint/{state_type}.pt")

    def __load_model_state(self, model_path):
        """Load checkpoint for further training"""
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if not self.scheduler is None:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        return checkpoint["current_epoch"], checkpoint["num_epoch"]
