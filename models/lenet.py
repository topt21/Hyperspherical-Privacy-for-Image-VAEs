import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl

from opacus import PrivacyEngine
from opacus.lightning import DPDataLoader

from .utils import conv_result_size


class LeNet(pl.LightningModule):
    def __init__(self, size, channels, num_classes):
        super().__init__()

        size = conv_result_size(size, 5, 1, 2)  # conv2d
        size = conv_result_size(size, 2, 2, 0)  # avgpool2d
        size = conv_result_size(size, 5, 1, 0)  # conv2d
        size = conv_result_size(size, 2, 2, 0)  # avgpool2d

        self.layers = nn.Sequential(
            nn.Conv2d(channels, 6, 5, 1, 2),
            nn.Sigmoid(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(6, 16, 5, 1, 0),
            nn.Sigmoid(),
            nn.AvgPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(size * size * 16, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, num_classes),
        )

        self.save_hyperparameters()

    def forward(self, x):
        return self.layers(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch

        # TODO: Temp fix because dataloader adds a fifth dimension for batches
        # but we create batches directly in the "dataset".
        x = x.squeeze(0)
        y = y.squeeze(0)

        z = self.layers(x)
        loss = self.loss(y, z)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        z = self.layers(x)
        loss = self.loss(y, z)
        acc = (torch.argmax(z, dim=1) == torch.argmax(y, dim=1)).float().mean()
        self.log("test_loss", loss)
        self.log("test_acc", acc)

    def loss(self, y, z):
        return F.cross_entropy(z, y)
