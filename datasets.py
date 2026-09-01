import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
import torchvision
import torchvision.transforms.v2.functional as FT
from torchvision import datasets
from torchvision.transforms import v2 as transforms
from pytorch_lightning import LightningDataModule
from matplotlib import pyplot as plt

from vmf_optimizer import random_VMF
from utils import add_noise


# NOTE: Needed to make the `num_workers` parameter work for the dataloader. A
# simple lambda does not work.
class OnehotTransform(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x):
        return F.one_hot(x, self.num_classes).float()


class Dataset(LightningDataModule):
    _dataset_map = {
        "mnist": datasets.MNIST,
        "fmnist": datasets.FashionMNIST,
        "celeba": datasets.CelebA,
    }

    _size_map = {
        "mnist": 32,
        "fmnist": 32,
        "celeba": 64,
    }

    _channels_map = {
        "mnist": 1,
        "fmnist": 1,
        "celeba": 3,
    }

    _classes_map = {
        "mnist": 10,
        "fmnist": 10,
        "celeba": 40,
    }

    _transform_map = {
        "mnist": [
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Resize(_size_map["mnist"]),
        ],
        "fmnist": [
            transforms.Normalize((0.2860,), (0.3530,)),
            transforms.Resize(_size_map["fmnist"]),
        ],
        "celeba": [
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(148),
            transforms.Resize(_size_map["celeba"]),
        ],
    }

    _target_transform_map = {
        "mnist": [
            torch.tensor,
            OnehotTransform(_classes_map["mnist"]),
        ],
        "fmnist": [
            torch.tensor,
            OnehotTransform(_classes_map["fmnist"]),
        ],
        "celeba": [transforms.ToDtype(torch.float32)],
    }

    _config_map = {
        "mnist": {"train": {"train": True}, "test": {"train": False}},
        "fmnist": {"train": {"train": True}, "test": {"train": False}},
        "celeba": {"train": {"split": "train"}, "test": {"split": "test"}},
    }

    def __init__(
        self, batch_size=64, dataset="mnist", num_workers=0, shuffle=True, subset=None
    ):
        super().__init__()
        # Remove to prevent message regarding validation loop.
        self.val_dataloader = None

        print("dataset", dataset)

        self.dataset = self._dataset_map[dataset]
        self.config = self._config_map[dataset]
        self.img_size = self._size_map[dataset]
        self.channels = self._channels_map[dataset]
        self.num_classes = self._classes_map[dataset]
        self.shuffle = shuffle
        self.subset = subset

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.Compose(
            [
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                *self._transform_map[dataset],
            ]
        )

        self.target_transform = transforms.Compose(
            [*self._target_transform_map[dataset]]
        )

    def mean_std(self):
        print("computing mean")
        if isinstance(self.train_data, (datasets.MNIST, datasets.FashionMNIST)):
            mean = self.train_data.data.float().mean() / 255
            std = self.train_data.data.float().std() / 255
        else:
            raise NotImplementedError(
                "mean and std computation not implemented for celeba"
            )
        return mean, std

    def prepare_data(self):
        self.dataset(
            root="data",
            download=True,
            **self.config["train"],
        )
        self.dataset(
            root="data",
            download=True,
            **self.config["test"],
        )

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_data = self.dataset(
                root="data",
                **self.config["train"],
                transform=self.transform,
                target_transform=self.target_transform,
            )
            if self.subset is not None:
                self.train_data = torch.utils.data.Subset(
                    self.train_data, range(int(len(self.train_data) * self.subset))
                )

        if stage == "test":
            self.test_data = self.dataset(
                root="data",
                **self.config["test"],
                transform=self.transform,
                target_transform=self.target_transform,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            pin_memory=True,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
        )


class SyntheticDataset(LightningDataModule):
    class Synthetic(torch.utils.data.IterableDataset):
        def __init__(self, model, data, batch_size=1, noise=None):
            super().__init__()
            self.model = model
            self.batch_size = batch_size
            self.classes = torch.stack([c for _, c in data.train_data], dim=0).cuda()
            self.noise = noise

            # self.epoch = 0

        def __iter__(self):
            # Shuffle order of classes.
            self.classes = self.classes[torch.randperm(self.classes.shape[0])]
            for i in range(0, len(self.classes), self.batch_size):
                c = self.classes[i : i + self.batch_size]
                z = self.model.sample_latent(len(c))  # , self.noise)
                z = add_noise(z, self.model, self.noise)
                x_hat = self.model.decoder(z, c).detach()

                yield (x_hat, c)

        def __len__(self):
            # Ceiling division, because 10_000 / 64 = 157 batches.
            return -(len(self.classes) // -self.batch_size)

    def __init__(self, model, data, batch_size=1, noise=None):
        super().__init__()
        self.model = self.Synthetic(model, data, batch_size, noise)
        self.dataset = data

    def prepare_data(self):
        self.dataset.prepare_data()

    def setup(self, stage=None):
        self.dataset.setup(stage=stage)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.model)

    def test_dataloader(self) -> DataLoader:
        return self.dataset.test_dataloader()
