#!/usr/bin/env python3
from torch.utils.data import DataLoader
import argparse
import logging
from threading import Thread
import itertools
import pathlib

import torch
from torch.nn import functional as F
import torchvision
import torchvision.transforms.v2.functional as FT

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import CSVLogger

from opacus.lightning import DPLightningDataModule

from matplotlib import pyplot as plt
from matplotlib.widgets import TextBox, Button, Slider
import numpy as np

import os
import sys

sys.path.append(
    os.path.realpath(filename=os.path.dirname(p=__file__)) + "/s-vae-pytorch"
)

from models.svae import VAE as HypersphericalVAE
from models.lenet import LeNet
from datasets import Dataset, SyntheticDataset
import attacks
from utils import add_noise


# Hide advertisements. https://github.com/Lightning-AI/pytorch-lightning/issues/21294#issuecomment-3410770397
logging.getLogger("pytorch_lightning.utilities.rank_zero").addFilter(
    lambda record: "💡 Tip" not in record.getMessage()
)


class Plot:
    def __init__(self, **kwargs):
        subplots = kwargs.get("subplots")
        if subplots is not None:
            self._fig, self._ax = plt.subplots(*subplots)
        else:
            self._fig, self._ax = plt.subplots(1, 1, squeeze=False)

        inputs = kwargs.get("inputs", [])
        if len(inputs) > 0:
            self._fig.subplots_adjust(bottom=0.1)

            margin = 0.02
            left = 0.05
            bottom = 0.05
            width = 0.05
            height = 0.025

            self._text_boxes = []
            self._buttons = []

            for input_type, label, value, pre_update_check in inputs:
                axbox = self._fig.add_axes((left, bottom, width, height))
                if input_type == "text":
                    self._text_boxes.append(TextBox(axbox, label))
                    self._text_boxes[-1].set_val(value)
                    self._text_boxes[-1].on_submit(
                        self._textbox_update(pre_update_check)
                    )
                elif input_type == "button":
                    self._buttons.append(Button(axbox, label))
                    self._buttons[-1].on_clicked(self._button_update(pre_update_check))
                left += width + margin

    def _textbox_update(self, pre_update_check):
        def inner(text):
            if pre_update_check(text):
                for ax in self._ax.flatten():
                    ax.clear()
                self.update()
                self._fig.canvas.draw_idle()

        return inner

    def _button_update(self, pre_update_check):
        def inner(event):
            if pre_update_check():
                for ax in self._ax.flatten():
                    ax.clear()
                self.update()
                self._fig.canvas.draw_idle()

        return inner

    def pre_update(self, label, text):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError


class PlotLatent(Plot):
    def __init__(self, model, data, num_classes, device, num_batches=100):
        self.c = 0
        self.noise = False
        self.noise_multiplier = 1.0
        inputs = [
            ("text", "c", self.c, self.pre_update_c),
            ("button", "full latent", 0, self.toggle_full_latent),
            ("button", "noise", 0, self.toggle_noise),
            ("text", "m", self.noise_multiplier, self.pre_update_noise_multiplier),
        ]
        super().__init__(inputs=inputs)

        self.model = model
        self.data = data
        self.num_classes = num_classes
        self.device = device
        self.num_batches = num_batches

        self.cb = None

        self.full_latent = True

        self.update()

    def pre_update_c(self, text):
        try:
            c = int(text)
        except ValueError:
            return False
        if c == self.c or c >= self.num_classes:
            return False
        self.c = c
        return True

    def pre_update_noise_multiplier(self, text):
        try:
            mult = float(text)
        except ValueError:
            return False
        if mult == self.noise_multiplier:
            return False
        self.noise_multiplier = mult
        return True

    def toggle_noise(self):
        self.noise = not self.noise
        return True

    def toggle_full_latent(self):
        self.full_latent = not self.full_latent
        return True

    def update(self):
        print("plotting latent")
        self._fig.suptitle(f"Latent space of {self.model.name()}")
        for i, (x, y) in enumerate(self.data):
            c = torch.argmax(y, 1)
            if not self.full_latent:
                x = x[torch.nonzero(c == self.c, as_tuple=True)]
                y = y[torch.nonzero(c == self.c, as_tuple=True)]
                c = c[torch.nonzero(c == self.c)]

            z, *_ = self.model.encoder(x.to(self.device), y.to(self.device))

            if self.noise:
                z = add_noise(z, self.model, self.noise_multiplier)

            z = z.to("cpu").detach().numpy()
            if self.full_latent:
                im = self._ax[0, 0].scatter(z[:, 0], z[:, 1], c=c, cmap="tab10")
            else:
                im = self._ax[0, 0].plot(z[:, 0], z[:, 1], f"oC{self.c}")
            if i > self.num_batches:
                if self.full_latent:
                    if self.cb is not None:
                        self.cb.update_normal(im)
                    else:
                        self.cb = self._fig.colorbar(im)
                break


class PlotSample(Plot):
    def __init__(
        self,
        model,
        num_classes,
        device,
        w=28,
        channels=1,
        n=12,
    ):
        self.c = 0
        self.noise_multiplier = 1.0
        inputs = [
            ("button", "uniform", 0, self.set_method("uniform")),
            ("button", "randn", 0, self.set_method("randn")),
            ("button", "prior", 0, self.set_method("prior")),
            ("button", "normalize", 0, self.toggle_normalize),
            ("button", "noise", 0, self.toggle_noise),
            ("text", "m", self.noise_multiplier, self.pre_update_noise_multiplier),
            ("text", "c", self.c, self.pre_update_c),
        ]
        super().__init__(inputs=inputs)
        self.img = np.zeros((n * w, n * w))

        self.model = model
        self.num_classes = num_classes
        self.device = device
        self.w = w
        self.channels = channels
        self.n = n

        self.method = "uniform"
        self.normalize = False
        self.noise = False

        self.update()

    def pre_update_c(self, text):
        try:
            c = int(text)
        except ValueError:
            return False
        if c == self.c or c >= self.num_classes:
            return False
        self.c = c
        return True

    def pre_update_noise_multiplier(self, text):
        try:
            mult = float(text)
        except ValueError:
            return False
        if mult == self.noise_multiplier:
            return False
        self.noise_multiplier = mult
        return True

    def set_method(self, method):
        def inner():
            self.method = method
            return True

        return inner

    def toggle_normalize(self):
        self.normalize = not self.normalize
        return True

    def toggle_noise(self):
        self.noise = not self.noise
        return True

    def update(self):
        print("plotting samples")
        self._fig.suptitle(
            f"{'normalized ' if self.normalize else ''}{self.method} sampling from {self.model.name()}"
        )

        if self.method == "uniform":
            z = (
                torch.distributions.Uniform(-3, 3)
                .sample((self.n * self.n, self.model.encoder.latent_dims))
                .to(self.device)
            )
        elif self.method == "randn":
            z = torch.randn(
                self.n * self.n, self.model.encoder.latent_dims, device="cuda"
            )
        elif self.method == "prior":
            z = self.model.sample_latent(self.n * self.n)

        if self.normalize:
            z = F.normalize(z)

        c = (
            F.one_hot(torch.tensor(self.c).repeat(self.n * self.n), self.num_classes)
            .float()
            .to(self.device)
        )

        if self.noise:
            z = add_noise(z, self.model, self.noise_multiplier)

        x_hat = self.model.decoder(z, c)
        x_hat = x_hat.reshape(-1, self.channels, self.w, self.w)

        grid = (
            torchvision.utils.make_grid(x_hat, self.n, padding=0, normalize=True)
            .detach()
            .cpu()
        )
        # grid = FT.to_grayscale(grid)
        grid = FT.to_pil_image(grid)
        self._ax[0, 0].imshow(grid)


class PlotCompare(Plot):
    def __init__(
        self,
        dataset,
        num_classes,
        device,
        w=28,
        channels=1,
        n=12,
    ):

        self.c = 0
        self.noise_multiplier = 1.0
        self.dataset = dataset
        self.glob = "[ns]*-" + dataset
        inputs = [
            ("button", "noise", 0, self.toggle_noise),
            ("text", "m", self.noise_multiplier, self.pre_update_noise_multiplier),
            ("text", "c", self.c, self.pre_update_c),
            ("button", "load", 0, self.load),
            ("text", "glob", self.glob, self.pre_update_glob),
            ("button", "save l", 0, self.save(0)),
            ("button", "save r", 0, self.save(1)),
        ]
        super().__init__(inputs=inputs, subplots=(1, 2))

        self.models_path = pathlib.Path("./lightning_logs/")
        self.models = []

        self.selected_model_names = [None, None]
        self.selected_models = [None, None]
        self.changed = [True, True]

        self._sliders = [
            Slider(self._fig.add_axes((0.05 + 4 * 0.12, 0.05, 0.2, 0.025)), "", 0, 1),
            Slider(self._fig.add_axes((0.15 + 5 * 0.12, 0.05, 0.2, 0.025)), "", 0, 1),
        ]

        self.img = np.zeros((n * w, n * w))

        self.num_classes = num_classes
        self.device = device
        self.w = w
        self.channels = channels
        self.n = n

        self.noise = False

        self.pre_update_glob(self.glob)
        self.update_selected_model(0)(0)
        self.update_selected_model(1)(1)
        self.update()

    def save(self, i):
        def inner():
            model = self.selected_models[i]
            name = "cmp-" + save_name(
                model, self.dataset, False, self.noise if self.noise else None
            )
            save_samples(model, name, overwrite=True)

        return inner

    def update_ax_titles(self, i):
        try:
            with open("./classification/" + self.selected_model_names[i] + ".txt") as f:
                acc = f.readline()
        except FileNotFoundError:
            acc = "???"
        self._ax[i].set_title(f"{self.selected_model_names[i]}\n{acc}")

    def update_selected_model(self, i):
        def inner(j):
            self.selected_model_names[i] = self.models[j].name
            self.update_ax_titles(i)
            self.changed[i] = True

        return inner

    def load(self):
        for i in range(2):
            if self.changed[i] and self.selected_model_names[i] is not None:
                path = pathlib.Path(
                    "./lightning_logs/" + self.selected_model_names[i] + "/checkpoints"
                )
                # TODO: This just selects an arbitrary checkpoint. If we have
                # multiple we may want to select the latest.
                ckpt = next(path.iterdir())
                self.selected_models[i] = HypersphericalVAE.load_from_checkpoint(
                    str(ckpt)
                )
                self.changed[i] = False
        return True

    def pre_update_glob(self, glob):
        self.glob = glob
        self.models = sorted(list(self.models_path.glob(self.glob)))
        valmax = len(self.models) - 1
        for i in range(2):
            s = self._sliders[i]
            valinit = s.val
            ax = s.ax
            ax.clear()
            self._sliders[i] = Slider(
                ax,
                "",
                0,
                valmax,
                valstep=1,
                valinit=valinit,
            )
            self._sliders[i].on_changed(self.update_selected_model(i))
        return False

    def pre_update_c(self, text):
        try:
            c = int(text)
        except ValueError:
            return False
        if c == self.c or c >= self.num_classes:
            return False
        self.c = c
        return True

    def pre_update_noise_multiplier(self, text):
        try:
            mult = float(text)
        except ValueError:
            return False
        if mult == self.noise_multiplier:
            return False
        self.noise_multiplier = mult
        return True

    def toggle_noise(self):
        self.noise = not self.noise
        return True

    def update(self):
        # print(f"{self.selected_model[0]} vs. {self.selected_model[1]}")
        # self._fig.suptitle(
        #     f"{self.selected_model_names[0]}\nvs.\n{self.selected_model_names[1]}"
        # )

        for i in range(2):
            if self.selected_models[i] is None:
                continue

            self.update_ax_titles(i)

            z = self.selected_models[i].sample_latent(self.n * self.n)

            if self.noise:
                # z += torch.normal(0, self.noise_multiplier, size=z.shape, device=z.device)
                z *= self.noise_multiplier

            c = (
                F.one_hot(
                    torch.tensor(self.c).repeat(self.n * self.n), self.num_classes
                )
                .float()
                .to(self.device)
            )

            x_hat = self.selected_models[i].decoder(z, c)
            x_hat = x_hat.reshape(-1, self.channels, self.w, self.w)

            grid = (
                torchvision.utils.make_grid(x_hat, self.n, padding=0, normalize=True)
                .detach()
                .cpu()
            )
            grid = FT.to_pil_image(grid)
            self._ax[i].imshow(grid)


def save_name(model, dataset, overfit, noise):
    name = []
    if overfit:
        name.append("overfit")
    name.append("s" if model.hyperspherical else "n")
    name.append("conv")
    if model.num_classes > 0:
        name.append("cond")
    if model.dirdp:
        name.append("dirdp")
    elif model.private:
        name.append("dp")
    if noise is not None:
        name.append(f"noise{noise}")
    # name.append(f"epoch{model.epochs}")
    name.append(f"latent{model.encoder.latent_dims}")
    # name.append(f"hidden{model.hparams.hidden_dims}")
    name.append(f"kl{round(model.kl_weight, 3)}")
    if model.private:
        name.append(f"eps{model.epsilon}")
        # name.append(f"delta{round(model.delta, 3)}")
        name.append(f"maxnorm{model.max_grad_norm}")
    name.append(dataset)
    return "-".join(name)


def save_samples(model, name, noise=None, overwrite=False):
    if not overwrite and pathlib.Path("samples/" + name + ".png").exists():
        return

    rows = 3
    num_classes = model.num_classes

    z = model.sample_latent(rows * num_classes)
    z = add_noise(z, model, noise)

    c = torch.cat(
        [
            F.one_hot(torch.tensor([i] * rows, device="cuda"), num_classes)
            for i in range(num_classes)
        ]
    ).reshape(rows * num_classes, num_classes)

    x_hat = model.decoder(z, c)
    x_hat = x_hat.reshape(
        -1, model.decoder.channels, model.encoder.img_size, model.encoder.img_size
    )

    grid = (
        torchvision.utils.make_grid(x_hat, rows, padding=0, normalize=True)
        .detach()
        .cpu()
    )
    print(grid.shape)
    grid = FT.to_pil_image(grid)
    plt.imsave("samples/" + name + ".png", grid)


def save_reconstructions(model, data: Dataset, name, noise=None):
    rows = 5
    cols = 5

    x, y = next(iter(DataLoader(data.test_data, batch_size=rows)))
    x = x.to(model.device)
    y = y.to(model.device)

    z = torch.cat([model.encoder(x, y)[0] for _ in range(cols)], dim=0)
    z = add_noise(z, model, noise)

    x_hat = model.decoder(z, y.repeat(cols, 1))
    x_hat = torch.cat([x, x_hat], dim=0)

    grid = (
        torchvision.utils.make_grid(x_hat, rows, padding=0, normalize=True)
        .detach()
        .cpu()
    )
    grid = FT.to_pil_image(grid)
    plt.imsave("reconstructions/" + name + ".png", grid)


def run_classifier(model, data, name, noise, device):
    synthetic = SyntheticDataset(model, data, batch_size=64, noise=noise)

    classifier = LeNet(data.img_size, data.channels, data.num_classes).to(device)
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=10,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(classifier, synthetic)
    classifier.eval()
    result = trainer.test(classifier, synthetic)
    # with open("classification/" + name + ".txt", "a") as f:
    #     f.write(str(result[0]["test_acc"]))
    #     f.write("\n")


def evaluate(
    model,
    dataset,
    save=False,
    plots=True,
    classify=False,
    attack=False,
    subset=None,
    noise=None,
):
    data = Dataset(128, dataset=dataset, num_workers=4)
    data.setup()
    data.setup("test")

    device = torch.accelerator.current_accelerator().type
    model = model.to(device)

    name = save_name(model, dataset, subset is not None, noise)
    print("EVALUATING", name)

    model.eval()
    if plots:
        _plots = [
            PlotLatent(
                model,
                data=data.train_dataloader(),
                num_classes=data.num_classes,
                device=device,
            ),
            PlotSample(
                model,
                num_classes=data.num_classes,
                device=device,
                w=data.img_size,
                channels=data.channels,
                n=8,
            ),
        ]

    if save:
        save_samples(model, name, noise)
        # save_reconstructions(model, data, name, noise)

    # Run classifier in a separate thread so plots display remain interactive.
    # NOTE: Not sure this is the best way to handle this (doing the opposite,
    # plots in child process, results in a CUDA warning). Also requires two
    # ctrl-c to interrupt.
    if classify:
        classifier = Thread(
            target=run_classifier, args=(model, data, name, noise, device)
        )
        classifier.start()
    if plots:
        plt.show()
    if classify:
        classifier.join()

    if attack:
        device = "cpu"
        model = model.cpu()
        data = Dataset(None, dataset=dataset, shuffle=False)
        data.setup()
        data.setup("test")
        monte_carlo_acc = 0.0
        for i in range(data.num_classes):
            label = F.one_hot(torch.tensor(i), data.num_classes).float()
            acc = attacks.monte_carlo(model, data, subset, label, 100, 10**3, noise)
            print(f"monte carlo accuracy for ({i}): {acc}")
            monte_carlo_acc += acc
        monte_carlo_acc /= data.num_classes
        print("monte carlo accuracy", monte_carlo_acc)
        if save:
            with open("attack/" + name + "-mc.txt", "a") as f:
                f.write(str(monte_carlo_acc))
                f.write("\n")

        recon_acc = 0.0
        for i in range(data.num_classes):
            label = F.one_hot(torch.tensor(i), data.num_classes).float()
            acc = attacks.reconstruction(model, data, subset, label, 100, 300, noise)
            print(f"recon accuracy for ({i}): {acc}")
            recon_acc += acc
        recon_acc /= data.num_classes
        print("recon accuracy", recon_acc)
        if save:
            with open("attack/" + name + "-recon.txt", "a") as f:
                f.write(str(recon_acc))
                f.write("\n")


def train(
    dataset="mnist",
    latent_dims=3,
    kl_weight=0.0025,
    hyperspherical=False,
    private=False,
    epsilon=1.0,
    delta=1e-5,
    max_grad_norm=1,
    dirdp=False,
    epochs=5,
    subset=None,
    checkpoint=False,
    eval=False,
    save=False,
    plots=True,
    classify=False,
    **kwargs,
):
    print("training")

    data = Dataset(64, dataset=dataset, num_workers=4, subset=subset)
    dp_data = DPLightningDataModule(data)
    # Remove again to prevent message regarding validation loop.
    if not data.val_dataloader:
        dp_data.val_dataloader = None
    print(data)

    # data.setup()
    # mean, std = data.mean_std()
    # print("mean, std", mean, std)
    # x, y = next(iter(data.train_dataloader()))
    # print(x.shape)
    # print(y.shape)
    # exit(0)

    # S-VAE
    model = HypersphericalVAE(
        data.img_size,
        data.channels,
        latent_dims,
        epochs,
        data.num_classes,
        hyperspherical=hyperspherical,
        kl_weight=kl_weight,
        private=private,
        dirdp=dirdp,
        epsilon=epsilon,
        # noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )
    print(model)

    name = save_name(model, dataset, subset is not None, noise=None)
    # Skip training if already it exists.
    if pathlib.Path("./lightning_logs/" + name).is_dir():
        if pathlib.Path("./classification/" + name + ".txt").exists():
            return None
        if pathlib.Path("./attack/" + name + "-mc.txt").exists():
            return None
        path = pathlib.Path("./lightning_logs/" + name + "/checkpoints/").iterdir()
        chkpt = next(path).name
        return HypersphericalVAE.load_from_checkpoint(
            "./lightning_logs/" + name + "/checkpoints/" + chkpt
        )

    if isinstance(checkpoint, str):
        logger = CSVLogger(save_dir=os.getcwd(), version=checkpoint)
    elif checkpoint:
        logger = CSVLogger(
            save_dir=os.getcwd(),
            version=name,
        )
    else:
        # Same as default. Setting this explicitly hides tensorboard message.
        logger = CSVLogger(save_dir=os.getcwd())

    trainer = pl.Trainer(
        accelerator="auto", devices=1, max_epochs=epochs, logger=logger
    )
    trainer.fit(model, dp_data)

    if eval:
        with torch.no_grad():
            evaluate(model, dataset, classify=classify, save=save, plots=plots)

    return model


def evaluate_from_checkpoint(
    dataset, checkpoint, save, classify, plots, attack, noise, **kwargs
):
    # S-VAE
    model = HypersphericalVAE.load_from_checkpoint(checkpoint)
    print(model)

    with torch.no_grad():
        evaluate(
            model,
            dataset,
            save,
            plots,
            classify,
            attack,
            subset=0.1 if attack else None,
            noise=noise,
        )


def run_experiments(**kwargs):
    datasets = ["mnist", "fmnist"]
    kl_weights = [64 / 60_000]
    epsilons = [0.5, 1.0, 5.0]
    max_grad_norms = [3.0]  # , 3.0]
    latent_dims = [2, 3, 10]
    is_convolutional = [True]
    is_hyperspherical = [False, True]
    is_overfit = [False, True]

    # Non-private.
    for (
        dataset,
        kl_weight,
        latent_dim,
        convolutional,
        hyperspherical,
        overfit,
    ) in itertools.product(
        datasets,
        kl_weights,
        latent_dims,
        is_convolutional,
        is_hyperspherical,
        is_overfit,
    ):
        subset = 0.1 if overfit else None
        args = argparse.Namespace(
            dataset=dataset,
            latent_dims=latent_dim,
            kl_weight=kl_weight,
            hyperspherical=hyperspherical,
            conditional=True,
            convolutional=convolutional,
            private=False,
            # epsilon=epsilon,
            # delta=1e-5,
            # max_grad_norm=max_grad_norm,
            # dirdp=dirdp,
            epochs=300 if overfit else 10,
            subset=subset,
            checkpoint=True,
            eval=False,
            # save=False,
            # plots=True,
            # classify=False,
        )
        print("TRAINING", args)
        model = train(**vars(args))
        if model is not None:
            with torch.no_grad():
                for _ in range(5):
                    evaluate(
                        model,
                        dataset,
                        classify=not overfit,
                        save=True,
                        plots=False,
                        subset=subset,
                        attack=overfit,
                    )
                for noise in [0.1, 0.5, 1.0]:
                    evaluate(
                        model,
                        dataset,
                        classify=True,
                        save=True,
                        plots=False,
                        subset=subset,
                        attack=overfit,
                        noise=noise,
                    )

    # DP-SGD.
    for (
        dataset,
        kl_weight,
        epsilon,
        max_grad_norm,
        latent_dim,
        convolutional,
        hyperspherical,
        overfit,
    ) in itertools.product(
        datasets,
        kl_weights,
        epsilons,
        max_grad_norms,
        latent_dims,
        is_convolutional,
        is_hyperspherical,
        is_overfit,
    ):
        subset = 0.1 if overfit else None
        args = argparse.Namespace(
            dataset=dataset,
            latent_dims=latent_dim,
            kl_weight=kl_weight,
            hyperspherical=hyperspherical,
            conditional=True,
            convolutional=convolutional,
            private=True,
            epsilon=epsilon,
            # delta=1e-5,
            max_grad_norm=max_grad_norm,
            dirdp=False,
            epochs=300 if overfit else 10,
            subset=subset,
            checkpoint=True,
            eval=False,
            # save=False,
            # plots=True,
            # classify=False,
        )
        print("TRAINING", args)
        model = train(**vars(args))
        if model is not None:
            with torch.no_grad():
                for _ in range(5):
                    evaluate(
                        model,
                        dataset,
                        classify=not overfit,
                        save=True,
                        plots=False,
                        subset=subset,
                        attack=overfit,
                    )

    # DirDP-SGD.
    for (
        dataset,
        kl_weight,
        epsilon,
        latent_dim,
        convolutional,
        hyperspherical,
        overfit,
    ) in itertools.product(
        datasets,
        kl_weights,
        epsilons,
        latent_dims,
        is_convolutional,
        is_hyperspherical,
        is_overfit,
    ):
        subset = 0.1 if overfit else None
        args = argparse.Namespace(
            dataset=dataset,
            latent_dims=latent_dim,
            kl_weight=kl_weight,
            hyperspherical=hyperspherical,
            conditional=True,
            convolutional=convolutional,
            private=True,
            epsilon=epsilon,
            # delta=1e-5,
            # max_grad_norm=max_grad_norm,
            dirdp=True,
            epochs=300 if overfit else 10,
            subset=subset,
            checkpoint=True,
            eval=False,
            # save=False,
            # plots=True,
            # classify=False,
        )
        print("TRAINING", args)
        model = train(**vars(args))
        if model is not None:
            with torch.no_grad():
                for _ in range(5):
                    evaluate(
                        model,
                        dataset,
                        classify=not overfit,
                        save=True,
                        plots=False,
                        subset=subset,
                        attack=overfit,
                    )


def compare(dataset, **kwargs):
    _plot = PlotCompare(dataset, num_classes=10, w=32, channels=1, n=8, device="cuda")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", action="store", type=int, default=1)

    subparsers = parser.add_subparsers(required=True)

    train_parser = subparsers.add_parser("train", help="train model")
    train_parser.set_defaults(func=train)
    train_parser.add_argument(
        "dataset",
        action="store",
        help="dataset to train on (mnist, fmnist, celeba)",
    )
    train_parser.add_argument(
        "--epochs",
        action="store",
        type=int,
        default=5,
        help="number of epochs to train (default: %(default)d)",
    )
    train_parser.add_argument(
        "-c",
        "--checkpoint",
        help="name of directory containing checkpoints (default: 'version_{n}')",
    )
    train_parser.add_argument(
        "-e", "--eval", action="store_true", help="evaluate at the end of training"
    )
    train_parser.add_argument(
        "-s",
        "--hyperspherical",
        action="store_true",
        dest="hyperspherical",
        help="enable hyperspherical model",
    )
    train_parser.add_argument(
        "--priv",
        "--private",
        action="store_true",
        dest="private",
        help="enable private training",
    )
    train_parser.add_argument(
        "-d",
        "--dims",
        action="store",
        type=int,
        default=3,
        dest="latent_dims",
        help="latent dimesions (default: %(default)d)",
    )
    train_parser.add_argument(
        "--dirdp",
        action="store_true",
        help="dirdp-sgd",
    )
    train_parser.add_argument("--classify", action="store_true", help="run classifier")

    eval_parser = subparsers.add_parser("eval", help="evaluate model")
    eval_parser.set_defaults(func=evaluate_from_checkpoint)
    eval_parser.add_argument("checkpoint", help="path to .ckpt file")
    eval_parser.add_argument(
        "dataset",
        action="store",
        help="dataset to test on (mnist, fmnist, celeba)",
    )
    eval_parser.add_argument(
        "-s", "--save", action="store_true", help="save sampled images"
    )
    eval_parser.add_argument("--classify", action="store_true", help="run classifier")
    eval_parser.add_argument(
        "--no-plots", action="store_false", dest="plots", help="run classifier"
    )
    eval_parser.add_argument("--attack", action="store_true", help="run attack test")
    eval_parser.add_argument(
        "--noise", action="store", type=float, default=None, help="run attack test"
    )

    exp_parser = subparsers.add_parser("exp", help="run experiments")
    exp_parser.set_defaults(func=run_experiments)

    cmp_parser = subparsers.add_parser("cmp", help="compare two models")
    cmp_parser.add_argument(
        "dataset",
        action="store",
        help="dataset to compare with (mnist, fmnist, celeba)",
    )
    cmp_parser.set_defaults(func=compare)

    args = parser.parse_args()

    print(args)

    if args.func is not None:
        seed_everything(args.seed, True)
        args.func(**vars(args))
