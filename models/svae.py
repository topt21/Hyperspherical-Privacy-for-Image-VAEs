# Based on https://github.com/nicola-decao/s-vae-pytorch

import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

from opacus import PrivacyEngine
import opacus.optimizers

from .utils import conv_result_size


def get_optimizer_class(clipping: str, distributed: bool, grad_sample_mode: str = None):
    from vmf_optimizer import VMFDPOptimizer

    if clipping == "vmf":
        return VMFDPOptimizer
    return opacus.optimizers.get_optimizer_class(
        clipping, distributed, grad_sample_mode
    )


def monkey_prepare_optimizer(
    self,
    *,
    optimizer,
    noise_multiplier,
    max_grad_norm,
    expected_batch_size,
    loss_reduction,
    distributed,
    clipping,
    noise_generator=None,
    grad_sample_mode="hooks",
    **kwargs,
):
    if isinstance(optimizer, opacus.optimizers.DPOptimizer):
        optimizer = optimizer.original_optimizer

    generator = None
    if self.secure_mode:
        generator = self.secure_rng
    elif noise_generator is not None:
        generator = noise_generator

    optim_class = get_optimizer_class(
        clipping=clipping,
        distributed=distributed,
        grad_sample_mode=grad_sample_mode,
    )

    return optim_class(
        optimizer=optimizer,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
        expected_batch_size=expected_batch_size,
        loss_reduction=loss_reduction,
        generator=generator,
        secure_mode=self.secure_mode,
        **kwargs,
    )


PrivacyEngine._prepare_optimizer = monkey_prepare_optimizer

from opacus.lightning import DPDataLoader

import importlib

VonMisesFisher = getattr(
    importlib.import_module(name="s-vae-pytorch.hyperspherical_vae.distributions"),
    "VonMisesFisher",
)
HypersphericalUniform = getattr(
    importlib.import_module(name="s-vae-pytorch.hyperspherical_vae.distributions"),
    "HypersphericalUniform",
)


# NOTE: Convolutional variant uses `InstanceNorm2d` instead of `BatchNorm2d`.
# See:
#     https://stats.stackexchange.com/questions/573599/should-i-be-using-batchnorm-and-or-dropout-in-a-vae-or-gan
#     https://arxiv.org/abs/2303.00654
class Encoder(pl.LightningModule):
    def __init__(
        self,
        img_size,
        channels,
        latent_dims,
        num_classes,
        hyperspherical=False,
    ):
        super().__init__()
        self.img_size = img_size
        self.latent_dims = latent_dims
        self.num_classes = num_classes
        self.hyperspherical = hyperspherical

        self.embed_class = nn.Linear(num_classes, img_size * img_size)
        channels += 1
        self.layers = nn.Sequential(
            nn.Conv2d(channels, 32, 3, 2, 1),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(),
        )
        conv_size = conv_result_size(img_size, 3, 2, 1, n=3)
        self.fc_mean = nn.Linear(128 * conv_size * conv_size, latent_dims)
        if hyperspherical:
            self.fc_var = nn.Linear(128 * conv_size * conv_size, 1)
        else:
            self.fc_var = nn.Linear(128 * conv_size * conv_size, latent_dims)

    def forward(self, x, y):
        y = self.embed_class(y)
        y = y.view(-1, self.img_size, self.img_size).unsqueeze(1)
        x = torch.cat([x, y], 1)

        x = self.layers(x)

        x = torch.flatten(x, start_dim=1)

        z_mean = self.fc_mean(x)

        z_var = F.softplus(self.fc_var(x))

        if self.hyperspherical:
            z_mean = F.normalize(z_mean)
            z_var += 1

        q_z, p_z = self.reparameterize(z_mean, z_var)

        z = q_z.rsample()

        return z, q_z, p_z

    def reparameterize(self, z_mean, z_var):
        if self.hyperspherical:
            q_z = VonMisesFisher(z_mean, z_var)
            p_z = HypersphericalUniform(self.latent_dims - 1, device="cuda")
        else:
            q_z = torch.distributions.Normal(z_mean, z_var)
            p_z = torch.distributions.Normal(
                torch.zeros_like(z_mean), torch.ones_like(z_var)
            )

        return q_z, p_z


class Decoder(pl.LightningModule):
    def __init__(
        self,
        img_size,
        channels,
        latent_dims,
        num_classes,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.channels = channels

        self.conv_size = conv_result_size(img_size, 3, 2, 1, n=3)
        self.fc0 = nn.Linear(
            latent_dims + num_classes, 128 * self.conv_size * self.conv_size
        )

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, channels, 3, 2, 1, 1),
        )

    def forward(self, z, y):
        z = torch.cat([z, y], 1)
        z = self.fc0(z)
        z = z.view(-1, 128, self.conv_size, self.conv_size)
        x = self.layers(z)
        return x.reshape(-1, self.channels, self.img_size, self.img_size)


class VAE(pl.LightningModule):
    def __init__(
        self,
        img_size,
        channels,
        latent_dims,
        epochs,
        num_classes,
        hyperspherical=False,
        kl_weight=1.0,
        private=False,
        dirdp=False,
        epsilon=10.0,
        delta=None,
        noise_multiplier=1.0,
        max_grad_norm=1.0,
    ):
        super().__init__()
        self.hyperspherical = hyperspherical
        self.encoder = Encoder(
            img_size, channels, latent_dims, num_classes, hyperspherical
        )
        self.decoder = Decoder(img_size, channels, latent_dims, num_classes)

        self.num_classes = num_classes
        self.kl_weight = kl_weight
        self.epochs = epochs

        self.private = private
        self.dirdp = dirdp
        self.epsilon = epsilon
        self.delta = delta
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        if self.private:
            self.privacy_engine = PrivacyEngine(accountant="rdp")

        self.save_hyperparameters()

    def dist_str(self):
        return "hyperspherical" if self.hyperspherical else "gaussian"

    def name(self):
        return f"{'private ' if self.private else ''}{'conditional ' if self.num_classes > 0 else ''}{self.dist_str()} vae"

    def extra_repr(self):
        ret = [
            self.dist_str(),
            "convolutional",
            f"conditional={self.num_classes > 0}",
            f"private={self.private}",
        ]
        if self.private:
            ret.append(f"  epsilon={self.epsilon}")
            ret.append(f"  delta={self.delta}")
            ret.append(f"  noise_multiplier={self.noise_multiplier}")
            ret.append(f"  max_grad_norm={self.max_grad_norm}")
        return "\n".join(ret)

    def forward(self, x, y):
        z, *_ = self.encoder(x, y)
        return self.decoder(z, y)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        # optimizer = torch.optim.SGD(self.parameters(), lr=1.5)

        if self.private:
            self.trainer.fit_loop.setup_data()
            data_loader = self.trainer.train_dataloader
            if hasattr(self, "dp"):
                self.dp["model"].remove_hooks()
            if self.dirdp:
                dp_model, optimizer, dataloader = self.privacy_engine.make_private(
                    module=self,
                    optimizer=optimizer,
                    data_loader=data_loader,
                    noise_multiplier=self.epsilon,
                    max_grad_norm=self.max_grad_norm,
                    poisson_sampling=isinstance(data_loader, DPDataLoader),
                    clipping="vmf",
                )
            else:
                if self.delta is None:
                    self.delta = 1 / len(data_loader.dataset)
                dp_model, optimizer, dataloader = (
                    self.privacy_engine.make_private_with_epsilon(
                        module=self,
                        optimizer=optimizer,
                        data_loader=data_loader,
                        target_epsilon=self.epsilon,
                        target_delta=self.delta,
                        epochs=self.epochs,
                        max_grad_norm=self.max_grad_norm,
                        poisson_sampling=isinstance(data_loader, DPDataLoader),
                    )
                )
            self.dp = {"model": dp_model}

        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        z, q_z, p_z, *_ = self.encoder(x, y)
        x_hat = self.decoder(z, y)
        loss = self.loss(x, x_hat, q_z, p_z)
        self.log("loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def loss(self, x, x_hat, q_z, p_z):
        loss_recon = F.mse_loss(x_hat, x)
        if self.hyperspherical:
            loss_kl = torch.distributions.kl.kl_divergence(q_z, p_z).mean()
        else:
            loss_kl = torch.distributions.kl.kl_divergence(q_z, p_z).sum(-1).mean()
        return loss_recon + self.kl_weight * loss_kl

    def on_train_epoch_end(self):
        if self.private and not self.dirdp:
            eps = self.privacy_engine.get_epsilon(self.delta)
            self.log("eps", eps, on_step=False, on_epoch=True, prog_bar=True)

    def sample_latent(self, n, noise=None):
        if self.hyperspherical:
            z = HypersphericalUniform(
                self.encoder.latent_dims - 1, device=self.device
            ).sample(n)
        else:
            z = torch.randn(n, self.encoder.latent_dims, device=self.device)

        if noise is not None:
            z += torch.normal(0, noise, size=z.shape, device=self.device)

        return z
