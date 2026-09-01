import torch

from vmf_optimizer import random_VMF


def add_noise(z, model, noise):
    if noise is not None:
        if noise >= 1.0 and model.hyperspherical:
            z = random_VMF(z, torch.tensor(noise, device=z.device), z.shape)
        else:
            z += torch.normal(0, noise, z.shape, device=z.device)
    return z
