# vMF-based differentially private SGD based on https://arxiv.org/abs/2211.04686
# Implementation of vMF-mechanism adapted from https://hal.science/hal-04004568v2

import torch
import torch.nn.functional as F
from opacus.optimizers.optimizer import _check_processed_flag, _mark_as_processed
from opacus.optimizers import DPOptimizer


def random_VMF(mu: torch.Tensor, kappa: torch.Tensor, size) -> torch.Tensor:
    mu = F.normalize(mu, dim=1)
    n = size[0]
    d = torch.tensor(size[1], device=mu.device)
    z = torch.normal(0, 1, size, device=mu.device)
    z /= torch.linalg.norm(z, dim=1, keepdim=True)
    z = z - torch.linalg.vecdot(z, mu)[:, None] * mu
    z /= torch.linalg.norm(z, dim=1, keepdim=True)
    cos = random_VMF_cos(d, kappa, n)
    sin = torch.sqrt(1 - cos**2)
    x = z * sin[:, None] + cos[:, None] * mu
    return x.reshape(mu.shape)


def random_VMF_cos(d: torch.Tensor, kappa: torch.Tensor, n: int) -> torch.Tensor:
    b = (d - 1) / (2 * kappa + (4 * kappa**2 + (d - 1) ** 2) ** 0.5)
    x0 = (1 - b) / (1 + b)
    c = kappa * x0 + (d - 1) * torch.log(1 - x0**2)
    found = 0
    out = []
    beta = torch.distributions.Beta((d - 1) / 2, (d - 1) / 2)
    exp = torch.distributions.Exponential(torch.ones(1, device=d.device).squeeze())
    while found < n:
        m = min(n, int((n - found) * 1.5))
        z = beta.sample((m,))
        t = (1 - (1 + b) * z) / (1 - (1 - b) * z)
        test = kappa * t + (d - 1) * torch.log(1 - x0 * t) - c
        accept = test >= -exp.sample((m,))
        out.append(t[accept])
        found += len(out[-1])
    return torch.cat(out)[:n]


def von_mises_fisher(grads: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    # TODO: CPU is currently faster. Remove `.detach().cpu()` to run on GPU.
    grads_flat = reshape_grads(grads).detach().cpu()
    kappa = torch.tensor(sigma, device=grads_flat.device)
    if grads_flat.shape[-1] == 1:
        grad = torch.distributions.VonMises(grads_flat, kappa).sample()
    else:
        grad = random_VMF(grads_flat, kappa, grads_flat.shape)
    return grad.to(grads.device)


def von_mises_fisher_flat(grads: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    grads_flat = grads.detach().cpu().flatten()
    kappa = torch.tensor(sigma, device=grads_flat.device)
    if grads_flat.shape[-1] == 1:
        grad = torch.distributions.VonMises(grads_flat, kappa).sample()
    else:
        grad = random_VMF(grads_flat, kappa, (1, *grads_flat.shape))
    return grad.to(grads.device)


def reshape_grads(grads: torch.Tensor) -> torch.Tensor:
    if grads.ndim == 1:
        return grads.unsqueeze(0)
    else:
        return grads.flatten(start_dim=1)


class VMFDPOptimizer(DPOptimizer):
    def add_noise(self):
        for p in self.params:
            _check_processed_flag(p.summed_grad)

            noise = von_mises_fisher(p.summed_grad, sigma=self.noise_multiplier)
            p.grad = noise.view_as(p)

            _mark_as_processed(p.summed_grad)

    def clip_and_accumulate(self):
        if len(self.grad_samples[0]) == 0:
            # Empty batch
            per_sample_clip_factor = torch.zeros(
                (0,), device=self.grad_samples[0].device
            )
        else:
            per_param_norms = [
                g.reshape(len(g), -1).norm(2, dim=-1) for g in self.grad_samples
            ]
            per_sample_norms = torch.stack(per_param_norms, dim=1).norm(2, dim=1)
            per_sample_clip_factor = self.max_grad_norm / (
                per_sample_norms + 1e-6
            )  # .clamp(max=1.0)

        for p in self.params:
            _check_processed_flag(p.grad_sample)
            grad_sample = self._get_flat_grad_sample(p)
            grad = torch.einsum("i,i...", per_sample_clip_factor, grad_sample)

            if p.summed_grad is not None:
                p.summed_grad += grad
            else:
                p.summed_grad = grad

            _mark_as_processed(p.grad_sample)
