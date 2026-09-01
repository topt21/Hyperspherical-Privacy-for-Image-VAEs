import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.decomposition import PCA

from datasets import Dataset
from models.svae import VAE
from utils import add_noise


def distance(pca, x, y):
    x = torch.tensor(pca.transform(x))
    y = torch.tensor(pca.transform(y))
    return (x - y).norm(dim=1)


def monte_carlo(
    model: VAE,
    data: Dataset,
    subset: float,
    label: torch.Tensor,
    M: int,
    n: int,
    noise: None | float = None,
):
    subset_size = int(len(data.train_data) * subset)

    # Data used in training.
    train_data = DataLoader(
        Subset(data.train_data, indices=range(subset_size)),
        batch_size=None,
        shuffle=True,
    )
    # Data not used in training.
    non_train_data = DataLoader(
        Subset(data.train_data, indices=range(subset_size, len(data.train_data))),
        batch_size=None,
        shuffle=True,
    )
    # Data used to define PCA.
    test_data = DataLoader(data.test_data, batch_size=len(data.test_data))

    xs = []
    for x, y in train_data:
        if torch.argmax(y) == torch.argmax(label):
            xs.append((True, x.flatten()))
        if len(xs) >= M:
            break
    # print("xs", len(xs), len(data.train_data), len(train_data))
    assert len(xs) == M
    for x, y in non_train_data:
        if torch.argmax(y) == torch.argmax(label):
            xs.append((False, x.flatten()))
        if len(xs) >= M * 2:
            break
    random.shuffle(xs)
    # print("xs created", len(xs))
    assert len(xs) == M * 2

    z = model.sample_latent(n)  # , noise=noise)
    z = add_noise(z, model, noise)
    gs = (
        model.decoder(z, label.expand(n, data.num_classes))  # + label_noise)
        .flatten(start_dim=1)
        .detach()
    )
    # print("gs created", gs.shape)
    assert len(gs) == n

    pca = PCA(40)
    pca.fit(next(iter(test_data))[0].flatten(start_dim=1))
    # print("pca created")

    Us = []
    for i, (train, x) in enumerate(xs):
        U_x = distance(pca, x.expand(n, -1), gs)
        # if i % 10 == 0:
        #     print(i, "done")
        Us.append((train, U_x))
    # print("Us created", Us[0][1].shape)

    eps = sorted(torch.min(U) for _, U in Us)[len(Us) // 2]
    # print("eps created", eps)

    Us_eps = [(train, sum(1 for d in U if d <= eps) / n) for train, U in Us]
    # print("Us_eps created")

    f_hats = Us_eps
    f_hats.sort(key=lambda x: x[1], reverse=True)
    # print("f_hats created")

    accuracy = sum(train for train, _ in f_hats[:M]) / M
    print("accuracy created")
    print(accuracy)

    return accuracy


def reconstruction(
    model: VAE,
    data: Dataset,
    subset: float,
    label: torch.Tensor,
    M: int,
    n: int,
    noise: None | float = None,
):
    subset_size = int(len(data.train_data) * subset)

    # Data used in training.
    train_data = DataLoader(
        Subset(data.train_data, indices=range(subset_size)),
        batch_size=None,
        shuffle=True,
    )
    # Data not used in training.
    non_train_data = DataLoader(
        Subset(data.train_data, indices=range(subset_size, len(data.train_data))),
        batch_size=None,
        shuffle=True,
    )

    xs = []
    for x, y in train_data:
        if torch.argmax(y) == torch.argmax(label):
            xs.append((True, x))
        if len(xs) >= M:
            break
    print("xs", len(xs), len(data.train_data), len(train_data))
    assert len(xs) == M
    for x, y in non_train_data:
        if torch.argmax(y) == torch.argmax(label):
            xs.append((False, x))
        if len(xs) >= M * 2:
            break
    random.shuffle(xs)
    print("xs created", len(xs))
    assert len(xs) == M * 2

    label = label.expand(n, -1)

    def f_hat(x: torch.Tensor):
        x = x.expand(n, *x.shape)
        z, _, _ = model.encoder(x, label)
        z = add_noise(z, model, noise)
        x_hat = model.decoder(z, label)
        norm = (x_hat - x).norm(dim=(2, 3))
        return -sum(norm) / n

    f_hats = [(t, f_hat(x)) for t, x in xs]
    f_hats.sort(key=lambda x: x[1], reverse=True)

    accuracy = sum(train for train, _ in f_hats[:M]) / M
    return accuracy
