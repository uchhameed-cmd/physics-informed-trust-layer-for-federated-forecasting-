"""Day-ahead forecaster: MLP mapping 24 h history + 24 h physics envelope + calendar
to the next 24 h of load (per unit of the home's training mean) and PV (kW per kW)."""
import numpy as np
import torch
import torch.nn as nn

from .data import H, LAG

IN_DIM = 2 * LAG + H + 5 + 2 * H
OUT_DIM = 2 * H


class Forecaster(nn.Module):
    def __init__(self, hidden: int = 96):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(IN_DIM, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, OUT_DIM))

    def forward(self, x):
        # residual form: start from the previous week's mean daily profile of load and PV
        # (last 48 inputs) and learn a correction
        return x[:, -2 * H:] + self.net(x)


def get_vec(m: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in m.parameters()]).clone()


def set_vec(m: nn.Module, v: torch.Tensor):
    i = 0
    with torch.no_grad():
        for p in m.parameters():
            n = p.numel(); p.copy_(v[i:i + n].view_as(p)); i += n


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def local_train(model: nn.Module, X: torch.Tensor, Y: torch.Tensor, epochs: int = 1,
                lr: float = 1e-4, batch: int = 128, gen: torch.Generator | None = None):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(X)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, generator=gen)
        for k in range(0, n, batch):
            idx = perm[k:k + batch]
            opt.zero_grad()
            loss = ((model(X[idx]) - Y[idx]) ** 2).mean()
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def predict(model: nn.Module, X: torch.Tensor) -> np.ndarray:
    model.eval()
    return model(X).numpy()
