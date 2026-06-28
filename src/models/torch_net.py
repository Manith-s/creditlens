"""Small PyTorch MLP for tabular fraud detection (CPU-friendly).

Imbalance is handled the PyTorch-idiomatic way: BCEWithLogitsLoss with `pos_weight`
(cost-sensitive learning), not pre-split resampling. Kept intentionally small
(2-3 dense layers) so it trains on CPU in seconds on the 284K-row fraud set.

`torch` is an optional dependency — `is_available()` lets the benchmark skip this
model gracefully when torch isn't installed.
"""
from __future__ import annotations

import numpy as np


def is_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def train_torch_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    epochs: int = 12,
    batch_size: int = 2048,
    lr: float = 1e-3,
    hidden: tuple[int, ...] = (64, 32),
    random_state: int = 42,
):
    """Train and return a fitted MLP wrapper exposing .predict_proba(X)."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(random_state)
    X = np.asarray(X_train, dtype=np.float32)
    y = np.asarray(y_train, dtype=np.float32)

    # standardize (fit on train only)
    mu = X.mean(axis=0)
    sigma = X.std(axis=0) + 1e-8
    Xs = (X - mu) / sigma

    layers: list[nn.Module] = []
    in_dim = Xs.shape[1]
    for h in hidden:
        layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(0.2)]
        in_dim = h
    layers += [nn.Linear(in_dim, 1)]
    net = nn.Sequential(*layers)

    pos = max(float((y == 1).sum()), 1.0)
    neg = float((y == 0).sum())
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    ds = TensorDataset(torch.from_numpy(Xs), torch.from_numpy(y))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    net.train()
    for _ in range(epochs):
        for xb, yb in dl:
            opt.zero_grad()
            logits = net(xb).squeeze(1)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()

    return _TorchWrapper(net, mu, sigma)


class _TorchWrapper:
    def __init__(self, net, mu, sigma):
        self.net = net
        self.mu = mu
        self.sigma = sigma
        self._creditlens_backend = "pytorch"

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        import torch

        Xs = ((np.asarray(X, dtype=np.float32) - self.mu) / self.sigma).astype(np.float32)
        self.net.eval()
        with torch.no_grad():
            p = torch.sigmoid(self.net(torch.from_numpy(Xs)).squeeze(1)).numpy()
        return np.column_stack([1 - p, p])
