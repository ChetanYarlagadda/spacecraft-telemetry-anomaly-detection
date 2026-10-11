"""A small per-channel LSTM forecaster (PyTorch) in the style of telemanom, sized for a laptop GPU budget.

Input: the previous ``l_s`` timesteps of every column (telemetry value + one-hot command flags).
Output: the next ``n_predictions`` telemetry values; the one-step-ahead value is the prediction used for scoring.
Early stopping uses the chronologically last 20% of the training windows.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class LSTMConfig:
    """Forecaster hyperparameters (telemanom used l_s=250, two layers of 80 units, up to 35 epochs, patience 10)."""

    l_s: int = 100
    n_predictions: int = 10
    hidden: int = 32
    layers: int = 2
    dropout: float = 0.2
    batch_size: int = 64
    lr: float = 1e-3
    max_epochs: int = 20
    patience: int = 3
    min_delta: float = 3e-4
    val_frac: float = 0.2
    seed: int = 0

    def key(self) -> str:
        """Short hash of the configuration, used in cache paths."""
        return hashlib.sha1(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:10]


class Forecaster(nn.Module):
    """Stacked LSTM followed by a linear head on the last hidden state."""

    def __init__(self, n_inputs: int, cfg: LSTMConfig) -> None:
        super().__init__()
        self.lstm = nn.LSTM(n_inputs, cfg.hidden, num_layers=cfg.layers, batch_first=True,
                            dropout=cfg.dropout if cfg.layers > 1 else 0.0)
        self.head = nn.Linear(cfg.hidden, cfg.n_predictions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


def windows(arr: np.ndarray, l_s: int, n_predictions: int) -> tuple[np.ndarray, np.ndarray]:
    """Sliding windows: X (m, l_s, d) and Y (m, n_predictions) with Y = the next values of column 0."""
    m = len(arr) - l_s - n_predictions
    if m <= 0:
        raise ValueError("series shorter than l_s + n_predictions")
    x = np.lib.stride_tricks.sliding_window_view(arr[: m - 1 + l_s], (l_s, arr.shape[1]))[:, 0]
    y = np.lib.stride_tricks.sliding_window_view(arr[l_s : l_s + m - 1 + n_predictions, 0], n_predictions)
    return np.ascontiguousarray(x), np.ascontiguousarray(y)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(train_arr: np.ndarray, cfg: LSTMConfig, device: torch.device | None = None) -> tuple[Forecaster, dict]:
    """Fit one forecaster with early stopping on a chronological validation split. Returns (model, history)."""
    device = device or _device()
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    x, y = windows(train_arr, cfg.l_s, cfg.n_predictions)
    n_val = max(int(len(x) * cfg.val_frac), 1)
    xt = torch.from_numpy(x[:-n_val]).to(device)
    yt = torch.from_numpy(y[:-n_val]).to(device)
    xv = torch.from_numpy(x[-n_val:]).to(device)
    yv = torch.from_numpy(y[-n_val:]).to(device)
    model = Forecaster(x.shape[2], cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()
    best, best_state, wait, hist = np.inf, None, 0, []
    t0 = time.perf_counter()
    for epoch in range(cfg.max_epochs):
        model.train()
        order = torch.from_numpy(rng.permutation(len(xt))).to(device)
        total = 0.0
        for i in range(0, len(order), cfg.batch_size):
            b = order[i : i + cfg.batch_size]
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xt[b]), yt[b])
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(b)
        model.eval()
        with torch.no_grad():
            val = float(loss_fn(model(xv), yv))
        hist.append({"epoch": epoch + 1, "train_mse": total / len(xt), "val_mse": val})
        if val < best - cfg.min_delta:
            best, wait = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= cfg.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    info = {"epochs": len(hist), "best_val_mse": float(min(h["val_mse"] for h in hist)), "train_windows": int(len(xt)),
            "train_seconds": time.perf_counter() - t0, "history": hist}
    return model, info


@torch.no_grad()
def predict(model: Forecaster, test_arr: np.ndarray, cfg: LSTMConfig, batch: int = 2048) -> np.ndarray:
    """One-step-ahead predictions for test indices l_s .. n - n_predictions - 1."""
    device = next(model.parameters()).device
    x, _ = windows(test_arr, cfg.l_s, cfg.n_predictions)
    model.eval()
    out = [model(torch.from_numpy(x[i : i + batch]).to(device))[:, 0].float().cpu().numpy() for i in range(0, len(x), batch)]
    return np.concatenate(out)


def n_params(n_inputs: int, cfg: LSTMConfig) -> int:
    """Trainable parameters of a forecaster with ``n_inputs`` input columns."""
    return sum(p.numel() for p in Forecaster(n_inputs, cfg).parameters())


def fit_predict_cached(chan_id: str, train_arr: np.ndarray, test_arr: np.ndarray, cfg: LSTMConfig, cache: Path) -> tuple[np.ndarray, dict]:
    """Train + predict one channel, cached on disk under the config hash and a hash of the data."""
    data_key = hashlib.sha1(train_arr.tobytes() + test_arr.tobytes()).hexdigest()[:10]
    path = cache / f"lstm-{cfg.key()}" / f"{chan_id}-{data_key}.npz"
    if path.exists():
        z = np.load(path, allow_pickle=False)
        return z["y_hat"], json.loads(str(z["info"]))
    model, info = train(train_arr, cfg)
    t0 = time.perf_counter()
    y_hat = predict(model, test_arr, cfg)
    info["predict_seconds"] = time.perf_counter() - t0
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, y_hat=y_hat, info=json.dumps(info))
    return y_hat, info
