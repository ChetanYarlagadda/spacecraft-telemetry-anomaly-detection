"""Baseline anomaly scores for one telemetry channel. Each returns one score per test timestep (higher = more
anomalous); the first ``history - 1`` values, where the window is incomplete, are filled with the first valid
score.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

os.environ.setdefault("NUMBA_NUM_THREADS", "2")


def _fill_head(score: np.ndarray, start: int) -> np.ndarray:
    out = np.asarray(score, dtype=float).copy()
    out[:start] = out[start]
    return out


def persistence_error(test: np.ndarray) -> np.ndarray:
    """|x[t] - x[t-1]|: the residual of the naive 'tomorrow equals today' forecaster."""
    x = test[:, 0].astype(float)
    return _fill_head(np.r_[np.nan, np.abs(np.diff(x))], 1)


def rolling_zscore(test: np.ndarray, history: int = 100, sd_floor: float = 0.01) -> np.ndarray:
    """|x[t] - mean| / max(sd, sd_floor) over the previous ``history`` values (trailing, causal)."""
    x = pd.Series(test[:, 0].astype(float))
    prev = x.shift(1).rolling(history)
    z = (x - prev.mean()).abs() / prev.std(ddof=0).clip(lower=sd_floor)
    return _fill_head(z.to_numpy(), history)


def isolation_forest(train: np.ndarray, test: np.ndarray, history: int = 100, seed: int = 0) -> np.ndarray:
    """Isolation Forest fitted on sliding windows of the training values, scored on test windows ending at t."""
    tr = np.lib.stride_tricks.sliding_window_view(train[:, 0].astype(float), history)
    te = np.lib.stride_tricks.sliding_window_view(test[:, 0].astype(float), history)
    model = IsolationForest(n_estimators=100, max_samples=256, random_state=seed, n_jobs=2).fit(tr)
    score = np.full(len(test), np.nan)
    score[history - 1 :] = -model.score_samples(te)
    return _fill_head(score, history - 1)


def matrix_profile(train: np.ndarray, test: np.ndarray, m: int = 100, normalize: bool = False) -> np.ndarray:
    """AB-join matrix profile: distance from each test subsequence (ending at t) to its nearest training subsequence.

    ``normalize=False`` uses plain Euclidean distance (stumpy.aamp); ``True`` the z-normalised distance (stumpy.stump).
    """
    import numba
    import stumpy

    numba.set_num_threads(min(2, numba.config.NUMBA_NUM_THREADS))
    a = test[:, 0].astype(np.float64)
    b = train[:, 0].astype(np.float64)
    if normalize:
        prof = stumpy.stump(a, m, T_B=b, ignore_trivial=False)[:, 0]
    else:
        prof = stumpy.aamp(a, m, T_B=b, ignore_trivial=False)[:, 0]
    score = np.full(len(test), np.nan)
    score[m - 1 :] = np.asarray(prof, dtype=float)
    score[~np.isfinite(score)] = np.nan
    head = m - 1
    if np.isnan(score[head:]).any():  # guard against non-finite distances
        score[head:] = pd.Series(score[head:]).ffill().bfill().to_numpy()
    return _fill_head(score, head)


def random_scores(n: int, seed: int) -> np.ndarray:
    """Uniform random scores: the no-skill reference."""
    return np.random.default_rng(seed).random(n)
