"""Nonparametric dynamic error thresholding with pruning (Hundman et al., KDD 2018), reimplemented with NumPy.

The procedure turns a per-timestep anomaly score (originally the LSTM's absolute prediction error) into a list
of anomalous sequences:

1. Smooth the scores with an exponentially weighted moving average (span = 5% of the history window).
2. Slide a window of ``window_size`` batches of ``batch_size`` values over the smoothed scores. In each window pick
   ``epsilon = mean + z * sd`` with ``z`` in {2.5, 3, ..., 11.5} maximising

       (delta_mean / mean + delta_sd / sd) / (|E_seq|^2 + |e_a|)

   where ``delta_*`` is the drop in mean / sd when the values above ``epsilon`` are removed, ``e_a`` the
   (buffered) anomalous values and ``E_seq`` the contiguous anomalous sequences. The same is done for the scores
   mirrored around their mean, so unusually *low* errors can also be anomalous.
3. Prune: sort the sequences' maxima (plus the largest non-anomalous value) in descending order and discard every
   sequence after the last step that drops by at least ``p`` (13%).

The control flow follows the reference implementation released with the paper (first public commit of
``khundman/telemanom``), including its small quirks, so that it reproduces the published reference run exactly;
the quirks are marked ``reference behaviour`` below. The loops over points are vectorised.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class NDTConfig:
    """Parameters of the dynamic thresholding (defaults are the published ones)."""

    l_s: int = 250  # length of the history the scorer needs; the first l_s smoothed values are flattened
    batch_size: int = 70  # new values added per window step
    window_size: int = 30  # history window = window_size batches
    smoothing_perc: float = 0.05
    error_buffer: int = 100
    p: float = 0.13  # minimum percent decrease between sequence maxima for pruning
    z_min: float = 2.5
    sd_lim: float = 12.0
    prune: bool = True
    value_guards: bool = True  # skip windows / points whose errors are tiny relative to the telemetry's own range
    inverse: bool = True  # also look for unusually low scores


def smooth(errors: np.ndarray, cfg: NDTConfig, flatten_start: bool = True) -> np.ndarray:
    """EWMA-smoothed scores; the first ``l_s`` values are replaced by the mean of the first ``2 * l_s``."""
    span = int(cfg.batch_size * cfg.window_size * cfg.smoothing_perc)
    e_s = pd.Series(np.asarray(errors, dtype=float)).ewm(span=span).mean().to_numpy()
    if flatten_start:
        e_s[: cfg.l_s] = np.mean(e_s[: cfg.l_s * 2])
    return e_s


def group_sequences(idx: np.ndarray) -> list[tuple[int, int]]:
    """Group sorted unique indices into runs of consecutive integers; single-point runs are dropped."""
    idx = np.asarray(idx, dtype=int)
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) != 1)
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]]
    return [(int(s), int(e)) for s, e in zip(starts, ends) if s != e]


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Row-wise boolean dilation: out[t] is True when mask has a True within [t - radius, t + radius]."""
    m = mask.shape[-1]
    cs = np.concatenate([np.zeros(mask.shape[:-1] + (1,), dtype=np.int64), np.cumsum(mask, axis=-1)], axis=-1)
    t = np.arange(m)
    hi = np.minimum(t + radius, m - 1) + 1
    lo = np.maximum(t - radius, 0)
    return (cs[..., hi] - cs[..., lo]) > 0


def find_z(e_s: np.ndarray, cfg: NDTConfig) -> float:
    """Threshold (in standard deviations above the mean) that maximises the paper's objective for one window."""
    e_s = np.asarray(e_s, dtype=float)
    n = e_s.size
    mean, sd = e_s.mean(), e_s.std()
    zs = np.arange(cfg.z_min, cfg.sd_lim, 0.5)
    eps = mean + sd * zs[:, None]
    above = e_s[None, :] > eps
    below = e_s[None, :] < eps
    covered = dilate(above, cfg.error_buffer - 1)
    n_anom = covered.sum(axis=1)
    n_seq = (covered[:, :1].astype(int) + (np.diff(covered.astype(np.int8), axis=1) == 1).sum(axis=1, keepdims=True)).ravel()
    n_below = below.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        pm = (below * e_s).sum(axis=1) / n_below
        pv = (below * (e_s - pm[:, None]) ** 2).sum(axis=1) / n_below
        score = ((mean - pm) / mean + (sd - np.sqrt(pv)) / sd) / (n_seq**2 + n_anom)
    best, max_s = cfg.sd_lim, 0.0
    for z, s, k, a in zip(zs, score, n_seq, n_anom):
        # reference behaviour: ties go to the larger z; at most 5 sequences; under half of the window flagged
        if a > 0 and s >= max_s and k <= 5 and a < n * 0.5:
            best, max_s = float(z), float(s)
    return best


def _compare(e_s: np.ndarray, epsilon: float, guard_ok: bool, inter_range: float, n_total: int, window: int,
             flagged: np.ndarray, offset: int, cfg: NDTConfig) -> tuple[list[tuple[int, int]], np.ndarray, float]:
    """Anomalous indices of one window above ``epsilon`` (only the newest batch counts after the first window)."""
    if not guard_ok:
        return [], np.array([], dtype=int), 0.0
    n = e_s.size
    hit = e_s > epsilon
    if cfg.value_guards:
        hit &= e_s > 0.05 * inter_range
    num_to_ignore = cfg.l_s * 2
    if n_total < 2500:
        num_to_ignore = cfg.l_s
    if n_total < 1800:
        num_to_ignore = 0
    pts = np.flatnonzero(hit)
    if pts.size:
        b = np.arange(cfg.error_buffer)
        cand = np.unique(np.concatenate([pts[:, None] + b, pts[:, None] - b]).ravel())
        cand = cand[(cand >= 0) & (cand <= n)]  # reference behaviour: index n (one past the window) is allowed
        if window == 0:
            cand = cand[cand >= num_to_ignore]
        else:
            cand = cand[cand >= n - cfg.batch_size]
    else:
        cand = np.array([], dtype=int)
    # largest value that is neither flagged here nor previously (reference behaviour: offset by one batch less)
    adj = np.arange(n) + offset - cfg.batch_size
    prev = np.zeros(n, dtype=bool)
    ok = (adj >= 0) & (adj < flagged.size)
    prev[ok] = flagged[adj[ok]]
    local = np.zeros(n, dtype=bool)
    local[cand[cand < n]] = True
    rest = e_s[~prev & ~local]
    non_anom_max = float(max(0.0, rest.max())) if rest.size else 0.0
    return group_sequences(cand), cand, non_anom_max


def _prune(seqs: list[tuple[int, int]], e_s: np.ndarray, non_anom_max: float, idx: np.ndarray, p: float) -> np.ndarray:
    """Drop sequences whose maximum is within ``p`` (relative) of the next lower maximum, after the last big drop."""
    seq_max = [float(e_s[s:e].max()) for s, e in seqs if e > s]  # reference behaviour: end index exclusive
    kept_seqs = [sq for sq in seqs if sq[1] > sq[0]]
    ordered = sorted(seq_max, reverse=True)
    if non_anom_max > 0:
        ordered.append(non_anom_max)
    remove: list[int] = []
    for i in range(len(ordered) - 1):
        if (ordered[i] - ordered[i + 1]) / ordered[i] < p:
            remove.append(seq_max.index(ordered[i]))
        else:
            remove = []
    keep = [sq for j, sq in enumerate(kept_seqs) if j not in set(remove)]
    if not keep:
        return np.array([], dtype=int)
    inside = np.zeros(idx.size, dtype=bool)
    for s, e in keep:
        inside |= (idx >= s) & (idx <= e)
    return idx[inside]


def detect(errors: np.ndarray, values: np.ndarray, cfg: NDTConfig = NDTConfig(), smoothed: np.ndarray | None = None,
           flatten_start: bool = True) -> tuple[list[tuple[int, int]], np.ndarray]:
    """Run dynamic thresholding over a whole series.

    Args:
        errors: raw per-timestep scores (e.g. |y_hat - y|); ignored when ``smoothed`` is given.
        values: telemetry targets aligned with ``errors``, shape (n,) or (n, n_predictions); used by the guards.
        cfg: thresholding parameters.
        smoothed: precomputed smoothed scores (to reuse the reference run's saved errors).
        flatten_start: flatten the first ``l_s`` smoothed values (the reference skips this for channel C-2).

    Returns:
        (sequences, smoothed scores); sequences are inclusive (start, end) indices into ``errors``.
    """
    e_s = smooth(errors, cfg, flatten_start) if smoothed is None else np.asarray(smoothed, dtype=float)
    values = np.asarray(values, dtype=float)
    n = e_s.size
    window_size = cfg.window_size
    n_windows = int((n - cfg.batch_size * window_size) / cfg.batch_size)
    while n_windows < 0:
        window_size = max(window_size - 1, 1)
        n_windows = int((n - cfg.batch_size * window_size) / cfg.batch_size)
        if window_size == 1 and n_windows < 0:
            raise ValueError("series shorter than one batch")
    flagged = np.zeros(n + 1, dtype=bool)  # one extra slot: see the reference behaviour in _compare
    for i in range(1, n_windows + 2):
        start = (i - 1) * cfg.batch_size
        end = n if i == n_windows + 1 else cfg.window_size * cfg.batch_size + start  # reference: full window_size
        w = e_s[start:end]
        if w.size < 2:
            continue
        vals = values[start:end]
        hi, lo = np.percentile(vals, [95, 5])
        inter_range = hi - lo
        mean, sd = w.mean(), w.std()
        chan_sd = np.std(vals)

        def guard(scores: np.ndarray) -> bool:
            if not cfg.value_guards:
                return True
            return bool((sd > 0.05 * chan_sd or scores.max() > 0.05 * inter_range) and scores.max() > 0.05)

        z = find_z(w, cfg)
        seqs, idx, nam = _compare(w, mean + z * sd, guard(w), inter_range, n, i - 1, flagged, start, cfg)
        if cfg.prune and seqs:
            idx = _prune(seqs, w, nam, idx, cfg.p)
        new = [idx]
        if cfg.inverse:
            w_inv = 2 * mean - w
            z_inv = find_z(w_inv, cfg)
            seqs_i, idx_i, nam_i = _compare(w_inv, mean + z_inv * sd, guard(w_inv), inter_range, n, i - 1, flagged, start, cfg)
            if cfg.prune and seqs_i:
                idx_i = _prune(seqs_i, w_inv, nam_i, idx_i, cfg.p)
            new.append(idx_i)
        idx_all = np.unique(np.concatenate(new)).astype(int) + start
        flagged[idx_all[idx_all <= n]] = True
    return group_sequences(np.flatnonzero(flagged)), e_s


def static_detect(errors: np.ndarray, cfg: NDTConfig = NDTConfig(), k: float = 3.0,
                  flatten_start: bool = True) -> tuple[list[tuple[int, int]], np.ndarray]:
    """Ablation: one global threshold ``mean + k * sd`` of the smoothed scores, same buffering and grouping."""
    e_s = smooth(errors, cfg, flatten_start)
    hit = e_s > e_s.mean() + k * e_s.std()
    covered = dilate(hit[None, :], cfg.error_buffer - 1)[0]
    return group_sequences(np.flatnonzero(covered)), e_s


def target_matrix(test_values: np.ndarray, l_s: int, n_predictions: int) -> np.ndarray:
    """(n - l_s - n_predictions, n_predictions) matrix of future values, as the reference passes to the guards."""
    n = len(test_values) - l_s - n_predictions
    return np.stack([test_values[l_s + k : l_s + k + n] for k in range(n_predictions)], axis=1)
