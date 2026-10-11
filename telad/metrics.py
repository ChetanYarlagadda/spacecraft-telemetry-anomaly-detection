"""Event-level and point-wise detection metrics, point adjustment, bootstrap and permutation tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Seq = tuple[int, int]


@dataclass
class EventCounts:
    """Event-level counts for one channel (Hundman et al. 2018 definition)."""

    tp: int  # labelled anomalies overlapped by at least one predicted sequence
    fp: int  # predicted sequences that overlap no labelled anomaly
    fn: int  # labelled anomalies overlapped by no predicted sequence
    n_pred: int  # predicted sequences


def overlaps(a: Seq, b: Seq) -> bool:
    """True when two inclusive index ranges share at least one index."""
    return a[0] <= b[1] and b[0] <= a[1]


def event_counts(pred: list[Seq], true: list[Seq]) -> EventCounts:
    """Match predicted sequences to labelled windows by any overlap.

    A predicted sequence that overlaps several labelled windows credits all of them; several predicted sequences
    on one window count it once and are not false positives.
    """
    matched = np.zeros(len(true), dtype=bool)
    fp = 0
    for p in pred:
        hit = [i for i, t in enumerate(true) if overlaps(p, t)]
        matched[hit] = True
        fp += not hit
    tp = int(matched.sum())
    return EventCounts(tp=tp, fp=fp, fn=len(true) - tp, n_pred=len(pred))


def prf(tp: float, fp: float, fn: float, beta: float = 0.5) -> tuple[float, float, float]:
    """Precision, recall and F-beta (0 where undefined)."""
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    b2 = beta**2
    f = (1 + b2) * p * r / (b2 * p + r) if p + r else 0.0
    return p, r, f


def to_mask(seqs: list[Seq], n: int) -> np.ndarray:
    """Boolean mask of length n from inclusive sequences (clipped to the array)."""
    m = np.zeros(n, dtype=bool)
    for s, e in seqs:
        m[max(s, 0) : min(e, n - 1) + 1] = True
    return m


def mask_to_seqs(mask: np.ndarray) -> list[Seq]:
    """Inclusive (start, end) runs of True values."""
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return [(int(s), int(e) - 1) for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))]


def point_adjust(pred: np.ndarray, label: np.ndarray) -> np.ndarray:
    """Point adjustment: if any point of a labelled segment is predicted, the whole segment counts as predicted."""
    out = pred.copy()
    for s, e in mask_to_seqs(label):
        if pred[s : e + 1].any():
            out[s : e + 1] = True
    return out


def pointwise_counts(pred: np.ndarray, label: np.ndarray) -> tuple[int, int, int]:
    """Point-wise TP, FP, FN."""
    return int((pred & label).sum()), int((pred & ~label).sum()), int((~pred & label).sum())


def best_f1(scores: np.ndarray, label: np.ndarray, adjust: bool = False) -> tuple[float, float, float, float]:
    """Best point-wise F1 over all thresholds (an oracle upper bound), with or without point adjustment.

    With point adjustment, a segment is detected at threshold t when its *maximum* score is >= t, so every score
    inside a labelled segment is replaced by the segment maximum before the threshold sweep.

    Returns (f1, precision, recall, threshold).
    """
    s = np.asarray(scores, dtype=float).copy()
    if adjust:
        for a, b in mask_to_seqs(label):
            s[a : b + 1] = s[a : b + 1].max()
    order = np.argsort(-s, kind="stable")
    s_sorted, y = s[order], label[order].astype(np.int64)
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    last = np.r_[np.flatnonzero(np.diff(s_sorted) != 0), len(s_sorted) - 1]  # only cut between distinct values
    tp, fp = tp[last], fp[last]
    p = tp / (tp + fp)
    r = tp / max(int(y.sum()), 1)
    f = np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-12), 0.0)
    k = int(np.argmax(f))
    return float(f[k]), float(p[k]), float(r[k]), float(s_sorted[last[k]])


def pooled_prf(counts: list[EventCounts], beta: float = 0.5) -> tuple[float, float, float]:
    """Precision, recall, F-beta from counts summed over channels."""
    tp = sum(c.tp for c in counts)
    fp = sum(c.fp for c in counts)
    fn = sum(c.fn for c in counts)
    return prf(tp, fp, fn, beta)


def bootstrap_channels(counts: np.ndarray, n_boot: int = 2000, seed: int = 0, beta: float = 0.5) -> np.ndarray:
    """Bootstrap F-beta over channels.

    Args:
        counts: array (n_methods, n_channels, 3) of TP, FP, FN per channel.

    Returns:
        array (n_boot, n_methods) of pooled F-beta; channels are resampled jointly for all methods (paired).
    """
    rng = np.random.default_rng(seed)
    n = counts.shape[1]
    idx = rng.integers(0, n, size=(n_boot, n))
    tot = counts[:, idx, :].sum(axis=2)  # (methods, boot, 3)
    tp, fp, fn = tot[..., 0], tot[..., 1], tot[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        r = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        b2 = beta**2
        f = np.where(p + r > 0, (1 + b2) * p * r / (b2 * p + r), 0.0)
    return f.T


def holm(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (monotone, capped at 1)."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj.tolist()


def random_placement(pred: list[Seq], n: int, rng: np.random.Generator) -> list[Seq]:
    """The same predicted sequence lengths placed uniformly at random inside [0, n)."""
    out = []
    for s, e in pred:
        length = min(e, n - 1) - max(s, 0) + 1
        start = int(rng.integers(0, max(n - length, 0) + 1))
        out.append((start, start + length - 1))
    return out
