"""Run the detector over every channel and turn detections into event-level and point-wise result tables."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from telad.data import Channel
from telad.metrics import (
    EventCounts,
    best_f1,
    bootstrap_channels,
    event_counts,
    holm,
    point_adjust,
    pointwise_counts,
    prf,
    random_placement,
    to_mask,
)
from telad.thresholding import NDTConfig, detect, dilate, group_sequences, smooth, static_detect, target_matrix

Seq = tuple[int, int]
NO_FLATTEN = {"C-2"}  # the reference run skips flattening the first l_s errors for C-2 (its first anomaly is early)


@dataclass
class Scored:
    """One channel prepared for scoring: the evaluated stretch [start, end) of the test series and its targets."""

    chan: Channel
    start: int
    end: int
    values: np.ndarray  # (end - start, n_predictions) future telemetry values, used by the detector's guards

    @property
    def labels(self) -> np.ndarray:
        """Label mask over [start, end)."""
        return self.chan.labels[self.start : self.end]


def prepare(chan: Channel, cfg: NDTConfig, n_predictions: int = 10) -> Scored:
    """Evaluate every method on test indices [l_s, n - n_predictions), the stretch the reference system scored."""
    values = target_matrix(chan.test[:, 0], cfg.l_s, n_predictions)
    return Scored(chan, cfg.l_s, cfg.l_s + len(values), values)


def excess_over_recent(score: np.ndarray, window: int = 2100) -> np.ndarray:
    """max(score - median of the previous ``window`` scores, 0): a residual-like score whose normal level is zero.

    The pruning step compares sequence maxima by relative drop, which assumes scores near zero in normal operation
    (true for prediction errors). Isolation Forest, matrix-profile, z-score and random scores sit on a large offset,
    so they are shifted by their own recent median (causal; the first value uses itself) before thresholding.
    """
    s = pd.Series(np.asarray(score, dtype=float))
    med = s.shift(1).rolling(window, min_periods=1).median().fillna(s.iloc[0])
    return np.clip(s.to_numpy() - med.to_numpy(), 0, None)


def detect_all(scores: dict[str, np.ndarray], scored: list[Scored], cfg: NDTConfig,
               method: str = "ndt", static_k: float = 3.0) -> tuple[dict[str, list[Seq]], dict[str, np.ndarray]]:
    """Detected sequences (in test-array indices) and smoothed scores for every channel."""
    seqs, smoothed = {}, {}
    for s in scored:
        cid = s.chan.chan_id
        flatten = cid not in NO_FLATTEN
        if method == "static":
            found, e_s = static_detect(scores[cid], cfg, k=static_k, flatten_start=flatten)
        else:
            found, e_s = detect(scores[cid], s.values, cfg, flatten_start=flatten)
        seqs[cid] = [(a + s.start, b + s.start) for a, b in found]
        smoothed[cid] = e_s
    return seqs, smoothed


def per_channel(detections: dict[str, list[Seq]], scored: list[Scored]) -> pd.DataFrame:
    """Event-level TP / FP / FN per channel."""
    rows = []
    for s in scored:
        c = event_counts(detections[s.chan.chan_id], s.chan.anomalies)
        rows.append({"chan_id": s.chan.chan_id, "spacecraft": s.chan.spacecraft, "tp": c.tp, "fp": c.fp, "fn": c.fn,
                     "n_detected": c.n_pred, "n_labelled": len(s.chan.anomalies)})
    return pd.DataFrame(rows)


def summarise(pc: pd.DataFrame) -> pd.DataFrame:
    """Pooled precision / recall / F0.5 per spacecraft and overall."""
    rows = []
    for sc, g in [("SMAP", pc[pc.spacecraft == "SMAP"]), ("MSL", pc[pc.spacecraft == "MSL"]), ("Total", pc)]:
        p, r, f = prf(g.tp.sum(), g.fp.sum(), g.fn.sum())
        rows.append({"spacecraft": sc, "channels": len(g), "labelled": int(g.n_labelled.sum()), "tp": int(g.tp.sum()),
                     "fp": int(g.fp.sum()), "fn": int(g.fn.sum()), "precision": p, "recall": r, "f05": f})
    return pd.DataFrame(rows)


def bootstrap_table(pcs: dict[str, pd.DataFrame], reference: str, n_boot: int = 2000, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Channel-bootstrap 95% CIs of F0.5 per method and spacecraft, and paired tests of ``reference`` vs each method.

    Returns (ci table, significance table with Holm-adjusted p-values within each spacecraft).
    """
    methods = list(pcs)
    ci_rows, sig_rows = [], []
    for sc in ["SMAP", "MSL", "Total"]:
        arr = []
        for m in methods:
            g = pcs[m] if sc == "Total" else pcs[m][pcs[m].spacecraft == sc]
            arr.append(g[["tp", "fp", "fn"]].to_numpy())
        boot = bootstrap_channels(np.stack(arr), n_boot=n_boot, seed=seed)
        for j, m in enumerate(methods):
            lo, hi = np.percentile(boot[:, j], [2.5, 97.5])
            ci_rows.append({"method": m, "spacecraft": sc, "f05_ci_low": lo, "f05_ci_high": hi})
        ref = methods.index(reference)
        rows = []
        for j, m in enumerate(methods):
            if j == ref:
                continue
            d = boot[:, ref] - boot[:, j]
            p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()) + 1 / n_boot)
            g_ref = arr[ref].sum(axis=0)
            g_m = arr[j].sum(axis=0)
            rows.append({"spacecraft": sc, "comparison": f"{reference} - {m}", "diff_f05": prf(*g_ref)[2] - prf(*g_m)[2],
                         "ci_low": np.percentile(d, 2.5), "ci_high": np.percentile(d, 97.5), "p_value": p})
        adj = holm([r["p_value"] for r in rows])
        for r, a in zip(rows, adj):
            r["p_holm"] = a
        sig_rows += rows
    return pd.DataFrame(ci_rows), pd.DataFrame(sig_rows)


def permutation_null(detections: dict[str, list[Seq]], scored: list[Scored], n_perm: int = 1000, seed: int = 0) -> pd.DataFrame:
    """F0.5 when each channel's detected sequences keep their lengths but move to random positions.

    Answers: how much of the score would a detector get from flagging the same amount of time at random?
    """
    rng = np.random.default_rng(seed)
    out = []
    by_sc: dict[str, list[Scored]] = {"SMAP": [], "MSL": []}
    for s in scored:
        by_sc[s.chan.spacecraft].append(s)
    by_sc["Total"] = scored
    sims = {sc: np.zeros((n_perm, 3)) for sc in by_sc}
    for s in scored:
        n = s.end - s.start
        labs = [(a - s.start, b - s.start) for a, b in s.chan.anomalies]
        pred = [(a - s.start, b - s.start) for a, b in detections[s.chan.chan_id]]
        for r in range(n_perm):
            c: EventCounts = event_counts(random_placement(pred, n, rng), labs)
            for sc in (s.chan.spacecraft, "Total"):
                sims[sc][r] += (c.tp, c.fp, c.fn)
    for sc, sim in sims.items():
        f = np.array([prf(*row)[2] for row in sim])
        rec = np.array([prf(*row)[1] for row in sim])
        out.append({"spacecraft": sc, "null_f05_mean": f.mean(), "null_f05_p95": np.percentile(f, 95), "null_recall_mean": rec.mean(),
                    "_null_f05": f})
    return pd.DataFrame(out)


def pointwise_table(method: str, detections: dict[str, list[Seq]], raw_scores: dict[str, np.ndarray], scored: list[Scored]) -> pd.DataFrame:
    """Point-wise metrics with and without point adjustment, for the detector output and for an oracle threshold.

    Oracle rows pool all channels of a spacecraft after rank-normalising each channel's raw scores to [0, 1], then
    pick the single threshold with the best F1 on the test labels themselves (an optimistic upper bound).
    """
    rows = []
    groups = {"SMAP": [s for s in scored if s.chan.spacecraft == "SMAP"], "MSL": [s for s in scored if s.chan.spacecraft == "MSL"]}
    groups["Total"] = scored
    for sc, ss in groups.items():
        pred = np.concatenate([to_mask([(a - s.start, b - s.start) for a, b in detections[s.chan.chan_id]], s.end - s.start) for s in ss])
        lab = np.concatenate([s.labels for s in ss])
        pa = np.concatenate([point_adjust(to_mask([(a - s.start, b - s.start) for a, b in detections[s.chan.chan_id]], s.end - s.start), s.labels) for s in ss])
        _, _, f_pt = prf(*pointwise_counts(pred, lab), beta=1.0)
        _, _, f_pa = prf(*pointwise_counts(pa, lab), beta=1.0)
        ranks = np.concatenate([pd.Series(raw_scores[s.chan.chan_id]).rank(pct=True).to_numpy() for s in ss])
        f_or, _, _, _ = best_f1(ranks, lab, adjust=False)
        f_or_pa, p_or_pa, r_or_pa, _ = best_f1(ranks, lab, adjust=True)
        rows.append({"method": method, "spacecraft": sc, "detector_point_f1": f_pt, "detector_pa_f1": f_pa,
                     "oracle_point_f1": f_or, "oracle_pa_f1": f_or_pa, "oracle_pa_precision": p_or_pa, "oracle_pa_recall": r_or_pa,
                     "flagged_share": float(pred.mean()), "labelled_share": float(lab.mean())})
    return pd.DataFrame(rows)


def recall_by_class(method: str, detections: dict[str, list[Seq]], scored: list[Scored]) -> pd.DataFrame:
    """Share of labelled anomalies found, split by the dataset's 'point' / 'contextual' class."""
    rows = []
    for s in scored:
        for (a, b), cls in zip(s.chan.anomalies, s.chan.classes):
            hit = any(p[0] <= b and a <= p[1] for p in detections[s.chan.chan_id])
            rows.append({"spacecraft": s.chan.spacecraft, "class": cls, "found": hit})
    df = pd.DataFrame(rows)
    out = df.groupby(["spacecraft", "class"]).found.agg(["sum", "size"]).reset_index()
    tot = df.groupby("class").found.agg(["sum", "size"]).reset_index().assign(spacecraft="Total")
    out = pd.concat([out, tot], ignore_index=True).rename(columns={"sum": "found", "size": "labelled"})
    out["recall"] = out.found / out.labelled
    out.insert(0, "method", method)
    return out


def oracle_event_table(method: str, raw_scores: dict[str, np.ndarray], scored: list[Scored], cfg: NDTConfig,
                       top_fracs: list[float]) -> pd.DataFrame:
    """Event-level metrics of a simple static rule, swept over its one parameter: in every channel flag the top
    ``frac`` of EWMA-smoothed scores, buffer by ``error_buffer`` and group into sequences.

    Picking the best ``frac`` on the test labels is an oracle (an optimistic upper bound for a static threshold); it
    separates the quality of a score from the quality of the dynamic threshold.
    """
    smoothed = {s.chan.chan_id: smooth(raw_scores[s.chan.chan_id], cfg, flatten_start=False) for s in scored}
    rows = []
    for frac in top_fracs:
        dets = {}
        for s in scored:
            e_s = smoothed[s.chan.chan_id]
            hit = e_s > np.quantile(e_s, 1 - frac)
            found = group_sequences(np.flatnonzero(dilate(hit[None, :], cfg.error_buffer - 1)[0]))
            dets[s.chan.chan_id] = [(a + s.start, b + s.start) for a, b in found]
        rows.append(summarise(per_channel(dets, scored)).assign(method=method, top_frac=frac))
    return pd.concat(rows, ignore_index=True)
