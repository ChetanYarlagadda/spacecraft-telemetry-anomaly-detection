"""All report figures, drawn from the result tables in reports/results/."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from telad import plotting
from telad.plotting import GRID, INK, INK_2, METHOD_COLOR, MUTED, SERIES, SPACECRAFT_LABEL

# Takeaway titles, written after reading the results.
TITLES = {
    "examples": "LSTM + dynamic threshold: 3 of 3 windows found on P-4, 2 false alarms on G-7, an 85-step anomaly missed on A-1",
    "event": "The 2018 LSTM leads; a 5x smaller LSTM is next, and a one-line persistence residual is statistically level with it",
    "point_adjust": "Point adjustment lifts random scores to F1 0.95, level with the LSTM (0.96); point-wise they score 0.23",
    "timeline": "The small LSTM detector finds 51 of 68 SMAP and 19 of 36 MSL windows, with 16 false alarms in total",
    "ablation": "Pruning keeps false alarms down: without it, SMAP false alarms rise from 14 to 126",
    "pred_error": "The small LSTM predicts worse than the 2018 model on 60% of channels",
}
LABEL_COLOR = SERIES[1]
DETECT_COLOR = SERIES[0]
FP_COLOR = SERIES[7]


def _suptitle(fig: plt.Figure, text: str, y: float = 0.995) -> None:
    fig.suptitle(text, x=0.01, y=y, ha="left", fontsize=12, fontweight="bold", color=INK)


def _spans(ax: plt.Axes, mask: np.ndarray, t: np.ndarray, color: str, alpha: float, ymin: float = 0, ymax: float = 1) -> None:
    d = np.diff(np.r_[0, mask.astype(int), 0])
    for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        ax.axvspan(t[s], t[e - 1], ymin=ymin, ymax=ymax, color=color, alpha=alpha, lw=0)


def fig_examples(series: pd.DataFrame, per_channel: pd.DataFrame, meta: dict, path: Path, margin: int = 1200) -> Path:
    """Telemetry, smoothed LSTM error, labelled windows and detections for the rule-picked example channels.

    Each panel is cropped to the labelled windows and detections plus ``margin`` timesteps on each side.
    """
    chans = meta["examples"]
    pc = per_channel[per_channel.method == "LSTM (this project)"].set_index("chan_id")
    fig = plt.figure(figsize=(10.5, 2.75 * len(chans) + 0.5))
    subs = fig.subfigures(len(chans), 1, hspace=0.04)
    for k, (sub, cid) in enumerate(zip(np.atleast_1d(subs), chans)):
        g = series[series.chan_id == cid]
        flagged = g.t[g.labelled | g.detected]
        lo, hi = max(g.t.min(), flagged.min() - margin), min(g.t.max(), flagged.max() + margin)
        g = g[(g.t >= lo) & (g.t <= hi)]
        t = g.t.to_numpy()
        ax, ax2 = sub.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [1.5, 1], "hspace": 0.08})
        for a in (ax, ax2):
            _spans(a, g.labelled.to_numpy(), t, LABEL_COLOR, 0.22)
            _spans(a, g.detected.to_numpy(), t, DETECT_COLOR, 0.95, ymin=0.0, ymax=0.09)
            a.set_xlim(t[0], t[-1])
        ax.plot(t, g.value, color=INK_2, lw=0.7)
        ax.set_ylabel("value\n(scaled)")
        ax2.plot(t, g.smoothed_error, color=DETECT_COLOR, lw=1.1)
        ax2.set_ylabel("smoothed\n|error|")
        if k == len(chans) - 1:
            ax2.set_xlabel("Test timestep")
        r = pc.loc[cid]
        sub.suptitle(f"{cid} ({r.spacecraft}): {int(r.tp)} of {int(r.n_labelled)} labelled window{'s' if r.n_labelled != 1 else ''} "
                     f"found, {int(r.fp)} false alarm{'s' if r.fp != 1 else ''}", x=0.08, ha="left", fontsize=10, color=INK)
    fig.legend(handles=[Patch(color=LABEL_COLOR, alpha=0.3, label="labelled anomaly window"),
                        Patch(color=DETECT_COLOR, label="detected sequence (strip at the bottom of each panel)"),
                        Line2D([], [], color=DETECT_COLOR, lw=1.1, label="LSTM prediction error, EWMA-smoothed")],
               loc="upper center", ncol=3, fontsize=8.5, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(TITLES["examples"], x=0.01, y=1.02, ha="left", fontsize=12, fontweight="bold", color=INK)
    plotting.save(fig, path)
    return path


def fig_event(summary: pd.DataFrame, oracle: pd.DataFrame, path: Path) -> Path:
    """F0.5 with channel-bootstrap CIs per method and spacecraft, the random-placement null and the oracle static rule."""
    methods = list(METHOD_COLOR)
    best = oracle.groupby(["method", "spacecraft"]).f05.max()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True, gridspec_kw={"wspace": 0.38})
    for ax, sc in zip(axes, ["SMAP", "MSL"]):
        g = summary[summary.spacecraft == sc].set_index("method").reindex(methods)
        y = np.arange(len(methods))[::-1]
        ax.hlines(y, g.f05_ci_low, g.f05_ci_high, color=[METHOD_COLOR[m] for m in methods], lw=2.2, alpha=0.55)
        ax.scatter(g.f05, y, color=[METHOD_COLOR[m] for m in methods], s=46, zorder=3)
        ax.scatter(g.null_f05_mean, y, marker="|", color=INK, s=90, lw=1.4, zorder=3)
        ax.scatter([best[(m, sc)] for m in methods], y, marker="D", facecolor="none", edgecolor=INK_2, s=30, lw=1, zorder=3)
        for yi, r in zip(y, g.itertuples()):
            ax.text(1.02, yi, f"P {r.precision:.2f}  R {r.recall:.2f}", va="center", fontsize=8, color=INK_2,
                    transform=ax.get_yaxis_transform())
        ax.set_xlim(0, 1)
        ax.set_title(f"{SPACECRAFT_LABEL[sc]}: {int(g.labelled.iloc[0])} labelled windows on {int(g.channels.iloc[0])} channels",
                     loc="left", fontsize=10, fontweight="normal", color=INK)
        ax.set_xlabel("Event-level F0.5")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(np.arange(len(methods))[::-1], methods)
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=MUTED, label="dynamic threshold (dot) with 95% channel-bootstrap CI"),
                        Line2D([], [], marker="|", ls="", color=INK, ms=10, mew=1.4, label="same detections at random positions (mean)"),
                        Line2D([], [], marker="D", ls="", mfc="none", mec=INK_2, label="best static top-x% rule, x picked on test labels (oracle)")],
               loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.45, -0.08))
    _suptitle(fig, TITLES["event"], y=1.03)
    fig.subplots_adjust(right=0.87)
    plotting.save(fig, path)
    return path


def fig_point_adjust(pa: pd.DataFrame, summary: pd.DataFrame, labels: pd.DataFrame, path: Path) -> Path:
    """Why point-adjusted F1 inflates: long labelled windows, and four metrics compared per method."""
    methods = list(METHOD_COLOR)
    fig, (ax0, ax) = plt.subplots(1, 2, figsize=(11.5, 4.8), gridspec_kw={"width_ratios": [1, 1.9], "wspace": 0.55})
    lab = labels.assign(length=labels.end - labels.start + 1)
    bins = np.logspace(np.log10(max(lab.length.min() * 0.8, 1)), np.log10(lab.length.max() * 1.2), 16)
    counts, _, _ = ax0.hist([lab[lab.spacecraft == sc].length for sc in ["SMAP", "MSL"]], bins=bins, stacked=True,
                            color=[SERIES[0], SERIES[2]], label=[SPACECRAFT_LABEL["SMAP"], SPACECRAFT_LABEL["MSL"]])
    ax0.set_ylim(0, np.max(counts) * 1.35)
    ax0.set_xscale("log")
    ax0.set_xlabel("Labelled window length (timesteps, log)")
    ax0.set_ylabel("Labelled windows")
    ax0.legend(fontsize=8, loc="upper right")
    share = (lab.length.sum() / labels.drop_duplicates("chan_id").test_len.sum())
    ax0.set_title(f"Median {int(lab.length.median())} steps; windows cover {share:.0%} of test data", loc="left",
                  fontsize=9.5, fontweight="normal", color=INK)

    tot = pa[pa.spacecraft == "Total"].set_index("method").reindex(methods)
    ev = summary[summary.spacecraft == "Total"].set_index("method").reindex(methods)
    metrics = [("Event-level F0.5 of the detections", ev.f05, "o", SERIES[0]),
               ("Point-wise F1 of the detections", tot.detector_point_f1, "s", SERIES[2]),
               ("Point-adjusted F1 of the detections", tot.detector_pa_f1, "^", SERIES[3]),
               ("Point-wise F1 at the best threshold of the raw score", tot.oracle_point_f1, "D", "none"),
               ("Point-adjusted F1 at the best threshold of the raw score", tot.oracle_pa_f1, "D", SERIES[1])]
    y = np.arange(len(methods))[::-1]
    lo = np.minimum(tot.detector_point_f1, tot.oracle_pa_f1)
    ax.hlines(y, lo, tot.oracle_pa_f1, color=GRID, lw=3, zorder=1)
    for name, vals, marker, color in metrics:
        if color == "none":
            ax.scatter(vals.to_numpy(), y, marker=marker, s=44, facecolor="none", edgecolor=SERIES[1], lw=1.3, label=name, zorder=3)
        else:
            ax.scatter(vals.to_numpy(), y, marker=marker, s=48, color=color, label=name, zorder=3)
    ax.set_yticks(y, methods)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Score, both spacecraft")
    ax.grid(axis="y", visible=False)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.3, -0.14), ncol=2)
    _suptitle(fig, TITLES["point_adjust"], y=1.02)
    plotting.save(fig, path)
    return path


def fig_timeline(labels: pd.DataFrame, detections: pd.DataFrame, channels: pd.DataFrame, path: Path) -> Path:
    """One row per channel: labelled windows and the LSTM detector's true and false detections."""
    det = detections[detections.method == "LSTM (this project)"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 8.6), gridspec_kw={"width_ratios": [1, 1], "wspace": 0.28})
    for ax, sc in zip(axes, ["SMAP", "MSL"]):
        ch = channels[channels.spacecraft == sc].reset_index(drop=True)
        n = len(ch)
        for i, r in ch.iterrows():
            y = n - 1 - i
            ax.hlines(y, 0, 1, color=GRID, lw=0.8)
            for lr in labels[labels.chan_id == r.chan_id].itertuples():
                ax.add_patch(plt.Rectangle((lr.start / r.test_len, y - 0.4), (lr.end - lr.start + 1) / r.test_len, 0.8,
                                           color=LABEL_COLOR, alpha=0.35, lw=0))
            for dr in det[det.chan_id == r.chan_id].itertuples():
                ax.add_patch(plt.Rectangle((dr.start / r.test_len, y - 0.18), max((dr.end - dr.start + 1) / r.test_len, 0.004), 0.36,
                                           color=DETECT_COLOR if dr.true_positive else FP_COLOR, lw=0))
        ax.set_yticks(range(n)[::-1], ch.chan_id, fontsize=6.5)
        ax.set_ylim(-0.8, n - 0.2)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Position in the test series (fraction of its length)")
        ax.grid(False)
        ax.set_title(SPACECRAFT_LABEL[sc], loc="left", fontsize=10, fontweight="normal", color=INK)
    fig.legend(handles=[Patch(color=LABEL_COLOR, alpha=0.4, label="labelled anomaly window"),
                        Patch(color=DETECT_COLOR, label="detection overlapping a labelled window"),
                        Patch(color=FP_COLOR, label="false alarm")], loc="lower center", ncol=3, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.005))
    _suptitle(fig, TITLES["timeline"])
    fig.subplots_adjust(top=0.94, bottom=0.08)
    plotting.save(fig, path)
    return path


def fig_ablation(abl: pd.DataFrame, path: Path) -> Path:
    """Precision, recall and F0.5 of thresholding variants applied to the same LSTM errors."""
    variants = list(dict.fromkeys(abl.variant))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True, gridspec_kw={"wspace": 0.32})
    y = np.arange(len(variants))[::-1]
    for ax, sc in zip(axes, ["SMAP", "MSL"]):
        g = abl[abl.spacecraft == sc].set_index("variant").reindex(variants)
        ax.hlines(y, g[["precision", "recall"]].min(axis=1), g[["precision", "recall"]].max(axis=1), color=GRID, lw=3, zorder=1)
        ax.scatter(g.precision, y, marker="s", s=44, color=SERIES[0], label="precision", zorder=3)
        ax.scatter(g.recall, y, marker="o", s=44, color=SERIES[1], label="recall", zorder=3)
        ax.scatter(g.f05, y, marker="D", s=40, facecolor="none", edgecolor=INK, lw=1.2, label="F0.5", zorder=4)
        for yi, r in zip(y, g.itertuples()):
            ax.text(1.03, yi, f"{int(r.fp)} false alarms", va="center", fontsize=8, color=INK_2, transform=ax.get_yaxis_transform())
        ax.set_xlim(0, 1)
        ax.set_xlabel("Event-level score")
        ax.grid(axis="y", visible=False)
        ax.set_title(SPACECRAFT_LABEL[sc], loc="left", fontsize=10, fontweight="normal", color=INK)
    axes[0].set_yticks(y, variants)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=3, fontsize=8.5, bbox_to_anchor=(0.5, 0.0))
    _suptitle(fig, TITLES["ablation"], y=1.03)
    fig.subplots_adjust(right=0.88)
    plotting.save(fig, path)
    return path


def fig_pred_error(channels: pd.DataFrame, path: Path) -> Path:
    """Normalised mean absolute prediction error per channel: this project's LSTM vs the 2018 reference LSTM."""
    fig, ax = plt.subplots(figsize=(7, 5.6))
    for sc, color in [("SMAP", SERIES[0]), ("MSL", SERIES[2])]:
        g = channels[channels.spacecraft == sc]
        ax.scatter(g.ref_lstm_normalized_error, g.lstm_normalized_error, s=24, color=color, alpha=0.85, label=SPACECRAFT_LABEL[sc])
    v = channels[["ref_lstm_normalized_error", "lstm_normalized_error"]].to_numpy()
    lo, hi = np.nanmin(v[v > 0]) * 0.7, np.nanmax(v) * 1.4
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls="--", label="equal error")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("2018 reference LSTM: mean |error| / range of the series")
    ax.set_ylabel("This project's LSTM: mean |error| / range")
    ax.legend(fontsize=8, loc="upper left")
    _suptitle(fig, TITLES["pred_error"], y=1.0)
    plotting.save(fig, path)
    return path


def make_all(results: Path, figs: Path) -> list[Path]:
    """Redraw every figure from the result tables."""
    plotting.apply_style()
    meta = json.loads((results / "run_metadata.json").read_text())
    summary = pd.read_csv(results / "event_results.csv")
    per_channel = pd.read_csv(results / "event_per_channel.csv")
    labels = pd.read_csv(results / "labels.csv")
    channels = pd.read_csv(results / "channels.csv")
    return [
        fig_examples(pd.read_csv(results / "example_series.csv"), per_channel, meta, figs / "01_detection_examples.png"),
        fig_event(summary, pd.read_csv(results / "oracle_static_threshold.csv"), figs / "02_event_level_results.png"),
        fig_point_adjust(pd.read_csv(results / "point_adjust.csv"), summary, labels, figs / "03_point_adjust_inflation.png"),
        fig_timeline(labels, pd.read_csv(results / "detections.csv"), channels, figs / "04_detection_timeline.png"),
        fig_ablation(pd.read_csv(results / "ablation.csv"), figs / "05_threshold_ablation.png"),
        fig_pred_error(channels, figs / "06_prediction_error.png"),
    ]
