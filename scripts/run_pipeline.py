"""Run every experiment end to end.

Writes reports/results/*.csv|json and reports/figures/*.png and logs each method to MLflow (./mlruns).
Expensive intermediates (LSTM predictions, matrix profiles and other scores, detections) are cached in
data/cache/ under keys that include their settings and a hash of the data, so re-runs only recompute what changed.

    python scripts/run_pipeline.py            # full run
    python scripts/run_pipeline.py --quick    # smoke test on 6 channels; outputs go to data/cache/smoke/
"""

from __future__ import annotations

import os

os.environ.setdefault("NUMBA_NUM_THREADS", "2")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import argparse  # noqa: E402
import ast  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from dataclasses import asdict, replace  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Callable  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402
import sklearn  # noqa: E402
import stumpy  # noqa: E402
import torch  # noqa: E402

from telad import baselines  # noqa: E402
from telad.data import (  # noqa: E402
    MIRROR_REPO, MIRROR_REVISION, PUBLISHED_RESULTS, Channel, crosscheck, deduplicate, download, load_channels, original_run_outputs,
)
from telad.evaluation import (  # noqa: E402
    detect_all,
    NO_FLATTEN, Scored, bootstrap_table, excess_over_recent, oracle_event_table, per_channel, permutation_null, pointwise_table, prepare,
    recall_by_class, summarise,
)
from telad.figures import make_all  # noqa: E402
from telad.lstm import LSTMConfig, fit_predict_cached, n_params  # noqa: E402
from telad.metrics import event_counts, prf  # noqa: E402
from telad.thresholding import NDTConfig, detect, smooth, target_matrix  # noqa: E402
from telad.tracking import init_mlflow, log_result  # noqa: E402

torch.set_num_threads(2)

CACHE = ROOT / "data" / "cache"
RAW = ROOT / "data" / "raw"
RESULTS = ROOT / "reports" / "results"
FIGS = ROOT / "reports" / "figures"

NDT = NDTConfig()  # published thresholding settings (l_s = 250 defines the evaluated stretch of every series)
LSTM = LSTMConfig()
HISTORY = 100  # every windowed baseline looks at the same 100 steps the LSTM reads
N_BOOT = 2000
N_PERM = 1000
REFERENCE = "LSTM (this project)"
METHODS = ["LSTM (this project)", "LSTM (2018 reference run)", "Persistence residual", "Rolling z-score", "Isolation Forest",
           "Matrix profile (raw)", "Matrix profile (z-norm)", "Random scores"]
ABLATIONS = {
    "Published settings": (NDT, "ndt"),
    "No pruning": (replace(NDT, prune=False), "ndt"),
    "No value-scale guards": (replace(NDT, value_guards=False), "ndt"),
    "No low-error search": (replace(NDT, inverse=False), "ndt"),
    "Static threshold (mean + 3 sd)": (NDT, "static"),
}
RESIDUAL = {"LSTM (this project)", "LSTM (2018 reference run)", "Persistence residual"}  # prediction errors: published NDT as is
TOP_FRACS = [0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2]
REFERENCE_RESULTS_URL = "https://raw.githubusercontent.com/khundman/telemanom/b0f3216947/results/2018-05-19_15.00.10.csv"
QUICK_CHANNELS = ["P-1", "A-1", "E-13", "C-2", "M-6", "T-9"]


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


class Timer:
    """Wall-clock per stage; stages flagged gpu=True add to the GPU total."""

    def __init__(self) -> None:
        self.stages: dict[str, float] = {}
        self.gpu: set[str] = set()

    def add(self, name: str, seconds: float, gpu: bool = False) -> None:
        self.stages[name] = self.stages.get(name, 0.0) + seconds
        if gpu and torch.cuda.is_available():
            self.gpu.add(name)


def data_key(c: Channel) -> str:
    return hashlib.sha1(c.train.tobytes() + c.test.tobytes()).hexdigest()[:10]


def cached_score(name: str, c: Channel, fn: Callable[[], np.ndarray]) -> np.ndarray:
    """Score array for one channel, cached under the scorer's name/settings and the data hash."""
    path = CACHE / "scores" / name / f"{c.chan_id}-{data_key(c)}.npy"
    if path.exists():
        return np.load(path)
    out = fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, out)
    return out


# --------------------------------------------------------------------------- reference reproduction


def parse_reference_log(text: str) -> pd.DataFrame:
    """Per-stream TP / FP / FN printed by the reference run's params.log."""
    rows = re.findall(r"Stream # (\d+): (\S+)\n.*?\n.*?TP: (\d+)\s+FP: (\d+)\s+FN: (\d+)", text)
    return pd.DataFrame(rows, columns=["row", "chan_id", "tp", "fp", "fn"]).astype({"row": int, "tp": int, "fp": int, "fn": int})


def reference_reproduction(root: Path, rows: list[Channel], only: set[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Re-threshold the reference run's saved smoothed errors and compare with its logged and released results.

    ``rows`` are all label rows in file order (the reference run processed them in this order); ``only`` limits the
    check to some channel ids (smoke test).
    """
    run_dir = root / "data" / "data" / "2018-05-19_15.00.10"
    ref = parse_reference_log((run_dir / "params.log").read_text())
    path = RAW / "telemanom_2018-05-19_results.csv"
    if not path.exists():
        r = requests.get(REFERENCE_RESULTS_URL, timeout=30)
        r.raise_for_status()
        path.write_bytes(r.content)
    released = pd.read_csv(path)
    out = []
    for k, c in enumerate(rows):
        if only is not None and c.chan_id not in only:
            continue
        _, e_s = original_run_outputs(root, c.chan_id)
        values = target_matrix(c.test[:, 0], NDT.l_s, LSTM.n_predictions)
        found, _ = detect(None, values, NDT, smoothed=e_s)
        found = sorted({(a + NDT.l_s, b + NDT.l_s) for a, b in found})
        ec = event_counts(found, c.anomalies)
        rel = released.iloc[k]
        assert rel.chan_id == c.chan_id and ref.chan_id[k] == c.chan_id
        rel_seqs = sorted({(int(a) + NDT.l_s, int(b) + NDT.l_s) for a, b in ast.literal_eval(rel.tp_sequences) + ast.literal_eval(rel.fp_sequences)})
        out.append({"row": k + 1, "chan_id": c.chan_id, "spacecraft": c.spacecraft, "tp": ec.tp, "fp": ec.fp, "fn": ec.fn,
                    "ref_tp": int(ref.tp[k]), "ref_fp": int(ref.fp[k]), "ref_fn": int(ref.fn[k]),
                    "counts_match": (ec.tp, ec.fp, ec.fn) == (ref.tp[k], ref.fp[k], ref.fn[k]),
                    "sequences": json.dumps(found), "ref_sequences": json.dumps(rel_seqs), "sequences_match": found == rel_seqs})
    df = pd.DataFrame(out)
    summ = []
    for sc in ["SMAP", "MSL", "Total"]:
        g = df if sc == "Total" else df[df.spacecraft == sc]
        p, r, f = prf(g.tp.sum(), g.fp.sum(), g.fn.sum())
        rp, rr, rf = prf(g.ref_tp.sum(), g.ref_fp.sum(), g.ref_fn.sum())
        pub = PUBLISHED_RESULTS.get(sc, {})
        summ.append({"spacecraft": sc, "rows": len(g), "tp": int(g.tp.sum()), "fp": int(g.fp.sum()), "fn": int(g.fn.sum()),
                     "precision": p, "recall": r, "f05": f, "ref_log_precision": rp, "ref_log_recall": rr, "ref_log_f05": rf,
                     "published_precision": pub.get("precision"), "published_recall": pub.get("recall"),
                     "channels_counts_match": int(g.counts_match.sum()), "channels_sequences_match": int(g.sequences_match.sum())})
    return df, pd.DataFrame(summ)


# --------------------------------------------------------------------------- scores


def compute_scores(scored: list[Scored], root: Path, lstm_cfg: LSTMConfig, timer: Timer) -> tuple[dict, pd.DataFrame]:
    """Raw anomaly scores on each channel's evaluated stretch, for every method."""
    scores: dict[str, dict[str, np.ndarray]] = {m: {} for m in METHODS}
    info_rows = []
    for i, s in enumerate(scored):
        c, a, b = s.chan, s.start, s.end
        y = c.test[a:b, 0]
        t0 = time.perf_counter()
        y_hat, info = fit_predict_cached(c.chan_id, c.train, c.test, lstm_cfg, CACHE)
        timer.add("lstm_train_predict_s", time.perf_counter() - t0, gpu=True)
        y_hat = y_hat[a - lstm_cfg.l_s :]
        scores["LSTM (this project)"][c.chan_id] = np.abs(y_hat - y)
        y_hat_ref, _ = original_run_outputs(root, c.chan_id)
        scores["LSTM (2018 reference run)"][c.chan_id] = np.abs(y_hat_ref - y)
        t0 = time.perf_counter()
        full = {
            "Persistence residual": cached_score("persistence", c, lambda: baselines.persistence_error(c.test)),
            "Rolling z-score": cached_score(f"zscore-h{HISTORY}", c, lambda: baselines.rolling_zscore(c.test, HISTORY)),
            "Isolation Forest": cached_score(f"iforest-h{HISTORY}", c, lambda: baselines.isolation_forest(c.train, c.test, HISTORY)),
            "Matrix profile (raw)": cached_score(f"mp-raw-m{HISTORY}", c, lambda: baselines.matrix_profile(c.train, c.test, HISTORY)),
            "Matrix profile (z-norm)": cached_score(f"mp-znorm-m{HISTORY}", c,
                                                    lambda: baselines.matrix_profile(c.train, c.test, HISTORY, normalize=True)),
            "Random scores": baselines.random_scores(len(c.test), seed=i),
        }
        timer.add("baselines_s", time.perf_counter() - t0)
        for m, v in full.items():
            scores[m][c.chan_id] = v[a:b]
        norm = lambda e: float(np.mean(e) / np.ptp(y)) if np.ptp(y) > 0 else float("nan")  # noqa: E731
        info_rows.append({
            "chan_id": c.chan_id, "spacecraft": c.spacecraft, "train_len": len(c.train), "test_len": len(c.test),
            "n_inputs": c.test.shape[1], "labelled_windows": len(c.anomalies), "labelled_share": float(c.labels.mean()),
            "train_value_constant": bool(np.ptp(c.train[:, 0]) == 0),
            "lstm_epochs": info["epochs"], "lstm_best_val_mse": info["best_val_mse"], "lstm_train_s": info["train_seconds"],
            "lstm_predict_s": info["predict_seconds"],
            "lstm_normalized_error": norm(scores["LSTM (this project)"][c.chan_id]),
            "ref_lstm_normalized_error": norm(scores["LSTM (2018 reference run)"][c.chan_id]),
        })
        if (i + 1) % 10 == 0:
            log(f"scores: {i + 1}/{len(scored)} channels")
    return scores, pd.DataFrame(info_rows)


def detections_cached(method: str, variant: str, cfg: NDTConfig, kind: str, scores: dict[str, np.ndarray],
                      scored: list[Scored], lstm_cfg: LSTMConfig) -> dict[str, list[tuple[int, int]]]:
    """Detected sequences per channel for one scorer and thresholding variant, cached as JSON.

    Prediction errors go through the published thresholding as is. Other scores are first shifted by their recent
    median (``excess_over_recent``) and skip the guards that compare errors with the telemetry's value range.
    """
    transform = "none" if method in RESIDUAL else "excess_over_recent+no_guards"
    keysrc = json.dumps([method, variant, kind, transform, asdict(cfg), lstm_cfg.key(), HISTORY,
                         [(s.chan.chan_id, data_key(s.chan)) for s in scored]], sort_keys=True)
    path = CACHE / "detections" / f"{hashlib.sha1(keysrc.encode()).hexdigest()[:16]}.json"
    if path.exists():
        return {k: [tuple(x) for x in v] for k, v in json.loads(path.read_text())["detections"].items()}
    if method not in RESIDUAL:
        scores = {k: excess_over_recent(v, cfg.batch_size * cfg.window_size) for k, v in scores.items()}
        cfg = replace(cfg, value_guards=False)
    seqs, _ = detect_all(scores, scored, cfg, method=kind)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"method": method, "variant": variant, "detections": seqs}))
    return seqs


def pick_examples(pc: pd.DataFrame) -> list[str]:
    """Three channels for the hero figure: a clean hit, a false alarm and a miss (rule-based, not hand-picked)."""
    pc = pc.assign(order=range(len(pc)))
    hit = pc[(pc.tp >= 1) & (pc.fp == 0) & (pc.fn == 0)]
    hit = hit.assign(n=hit.n_labelled).sort_values(["n", "order"], ascending=[False, True])
    fa = pc[pc.fp >= 1].sort_values(["fp", "order"], ascending=[False, True])
    miss = pc[(pc.fn >= 1) & (pc.tp == 0)].sort_values("order")
    picks: list[str] = []
    for g in (hit, fa, miss):
        for cid in g.chan_id:
            if cid not in picks:
                picks.append(cid)
                break
    return picks


def main(quick: bool) -> None:
    t_start = time.perf_counter()
    timer = Timer()
    results = RESULTS if not quick else CACHE / "smoke" / "results"
    figs = FIGS if not quick else CACHE / "smoke" / "figures"
    results.mkdir(parents=True, exist_ok=True)
    lstm_cfg = LSTM if not quick else replace(LSTM, max_epochs=2)
    n_boot, n_perm = (N_BOOT, N_PERM) if not quick else (200, 50)

    # ---- data
    root = download(RAW)
    rows = load_channels(root)
    check = crosscheck(root, rows)
    check.to_csv(results / "data_crosscheck.csv", index=False)
    log(f"data: {len(rows)} label rows; cross-check {int(check.match.sum())}/{len(check)} match")
    channels = deduplicate(rows)
    if quick:
        channels = [c for c in channels if c.chan_id in QUICK_CHANNELS]
    log(f"data: {len(channels)} distinct labelled channels, {sum(len(c.anomalies) for c in channels)} anomaly windows")

    # ---- 1. reproduce the reference run's thresholding from its saved errors
    t0 = time.perf_counter()
    repro, repro_summary = reference_reproduction(root, rows, set(QUICK_CHANNELS) if quick else None)
    timer.add("reference_reproduction_s", time.perf_counter() - t0)
    repro.to_csv(results / "reference_reproduction.csv", index=False)
    repro_summary.to_csv(results / "reference_reproduction_summary.csv", index=False)
    log(f"reference: sequences identical on {int(repro.sequences_match.sum())}/{len(repro)} rows; "
        f"counts {repro.tp.sum()}/{repro.fp.sum()}/{repro.fn.sum()} vs log {repro.ref_tp.sum()}/{repro.ref_fp.sum()}/{repro.ref_fn.sum()}")

    # ---- 2. scores
    scored = [prepare(c, NDT, lstm_cfg.n_predictions) for c in channels]
    scores, chan_info = compute_scores(scored, root, lstm_cfg, timer)
    chan_info.to_csv(results / "channels.csv", index=False)
    log(f"scores: LSTM {timer.stages.get('lstm_train_predict_s', 0):.0f}s, baselines {timer.stages.get('baselines_s', 0):.0f}s")

    # ---- 3. detection (same dynamic thresholding for every scorer)
    t0 = time.perf_counter()
    dets = {m: detections_cached(m, "Published settings", NDT, "ndt", scores[m], scored, lstm_cfg) for m in METHODS}
    pcs = {m: per_channel(dets[m], scored) for m in METHODS}
    abl_dets = {v: detections_cached(REFERENCE, v, cfg, kind, scores[REFERENCE], scored, lstm_cfg) for v, (cfg, kind) in ABLATIONS.items()}
    timer.add("thresholding_s", time.perf_counter() - t0)

    summary = pd.concat([summarise(pcs[m]).assign(method=m) for m in METHODS], ignore_index=True)
    ci, sig = bootstrap_table(pcs, REFERENCE, n_boot=n_boot)
    t0 = time.perf_counter()
    nulls = pd.concat([permutation_null(dets[m], scored, n_perm=n_perm).assign(method=m) for m in METHODS], ignore_index=True)
    timer.add("permutation_s", time.perf_counter() - t0)
    summary = summary.merge(ci, on=["method", "spacecraft"]).merge(nulls.drop(columns="_null_f05"), on=["method", "spacecraft"])
    summary["null_p_value"] = [
        (1 + int((nulls[(nulls.method == r.method) & (nulls.spacecraft == r.spacecraft)]._null_f05.iloc[0] >= r.f05).sum())) / (1 + n_perm)
        for r in summary.itertuples()
    ]
    summary = summary[["method", "spacecraft", "channels", "labelled", "tp", "fp", "fn", "precision", "recall", "f05", "f05_ci_low",
                       "f05_ci_high", "null_f05_mean", "null_f05_p95", "null_recall_mean", "null_p_value"]]
    summary.to_csv(results / "event_results.csv", index=False)
    sig.to_csv(results / "significance.csv", index=False)
    pd.concat([pcs[m].assign(method=m) for m in METHODS], ignore_index=True).to_csv(results / "event_per_channel.csv", index=False)
    det_rows = []
    for m in METHODS:
        for s in scored:
            for a, b in dets[m][s.chan.chan_id]:
                det_rows.append({"method": m, "chan_id": s.chan.chan_id, "start": a, "end": b,
                                 "true_positive": any(a <= e and s0 <= b for s0, e in s.chan.anomalies)})
    pd.DataFrame(det_rows).to_csv(results / "detections.csv", index=False)
    pd.DataFrame([{"chan_id": c.chan_id, "spacecraft": c.spacecraft, "start": a, "end": b, "class": cl, "test_len": len(c.test)}
                  for c in channels for (a, b), cl in zip(c.anomalies, c.classes)]).to_csv(results / "labels.csv", index=False)
    for r in summary[summary.spacecraft != "Total"].itertuples():
        log(f"  {r.method:28s} {r.spacecraft:5s} P {r.precision:.3f} R {r.recall:.3f} F0.5 {r.f05:.3f} (null {r.null_f05_mean:.3f})")

    abl = pd.concat([summarise(per_channel(d, scored)).assign(variant=v) for v, d in abl_dets.items()], ignore_index=True)
    abl.to_csv(results / "ablation.csv", index=False)

    # ---- 4. point adjustment and per-class recall
    t0 = time.perf_counter()
    pa = pd.concat([pointwise_table(m, dets[m], scores[m], scored) for m in METHODS], ignore_index=True)
    timer.add("pointwise_s", time.perf_counter() - t0)
    pa.to_csv(results / "point_adjust.csv", index=False)
    t0 = time.perf_counter()
    oracle = pd.concat([oracle_event_table(m, scores[m], scored, NDT, TOP_FRACS) for m in METHODS], ignore_index=True)
    timer.add("oracle_s", time.perf_counter() - t0)
    oracle.to_csv(results / "oracle_static_threshold.csv", index=False)
    pd.concat([recall_by_class(m, dets[m], scored) for m in METHODS], ignore_index=True).to_csv(results / "recall_by_class.csv", index=False)

    # ---- 5. example series for the hero figure
    examples = pick_examples(pcs[REFERENCE])
    ex_rows = []
    for s in scored:
        cid = s.chan.chan_id
        if cid not in examples:
            continue
        e_s = smooth(scores[REFERENCE][cid], NDT, flatten_start=cid not in NO_FLATTEN)
        mask = np.zeros(s.end - s.start, dtype=bool)
        for a, b in dets[REFERENCE][cid]:
            mask[max(a - s.start, 0) : min(b - s.start, len(mask) - 1) + 1] = True
        ex_rows.append(pd.DataFrame({"chan_id": cid, "t": np.arange(s.start, s.end), "value": s.chan.test[s.start : s.end, 0],
                                     "smoothed_error": e_s, "labelled": s.labels, "detected": mask}))
    pd.concat(ex_rows, ignore_index=True).round(5).to_csv(results / "example_series.csv", index=False)

    # ---- tracking + metadata
    if not quick:
        init_mlflow(ROOT)
        for m in METHODS:
            g = summary[summary.method == m].set_index("spacecraft")
            log_result(m, {"thresholding": "NDT (published settings)", "history": HISTORY, **({f"lstm_{k}": v for k, v in asdict(lstm_cfg).items()} if m == REFERENCE else {})},
                       {f"{k}_{sc}": float(g.loc[sc, k]) for sc in g.index for k in ["precision", "recall", "f05"]})
    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "quick": quick,
        "data": {"mirror": MIRROR_REPO, "revision": MIRROR_REVISION, "label_rows": len(rows), "channels": len(channels),
                 "anomaly_windows": sum(len(c.anomalies) for c in channels),
                 "evaluated_timesteps": int(sum(s.end - s.start for s in scored))},
        "ndt_config": asdict(NDT), "lstm_config": asdict(lstm_cfg), "baseline_history": HISTORY,
        "lstm_params": {"SMAP (25 inputs)": n_params(25, lstm_cfg), "MSL (55 inputs)": n_params(55, lstm_cfg)},
        "examples": examples, "n_boot": n_boot, "n_perm": n_perm,
        "timings_s": {k: round(v, 1) for k, v in timer.stages.items()},
        "lstm_gpu_s_when_trained": round(float(chan_info.lstm_train_s.sum() + chan_info.lstm_predict_s.sum()), 1),
        "gpu_stages": sorted(timer.gpu),
        "total_s": round(time.perf_counter() - t_start, 1),
        "versions": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__,
                     "scikit-learn": sklearn.__version__, "stumpy": stumpy.__version__},
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    (results / "run_metadata.json").write_text(json.dumps(meta, indent=2))
    for p in make_all(results, figs):
        log(f"figure {p.name}")
    log(f"done in {meta['total_s']:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smoke test on 6 channels")
    main(ap.parse_args().quick)
