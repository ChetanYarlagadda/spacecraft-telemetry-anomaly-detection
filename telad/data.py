"""Download, checksum and load the NASA SMAP / MSL telemetry anomaly benchmark (Hundman et al., KDD 2018).

The original S3 link in the telemanom README is no longer public, so the files come from a Hugging Face mirror
of the original ``data.zip`` pinned to one commit. Every file is checked against the SHA-256 the Hub records for
it, and the labels are compared with ``labeled_anomalies.csv`` in the original GitHub repository.
"""

from __future__ import annotations

import ast
import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests

MIRROR_REPO = "appleparan/telemanom"
MIRROR_REVISION = "2d22e1061be83a88b7b9e48df35163d5147adc9d"
ORIGINAL_RUN = "2018-05-19_15.00.10"  # the reference run shipped with the original data.zip
GITHUB_LABELS_URL = "https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv"
PATTERNS = [
    "labeled_anomalies.csv",
    "LICENSE.txt",
    "data/data/train/*.npy",
    "data/data/test/*.npy",
    f"data/data/{ORIGINAL_RUN}/y_hat/*.npy",
    f"data/data/{ORIGINAL_RUN}/smoothed_errors/*.npy",
    f"data/data/{ORIGINAL_RUN}/params.log",
]

# Dataset statistics published in the telemanom README / paper (Table 1), used as a cross-check.
PUBLISHED_STATS = {
    "SMAP": {"anomaly_sequences": 69, "point": 43, "contextual": 26, "channels": 55, "evaluated_values": 429_735},
    "MSL": {"anomaly_sequences": 36, "point": 19, "contextual": 17, "channels": 27, "evaluated_values": 66_709},
}
# The published "telemetry values evaluated" count every test value after the first l_s = 250 and before the last
# n_predictions = 10, i.e. the predicted values, per label row.
EVAL_OFFSET = 260
# Event-level results of the original LSTM + dynamic thresholding system (telemanom README, default parameters).
PUBLISHED_RESULTS = {
    "SMAP": {"precision": 0.855, "recall": 0.855, "f05": 0.71},
    "MSL": {"precision": 0.926, "recall": 0.694, "f05": 0.69},
}


@dataclass
class Channel:
    """One telemetry channel: train/test arrays (column 0 = telemetry value, the rest = one-hot commands)."""

    chan_id: str
    spacecraft: str
    train: np.ndarray
    test: np.ndarray
    anomalies: list[tuple[int, int]]  # inclusive [start, end] indices into the test array
    classes: list[str]

    @property
    def labels(self) -> np.ndarray:
        """Boolean per test timestep: inside a labelled anomaly window."""
        y = np.zeros(len(self.test), dtype=bool)
        for s, e in self.anomalies:
            y[s : e + 1] = True
        return y


def sha256(path: Path) -> str:
    """Hex SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(raw_dir: Path) -> Path:
    """Download the pinned mirror snapshot into ``raw_dir`` (once) and verify every file's SHA-256.

    Returns the directory that holds ``labeled_anomalies.csv`` and ``data/``.
    """
    from huggingface_hub import HfApi, snapshot_download

    root = raw_dir / f"telemanom-{MIRROR_REVISION[:10]}"
    marker = root / ".verified"
    if marker.exists():
        return root
    snapshot_download(MIRROR_REPO, repo_type="dataset", revision=MIRROR_REVISION, allow_patterns=PATTERNS, local_dir=root)
    api = HfApi()
    expected: dict[str, str] = {}
    for folder in ["data/data/train", "data/data/test"]:
        for entry in api.list_repo_tree(MIRROR_REPO, path_in_repo=folder, repo_type="dataset", revision=MIRROR_REVISION, expand=True):
            if getattr(entry, "lfs", None) is not None:
                expected[entry.path] = entry.lfs.sha256
    bad = [p for p, digest in expected.items() if sha256(root / p) != digest]
    if bad or len(expected) != 164:
        raise RuntimeError(f"checksum mismatch for {len(bad)} files (checked {len(expected)})")
    marker.write_text(f"{len(expected)} train/test files verified against Hub SHA-256 at revision {MIRROR_REVISION}\n")
    return root


def parse_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Parse the ``anomaly_sequences`` and ``class`` columns of ``labeled_anomalies.csv`` into Python lists."""
    out = df.copy()
    out["anomaly_sequences"] = out["anomaly_sequences"].map(lambda s: [tuple(map(int, x)) for x in ast.literal_eval(s)])
    out["class"] = out["class"].map(lambda s: [c.strip() for c in s.strip("[]").split(",")])
    return out


def load_labels(root: Path) -> pd.DataFrame:
    """Labelled anomaly windows per channel, in file order (SMAP rows first, then MSL)."""
    return parse_labels(pd.read_csv(root / "labeled_anomalies.csv"))


def load_channels(root: Path, chan_ids: list[str] | None = None) -> list[Channel]:
    """Load every channel (or a subset) with its train/test arrays and labelled windows."""
    labels = load_labels(root)
    if chan_ids is not None:
        labels = labels[labels.chan_id.isin(chan_ids)]
    out = []
    for chan_id, sc, seqs, classes in zip(labels.chan_id, labels.spacecraft, labels.anomaly_sequences, labels["class"]):
        train = np.load(root / "data" / "data" / "train" / f"{chan_id}.npy").astype(np.float32)
        test = np.load(root / "data" / "data" / "test" / f"{chan_id}.npy").astype(np.float32)
        order = np.argsort([s for s, _ in seqs])
        out.append(Channel(chan_id, sc, train, test, [seqs[i] for i in order], [classes[i] for i in order]))
    return out


def merge_intervals(seqs: list[tuple[int, int]], classes: list[str]) -> tuple[list[tuple[int, int]], list[str]]:
    """Union of overlapping inclusive intervals; a merged interval keeps the class of its earliest member."""
    order = np.argsort([s for s, _ in seqs], kind="stable")
    out: list[list[int]] = []
    out_cls: list[str] = []
    for i in order:
        s, e = seqs[i]
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
            out_cls.append(classes[i])
    return [(a, b) for a, b in out], out_cls


def deduplicate(channels: list[Channel]) -> list[Channel]:
    """One entry per channel id. The labels file lists P-2 twice with overlapping windows; they are merged."""
    by_id: dict[str, Channel] = {}
    for c in channels:
        if c.chan_id in by_id:
            prev = by_id[c.chan_id]
            seqs, cls = merge_intervals(prev.anomalies + c.anomalies, prev.classes + c.classes)
            by_id[c.chan_id] = Channel(c.chan_id, c.spacecraft, c.train, c.test, seqs, cls)
        else:
            by_id[c.chan_id] = c
    return list(by_id.values())


def crosscheck(root: Path, channels: list[Channel]) -> pd.DataFrame:
    """Compare the mirror with the original GitHub labels file and the statistics published in the paper/README."""
    github = requests.get(GITHUB_LABELS_URL, timeout=30)
    github.raise_for_status()
    gh = pd.read_csv(io.StringIO(github.text))
    mirror = pd.read_csv(root / "labeled_anomalies.csv")
    key = ["chan_id", "spacecraft", "anomaly_sequences", "class", "num_values"]
    same_labels = gh[key].sort_values("chan_id").reset_index(drop=True).equals(mirror[key].sort_values("chan_id").reset_index(drop=True))
    rows = [{"check": "labeled_anomalies.csv identical to the original GitHub file", "spacecraft": "both",
             "expected": "True", "observed": str(same_labels), "match": bool(same_labels)}]
    lab = parse_labels(mirror)
    for sc, pub in PUBLISHED_STATS.items():
        chans = [c for c in channels if c.spacecraft == sc]
        sub = lab[lab.spacecraft == sc]
        classes = [c for cl in sub["class"] for c in cl]
        observed = {
            "anomaly_sequences": int(sum(len(s) for s in sub.anomaly_sequences)),
            "point": classes.count("point"),
            "contextual": classes.count("contextual"),
            "channels": len(chans),
            "evaluated_values": int(sum(len(c.test) - EVAL_OFFSET for c in chans)),
        }
        for k, v in pub.items():
            rows.append({"check": k, "spacecraft": sc, "expected": str(v), "observed": str(observed[k]), "match": observed[k] == v})
    n_mismatch = sum(int(len(c.test) != int(nv)) for c, nv in zip(channels, lab.num_values))
    rows.append({"check": "test array length equals num_values for every label row", "spacecraft": "both",
                 "expected": "0 mismatches", "observed": f"{n_mismatch} mismatches", "match": n_mismatch == 0})
    dup = sorted(lab.chan_id[lab.chan_id.duplicated()].unique())
    files = {p.stem for p in (root / "data" / "data" / "test").glob("*.npy")}
    unlabelled = sorted(files - set(lab.chan_id))
    rows.append({"check": "channel ids listed more than once (finding, not a published number)", "spacecraft": "both",
                 "expected": "none", "observed": ", ".join(dup) or "none", "match": not dup})
    rows.append({"check": "channels with data but no label row (finding, not a published number)", "spacecraft": "both",
                 "expected": "none", "observed": ", ".join(unlabelled) or "none", "match": not unlabelled})
    return pd.DataFrame(rows)


def original_run_outputs(root: Path, chan_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Predictions (y_hat) and smoothed errors saved by the original telemanom reference run for one channel."""
    base = root / "data" / "data" / ORIGINAL_RUN
    return np.load(base / "y_hat" / f"{chan_id}.npy").ravel(), np.load(base / "smoothed_errors" / f"{chan_id}.npy").ravel()
