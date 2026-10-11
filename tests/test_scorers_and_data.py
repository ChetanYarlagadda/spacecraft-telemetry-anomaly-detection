from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from telad import baselines
from telad.data import Channel, deduplicate, merge_intervals, parse_labels
from telad.evaluation import excess_over_recent
from telad.lstm import LSTMConfig, predict, train, windows


def test_rolling_zscore_peaks_at_a_level_shift():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.05, 1000)
    x[600:] += 1.0
    z = baselines.rolling_zscore(x[:, None], history=100)
    assert int(np.argmax(z)) == 600
    assert np.median(z[200:590]) < 2


def test_isolation_forest_cannot_score_constant_training_data():
    train = np.zeros((500, 1))
    test = np.r_[np.zeros(300), np.ones(50), np.zeros(300)][:, None]
    s = baselines.isolation_forest(train, test, history=20)
    assert np.allclose(s, s[0])  # every window looks equally normal: nothing to split on


def test_isolation_forest_flags_an_unseen_pattern():
    rng = np.random.default_rng(1)
    train = rng.normal(0, 0.1, (2000, 1))
    test = rng.normal(0, 0.1, (1000, 1))
    test[500:520, 0] += 3.0
    s = baselines.isolation_forest(train, test, history=20)
    assert 495 <= int(np.argmax(s)) <= 540


def test_excess_over_recent_removes_a_constant_offset():
    s = np.full(3000, 0.5)
    s[2000:2010] = 0.9
    x = excess_over_recent(s, window=500)
    assert np.allclose(x[:2000], 0)
    assert x[2005] == pytest.approx(0.4)


def test_lstm_windows_align_inputs_and_targets():
    arr = np.stack([np.arange(50, dtype=np.float32), np.zeros(50, dtype=np.float32)], axis=1)
    x, y = windows(arr, l_s=5, n_predictions=3)
    assert x.shape == (42, 5, 2) and y.shape == (42, 3)
    assert x[7, -1, 0] == 11 and y[7, 0] == 12 and y[7, -1] == 14


def test_lstm_learns_a_sine_wave_on_cpu():
    torch.manual_seed(0)
    t = np.arange(1200, dtype=np.float32)
    arr = np.stack([np.sin(t / 8), np.zeros_like(t)], axis=1).astype(np.float32)
    cfg = LSTMConfig(l_s=20, n_predictions=2, hidden=16, layers=1, lr=1e-2, batch_size=32, max_epochs=15, patience=15)
    model, info = train(arr[:900], cfg, device=torch.device("cpu"))
    assert info["history"][-1]["val_mse"] < info["history"][0]["val_mse"]
    y_hat = predict(model, arr[900:], cfg)
    target = arr[900 + cfg.l_s : 900 + cfg.l_s + len(y_hat), 0]
    assert len(y_hat) == 300 - 20 - 2
    assert np.mean((y_hat - target) ** 2) < 0.05


def test_labels_parse_and_duplicates_merge():
    df = parse_labels(pd.DataFrame({"chan_id": ["P-2", "P-2"], "spacecraft": ["SMAP"] * 2,
                                    "anomaly_sequences": ["[[5350, 6575]]", "[[5300, 6420]]"],
                                    "class": ["[point]", "[point]"], "num_values": [8209, 8209]}))
    assert df.anomaly_sequences[0] == [(5350, 6575)] and df["class"][1] == ["point"]
    arr = np.zeros((10, 2), dtype=np.float32)
    chans = [Channel("P-2", "SMAP", arr, arr, s, c) for s, c in zip(df.anomaly_sequences, df["class"])]
    out = deduplicate(chans)
    assert len(out) == 1 and out[0].anomalies == [(5300, 6575)]
    assert merge_intervals([(1, 5), (10, 12), (4, 8)], ["a", "b", "c"]) == ([(1, 8), (10, 12)], ["a", "b"])


DATA = Path(__file__).resolve().parents[1] / "data" / "raw" / "telemanom-2d22e1061b"


@pytest.mark.skipif(not DATA.exists(), reason="raw data not downloaded (run scripts/run_pipeline.py)")
def test_thresholding_reproduces_the_reference_run_on_saved_errors():
    """The 2018 reference run found (1881, 2100) and (4270, 4340) as hits and (2940, 3010) as a false alarm on P-1."""
    from telad.data import load_channels, original_run_outputs
    from telad.thresholding import NDTConfig, detect, target_matrix

    c = load_channels(DATA, ["P-1"])[0]
    _, e_s = original_run_outputs(DATA, "P-1")
    seqs, _ = detect(None, target_matrix(c.test[:, 0], 250, 10), NDTConfig(), smoothed=e_s)
    assert seqs == [(1881, 2100), (2940, 3010), (4270, 4340)]
