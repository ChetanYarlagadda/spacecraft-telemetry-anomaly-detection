import numpy as np
import pytest

from telad.thresholding import NDTConfig, _prune, detect, find_z, group_sequences, smooth, static_detect


def test_group_sequences_drops_single_points():
    assert group_sequences(np.array([1, 2, 3, 7, 10, 11])) == [(1, 3), (10, 11)]
    assert group_sequences(np.array([], dtype=int)) == []


def test_smooth_flattens_the_start():
    cfg = NDTConfig(l_s=10)
    e_s = smooth(np.arange(100, dtype=float), cfg)
    assert np.allclose(e_s[:10], e_s[0])
    assert e_s[50] > e_s[10]


def test_detect_finds_an_injected_error_burst_and_nothing_else():
    rng = np.random.default_rng(0)
    n = 6000
    errors = 0.02 * np.abs(rng.normal(size=n))
    errors[3000:3060] += 1.0
    values = np.sin(np.linspace(0, 60, n))[:, None]  # telemetry range ~2 for the value guards
    seqs, _ = detect(errors, values, NDTConfig())
    assert len(seqs) == 1
    s, e = seqs[0]
    assert s <= 3060 and e >= 3000


def test_value_guards_ignore_errors_tiny_relative_to_the_signal():
    rng = np.random.default_rng(1)
    n = 5000
    errors = 0.001 * np.abs(rng.normal(size=n))
    errors[2500:2550] += 0.01  # a clear relative bump, but far below 5% of the telemetry range
    values = np.sin(np.linspace(0, 40, n))[:, None]
    assert detect(errors, values, NDTConfig())[0] == []
    assert detect(errors, values, NDTConfig(value_guards=False))[0] != []


def test_prune_keeps_only_sequences_before_the_last_big_drop():
    e_s = np.zeros(100)
    e_s[5] = 1.0
    e_s[55] = 0.5
    seqs = [(0, 10), (50, 60)]
    idx = np.r_[np.arange(0, 11), np.arange(50, 61)]
    # maxima 1.0 -> 0.5 is a big drop; 0.5 -> 0.48 (largest normal value) is a small one, so the second goes
    kept = _prune(seqs, e_s, 0.48, idx, p=0.13)
    assert kept.min() == 0 and kept.max() == 10
    # if the largest normal value is far below, both sequences stay
    assert _prune(seqs, e_s, 0.1, idx, p=0.13).size == idx.size


def test_find_z_defaults_to_the_limit_on_constant_scores():
    assert find_z(np.full(500, 0.3), NDTConfig()) == NDTConfig().sd_lim


def test_static_detect_flags_values_far_above_the_mean():
    errors = np.zeros(3000)
    errors[1500:1520] = 5.0
    seqs, _ = static_detect(errors, NDTConfig(l_s=50))
    assert len(seqs) == 1 and seqs[0][0] <= 1520 and seqs[0][1] >= 1500


@pytest.mark.parametrize("n", [900, 1700])
def test_detect_handles_series_shorter_than_one_history_window(n):
    errors = np.full(n, 0.01)
    errors[n // 2 : n // 2 + 20] = 1.0
    values = np.linspace(-1, 1, n)[:, None]
    seqs, _ = detect(errors, values, NDTConfig())
    assert any(s <= n // 2 + 20 and e >= n // 2 for s, e in seqs)
