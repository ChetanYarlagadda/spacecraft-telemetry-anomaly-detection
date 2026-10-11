import numpy as np
import pytest

from telad.metrics import (
    best_f1,
    bootstrap_channels,
    event_counts,
    holm,
    mask_to_seqs,
    point_adjust,
    pointwise_counts,
    prf,
    random_placement,
    to_mask,
)


def test_event_counts_follow_the_overlap_rule():
    true = [(100, 200), (500, 600), (900, 950)]
    pred = [(150, 160), (170, 180), (300, 310), (590, 920)]
    c = event_counts(pred, true)
    # two hits on the first window count once; (590, 920) spans the 2nd and 3rd windows and credits both
    assert (c.tp, c.fp, c.fn, c.n_pred) == (3, 1, 0, 4)
    assert event_counts([], true).fn == 3


def test_f_beta_weights_precision_more_than_recall():
    p, r, f05 = prf(8, 2, 8)  # precision 0.8, recall 0.5
    assert p == pytest.approx(0.8) and r == pytest.approx(0.5)
    assert f05 == pytest.approx(1.25 * 0.8 * 0.5 / (0.25 * 0.8 + 0.5))
    assert f05 > prf(8, 2, 8, beta=1.0)[2]
    assert prf(0, 0, 5) == (0.0, 0.0, 0.0)


def test_mask_round_trip():
    seqs = [(2, 4), (8, 8)]
    assert mask_to_seqs(to_mask(seqs, 12)) == seqs


def test_point_adjustment_inflates_random_scores_on_long_windows():
    rng = np.random.default_rng(0)
    n = 20_000
    label = to_mask([(2000, 2999), (9000, 10999), (15000, 15999)], n)  # 20% of points, 3 long windows
    scores = rng.random(n)
    f_point, *_ = best_f1(scores, label)
    f_pa, p_pa, r_pa, _ = best_f1(scores, label, adjust=True)
    assert f_point < 0.4  # about the label share: random scores carry no information point-wise
    assert f_pa > 0.9  # a random spike inside each window marks the whole window as found
    assert r_pa == 1.0


def test_point_adjust_on_binary_predictions():
    label = to_mask([(10, 19)], 30)
    pred = np.zeros(30, dtype=bool)
    pred[12] = True
    assert pointwise_counts(pred, label) == (1, 0, 9)
    assert pointwise_counts(point_adjust(pred, label), label) == (10, 0, 0)


def test_holm_matches_hand_computation():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_bootstrap_is_paired_across_methods():
    counts = np.array([[[1, 0, 0], [0, 1, 1], [2, 0, 1]]] * 2)  # two identical methods
    boot = bootstrap_channels(counts, n_boot=50)
    assert boot.shape == (50, 2)
    assert np.allclose(boot[:, 0], boot[:, 1])


def test_random_placement_keeps_lengths_inside_the_series():
    rng = np.random.default_rng(3)
    out = random_placement([(0, 9), (50, 149)], 400, rng)
    assert [e - s for s, e in out] == [9, 99]
    assert all(0 <= s and e < 400 for s, e in out)


def test_summary_pools_counts_over_channels():
    from telad.data import Channel
    from telad.evaluation import Scored, per_channel, summarise

    arr = np.zeros((1000, 2), dtype=np.float32)
    chans = [Channel("A-1", "SMAP", arr, arr, [(300, 350)], ["point"]),
             Channel("M-1", "MSL", arr, arr, [(400, 420), (700, 800)], ["point", "contextual"])]
    scored = [Scored(c, 250, 990, np.zeros((740, 10))) for c in chans]
    det = {"A-1": [(320, 330), (600, 650)], "M-1": [(750, 760)]}
    pc = per_channel(det, scored)
    assert pc[["tp", "fp", "fn"]].to_numpy().tolist() == [[1, 1, 0], [1, 0, 1]]
    tot = summarise(pc).set_index("spacecraft").loc["Total"]
    assert (tot.tp, tot.fp, tot.fn) == (2, 1, 1)
    assert tot.precision == pytest.approx(2 / 3) and tot.recall == pytest.approx(2 / 3)
