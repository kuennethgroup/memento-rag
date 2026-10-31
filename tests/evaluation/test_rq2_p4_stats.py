# tests/evaluation/test_rq2_p4_stats.py
import math

import pytest

from evaluation.RQ2.harness import stats


def test_mcnemar_and_sign_test():
    assert stats.mcnemar_exact(0, 0) == 1.0
    assert stats.mcnemar_exact(9, 1) == pytest.approx(0.021484375)
    assert stats.sign_test(9, 1) == pytest.approx(0.021484375)


def test_wilcoxon_handles_all_ties():
    assert stats.wilcoxon_paired([1, 1, 1], [1, 1, 1])["p"] == 1.0
    assert stats.wilcoxon_paired([1, 2, 3, 4, 5, 6, 7, 8], [0] * 8)["p"] < 0.05


def test_bootstrap_ci_contains_mean():
    lo, hi = stats.bootstrap_ci([0.1] * 20 + [0.3] * 20, seed=1)
    assert lo <= 0.2 <= hi


def test_bh_is_monotone():
    adj = stats.bh({"h1": 0.01, "h2": 0.04, "h3": 0.03})
    assert adj["h1"] == pytest.approx(0.03) and adj["h2"] == pytest.approx(0.04) and adj["h3"] == pytest.approx(0.04)


def test_tost_equivalent_and_not():
    assert stats.tost_paired([0.0, 0.01, -0.01] * 20, 0.03) < 0.05
    assert stats.tost_paired([0.2, 0.25, 0.15] * 20, 0.03) > 0.5


def test_mcnemar_n_matches_formula():
    n = stats.mcnemar_n(0.20, 0.05)
    pd, d = 0.25, 0.15
    expect = math.ceil(((1.959964 * math.sqrt(pd) + 0.841621 * math.sqrt(pd - d * d)) / d) ** 2)
    assert n == expect


def test_tost_with_zero_variance():
    # control pairwise under the real router is almost all ties: zero variance must not give NaN
    assert stats.tost_paired([0.0] * 30, 0.03) == 0.0
    assert stats.tost_paired([0.05] * 30, 0.03) == 1.0
