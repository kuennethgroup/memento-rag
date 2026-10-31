"""Paired tests, CIs, multiplicity and power. scipy only."""
from __future__ import annotations

import math

import numpy as np
from scipy import stats as st


def wilcoxon_paired(x, y):
    d = np.asarray(x, float) - np.asarray(y, float)
    if not np.any(d):
        return {"n": len(d), "stat": 0.0, "p": 1.0}
    r = st.wilcoxon(d, zero_method="zsplit", alternative="two-sided")
    return {"n": len(d), "stat": float(r.statistic), "p": float(r.pvalue)}


def mcnemar_exact(b, c):
    return 1.0 if b + c == 0 else float(st.binomtest(b, b + c, 0.5).pvalue)


def sign_test(wins, losses):
    return mcnemar_exact(wins, losses)


def bootstrap_ci(diffs, *, n=10000, seed=0, alpha=0.05):
    d = np.asarray(diffs, float)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def bh(pvals):
    keys = sorted(pvals, key=pvals.get)
    m, out, prev = len(keys), {}, 1.0
    for rank in range(m, 0, -1):
        k = keys[rank - 1]
        prev = min(prev, pvals[k] * m / rank)
        out[k] = prev
    return out


def tost_paired(diffs, margin):
    d = np.asarray(diffs, float)
    lo = st.ttest_1samp(d, -margin, alternative="greater").pvalue
    hi = st.ttest_1samp(d, margin, alternative="less").pvalue
    return float(max(lo, hi))


def mcnemar_n(p10, p01, *, alpha=0.05, power=0.8):
    pd, delta = p10 + p01, p10 - p01
    za, zb = st.norm.ppf(1 - alpha / 2), st.norm.ppf(power)
    return math.ceil(((za * math.sqrt(pd) + zb * math.sqrt(pd - delta ** 2)) / abs(delta)) ** 2)
