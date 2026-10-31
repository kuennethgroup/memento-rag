"""RQ3 step 7 — pre-registered paired statistics (spec: winner rule = completeness non-inferiority vs P1 (margin 0.05)
→ correctness → faithfulness → fewer harmed; paired per question; Benjamini–Hochberg; both corpora).

Per corpus and pooled, for C, A (and P0 as reference) vs P1, on the questions judged for every arm:
  completeness (group score) · correctness (no contradiction, binary) · faithfulness (secondary: the judge shares the loop's
  verifier components) · abstention (unanswerable) — mean difference, 95 % bootstrap CI (10 000 resamples over questions),
  p (McNemar exact for binary, Wilcoxon signed-rank otherwise), BH-adjusted q; harm/help counts; latency incl. P1.
Sets: all questions (headline) and held-out (pilot questions excluded, sensitivity).
  .venv/bin/python evaluation/RQ3/eval/stats.py full   → data/rq3/runs/full/stats.md + stats.json
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import binomtest, wilcoxon

ROOT = Path(__file__).resolve().parents[3]
RUN = sys.argv[1] if len(sys.argv) > 1 else "full"
CORPORA = ("ceramics", "polymer-engineering")
ARMS = ("P0", "P1", "C", "A")
MARGIN = 0.05
RNG = np.random.default_rng(27)


def jl(p):
    return [json.loads(l) for l in open(p)] if Path(p).exists() else []


def load(corpus):
    d = ROOT / f"data/rq3/runs/{RUN}/{corpus}"
    J = {r["key"]: r for r in jl(d / "judgements.jsonl") if not r.get("error")}
    A = {r["key"]: r for r in jl(d / "answers.jsonl") if not r.get("error")}
    qids = sorted({k.split("|")[0] for k in J})
    full = [q for q in qids if all(f"{q}|{a}" in J for a in ARMS)]
    pilot = set(json.loads((ROOT / f"data/rq3/eval/pilot_qids_{corpus}.json").read_text()))
    return J, A, full, pilot


def boot_ci(d):
    d = np.asarray(d, float)
    if len(d) < 2:
        return (float("nan"), float("nan"))
    m = RNG.integers(0, len(d), (10000, len(d)))
    b = d[m].mean(1)
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def paired(J, qs, arm, metric, binary=False):
    pairs = [(J[f"{q}|P1"].get(metric), J[f"{q}|{arm}"].get(metric)) for q in qs]
    pairs = [(float(a), float(b)) for a, b in pairs if a is not None and b is not None]
    if not pairs:
        return None
    d = [b - a for a, b in pairs]
    worse, better = sum(x < 0 for x in d), sum(x > 0 for x in d)
    if binary:
        p = binomtest(better, better + worse, 0.5).pvalue if better + worse else 1.0
    else:
        nz = [x for x in d if x != 0]
        p = wilcoxon(nz).pvalue if len(nz) >= 1 and any(nz) else 1.0
    lo, hi = boot_ci(d)
    return {"n": len(d), "p1_mean": float(np.mean([a for a, _ in pairs])), "arm_mean": float(np.mean([b for _, b in pairs])),
            "diff": float(np.mean(d)), "ci": [lo, hi], "p": float(p), "worse": worse, "better": better}


def bh(ps):
    ps = np.asarray(ps, float); n = len(ps)
    order = np.argsort(ps); q = np.empty(n); prev = 1.0
    for rank, i in reversed(list(enumerate(order, 1))):
        prev = min(prev, ps[i] * n / rank); q[i] = prev
    return q


def latency(A, qs, arm):
    tot = []
    for q in qs:
        a, p1 = A.get(f"{q}|{arm}"), A.get(f"{q}|P1")
        if a and p1:
            tot.append(a["latency_s"] + (p1["latency_s"] if arm in ("C", "A") else 0))
    return (float(np.median(tot)), float(np.percentile(tot, 90))) if tot else (float("nan"), float("nan"))


METRICS = [("completeness", False, "primary"), ("corr_correct", True, "primary"), ("faithfulness", False, "secondary"),
           ("abstains", True, "unanswerable")]


PRIMARY_GROUPS = {"multihop", "control", "unanswerable"}   # pre-registered: list/compare analysed separately


def select(J, full, pilot, groupset, heldout):
    return [q for q in full if J[f"{q}|P1"].get("group") in groupset and (not heldout or q not in pilot)]


def main():
    data = {c: load(c) for c in CORPORA}
    # (set name, groups, held-out?, family)
    sets = [("primary", PRIMARY_GROUPS, False, "primary"),
            ("multihop", {"multihop"}, False, "all"), ("control", {"control"}, False, "all"),
            ("list", {"list"}, False, "all"), ("unanswerable", {"unanswerable"}, False, "all"),
            ("all incl. list", PRIMARY_GROUPS | {"list"}, False, "all"),
            ("primary, held-out", PRIMARY_GROUPS, True, "all")]
    rows = []
    for sname, groups, heldout, family in sets:
        for corpus in CORPORA + ("pooled",):
            srcs = CORPORA if corpus == "pooled" else (corpus,)
            for arm in ("C", "A", "P0"):
                for metric, binary, kind in METRICS:
                    Jall, qs = {}, []
                    for c in srcs:
                        J, A, full, pilot = data[c]
                        use = select(J, full, pilot, groups, heldout)
                        if metric == "abstains":
                            use = [q for q in use if J[f"{q}|P1"].get("group") == "unanswerable"]
                        Jall.update(J); qs += use
                    r = paired(Jall, qs, arm, metric, binary)
                    if r:
                        fam = "primary" if (family == "primary" and arm in ("C", "A")
                                            and metric in ("completeness", "corr_correct", "faithfulness")) else "all"
                        rows.append({"set": sname, "corpus": corpus, "arm": arm, "metric": metric, "kind": kind,
                                     "family": fam, **r})
    for fam in ("primary", "all"):
        idx = [i for i, r in enumerate(rows) if fam == "all" or r["family"] == "primary"]
        qs_ = bh([rows[i]["p"] for i in idx])
        for i, qq in zip(idx, qs_):
            rows[i]["q_" + fam] = float(qq)
    for r in rows:
        if r["metric"] == "completeness":
            r["noninferior"] = r["ci"][0] > -MARGIN
    lines = [f"# RQ3 {RUN} — paired statistics vs P1\n",
             f"Pre-registered: winner rule on multi-hop + control + unanswerable (\"primary\"); list/compare analysed as its "
             f"own group; completeness non-inferiority margin {MARGIN}, checked on the primary set and per group. "
             "CI = 95 % bootstrap over questions. q(primary) = Benjamini–Hochberg over the primary family (A and C × "
             "completeness, correctness, faithfulness × ceramics, polymer, pooled on the primary set); q(all) = BH over every "
             "test in this file (exploratory). Faithfulness is secondary (the judge shares the loop's verifier components).\n"]
    for sname, *_ in sets:
        lines.append(f"\n## {sname}\n")
        lines.append("| corpus | arm | metric | n | P1 | arm | Δ | 95 % CI | p | q(primary) | q(all) | worse / better | non-inferior |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            if r["set"] != sname:
                continue
            ni = ("yes" if r["noninferior"] else "**no**") if "noninferior" in r else ""
            qp = f"{r['q_primary']:.3g}" if "q_primary" in r else ""
            lines.append(f"| {r['corpus']} | {r['arm']} | {r['metric']} | {r['n']} | {r['p1_mean']:.3f} | {r['arm_mean']:.3f} | "
                         f"{r['diff']:+.3f} | [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}] | {r['p']:.3g} | {qp} | {r['q_all']:.3g} | "
                         f"{r['worse']} / {r['better']} | {ni} |")
    lines.append("\n## Latency (seconds, incl. the P1 answer for C/A; all questions; measured under load)\n")
    lines.append("| corpus | P0 median / p90 | P1 | C | A |")
    lines.append("|---|---|---|---|---|")
    for c in CORPORA:
        J, A, full, _ = data[c]
        cells = [f"{m:.0f} / {p:.0f}" for m, p in (latency(A, full, a) for a in ("P0", "P1", "C", "A"))]
        lines.append(f"| {c} | " + " | ".join(cells) + " |")
    lines.append(f"\nQuestions with every arm judged: " + ", ".join(f"{c} {len(data[c][2])}" for c in CORPORA))
    text = "\n".join(lines)
    out = ROOT / f"data/rq3/runs/{RUN}"
    (out / "stats.md").write_text(text + "\n")
    (out / "stats.json").write_text(json.dumps(rows, indent=1))
    print(text)


if __name__ == "__main__":
    main()
