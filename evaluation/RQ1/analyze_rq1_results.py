#!/usr/bin/env python3
"""
evaluation/RQ1/analyze_rq1_results.py — statistical analysis of rq1_eval.py's
per-query results: per-condition tables (overall and by target_type), paired
significance tests for the three real RQ1 contrasts, and summary plots.

Contrasts (paired by query — same 125 queries, different retrieval setting):
  1. VLM vs OCR              — at each retrieval mode (dense / hybrid / hybrid+rerank)
  2. multimodal vs text-only — VLM-hybrid+rerank vs VLM-hybrid+rerank-textonly
  3. dense vs hybrid vs +rerank — within each collection

Paired t-test + Cohen's d (dz, paired-difference standardized) on MRR and
each Hit@k. Hit@k is binary per query, so the t-test is the same large-
sample-normal-approximation used throughout the IR literature for this —
McNemar's test is the textbook-exact alternative for binary paired data and
is reported alongside as a cross-check, not a replacement.

Multiple comparisons: every paired-t-test p-value computed across the whole
run (all contrasts, all metrics, all subgroups) is collected and
Benjamini-Hochberg (FDR) corrected at the end — see `_ALL_TESTS` and
`benjamini_hochberg()`. Both raw and BH-adjusted p are reported; raw p is
never removed, so a reader can see exactly what correction changed.

Usage:
    python evaluation/RQ1/analyze_rq1_results.py --in evaluation/RQ1/rq1_results.json \
        --out-dir evaluation/rq1_plots
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

K_VALUES = [1, 5, 10, 20]
# Recall@k (added alongside hit@k/mrr): Hit@k and MRR both stop at the FIRST
# relevant page found, so they saturate near 1.0 once retrieval is "good
# enough" for the ~64% of queries with a single relevant page — Recall@k
# (found-in-top-k / total-relevant) keeps discriminating for the
# multi-relevant-page queries even after Hit@k has ceilinged out. See
# rq1_eval.py's score_pages().
METRICS = ([f"hit@{k}" for k in K_VALUES] + ["mrr"] + [f"recall@{k}" for k in K_VALUES]
           + ["ndcg@5", "ndcg@10"])
# Pre-declared primary family (v4): overall (not subgroup) tests on these metrics. Hit@5+ saturates on the reranked
# conditions (ceramics ~0.98), so only rank-sensitive metrics still separate them. Everything else (other metrics,
# per-type / per-language subgroups) is exploratory.
PRIMARY_METRICS = {"hit@1", "mrr", "ndcg@10"}

# Shared modern figure style + palette.
# Categorical hues are the validated dataviz reference slots, assigned in fixed
# order; VLM/OCR/Hybrid and multimodal/text-only get stable identity colors.
_CYCLE = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#eda100",
          "#e34948", "#008300", "#e87ba4"]
PALETTE = {
    "VLM": "#2a78d6", "OCR": "#eb6834", "Hybrid": "#1baf7a",
    "multimodal": "#2a78d6", "text-only": "#4a3aa7",
}
_STYLE_APPLIED = False

# Every paired-t-test p-value from every report_contrast() call this run,
# for the end-of-run Benjamini-Hochberg pass. Reset per invocation (module
# is re-imported fresh each `python analyze_rq1_results.py` run).
_ALL_TESTS: list[dict] = []


def benjamini_hochberg(pvals: list[float]) -> list[float]:
    """Standard BH step-up FDR adjustment. Returns adjusted p-values in the
    same order as the input (not sorted)."""
    n = len(pvals)
    if n == 0:
        return []
    arr = np.asarray(pvals, dtype=float)
    order = np.argsort(arr)
    sorted_p = arr[order]
    ranks = np.arange(1, n + 1)
    adj_sorted = sorted_p * n / ranks
    # enforce monotonicity (step-up): adjusted p must not decrease going
    # from the largest raw p down to the smallest
    adj_sorted = np.minimum.accumulate(adj_sorted[::-1])[::-1]
    adj_sorted = np.clip(adj_sorted, 0, 1)
    out = np.empty(n)
    out[order] = adj_sorted
    return out.tolist()


def _apply_modern_style():
    """Light, clean, thesis-ready matplotlib defaults — applied once."""
    global _STYLE_APPLIED
    import matplotlib.pyplot as plt
    from cycler import cycler
    if _STYLE_APPLIED:
        return
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True,
        "grid.color": "#e6e5df", "grid.linewidth": 0.9,
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.titlesize": 13.5, "axes.titleweight": "bold", "axes.titlepad": 12,
        "axes.titlecolor": "#0b0b0b", "axes.labelcolor": "#52514e",
        "axes.labelsize": 11,
        "xtick.color": "#898781", "ytick.color": "#898781",
        "xtick.labelcolor": "#52514e", "ytick.labelcolor": "#52514e",
        "text.color": "#0b0b0b", "legend.frameon": False, "legend.fontsize": 9,
        "figure.dpi": 150, "axes.prop_cycle": cycler(color=_CYCLE),
    })
    _STYLE_APPLIED = True


def load(path: str) -> list[dict]:
    return json.loads(Path(path).read_text())["per_query"]


def condition_means(per_query: list[dict], condition: str) -> dict:
    out = {}
    for m in METRICS:
        vals = [r["conditions"][condition][m] for r in per_query]
        out[m] = float(np.mean(vals)) if vals else float("nan")
    out["n"] = len(per_query)
    return out


def print_table(per_query: list[dict], conditions: list[str], title: str):
    print(f"\n{'='*90}\n  {title}  (n={len(per_query)})\n{'='*90}")
    header = f"  {'Condition':<30}" + "".join(f"{m:>10}" for m in METRICS)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for cond in conditions:
        means = condition_means(per_query, cond)
        row = f"  {cond:<30}" + "".join(f"{means[m]:>10.3f}" for m in METRICS)
        print(row)


def paired_test(per_query: list[dict], cond_a: str, cond_b: str, metric: str):
    a = np.array([r["conditions"][cond_a][metric] for r in per_query])
    b = np.array([r["conditions"][cond_b][metric] for r in per_query])
    diff = a - b
    n = len(diff)
    if n < 2 or np.allclose(diff, 0):
        return {"mean_a": float(a.mean()), "mean_b": float(b.mean()),
                "t": 0.0, "p": 1.0, "cohens_dz": 0.0, "n": n}
    t, p = stats.ttest_rel(a, b)
    sd = diff.std(ddof=1)
    dz = diff.mean() / sd if sd > 0 else 0.0
    return {"mean_a": float(a.mean()), "mean_b": float(b.mean()),
            "t": float(t), "p": float(p), "cohens_dz": float(dz), "n": n}


def mcnemar(per_query: list[dict], cond_a: str, cond_b: str, metric: str):
    """Exact-binomial McNemar cross-check for binary hit@k contrasts."""
    a = np.array([r["conditions"][cond_a][metric] for r in per_query])
    b = np.array([r["conditions"][cond_b][metric] for r in per_query])
    b_only = int(np.sum((a == 0) & (b == 1)))
    a_only = int(np.sum((a == 1) & (b == 0)))
    n_disc = a_only + b_only
    if n_disc == 0:
        return {"a_only": a_only, "b_only": b_only, "p": 1.0}
    p = stats.binomtest(min(a_only, b_only), n_disc, 0.5).pvalue
    return {"a_only": a_only, "b_only": b_only, "p": float(p)}


def report_contrast(per_query: list[dict], cond_a: str, cond_b: str, label: str,
                    subgroup: str | None = None):
    """subgroup=None marks an overall test (eligible for the primary family)."""
    conds = per_query[0]["conditions"] if per_query else {}
    if cond_a not in conds or cond_b not in conds:
        print(f"\n  -- {label}: SKIPPED ({cond_a if cond_a not in conds else cond_b} not in this file) --")
        return
    print(f"\n  -- {label}: {cond_a}  vs  {cond_b} --")
    for m in METRICS:
        r = paired_test(per_query, cond_a, cond_b, m)
        sig = "*" if r["p"] < 0.05 else " "
        line = (f"    {m:<8} mean_a={r['mean_a']:.3f}  mean_b={r['mean_b']:.3f}  "
                f"t={r['t']:+.2f}  p={r['p']:.4f}{sig}  d_z={r['cohens_dz']:+.2f}  (n={r['n']})")
        if m.startswith("hit@"):
            mc = mcnemar(per_query, cond_a, cond_b, m)
            line += f"   [McNemar: a-only={mc['a_only']} b-only={mc['b_only']} p={mc['p']:.4f}]"
        print(line)
        _ALL_TESTS.append({
            "label": label, "cond_a": cond_a, "cond_b": cond_b, "metric": m,
            "mean_a": r["mean_a"], "mean_b": r["mean_b"],
            "p": r["p"], "d_z": r["cohens_dz"], "n": r["n"], "subgroup": subgroup,
            "primary": subgroup is None and m in PRIMARY_METRICS,
        })


def plot_hit_at_k(per_query: list[dict], conditions: list[str], out_dir: Path, filename: str, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    _apply_modern_style()

    series = {cond: condition_means(per_query, cond) for cond in conditions}
    all_vals = [series[c][f"hit@{k}"] for c in conditions for k in K_VALUES]
    n = len(conditions)
    # Most curves saturate near 1.0 by k=5 — a fixed [0,1] axis crams all the
    # actually-interesting variation (concentrated at k=1) into a thin top
    # strip and makes converged lines indistinguishable. Zoom to the real
    # data range instead; a line chart's y-axis isn't a magnitude encoding
    # the way a bar's is, so truncating it here doesn't mislead. Margin
    # scales with n since the k=1 value-label stack (below) needs headroom
    # proportional to how many labels are ladder-stacked beneath the lowest point.
    ymin = max(0.0, min(all_vals) - (0.05 + 0.014 * n))
    fig_w = 8.5 + (2.2 if n > 4 else 0)  # widen for the outside-right legend when there are many series
    fig, ax = plt.subplots(figsize=(fig_w, 5.2))
    linestyles = ["-", "--", ":", "-."]
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    lines = []
    for i, cond in enumerate(conditions):
        means = series[cond]
        ys = [means[f"hit@{k}"] for k in K_VALUES]
        line, = ax.plot(K_VALUES, ys, marker=markers[i % len(markers)],
                          linestyle=linestyles[i % len(linestyles)], label=cond,
                          markersize=6.5, linewidth=2.2,
                          markeredgecolor="white", markeredgewidth=0.8)
        lines.append((cond, ys, line.get_color()))

    # Annotate k=1 — the point where conditions actually differ — so
    # overlapping/converged lines at higher k don't hide the real story.
    # Values often cluster closely (e.g. several near 0.85): a fixed
    # above/below alternation collides in that case, so rank by k=1 value
    # and stack labels in points-space (pixel offsets, robust regardless of
    # axis scale) so close values still get pushed visually apart.
    order = sorted(range(len(lines)), key=lambda idx: lines[idx][1][0])
    n_labels = len(order)
    label_spacing_pt = 24  # vertical pixels between adjacent stacked labels
    for rank, idx in enumerate(order):
        cond, ys, color = lines[idx]
        # Centered ladder of offsets, e.g. for n=6: -32,-19,-6,6,19,32 (points).
        # Placed to the right of the k=1 marker, in the open space before
        # k=5 — the left side is too cramped against the y-axis tick labels.
        offset_y = (rank - (n_labels - 1) / 2) * label_spacing_pt
        ax.annotate(f"{ys[0]:.3f}", (K_VALUES[0], ys[0]),
                     textcoords="offset points", xytext=(14, offset_y),
                     fontsize=8, fontweight="bold", color=color, ha="left", va="center",
                     path_effects=[pe.withStroke(linewidth=2.4, foreground="white")])
    ax.set_xlabel("k")
    ax.set_ylabel("Hit@k")
    ax.set_title(title)
    ax.set_xticks(K_VALUES)
    ax.set_ylim(ymin, 1.02)
    ax.grid(True, alpha=0.7)
    if n > 4:
        ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    else:
        ax.legend(fontsize=8, loc="lower right")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_by_content_type(by_type: dict, out_dir: Path, filename: str, backends: list[str] | None = None):
    """Grouped bar chart: Hit@5 per content type, dense vs hybrid+rerank side
    by side, which the aggregate Hit@k line chart can't show.
    Shows VLM/OCR by default, or any backend list (e.g. + Hybrid) passed in."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _apply_modern_style()

    backends = backends or ["VLM", "OCR"]
    colors = PALETTE
    types = [t for t in ("text", "table", "image", "chart", "equation") if t in by_type]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    n = len(backends)
    w = 0.8 / n
    for ax, mode in zip(axes, ("dense", "hybrid+rerank")):
        x = np.arange(len(types))
        for i, b in enumerate(backends):
            vals = [condition_means(by_type[t], f"{b}-{mode}")["hit@5"] for t in types]
            offset = (i - (n - 1) / 2) * w
            bars = ax.bar(x + offset, vals, w, label=b, color=colors.get(b))
            ax.bar_label(bars, fmt="%.2f", fontsize=7.5, padding=2, color="#52514e")
        ax.set_xticks(x)
        ax.set_xticklabels(types)
        ax.set_title(mode, fontsize=11.5)
        # Real variation lives in 0.4-1.0 here too, but this metric (Hit@5)
        # rarely drops far below that across conditions, so a touch of
        # headroom above 1.0 for the bar_label text is the main need.
        ax.set_ylim(0, 1.12)
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("Hit@5")
    axes[0].legend(loc="lower left", fontsize=9, frameon=True, facecolor="white",
                   framealpha=0.9, edgecolor="none")
    fig.suptitle(f"{' vs '.join(backends)} — Hit@5 by content type")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_modality_ablation(per_query: list[dict], visual_rows: list[dict], out_dir: Path, filename: str):
    """Bar chart for Contrast 2 (multimodal vs text-only) — the companion
    notes call this 'the strongest result in the whole study,' but it had
    no dedicated chart before this."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _apply_modern_style()

    groups = [("Overall (n=125)", per_query), ("Visual queries only (n=image+chart)", visual_rows)]
    metrics = ["hit@1", "hit@5", "mrr"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True)
    for ax, (label, rows) in zip(axes, groups):
        multi = condition_means(rows, "VLM-hybrid+rerank")
        textonly = condition_means(rows, "VLM-hybrid+rerank-textonly")
        x = np.arange(len(metrics))
        w = 0.35
        b1 = ax.bar(x - w / 2, [multi[m] for m in metrics], w, label="multimodal", color=PALETTE["multimodal"])
        b2 = ax.bar(x + w / 2, [textonly[m] for m in metrics], w, label="text-only", color=PALETTE["text-only"])
        ax.bar_label(b1, fmt="%.2f", fontsize=8, padding=2, color="#52514e")
        ax.bar_label(b2, fmt="%.2f", fontsize=8, padding=2, color="#52514e")
        ax.set_xticks(x)
        ax.set_xticklabels(["Hit@1", "Hit@5", "MRR"])
        ax.set_title(label, fontsize=11.5)
        ax.set_ylim(0, 1.12)
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("score")
    axes[0].legend(loc="lower center", fontsize=9, frameon=True, facecolor="white",
                   framealpha=0.9, edgecolor="none")
    fig.suptitle("Multimodal vs text-only retrieval (same parser, same embedder)")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=str(Path(__file__).parent / "rq1_results.json"))
    ap.add_argument("--out-dir", required=True,
                    help="per-run plot folder (never the v3 rq1_plots/ that older docs reference)")
    ap.add_argument("--keep-empty", action="store_true",
                    help="keep queries with no relevant page (they score 0 under every condition)")
    ap.add_argument("--match-nonempty", metavar="FILE",
                    help="also drop queries that have no relevant page in FILE (matched query set "
                         "for a cross-GT comparison)")
    ap.add_argument("--lang-map", metavar="JSON",
                    help="{target_paper: 'de'|'en'} — adds per-language tables and contrasts")
    ap.add_argument("--tests-json", metavar="FILE", help="write every test (raw + BH p) here")
    args = ap.parse_args()

    per_query = load(args.inp)
    n_all = len(per_query)
    if not args.keep_empty:
        per_query = [r for r in per_query if r["n_relevant"] > 0]
    if args.match_nonempty:
        other = {r["query_id"]: r["n_relevant"] for r in load(args.match_nonempty)}
        per_query = [r for r in per_query if other.get(r["query_id"], 0) > 0]
    print(f"queries: {len(per_query)} of {n_all} (dropped {n_all - len(per_query)} with an empty "
          f"relevant set{' here or in ' + args.match_nonempty if args.match_nonempty else ''})")
    conditions = list(per_query[0]["conditions"].keys())
    out_dir = Path(args.out_dir)

    # ── Overall table ────────────────────────────────────────────────────────
    print_table(per_query, conditions, "OVERALL (all target_types)")

    # ── Per target_type breakdown ───────────────────────────────────────────
    by_type = defaultdict(list)
    for r in per_query:
        by_type[r["target_type"]].append(r)
    for t in sorted(by_type):
        print_table(by_type[t], conditions, f"BY TARGET_TYPE = {t}")

    # ── Contrast 1: parsing-backend pairs, at each retrieval mode ───────────
    # Three backends if Hybrid (MinerU hybrid-engine) is present in this
    # results file: VLM (vlm-engine), OCR (pipeline), Hybrid (hybrid-engine).
    # Older results files only have VLM/OCR — guard so this script still
    # works against them.
    backend_pairs = [("VLM", "OCR")]
    if "Hybrid-dense" in conditions:
        backend_pairs += [("VLM", "Hybrid"), ("OCR", "Hybrid")]
    print(f"\n{'='*90}\n  CONTRAST 1 — parsing backend, at each retrieval mode\n{'='*90}")
    for a, b in backend_pairs:
        print(f"\n  -- {a} vs {b} --")
        for mode in ("dense", "dense+rerank", "hybrid", "hybrid+rerank"):
            report_contrast(per_query, f"{a}-{mode}", f"{b}-{mode}", f"parsing @ {mode}")

    # The aggregate (above) can hide a per-modality effect — H1 specifically
    # predicts VLM's advantage concentrates on table/figure/equation content,
    # not text. Re-run the same contrast within each target_type bucket
    # (n=25 each — enough to detect a large effect, not a small one; see
    # project_rq1 sizing discussion).
    print(f"\n{'='*90}\n  CONTRAST 1b — parsing backend, WITHIN each target_type (n=25/bucket)\n{'='*90}")
    for a, b in backend_pairs:
        print(f"\n  == {a} vs {b} ==")
        for t in sorted(by_type):
            print(f"\n  ### target_type = {t} (n={len(by_type[t])}) ###")
            for mode in ("dense", "hybrid+rerank"):
                report_contrast(by_type[t], f"{a}-{mode}", f"{b}-{mode}", f"{a} vs {b} @ {mode}", subgroup=t)

    # ── Contrast 2: multimodal vs text-only (modality-content ablation) ───────────────
    # Three levels per the thesis proposal: text-only < text+table <
    # text+table+figure (= the unrestricted VLM-hybrid+rerank condition).
    print(f"\n{'='*90}\n  CONTRAST 2 — modality-content ablation (VLM collection)\n{'='*90}")
    report_contrast(per_query, "VLM-hybrid+rerank", "VLM-hybrid+rerank-textonly", "full vs text-only")
    report_contrast(per_query, "VLM-dense", "VLM-dense-textonly", "full vs text-only @ dense")
    if "VLM-hybrid+rerank-text+table" in conditions:
        report_contrast(per_query, "VLM-hybrid+rerank-text+table", "VLM-hybrid+rerank-textonly",
                        "text+table vs text-only")
        report_contrast(per_query, "VLM-hybrid+rerank", "VLM-hybrid+rerank-text+table",
                        "full vs text+table")
    print("\n  -- same contrast(s), visual queries only --")
    visual_rows = by_type.get("image", []) + by_type.get("chart", [])
    if visual_rows:
        report_contrast(visual_rows, "VLM-hybrid+rerank", "VLM-hybrid+rerank-textonly",
                        "full vs text-only (visual queries)", subgroup="visual")
        if "VLM-hybrid+rerank-text+table" in conditions:
            report_contrast(visual_rows, "VLM-hybrid+rerank", "VLM-hybrid+rerank-text+table",
                            "full vs text+table (visual queries)", subgroup="visual")

    # ── Contrast 3: dense vs hybrid vs +rerank, within each collection ──────
    colls = ["VLM", "OCR"] + (["Hybrid"] if "Hybrid-dense" in conditions else [])
    print(f"\n{'='*90}\n  CONTRAST 3 — retrieval mode (within each collection)\n{'='*90}")
    for coll in colls:
        report_contrast(per_query, f"{coll}-dense", f"{coll}-hybrid", f"{coll}: dense vs hybrid")
        report_contrast(per_query, f"{coll}-dense", f"{coll}-dense+rerank", f"{coll}: dense vs dense+rerank")
        report_contrast(per_query, f"{coll}-hybrid", f"{coll}-hybrid+rerank", f"{coll}: hybrid vs hybrid+rerank")

    # ── Contrast 4: embedding-model ablation, VL vs text-only embedder ──
    # v4: the -text collection holds the same chunks as the VLM collection (all types; figures enter the text model
    # through their caption/description text), embedded by Qwen3-Embedding-8B. So the embedder contrast runs over all
    # queries, at every retrieval mode, with the same (VL) reranker; the text-type-only row is kept, labelled as a
    # subset, for continuity with v3.
    if "TEXT-8b-dense" in conditions:
        print(f"\n{'='*90}\n  CONTRAST 4 — embedding model: Qwen3-VL-Embedding-8B (VLM) vs Qwen3-Embedding-8B (TEXT-8b)\n{'='*90}")
        for mode in ("dense", "dense+rerank", "hybrid", "hybrid+rerank"):
            report_contrast(per_query, f"VLM-{mode}", f"TEXT-8b-{mode}", f"embedder @ {mode}")
        print(f"\n  -- same contrast, WITHIN each target_type --")
        for t in sorted(by_type):
            print(f"\n  ### target_type = {t} (n={len(by_type[t])}) ###")
            for mode in ("dense", "hybrid+rerank"):
                report_contrast(by_type[t], f"VLM-{mode}", f"TEXT-8b-{mode}", f"embedder @ {mode}", subgroup=t)
        text_rows = by_type.get("text", [])
        if text_rows and "VLM-dense-textonly" in conditions:
            report_contrast(text_rows, "VLM-dense-textonly", "TEXT-8b-dense",
                            "embedder, text-type queries, text chunks only (v3-style subset)", subgroup="text")

    # ── Contrast 8: reranker model — Qwen3-VL-Reranker-8B vs Qwen3-Reranker-8B ─
    # 2×2 with Contrast 4: {VLM, TEXT-8b} collections × {VL, text} reranker, at
    # dense and hybrid first stages. The VL reranker scores figure chunks on
    # image + caption; the text reranker on the caption text alone. Also the
    # end-to-end "fully text" vs "fully VL" pipeline.
    if "VLM-dense+textrerank" in conditions:
        print(f"\n{'='*90}\n  CONTRAST 8 — reranker model: VL reranker (+rerank) vs text reranker (+textrerank)\n{'='*90}")
        for coll in ("VLM", "TEXT-8b"):
            for mode in ("dense", "hybrid"):
                report_contrast(per_query, f"{coll}-{mode}+rerank", f"{coll}-{mode}+textrerank",
                                f"reranker @ {coll}-{mode}")
        report_contrast(per_query, "VLM-hybrid+rerank", "TEXT-8b-hybrid+textrerank",
                        "fully VL vs fully text pipeline (hybrid)")
        report_contrast(per_query, "VLM-dense+textrerank", "VLM-dense", "text reranker vs no rerank @ VLM-dense")
        print(f"\n  -- same contrasts, WITHIN each target_type --")
        for t in sorted(by_type):
            print(f"\n  ### target_type = {t} (n={len(by_type[t])}) ###")
            report_contrast(by_type[t], "VLM-hybrid+rerank", "VLM-hybrid+textrerank", "reranker @ VLM-hybrid", subgroup=t)
            report_contrast(by_type[t], "VLM-hybrid+rerank", "TEXT-8b-hybrid+textrerank",
                            "fully VL vs fully text pipeline (hybrid)", subgroup=t)

    # ── Contrast 9: raw-parse collections, VL vs text embedder ─────────────
    # NOT a pure embedder contrast: the NoEnrichTXT build drops figure chunks
    # without usable text (ceramics 2,736 vs 5,284 images, 1,071 vs 1,375
    # charts), so candidate population differs too. Reported for completeness;
    # Contrast 4 is the headline embedder test.
    if "NoEnrichTXT-dense" in conditions:
        print(f"\n{'='*90}\n  CONTRAST 9 — raw parse: NoEnrich (VL embedder) vs NoEnrichTXT (text embedder; fewer figure chunks)\n{'='*90}")
        for mode in ("dense", "dense+rerank", "hybrid", "hybrid+rerank"):
            report_contrast(per_query, f"NoEnrich-{mode}", f"NoEnrichTXT-{mode}", f"raw-parse embedder @ {mode}")

    # ── Per-language breakdown (cross-lingual: English queries) ─────────────
    if args.lang_map:
        lang = json.loads(Path(args.lang_map).read_text())
        by_lang = defaultdict(list)
        for r in per_query:
            by_lang[lang.get(r["target_paper"], "unknown")].append(r)
        for lg in sorted(by_lang):
            print_table(by_lang[lg], conditions, f"BY TARGET-DOCUMENT LANGUAGE = {lg}")
        print(f"\n{'='*90}\n  CROSS-LINGUAL — key contrasts within each document language\n{'='*90}")
        for lg in sorted(by_lang):
            rows = by_lang[lg]
            if len(rows) < 5:
                continue
            print(f"\n  ### language = {lg} (n={len(rows)}) ###")
            report_contrast(rows, "VLM-hybrid+rerank", "VLM-hybrid+rerank-textonly", "full vs text-only", subgroup=f"lang={lg}")
            report_contrast(rows, "VLM-hybrid+rerank", "TEXT-8b-hybrid+rerank", "embedder @ hybrid+rerank", subgroup=f"lang={lg}")
            report_contrast(rows, "VLM-dense", "OCR-dense", "VLM vs OCR @ dense", subgroup=f"lang={lg}")

    # ── Contrast 5: UniDoc-Bench retrieval architecture (RQ1 sub-study) ─────
    # Joint multimodal embedding (Memento's actual architecture, here
    # VLM-hybrid+rerank) vs Image-Only late-interaction (ColQwen) vs explicit
    # Fusion of separate text-only + image-only retrievers — replicating
    # UniDoc-Bench's "fusion beats joint" finding on our own corpus.
    if "ColQwen-only" in conditions:
        print(f"\n{'='*90}\n  CONTRAST 5 — UniDoc-Bench architecture (Joint vs ColQwen-only vs Fusion)\n{'='*90}")
        report_contrast(per_query, "VLM-hybrid+rerank", "ColQwen-only+rerank", "Joint vs ColQwen-only (+rerank)")
        report_contrast(per_query, "VLM-hybrid+rerank", "Fusion-text+colqwen+rerank", "Joint vs Fusion (+rerank)")
        report_contrast(per_query, "ColQwen-only+rerank", "Fusion-text+colqwen+rerank", "ColQwen-only vs Fusion (+rerank)")
        print("\n  -- same contrasts, image queries only --")
        image_rows = by_type.get("image", [])
        if image_rows:
            report_contrast(image_rows, "VLM-hybrid+rerank", "ColQwen-only+rerank", "Joint vs ColQwen-only (+rerank, image queries)")
            report_contrast(image_rows, "VLM-hybrid+rerank", "Fusion-text+colqwen+rerank", "Joint vs Fusion (+rerank, image queries)")

    # ── Contrast 6: Contextual Retrieval (Anthropic situating context) ──────
    # Same VLM parse + local embedder; the only variable is the LLM-generated
    # situating sentence prepended to embed_text at ingest (CONTEXT_BM25 was
    # false, so sparse vectors are identical — the hybrid contrast isolates
    # the dense-side effect). Fully paired against the matching VLM-* modes.
    # Expectation from the literature: effect concentrates at dense retrieval
    # and narrows after reranking (same pattern as parsing-backend effects).
    if "CTX-dense" in conditions:
        print(f"\n{'='*90}\n  CONTRAST 6 — Contextual Retrieval (CTX) vs plain embedding (VLM)\n{'='*90}")
        for mode in ("dense", "dense+rerank", "hybrid", "hybrid+rerank"):
            report_contrast(per_query, f"CTX-{mode}", f"VLM-{mode}", f"context @ {mode}")
        print(f"\n  -- same contrast, WITHIN each target_type (n=25/bucket) --")
        for t in sorted(by_type):
            print(f"\n  ### target_type = {t} (n={len(by_type[t])}) ###")
            for mode in ("dense", "hybrid+rerank"):
                report_contrast(by_type[t], f"CTX-{mode}", f"VLM-{mode}", f"context @ {mode}", subgroup=t)

    # ── Contrast 7: enrichment ablation (raw MinerU parse vs enriched) ──────
    # Same VLM parse + local embedder; the only variable is whether
    # enrich_content_list() ran at ingest (--no-enrich skips it): no figure/
    # table/equation context grounding, no section assignment, no
    # junk-fragment filtering. Fully paired against the matching VLM-* modes.
    # Expectation: effect should concentrate on image/chart/equation queries
    # (where enrichment injects the most additional retrievable context) and
    # on Keywords/short-fragment-heavy papers (where junk filtering matters).
    if "NoEnrich-dense" in conditions:
        print(f"\n{'='*90}\n  CONTRAST 7 — enrichment ablation (NoEnrich) vs enriched parse (VLM)\n{'='*90}")
        for mode in ("dense", "dense+rerank", "hybrid", "hybrid+rerank"):
            report_contrast(per_query, f"VLM-{mode}", f"NoEnrich-{mode}", f"enrichment @ {mode}")
        print(f"\n  -- same contrast, WITHIN each target_type (n=25/bucket) --")
        for t in sorted(by_type):
            print(f"\n  ### target_type = {t} (n={len(by_type[t])}) ###")
            for mode in ("dense", "hybrid+rerank"):
                report_contrast(by_type[t], f"VLM-{mode}", f"NoEnrich-{mode}", f"enrichment @ {mode}", subgroup=t)

    # ── Multiple comparisons correction ─────────────────────────────────────
    # Every paired-t-test p-value collected above (all contrasts, all
    # metrics, all subgroups), Benjamini-Hochberg FDR-adjusted. Report raw
    # and adjusted together, sorted by raw p, so nothing is hidden — this is
    # the direct fix for the RQ1 audit's finding that per-subgroup
    # "significant" results (aggregate null, image-subgroup positive) were
    # exactly the multiple-testing signature.
    print(f"\n{'='*90}\n  MULTIPLE COMPARISONS CORRECTION — Benjamini-Hochberg (FDR)\n"
          f"  {len(_ALL_TESTS)} paired tests collected across this run\n{'='*90}")
    raw_ps = [t["p"] for t in _ALL_TESTS]
    adj_ps = benjamini_hochberg(raw_ps)
    prim_idx = [i for i, t in enumerate(_ALL_TESTS) if t["primary"]]
    prim_adj = dict(zip(prim_idx, benjamini_hochberg([raw_ps[i] for i in prim_idx])))
    for i, t in enumerate(_ALL_TESTS):
        t["p_bh_all"] = adj_ps[i]
        t["p_bh_primary"] = prim_adj.get(i)
    print(f"\n  PRIMARY family (overall tests on {sorted(PRIMARY_METRICS)}): {len(prim_idx)} tests, "
          f"BH p<0.05: {sum(v < 0.05 for v in prim_adj.values())}")
    print(f"  {'contrast':<50}{'metric':<9}{'mean_a':>8}{'mean_b':>8}{'d_z':>7}{'raw p':>10}{'BH p':>10}")
    for i in sorted(prim_idx, key=lambda i: raw_ps[i]):
        t = _ALL_TESTS[i]
        print(f"  {t['label'][:49]:<50}{t['metric']:<9}{t['mean_a']:>8.3f}{t['mean_b']:>8.3f}"
              f"{t['d_z']:>+7.2f}{t['p']:>10.4f}{prim_adj[i]:>10.4f}  {'*' if prim_adj[i] < 0.05 else ''}")
    if args.tests_json:
        Path(args.tests_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.tests_json).write_text(json.dumps(
            {"results": args.inp, "n_queries": len(per_query), "tests": _ALL_TESTS}, indent=1))
    print(f"\n  ALL tests (primary + exploratory), BH over the whole run:")
    n_raw_sig = sum(p < 0.05 for p in raw_ps)
    n_adj_sig = sum(p < 0.05 for p in adj_ps)
    print(f"  raw p<0.05: {n_raw_sig}/{len(raw_ps)}   BH-adjusted p<0.05: {n_adj_sig}/{len(raw_ps)}")
    print(f"\n  {'contrast':<45}{'metric':<9}{'n':>5}{'raw p':>10}{'BH p':>10}  sig(BH)")
    ordering = sorted(range(len(_ALL_TESTS)), key=lambda i: raw_ps[i])
    for i in ordering:
        t = _ALL_TESTS[i]
        sig = "*" if adj_ps[i] < 0.05 else " "
        print(f"  {t['label']:<45}{t['metric']:<9}{t['n']:>5}{t['p']:>10.4f}{adj_ps[i]:>10.4f}  {sig}")

    # ── Plots ────────────────────────────────────────────────────────────────
    parsing_conds = ["VLM-dense", "VLM-hybrid+rerank", "OCR-dense", "OCR-hybrid+rerank"]
    if "Hybrid-dense" in conditions:
        parsing_conds += ["Hybrid-dense", "Hybrid-hybrid+rerank"]
    plot_hit_at_k(per_query, parsing_conds,
                  out_dir, "hit_at_k_parsing.png", "Hit@k — parsing × retrieval mode")
    plot_hit_at_k(per_query, [c for c in conditions if c.startswith("VLM")],
                  out_dir, "hit_at_k_vlm_modes.png", "Hit@k — VLM, all retrieval modes")
    plot_hit_at_k(per_query, [c for c in conditions if c.startswith("OCR")],
                  out_dir, "hit_at_k_ocr_modes.png", "Hit@k — OCR, all retrieval modes")
    if "Hybrid-dense" in conditions:
        plot_hit_at_k(per_query, [c for c in conditions if c.startswith("Hybrid-")],
                      out_dir, "hit_at_k_hybrid_modes.png", "Hit@k — Hybrid-engine, all retrieval modes")
    if ("TEXT-8b-hybrid+rerank" in conditions and "VLM-dense-textonly" in conditions
            and by_type.get("text")):
        plot_hit_at_k(by_type["text"], ["VLM-dense-textonly", "VLM-hybrid+rerank-textonly",
                                        "TEXT-8b-dense", "TEXT-8b-hybrid+rerank"],
                      out_dir, "hit_at_k_embedding_model.png",
                      "Hit@k — embedding model ablation (text-type queries)")
    if "CTX-dense" in conditions:
        plot_hit_at_k(per_query, ["VLM-dense", "CTX-dense", "VLM-hybrid+rerank", "CTX-hybrid+rerank"],
                      out_dir, "hit_at_k_contextual_retrieval.png",
                      "Hit@k — Contextual Retrieval vs plain embedding")
    if "NoEnrich-dense" in conditions:
        plot_hit_at_k(per_query, ["VLM-dense", "NoEnrich-dense", "VLM-hybrid+rerank", "NoEnrich-hybrid+rerank"],
                      out_dir, "hit_at_k_enrichment_ablation.png",
                      "Hit@k — enrichment ablation vs raw MinerU parse")
    backends = ["VLM", "OCR"] + (["Hybrid"] if "Hybrid-dense" in conditions else [])
    plot_by_content_type(by_type, out_dir, "hit_at_5_by_content_type.png", backends=backends)
    plot_modality_ablation(per_query, visual_rows, out_dir, "modality_ablation.png")
    if "ColQwen-only" in conditions:
        plot_hit_at_k(per_query, ["VLM-hybrid+rerank", "ColQwen-only+rerank", "Fusion-text+colqwen+rerank"],
                      out_dir, "hit_at_k_unidoc_bench.png",
                      "Hit@k — UniDoc-Bench architecture (Joint vs ColQwen-only vs Fusion)")
        if by_type.get("image"):
            plot_hit_at_k(by_type["image"], ["VLM-hybrid+rerank", "ColQwen-only+rerank", "Fusion-text+colqwen+rerank"],
                          out_dir, "hit_at_k_unidoc_bench_image.png",
                          "Hit@k — UniDoc-Bench architecture, image queries only")
    print(f"\nPlots written to {out_dir}/")


if __name__ == "__main__":
    main()
