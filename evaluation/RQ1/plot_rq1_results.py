#!/usr/bin/env python3
"""
plot_rq1_results.py — publication figures for the RQ1 retrieval evaluation.

Reads evaluation/RQ1/rq1_results.json directly (per-query per-condition scores)
and produces four focused figures, each built around one finding from
RQ1_RESULTS_ANALYSIS.md rather than a generic dump of all 36 conditions:

  fig1_reranking_effect.png   — recall@k curves, dense vs +rerank, one panel/backbone
  fig2_modality_ablation.png  — recall@k, full vs text+table vs text-only
  fig3_null_results.png       — parsing backend / retrieval mode / CTX / enrichment
  fig4_effect_size_forest.png — headline: all contrasts ranked by |Cohen's d_z|

Usage:
    python3 evaluation/RQ1/plot_rq1_results.py --in evaluation/RQ1/rq1_results.json \
        --out-dir evaluation/rq1_analysis_out_2026-08-03/figures
"""
import argparse
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── validated colorblind-safe categorical palette (dataviz skill reference) ──
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
GRAY = "#9a9a94"
GRID = "#e3e2dc"
TEXT = "#26251f"

K_VALUES = [1, 5, 10, 20]

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT,
    "text.color": TEXT,
    "xtick.color": TEXT,
    "ytick.color": TEXT,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "font.family": "DejaVu Sans",
})


def load(path):
    d = json.load(open(path))
    return d["per_query"]


def cond_metric_values(per_query, cond_name, metric, target_type=None):
    """Per-query list of metric values for one condition, aligned by query order.

    target_type: if given, restrict to queries of that target_type (e.g. "text").
    """
    return [
        pq["conditions"][cond_name][metric]
        for pq in per_query
        if cond_name in pq["conditions"] and (target_type is None or pq["target_type"] == target_type)
    ]


def mean_recall_curve(per_query, cond_name):
    return [statistics.mean(cond_metric_values(per_query, cond_name, f"recall@{k}")) for k in K_VALUES]


def cohens_d_z(a_vals, b_vals):
    diffs = [a - b for a, b in zip(a_vals, b_vals)]
    m = statistics.mean(diffs)
    sd = statistics.stdev(diffs) if len(diffs) > 1 else 0.0
    return (m / sd) if sd > 0 else 0.0


def bh_adjust(pvals):
    """Benjamini-Hochberg FDR adjustment. Returns adjusted p-values in input order."""
    n = len(pvals)
    order = sorted(range(n), key=lambda i: pvals[i])
    adjusted = [0.0] * n
    prev_min = 1.0
    for rank in range(n - 1, -1, -1):
        idx = order[rank]
        val = min(pvals[idx] * n / (rank + 1), 1.0)
        prev_min = min(prev_min, val)
        adjusted[idx] = prev_min
    return adjusted


def paired_ttest_p(a_vals, b_vals):
    diffs = np.array(a_vals) - np.array(b_vals)
    n = len(diffs)
    if n < 2 or np.allclose(diffs, 0):
        return 1.0
    m = diffs.mean()
    sd = diffs.std(ddof=1)
    if sd == 0:
        return 1.0
    t = m / (sd / np.sqrt(n))
    # two-sided p via normal approximation is fine for n=125/50; use scipy if present
    try:
        from scipy import stats as sstats
        return 2 * (1 - sstats.t.cdf(abs(t), df=n - 1))
    except ImportError:
        from math import erf, sqrt
        return 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))


def style_ax(ax, ymax=1.0):
    ax.set_ylim(0, ymax)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def fig1_reranking_effect(per_query, out_dir):
    backbones = [("VLM", "VLM"), ("OCR", "OCR"), ("Hybrid-parse", "Hybrid")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
    x = np.arange(len(K_VALUES))

    for ax, (title, prefix) in zip(axes, backbones):
        no_rr = mean_recall_curve(per_query, f"{prefix}-hybrid")
        rr = mean_recall_curve(per_query, f"{prefix}-hybrid+rerank")
        ax.plot(x, no_rr, marker="o", markersize=7, linewidth=2.5, color=GRAY,
                 label="no rerank", zorder=3)
        ax.plot(x, rr, marker="o", markersize=7, linewidth=2.5, color=BLUE,
                 label="+ rerank", zorder=3)
        for xi, (a, b) in enumerate(zip(no_rr, rr)):
            ax.annotate(f"+{(b - a) * 100:.0f}pp", (xi, b), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=9, color=BLUE, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"@{k}" for k in K_VALUES])
        ax.set_title(title)
        ax.set_xlabel("Recall@k")
        style_ax(ax)

    axes[0].set_ylabel("Recall")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.06))
    fig.suptitle("Reranking is the dominant lever across every parsing backend",
                 y=1.14, fontsize=14, fontweight="bold")
    fig.text(0.5, -0.02,
              "Hybrid retrieval mode shown; dense mode shows a similarly large rerank effect (not plotted). "
              "Dense vs. hybrid choice itself is not a significant factor (Fig. 3). n=125 queries. "
              "Labels show the +rerank gain in percentage points.",
              ha="center", fontsize=9, color=GRAY)
    fig.tight_layout()
    fig.savefig(out_dir / "fig1_reranking_effect.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig2_modality_ablation(per_query, out_dir):
    conditions = [
        ("Full (image+table+text)", "VLM-hybrid+rerank", BLUE),
        ("Text + table only", "VLM-hybrid+rerank-text+table", AQUA),
        ("Text only", "VLM-hybrid+rerank-textonly", ORANGE),
    ]
    x = np.arange(len(K_VALUES))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8.5, 5))

    for i, (label, cond, color) in enumerate(conditions):
        vals = mean_recall_curve(per_query, cond)
        offset = (i - 1) * width
        bars = ax.bar(x + offset, vals, width=width * 0.9, color=color, label=label,
                       zorder=3, edgecolor="white", linewidth=0.5)
        for rect, v in zip(bars, vals):
            ax.annotate(f"{v * 100:.0f}", (rect.get_x() + rect.get_width() / 2, v),
                        textcoords="offset points", xytext=(0, 4), ha="center",
                        fontsize=8.5, color=TEXT)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Recall@{k}" for k in K_VALUES])
    style_ax(ax)
    ax.set_ylabel("Recall")
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Dropping modalities measurably hurts recall\n(VLM-hybrid+rerank, n=125)")
    fig.text(0.5, -0.04,
              "Full vs. text-only and full vs. text+table are BH-significant at every k shown. "
              "Text+table vs. text-only is BH-significant at recall@5/10/20 but not at recall@1.",
              ha="center", fontsize=8.5, color=GRAY)
    fig.tight_layout()
    fig.savefig(out_dir / "fig2_modality_ablation.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig3_null_results(per_query, out_dir):
    panels = [
        ("Parsing backend\n(hybrid+rerank)", [
            ("VLM", "VLM-hybrid+rerank"),
            ("OCR", "OCR-hybrid+rerank"),
            ("Hybrid-parse", "Hybrid-hybrid+rerank"),
        ]),
        ("Retrieval mode\n(VLM, no rerank)", [
            ("Dense", "VLM-dense"),
            ("Hybrid (RRF)", "VLM-hybrid"),
        ]),
        ("Contextual Retrieval\n(hybrid+rerank)", [
            ("Plain", "VLM-hybrid+rerank"),
            ("+ CTX", "CTX-hybrid+rerank"),
        ]),
        ("Enrichment\n(hybrid+rerank)", [
            ("Enriched", "VLM-hybrid+rerank"),
            ("No enrich", "NoEnrich-hybrid+rerank"),
        ]),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(13, 4.2), sharey=True)
    colors = [BLUE, ORANGE, AQUA]

    for ax, (title, bars) in zip(axes, panels):
        vals = [statistics.mean(cond_metric_values(per_query, cond, "recall@10")) for _, cond in bars]
        labels = [lab for lab, _ in bars]
        xs = np.arange(len(bars))
        bar_objs = ax.bar(xs, vals, color=colors[: len(bars)], width=0.6, zorder=3,
                           edgecolor="white", linewidth=0.5)
        for rect, v in zip(bar_objs, vals):
            ax.annotate(f"{v * 100:.1f}%", (rect.get_x() + rect.get_width() / 2, v),
                        textcoords="offset points", xytext=(0, 5), ha="center",
                        fontsize=9.5, fontweight="bold")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=9.5)
        ax.set_title(title, fontsize=11)
        style_ax(ax)

    axes[0].set_ylabel("Recall@10")
    fig.suptitle("No BH-significant difference: backend, retrieval mode, CTX, or enrichment",
                 y=1.08, fontsize=14, fontweight="bold")
    fig.text(0.5, -0.03,
              "Bars within each panel are statistically indistinguishable after BH-FDR correction (n=125). "
              "Deliberately drawn to the same recall@10 scale as Fig. 1–2 to show how small these gaps are by comparison.",
              ha="center", fontsize=9, color=GRAY)
    fig.tight_layout()
    fig.savefig(out_dir / "fig3_null_results.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig4_effect_size_forest(per_query, out_dir):
    contrasts = [
        ("VLM: hybrid → +rerank", "VLM-hybrid", "VLM-hybrid+rerank"),
        ("OCR: hybrid → +rerank", "OCR-hybrid", "OCR-hybrid+rerank"),
        ("Hybrid-parse: hybrid → +rerank", "Hybrid-hybrid", "Hybrid-hybrid+rerank"),
        ("VLM: dense → +rerank", "VLM-dense", "VLM-dense+rerank"),
        ("OCR: dense → +rerank", "OCR-dense", "OCR-dense+rerank"),
        ("Hybrid-parse: dense → +rerank", "Hybrid-dense", "Hybrid-dense+rerank"),
        ("Full vs text-only", "VLM-hybrid+rerank-textonly", "VLM-hybrid+rerank"),
        ("Full vs text+table", "VLM-hybrid+rerank-text+table", "VLM-hybrid+rerank"),
        ("Text+table vs text-only", "VLM-hybrid+rerank-textonly", "VLM-hybrid+rerank-text+table"),
        # cond_a is always the first-named entity in the label, cond_b the second —
        # d = cond_a - cond_b, so "A vs B" > 0 means A retrieved better than B.
        ("OCR vs Hybrid-parse (dense)", "Hybrid-dense", "OCR-dense"),
        ("VLM vs OCR (dense)", "OCR-dense", "VLM-dense"),
        ("Dense vs hybrid (VLM)", "VLM-hybrid", "VLM-dense"),
        ("Dense vs hybrid (OCR)", "OCR-hybrid", "OCR-dense"),
        ("CTX vs plain (dense)", "VLM-dense", "CTX-dense"),
        ("CTX vs plain (hybrid+rerank)", "VLM-hybrid+rerank", "CTX-hybrid+rerank"),
        ("Enrichment vs plain (dense)", "NoEnrich-dense", "VLM-dense"),
        ("Enrichment vs plain (hybrid+rerank)", "NoEnrich-hybrid+rerank", "VLM-hybrid+rerank"),
    ]
    # Kept separate: restricted to the 25 text-type queries only (matches Contrast 4
    # in the written analysis), not the full 125 — a different query set than every
    # other row above, so it can't share the same "n" as the rest.
    text_only_contrast = (
        "Multimodal vs text embedder (text queries, n=25)",
        "TEXT-8b-hybrid+rerank", "VLM-hybrid+rerank-textonly", "text",
    )

    rows = []
    for label, cond_b, cond_a in contrasts:
        a = cond_metric_values(per_query, cond_a, "recall@10")
        b = cond_metric_values(per_query, cond_b, "recall@10")
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
        d = cohens_d_z(a, b)
        p = paired_ttest_p(a, b)
        rows.append((label, d, p, n))

    label, cond_b, cond_a, ttype = text_only_contrast
    a = cond_metric_values(per_query, cond_a, "recall@10", target_type=ttype)
    b = cond_metric_values(per_query, cond_b, "recall@10", target_type=ttype)
    rows.append((label, cohens_d_z(a, b), paired_ttest_p(a, b), len(a)))

    rows.sort(key=lambda r: r[1])
    labels = [r[0] for r in rows]
    ds = [r[1] for r in rows]
    raw_ps = [r[2] for r in rows]
    bh_ps = bh_adjust(raw_ps)
    sig = [p < 0.05 for p in bh_ps]

    fig, ax = plt.subplots(figsize=(9, 8))
    y = np.arange(len(rows))
    colors = [BLUE if s else GRAY for s in sig]
    ax.barh(y, ds, color=colors, height=0.6, zorder=3, edgecolor="white", linewidth=0.5)
    ax.axvline(0, color=TEXT, linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_xlabel("Effect size on Recall@10  (Cohen's d$_z$, paired)")
    ax.grid(axis="x", zorder=0, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)

    # Anchor every label at a fixed point to the right of the largest bar tip
    # (never at the bar's own tip) so small/negative bars never collide with
    # the y-tick labels sitting just left of x=0.
    label_x = max(max(ds), 0) + 0.06
    for yi, d, bh_p in zip(range(len(rows)), ds, bh_ps):
        txt = f"d={d:+.2f}" + ("  *" if bh_p < 0.05 else "  n.s.")
        ax.annotate(txt, (label_x, yi), ha="left", va="center",
                    fontsize=8.5, color=(BLUE if bh_p < 0.05 else GRAY))
    ax.set_xlim(min(ds) - 0.05, label_x + 0.32)

    handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=GRAY)]
    ax.legend(handles, ["BH-adjusted p < 0.05", "not significant"], loc="upper right",
              bbox_to_anchor=(0.62, 1.0), frameon=False, fontsize=9.5)
    ax.set_title("Effect sizes across all RQ1 contrasts (Recall@10)\nReranking and modality dominate; backend/CTX/enrichment are near-zero",
                 fontsize=13)
    fig.text(0.5, -0.02,
              "Significance is BH-FDR-corrected across this figure's 18 contrasts (own family, separate from "
              "the 774-test correction in the written analysis). Last row uses the 25 text-type queries only; "
              "all others use n=125.",
              ha="center", fontsize=8.5, color=GRAY)
    fig.tight_layout()
    fig.savefig(out_dir / "fig4_effect_size_forest.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", default=str(Path(__file__).parent / "rq1_results.json"))
    ap.add_argument("--out-dir", default="evaluation/rq1_analysis_out_2026-08-03/figures")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_query = load(args.infile)

    fig1_reranking_effect(per_query, out_dir)
    fig2_modality_ablation(per_query, out_dir)
    fig3_null_results(per_query, out_dir)
    fig4_effect_size_forest(per_query, out_dir)

    print(f"Wrote 4 figures to {out_dir}/")


if __name__ == "__main__":
    main()
