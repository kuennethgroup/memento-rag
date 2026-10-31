#!/usr/bin/env python3
"""RQ1 Layer A: Pairwise backend comparison for document parsing quality.

Compares OCR (auto), VLM (vlm), and Hybrid (hybrid_auto) parsing outputs
element-by-element using:
  - Normalized Edit Distance (NED) for text, headers, lists, equations
  - Element count / coverage comparison per type
  - Disagreement catalog for manual inspection

Uses MinerU's content_list.json which provides per-element type, text, bbox,
and page_idx — enabling alignment across backends via page + bbox IoU.

Usage:
    python evaluation/RQ1/parse_quality.py \
        --output-dir data/output/Ceramics \
        --out evaluation/RQ1/parse_quality_ceramics.json \
        --limit 50
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import editdistance


BACKENDS = ["auto", "vlm", "hybrid_auto"]
BACKEND_LABELS = {"auto": "OCR", "vlm": "VLM", "hybrid_auto": "Hybrid"}
TEXT_TYPES = {"text", "header", "list", "page_footnote", "aside_text", "footer"}
FORMULA_TYPES = {"equation", "interline_formula"}
TABLE_TYPES = {"table"}
SKIP_TYPES = {"page_number"}


def normalized_edit_distance(a: str, b: str) -> float:
    if not a and not b:
        return 0.0
    dist = editdistance.eval(a, b)
    return dist / max(len(a), len(b))


def extract_text(el: dict) -> str:
    """Extract comparable text from any element type."""
    t = el.get("type", "")
    if t == "list":
        items = el.get("list_items", [])
        return "\n".join(items) if items else ""
    if t == "table":
        body = el.get("table_body", "")
        caption = " ".join(el.get("table_caption", []))
        return f"{caption}\n{body}".strip() if caption else (body or "")
    if t in ("image", "chart"):
        content = el.get("content", "")
        cap_key = "image_caption" if t == "image" else "chart_caption"
        caption = " ".join(el.get(cap_key, []))
        return f"{caption}\n{content}".strip() if caption else (content or "")
    return el.get("text", "")


def bbox_iou(a: list, b: list) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def load_content_list(output_dir: str, doc: str, backend: str) -> list | None:
    p = Path(output_dir) / doc / backend / f"{doc}_content_list.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def classify_element(el: dict) -> str:
    t = el.get("type", "")
    if t in TEXT_TYPES:
        return "text"
    if t in FORMULA_TYPES:
        return "formula"
    if t in TABLE_TYPES:
        return "table"
    if t in {"image", "chart"}:
        return "figure"
    if t in SKIP_TYPES:
        return "skip"
    return "other"


def align_elements(list_a: list, list_b: list, iou_threshold: float = 0.3):
    """Match elements from list_a to list_b by page + bbox IoU.

    Returns list of (idx_a, idx_b, iou) for matched pairs,
    plus unmatched indices from each side.
    """
    used_b = set()
    matches = []

    for i, ea in enumerate(list_a):
        page_a = ea.get("page_idx", -1)
        bbox_a = ea.get("bbox")
        if not bbox_a:
            continue

        best_j, best_iou = -1, 0.0
        for j, eb in enumerate(list_b):
            if j in used_b:
                continue
            if eb.get("page_idx", -1) != page_a:
                continue
            bbox_b = eb.get("bbox")
            if not bbox_b:
                continue
            iou = bbox_iou(bbox_a, bbox_b)
            if iou > best_iou:
                best_iou = iou
                best_j = j

        if best_iou >= iou_threshold and best_j >= 0:
            matches.append((i, best_j, best_iou))
            used_b.add(best_j)

    unmatched_a = [i for i in range(len(list_a))
                   if i not in {m[0] for m in matches}]
    unmatched_b = [j for j in range(len(list_b)) if j not in used_b]
    return matches, unmatched_a, unmatched_b


def compare_pair(list_a: list, list_b: list, label_a: str, label_b: str):
    """Compare two backend outputs element-by-element."""
    matches, unmatched_a, unmatched_b = align_elements(list_a, list_b)

    results = {
        "matched": len(matches),
        "unmatched_a": len(unmatched_a),
        "unmatched_b": len(unmatched_b),
        "mean_iou": 0.0,
        "by_type": defaultdict(lambda: {"count": 0, "ned_sum": 0.0, "neds": []}),
        "disagreements": [],
    }

    if matches:
        results["mean_iou"] = sum(m[2] for m in matches) / len(matches)

    for idx_a, idx_b, iou in matches:
        ea, eb = list_a[idx_a], list_b[idx_b]
        cat = classify_element(ea)
        if cat == "skip":
            continue

        text_a = extract_text(ea)
        text_b = extract_text(eb)
        ned = normalized_edit_distance(text_a, text_b)

        bucket = results["by_type"][cat]
        bucket["count"] += 1
        bucket["ned_sum"] += ned
        bucket["neds"].append(ned)

        if ned > 0.1:
            results["disagreements"].append({
                "page": ea.get("page_idx"),
                "type": ea.get("type"),
                "category": cat,
                "ned": round(ned, 4),
                "iou": round(iou, 4),
                f"bbox_{label_a}": ea.get("bbox"),
                f"bbox_{label_b}": eb.get("bbox"),
                f"text_{label_a}": text_a[:200],
                f"text_{label_b}": text_b[:200],
            })

    for cat, bucket in results["by_type"].items():
        if bucket["count"] > 0:
            bucket["mean_ned"] = round(bucket["ned_sum"] / bucket["count"], 4)
        else:
            bucket["mean_ned"] = 0.0
        del bucket["ned_sum"]
        del bucket["neds"]

    results["by_type"] = dict(results["by_type"])
    return results


def analyze_document(output_dir: str, doc: str):
    loaded = {}
    for b in BACKENDS:
        cl = load_content_list(output_dir, doc, b)
        if cl is None:
            return None
        loaded[b] = cl

    coverage = {}
    for b in BACKENDS:
        type_counts = defaultdict(int)
        for el in loaded[b]:
            cat = classify_element(el)
            if cat != "skip":
                type_counts[cat] += 1
        coverage[BACKEND_LABELS[b]] = dict(type_counts)

    pairs = [
        ("auto", "vlm"),
        ("auto", "hybrid_auto"),
        ("vlm", "hybrid_auto"),
    ]
    comparisons = {}
    for ba, bb in pairs:
        label = f"{BACKEND_LABELS[ba]}_vs_{BACKEND_LABELS[bb]}"
        comp = compare_pair(loaded[ba], loaded[bb],
                            BACKEND_LABELS[ba], BACKEND_LABELS[bb])
        comparisons[label] = comp

    return {"doc": doc, "coverage": coverage, "comparisons": comparisons}


def aggregate_results(doc_results: list):
    agg = {
        "n_docs": len(doc_results),
        "coverage_totals": defaultdict(lambda: defaultdict(int)),
        "pair_summary": {},
    }

    pairs = ["OCR_vs_VLM", "OCR_vs_Hybrid", "VLM_vs_Hybrid"]
    for pair in pairs:
        agg["pair_summary"][pair] = {
            "total_matched": 0,
            "total_unmatched_a": 0,
            "total_unmatched_b": 0,
            "by_type": defaultdict(lambda: {"count": 0, "ned_sum": 0.0}),
            "total_disagreements": 0,
        }

    for dr in doc_results:
        for backend, counts in dr["coverage"].items():
            for cat, n in counts.items():
                agg["coverage_totals"][backend][cat] += n

        for pair in pairs:
            comp = dr["comparisons"].get(pair)
            if not comp:
                continue
            ps = agg["pair_summary"][pair]
            ps["total_matched"] += comp["matched"]
            ps["total_unmatched_a"] += comp["unmatched_a"]
            ps["total_unmatched_b"] += comp["unmatched_b"]
            ps["total_disagreements"] += len(comp["disagreements"])

            for cat, bucket in comp["by_type"].items():
                abt = ps["by_type"][cat]
                abt["count"] += bucket["count"]
                abt["ned_sum"] += bucket["mean_ned"] * bucket["count"]

    for pair in pairs:
        ps = agg["pair_summary"][pair]
        for cat in ps["by_type"]:
            b = ps["by_type"][cat]
            if b["count"] > 0:
                b["mean_ned"] = round(b["ned_sum"] / b["count"], 4)
            else:
                b["mean_ned"] = 0.0
            del b["ned_sum"]
        ps["by_type"] = dict(ps["by_type"])

    agg["coverage_totals"] = {k: dict(v) for k, v in agg["coverage_totals"].items()}
    return agg


def main():
    parser = argparse.ArgumentParser(description="RQ1 Layer A: Pairwise parse quality comparison")
    parser.add_argument("--output-dir", required=True, help="MinerU output directory")
    parser.add_argument("--out", required=True, help="Output JSON path")
    parser.add_argument("--limit", type=int, default=0, help="Max docs to compare (0=all)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir
    docs = sorted(d for d in os.listdir(output_dir)
                  if os.path.isdir(os.path.join(output_dir, d)))

    all3 = [d for d in docs
            if all(os.path.isdir(os.path.join(output_dir, d, b)) for b in BACKENDS)]

    if args.limit > 0:
        all3 = all3[:args.limit]

    print(f"Comparing {len(all3)} documents with all 3 backends")

    doc_results = []
    top_disagreements = []

    for i, doc in enumerate(all3):
        result = analyze_document(output_dir, doc)
        if result is None:
            continue
        doc_results.append(result)

        for pair, comp in result["comparisons"].items():
            for d in comp["disagreements"]:
                d["doc"] = doc
                d["pair"] = pair
                top_disagreements.append(d)

        if args.verbose and (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(all3)} done")

    agg = aggregate_results(doc_results)

    top_disagreements.sort(key=lambda x: x["ned"], reverse=True)

    import random
    random.seed(42)
    ned_bins = {
        "1.0": [d for d in top_disagreements if d["ned"] >= 0.99],
        "0.7-0.99": [d for d in top_disagreements if 0.7 <= d["ned"] < 0.99],
        "0.5-0.7": [d for d in top_disagreements if 0.5 <= d["ned"] < 0.7],
        "0.3-0.5": [d for d in top_disagreements if 0.3 <= d["ned"] < 0.5],
        "0.1-0.3": [d for d in top_disagreements if 0.1 < d["ned"] < 0.3],
    }
    stratified = []
    per_bin = 100
    for bin_name, items in ned_bins.items():
        sample = random.sample(items, min(per_bin, len(items)))
        stratified.extend(sample)
    stratified.sort(key=lambda x: x["ned"], reverse=True)

    output = {
        "config": {
            "output_dir": output_dir,
            "n_docs": len(doc_results),
            "iou_threshold": 0.3,
            "ned_disagreement_threshold": 0.1,
            "ned_bin_counts": {k: len(v) for k, v in ned_bins.items()},
            "total_disagreements": len(top_disagreements),
        },
        "aggregate": agg,
        "top_disagreements": stratified,
        "per_document": [
            {
                "doc": dr["doc"],
                "coverage": dr["coverage"],
                "pair_neds": {
                    pair: {cat: bucket["mean_ned"]
                           for cat, bucket in comp["by_type"].items()}
                    for pair, comp in dr["comparisons"].items()
                },
            }
            for dr in doc_results
        ],
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {args.out}")
    print(f"\n{'='*60}")
    print(f"AGGREGATE RESULTS ({len(doc_results)} documents)")
    print(f"{'='*60}")

    print(f"\n--- Element Coverage ---")
    all_cats = set()
    for b in agg["coverage_totals"].values():
        all_cats.update(b.keys())
    all_cats = sorted(all_cats)
    header = f"{'Type':<12}" + "".join(f"{b:>10}" for b in ["OCR", "VLM", "Hybrid"])
    print(header)
    for cat in all_cats:
        row = f"{cat:<12}"
        for b in ["OCR", "VLM", "Hybrid"]:
            row += f"{agg['coverage_totals'].get(b, {}).get(cat, 0):>10}"
        print(row)

    print(f"\n--- Pairwise NED (lower = more similar) ---")
    for pair, ps in agg["pair_summary"].items():
        print(f"\n  {pair}:")
        print(f"    Matched elements: {ps['total_matched']}")
        print(f"    Disagreements (NED>0.1): {ps['total_disagreements']}")
        for cat, b in ps["by_type"].items():
            print(f"    {cat}: mean NED = {b['mean_ned']:.4f} (n={b['count']})")


if __name__ == "__main__":
    main()
