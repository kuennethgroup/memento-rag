#!/usr/bin/env python3
"""RQ1 Layer A: VLM-as-judge for parsing disagreements.

Takes disagreements from parse_quality.py output, crops the relevant region
from the original PDF page, and asks a VLM to judge which parser output
is more faithful to the original document.

Usage:
    python evaluation/RQ1/parse_judge.py \
        --disagreements evaluation/RQ1/parse_quality_ceramics_full.json \
        --output-dir data/output/Ceramics \
        --out evaluation/RQ1/parse_judge_ceramics.json \
        --limit 50
"""
import argparse
import base64
import io
import json
import os
import sys
import time
from pathlib import Path

import fitz  # PyMuPDF
import requests

API_URL = os.environ.get("REASONING_URL", "http://localhost:8642")
API_KEY = os.environ.get("REASONING_API_KEY", "")
MODEL = os.environ.get("REASONING_MODEL_NAME", "qwen3.8-27b")

BACKEND_FOLDERS = {"auto": "auto", "vlm": "vlm", "hybrid_auto": "hybrid_auto"}

JUDGE_PROMPT = """You are evaluating the accuracy of two document parsers. Below is a cropped region from a PDF page. Two parsers produced different text for this region.

**Parser A output:**
{text_a}

**Parser B output:**
{text_b}

Compare both outputs against what you see in the image. Judge which parser output is more faithful to the original document content.

Respond with EXACTLY one of these JSON objects (no other text):
{{"verdict": "A", "reason": "<one sentence>"}}
{{"verdict": "B", "reason": "<one sentence>"}}
{{"verdict": "TIE", "reason": "<one sentence>"}}
{{"verdict": "BOTH_WRONG", "reason": "<one sentence>"}}"""


def find_origin_pdf(output_dir: str, doc: str) -> str | None:
    """Find the _origin.pdf that MinerU stores alongside parsed output.

    Searches both flat (output_dir/<doc>/) and per-corpus
    (output_dir/<corpus>/<doc>/) layouts.
    """
    od = Path(output_dir)
    for backend in ["auto", "vlm", "hybrid_auto"]:
        p = od / doc / backend / f"{doc}_origin.pdf"
        if p.exists():
            return str(p)
    for corpus_dir in od.iterdir():
        if not corpus_dir.is_dir() or corpus_dir.name.startswith(("_", ".")):
            continue
        for backend in ["auto", "vlm", "hybrid_auto"]:
            p = corpus_dir / doc / backend / f"{doc}_origin.pdf"
            if p.exists():
                return str(p)
    return None


def bbox_to_pdf_points(bbox: list, page_width: float, page_height: float) -> list:
    """Convert MinerU 0-1000 normalized bbox to PDF points."""
    return [
        bbox[0] / 1000.0 * page_width,
        bbox[1] / 1000.0 * page_height,
        bbox[2] / 1000.0 * page_width,
        bbox[3] / 1000.0 * page_height,
    ]


def crop_bbox_from_pdf(pdf_path: str, page_idx: int, bbox: list,
                       dpi: int = 200, padding: int = 10) -> str | None:
    """Render a PDF page region and return base64 PNG.

    bbox is in MinerU's 0-1000 normalized coordinate system.
    """
    try:
        doc = fitz.open(pdf_path)
        if page_idx >= len(doc):
            doc.close()
            return None
        page = doc[page_idx]

        pdf_bbox = bbox_to_pdf_points(
            bbox, page.rect.width, page.rect.height)

        clip = fitz.Rect(
            max(0, pdf_bbox[0] - padding),
            max(0, pdf_bbox[1] - padding),
            min(page.rect.width, pdf_bbox[2] + padding),
            min(page.rect.height, pdf_bbox[3] + padding),
        )

        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, clip=clip)

        if pix.width <= 0 or pix.height <= 0:
            doc.close()
            return None

        png_bytes = pix.tobytes("png")
        doc.close()
        return base64.b64encode(png_bytes).decode("ascii")
    except Exception as e:
        print(f"  Crop error: {e}", file=sys.stderr)
        return None


def call_vlm_judge(image_b64: str, text_a: str, text_b: str,
                   max_retries: int = 2) -> dict:
    prompt = JUDGE_PROMPT.format(text_a=text_a[:500], text_b=text_b[:500])

    payload = {
        "model": MODEL,
        "max_tokens": 200,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                {"type": "text", "text": prompt},
            ]
        }]
    }

    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(
                f"{API_URL}/v1/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}"},
                json=payload,
                timeout=60,
            )
            if resp.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"].get("content", "")
            if not content:
                return {"verdict": "ERROR", "reason": "empty response", "raw": ""}
            content = content.strip()
            if content.startswith("{"):
                return json.loads(content)
            for line in content.split("\n"):
                line = line.strip()
                if line.startswith("{"):
                    try:
                        return json.loads(line)
                    except json.JSONDecodeError:
                        continue
            return {"verdict": "PARSE_FAIL", "reason": "could not parse JSON", "raw": content[:300]}
        except requests.exceptions.Timeout:
            if attempt < max_retries:
                time.sleep(2)
                continue
            return {"verdict": "ERROR", "reason": "timeout"}
        except Exception as e:
            return {"verdict": "ERROR", "reason": str(e)[:200]}

    return {"verdict": "ERROR", "reason": "max retries exceeded"}


def main():
    parser = argparse.ArgumentParser(description="RQ1: VLM judge for parse disagreements")
    parser.add_argument("--disagreements", required=True, help="parse_quality JSON file")
    parser.add_argument("--output-dir", required=True, help="MinerU output directory")
    parser.add_argument("--out", required=True, help="Output JSON path")
    parser.add_argument("--limit", type=int, default=50, help="Max disagreements to judge")
    parser.add_argument("--min-ned", type=float, default=0.2,
                        help="Only judge disagreements with NED >= this")
    args = parser.parse_args()

    with open(args.disagreements) as f:
        data = json.load(f)

    disags = data.get("top_disagreements", [])
    disags = [d for d in disags if d.get("ned", 0) >= args.min_ned]
    if args.limit > 0:
        disags = disags[:args.limit]

    print(f"Judging {len(disags)} disagreements (NED >= {args.min_ned})")

    pair_map = {
        "OCR_vs_VLM": ("OCR", "VLM"),
        "OCR_vs_Hybrid": ("OCR", "Hybrid"),
        "VLM_vs_Hybrid": ("VLM", "Hybrid"),
    }

    results = []
    verdicts = {"A": 0, "B": 0, "TIE": 0, "BOTH_WRONG": 0, "ERROR": 0, "PARSE_FAIL": 0}

    for i, d in enumerate(disags):
        doc = d["doc"]
        page = d["page"]
        pair = d["pair"]
        label_a, label_b = pair_map.get(pair, ("A", "B"))

        text_a_key = f"text_{label_a}"
        text_b_key = f"text_{label_b}"
        text_a = d.get(text_a_key, "")
        text_b = d.get(text_b_key, "")

        pdf_path = find_origin_pdf(args.output_dir, doc)
        if not pdf_path:
            print(f"  [{i+1}] {doc} — no origin PDF, skipping")
            continue

        bbox_a = d.get(f"bbox_{label_a}")
        bbox_b = d.get(f"bbox_{label_b}")
        if bbox_a and bbox_b:
            bbox = [
                min(bbox_a[0], bbox_b[0]),
                min(bbox_a[1], bbox_b[1]),
                max(bbox_a[2], bbox_b[2]),
                max(bbox_a[3], bbox_b[3]),
            ]
        else:
            bbox = bbox_a or bbox_b or d.get("bbox")
        if not bbox:
            print(f"  [{i+1}] {doc} p{page} — no bbox, skipping")
            continue

        image_b64 = crop_bbox_from_pdf(pdf_path, page, bbox)
        if not image_b64:
            print(f"  [{i+1}] {doc} p{page} — crop failed, skipping")
            continue

        verdict = call_vlm_judge(image_b64, text_a, text_b)
        v = verdict.get("verdict", "ERROR")
        verdicts[v] = verdicts.get(v, 0) + 1

        winner = "neither"
        if v == "A":
            winner = label_a
        elif v == "B":
            winner = label_b

        result = {
            "doc": doc,
            "page": page,
            "type": d.get("type"),
            "pair": pair,
            "ned": d.get("ned"),
            "bbox": bbox,
            "verdict": v,
            "winner": winner,
            "reason": verdict.get("reason", ""),
            f"text_{label_a}": text_a[:300],
            f"text_{label_b}": text_b[:300],
        }
        results.append(result)

        print(f"  [{i+1}/{len(disags)}] {doc} p{page} {d.get('type')} — "
              f"NED={d.get('ned'):.2f} → {v} ({winner}): {verdict.get('reason','')[:60]}")

    pair_wins = {}
    for r in results:
        pair = r["pair"]
        if pair not in pair_wins:
            pair_wins[pair] = {}
        w = r.get("winner", "neither")
        pair_wins[pair][w] = pair_wins[pair].get(w, 0) + 1

    output = {
        "config": {
            "model": MODEL,
            "n_judged": len(results),
            "min_ned": args.min_ned,
        },
        "verdict_counts": verdicts,
        "pair_wins": pair_wins,
        "judgments": results,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {args.out}")
    print(f"\n{'='*50}")
    print(f"VERDICT SUMMARY ({len(results)} judgments)")
    print(f"{'='*50}")
    for v, c in sorted(verdicts.items(), key=lambda x: -x[1]):
        if c > 0:
            print(f"  {v}: {c}")

    print(f"\nPer-pair wins:")
    for pair, wins in pair_wins.items():
        print(f"  {pair}: {wins}")


if __name__ == "__main__":
    main()
