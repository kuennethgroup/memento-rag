#!/usr/bin/env python3
"""
evaluation/RQ1/collections/render_pages.py — RQ1 supplementary study (UniDoc-Bench architecture):
render every page of every corpus PDF to a PNG, for the visual-document retrieval
channel (TomoroAI/tomoro-colqwen3-embed-4b in build_colqwen_collection.py).

Mirrors services/file_server.py:render_page's PyMuPDF (fitz) rendering exactly
(same 150 DPI), just done in bulk to disk instead of on-demand over HTTP.

Pages are named 0-indexed (page_000.png = first page) to match the `page`
payload field convention used everywhere else in the corpus (query_pipe.py,
memento/retrieval/search.py, etc. all treat page as 0-indexed).

Usage:
    python evaluation/RQ1/collections/render_pages.py --all
    python evaluation/RQ1/collections/render_pages.py --pdf paper.pdf
"""
import argparse
import sys
import time
from pathlib import Path

import fitz  # PyMuPDF

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from memento.config.settings import INPUT_DIR, OUTPUT_DIR

PAGE_RENDER_DPI = 150  # matches services/file_server.py


def fmt(s: float) -> str:
    if s < 60:
        return f"{s:.1f}s"
    m, sec = divmod(int(s), 60)
    return f"{m}m {sec}s"


def render_paper(pdf_path: Path) -> int:
    """Render every page of one PDF to <OUTPUT_DIR>/<stem>/page_images/page_NNN.png.
    Idempotent — skips if the expected number of images already exists."""
    stem = pdf_path.stem
    out_dir = OUTPUT_DIR / stem / "page_images"

    doc = fitz.open(pdf_path)
    n_pages = doc.page_count

    if out_dir.exists() and len(list(out_dir.glob("page_*.png"))) == n_pages:
        doc.close()
        return n_pages

    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_pages):
        page = doc[i]
        pix = page.get_pixmap(dpi=PAGE_RENDER_DPI)
        pix.save(out_dir / f"page_{i:03d}.png")
    doc.close()
    return n_pages


def main():
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--pdf", help="Single PDF filename in data/input/")
    group.add_argument("--all", action="store_true")
    args = ap.parse_args()

    if args.pdf:
        pdfs = [INPUT_DIR / args.pdf]
    else:
        pdfs = sorted(INPUT_DIR.glob("*.pdf"))

    print(f"Rendering pages for {len(pdfs)} PDF(s) at {PAGE_RENDER_DPI} DPI...")
    t0 = time.perf_counter()
    total_pages = 0
    for i, pdf_path in enumerate(pdfs):
        n = render_paper(pdf_path)
        total_pages += n
        if (i + 1) % 20 == 0 or (i + 1) == len(pdfs):
            elapsed = time.perf_counter() - t0
            print(f"  [{i+1}/{len(pdfs)}] {pdf_path.stem}: {n} pages "
                  f"(total {total_pages} pages, {fmt(elapsed)} elapsed)")

    print(f"\nDone. {total_pages} pages rendered across {len(pdfs)} papers in "
          f"{fmt(time.perf_counter() - t0)}.")


if __name__ == "__main__":
    main()
