"""Dump every chunk's extraction text for the G2-lite-gliner rung (RQ2).

Uses build_graph's own chunking (`_embeddable_chunks`, `extract_text`), so GLiNER sees exactly the
text the LLM extractor saw, with the same chunk ids. One JSON line per chunk with text:
{"source", "chunk_id", "text"}.

    python evaluation/RQ2/build/rq2_dump_chunk_texts.py --domain ceramics --out data/rq2/gliner/ceramics/chunks.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from memento.config.cli_domain import apply_domain_arg  # noqa: E402

apply_domain_arg()
import memento.graph.build.build_graph as bg  # noqa: E402
from memento.config.settings import COLLECTION, OUTPUT_DIR  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    bg._OUTPUT_DIR = Path(OUTPUT_DIR).resolve()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    n_docs = n_chunks = 0
    missing = []
    with a.out.open("w") as f:
        for pid in bg.list_collection_sources(COLLECTION):
            pdir = bg._find_paper_dir(pid)
            cl = bg._find_content_list(pdir) if pdir else None
            if cl is None:
                missing.append(pid)
                continue
            items = json.loads(Path(bg.enrich_content_list(cl)).read_text())
            for c in bg._embeddable_chunks(items, pdir, pid):
                if c["extract_text"]:
                    f.write(json.dumps({"source": pid, "chunk_id": c["chunk_id"], "text": c["extract_text"]},
                                       ensure_ascii=False) + "\n")
                    n_chunks += 1
            n_docs += 1
    print(f"[dump] {COLLECTION}: documents={n_docs} chunks_with_text={n_chunks} missing={len(missing)} {missing[:5]}")
    if missing:
        sys.exit(1)


if __name__ == "__main__":
    main()
