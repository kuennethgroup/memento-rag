"""GLiNER entity extraction for the G2-lite-gliner rung (RQ2). Zero LLM calls.

Runs in its own venv (GLINER_VENV; the project venv's transformers is not touched) and
imports nothing from the project. Input: the chunk dump of evaluation/RQ2/build/rq2_dump_chunk_texts.py. Output:
one file per document, <out-dir>/<source>.json = {chunk_id: {"entities": [{"name", "type"}],
"relations": []}} -- the LLM extraction's format, read by `build_graph.py --no-llm --gliner-dir`,
which applies the same fusion (registry aliases, junk filters, type check).

Labels are the five entity types of the LLM prompt (memento/graph/registry_base.py DEFAULT_ENTITY_TYPES),
spelled as GLiNER-friendly phrases. Long chunks are split into overlapping word windows (GLiNER reads
at most 384 words).

    CUDA_VISIBLE_DEVICES=2 $GLINER_VENV/bin/python evaluation/RQ2/build/rq2_gliner_extract.py \
        --chunks data/rq2/gliner/ceramics/chunks.jsonl --out-dir data/rq2/gliner/ceramics/entities
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

MODEL = "urchade/gliner_multi-v2.1"          # multilingual (English + German corpora)
LABELS = {
    "material": "Material",
    "process": "Process",
    "property": "Property",
    "condition": "Condition",
    "characterization technique": "Characterization",
}
WINDOW, STRIDE = 200, 170                    # words; 30-word overlap


def windows(text: str) -> list[str]:
    words = text.split()
    if len(words) <= WINDOW:
        return [" ".join(words)]
    return [" ".join(words[i:i + WINDOW]) for i in range(0, max(1, len(words) - WINDOW + STRIDE), STRIDE)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--batch-size", type=int, default=32)
    a = ap.parse_args()

    import torch
    from gliner import GLiNER
    model = GLiNER.from_pretrained(MODEL).to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    rows = [json.loads(line) for line in a.chunks.open()]
    jobs = [(r["source"], r["chunk_id"], w) for r in rows for w in windows(r["text"])]
    print(f"[gliner] model={MODEL} chunks={len(rows)} windows={len(jobs)} threshold={a.threshold}", flush=True)

    found: dict[tuple[str, str], dict[tuple[str, str], str]] = defaultdict(dict)
    for r in rows:                                   # every chunk gets an entry, even with no entities
        found[(r["source"], r["chunk_id"])]
    t0 = time.time()
    labels = list(LABELS)
    for i in range(0, len(jobs), a.batch_size):
        batch = jobs[i:i + a.batch_size]
        with torch.no_grad():
            preds = model.batch_predict_entities([w for _, _, w in batch], labels, threshold=a.threshold)
        for (src, cid, _), ents in zip(batch, preds):
            for e in ents:
                name = " ".join(e["text"].split())
                if name:
                    found[(src, cid)].setdefault((name.lower(), LABELS[e["label"]]), name)
        if (i // a.batch_size) % 200 == 0:
            done = i + len(batch)
            rate = done / max(1e-9, time.time() - t0)
            print(f"[gliner] {done}/{len(jobs)} windows, {rate:.0f}/s, "
                  f"eta {(len(jobs) - done) / max(rate, 1e-9) / 60:.0f} min", flush=True)

    per_doc: dict[str, dict] = defaultdict(dict)
    n_ent = 0
    for (src, cid), ents in found.items():
        per_doc[src][cid] = {"entities": [{"name": n, "type": t} for (_, t), n in ents.items()], "relations": []}
        n_ent += len(ents)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for src, d in per_doc.items():
        (a.out_dir / f"{src}.json").write_text(json.dumps(d, ensure_ascii=False))
    print(f"[gliner] DONE documents={len(per_doc)} chunks={len(found)} entity_mentions={n_ent} "
          f"seconds={time.time() - t0:.0f}", flush=True)


if __name__ == "__main__":
    main()
