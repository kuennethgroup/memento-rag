"""Build the per-document fact sheets of one corpus and index them in Qdrant (RQ2 fact-sheets arm).

Texts: the chunk dump of evaluation/RQ2/build/rq2_dump_chunk_texts.py (build_graph's own chunking, in document order),
so the sheets see exactly the text the graph extractors saw. Writes <doc>_fact_sheet.json (+ a raw
response cache) next to each document's parsed output, then (re)creates the `<prefix>-sheets` collection.
Resumable: documents with a sheet are skipped unless --force.

    python evaluation/RQ2/build/rq2_fact_sheets_build.py --domain ceramics --url http://localhost:8091/v1 --workers 8
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from memento.config.cli_domain import apply_domain_arg  # noqa: E402

apply_domain_arg()
import memento.graph.build.build_graph as bg  # noqa: E402
from memento.config.settings import COLLECTION, OUTPUT_DIR, QDRANT_URL  # noqa: E402
from memento.graph.build import fact_sheets as fs  # noqa: E402
from memento.graph.build.extraction_cache import RawResponseCache  # noqa: E402

MODEL = "qwen3.8-27b"


class FailoverRoundRobin:
    """OpenAI-client stand-in over identical servers: calls go round-robin; a connection error, timeout or
    5xx moves the call to the next server, so one dying server never crash-loops the build. Other errors
    (e.g. 400 "context too long", handled by extract_sheet) are raised unchanged."""

    def __init__(self, clients):
        import itertools
        import threading
        self._clients, self._next, self._lock = clients, itertools.count(), threading.Lock()
        self.chat = self.completions = self

    def create(self, **kw):
        import openai
        with self._lock:
            start = next(self._next)
        last = None
        for k in range(len(self._clients)):
            c = self._clients[(start + k) % len(self._clients)]
            try:
                return c.chat.completions.create(**kw)
            except (openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError) as exc:
                last = exc
        raise last


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--url", default="http://localhost:8091/v1", help="one server, or several comma-separated")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="only the first N documents (smoke test)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-index", action="store_true")
    a = ap.parse_args()
    from openai import OpenAI
    from qdrant_client import QdrantClient
    from memento.graph import canonicalize

    bg._OUTPUT_DIR = Path(OUTPUT_DIR).resolve()
    texts: dict[str, list[str]] = defaultdict(list)
    for line in Path(f"data/rq2/gliner/{a.domain}/chunks.jsonl").open():
        r = json.loads(line)
        texts[r["source"]].append(r["text"])
    docs = sorted(texts)[: a.limit or None]
    clients = [OpenAI(base_url=u.strip(), api_key="none", timeout=900, max_retries=3)
               for u in a.url.split(",") if u.strip()]
    llm = clients[0] if len(clients) == 1 else FailoverRoundRobin(clients)
    print(f"[fact sheets] servers: {[str(c.base_url) for c in clients]} workers={a.workers}", flush=True)
    t0 = time.time()

    def one(pid: str) -> dict:
        pdir = bg._find_paper_dir(pid)
        out = pdir / f"{pid}_fact_sheet.json"
        if out.exists() and not a.force:
            return json.loads(out.read_text())
        cache = RawResponseCache(pdir / f"{pid}_fact_sheet_cache.json")
        try:
            sheet = fs.extract_sheet(llm, MODEL, pid, texts[pid], cache=cache)
        except Exception as exc:        # one document never stops the others; the rerun retries it
            print(f"[FAIL] {pid}: {exc!r}"[:500], flush=True)
            return None
        finally:
            cache.save()
        out.write_text(json.dumps(sheet, ensure_ascii=False, indent=1))
        print(f"[{pid}] parts={sheet['n_parts']} failed={sheet['n_failed_parts']} "
              f"materials={len(sheet['materials'])} properties={len(sheet['properties'])} "
              f"calls={sheet['usage']['calls']} out_tokens={sheet['usage']['completion_tokens']}", flush=True)
        return sheet

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        results = list(pool.map(one, docs))
    failed_docs = [d for d, s_ in zip(docs, results) if s_ is None]
    sheets = [s_ for s_ in results if s_ is not None]
    tot = {k: sum(s["usage"].get(k, 0) for s in sheets) for k in ("calls", "cache_hits", "prompt_tokens", "completion_tokens")}
    print(json.dumps({"documents": len(sheets), "parts": sum(s["n_parts"] for s in sheets),
                      "failed_parts": sum(s["n_failed_parts"] for s in sheets),
                      "docs_with_failed_parts": sum(1 for s in sheets if s["n_failed_parts"]),
                      "failed_documents": len(failed_docs), "seconds": round(time.time() - t0), **tot}), flush=True)
    if failed_docs:
        # no index from a partial set: exit non-zero so the chain reruns; finished documents are skipped
        print(f"[fact sheets] {len(failed_docs)} documents failed, e.g. {failed_docs[:3]}; not indexing", flush=True)
        sys.exit(1)
    if not a.no_index:
        coll = fs.sheets_collection(COLLECTION)
        n = fs.index_sheets(QdrantClient(url=QDRANT_URL, timeout=120), coll,
                            [fs.payload(s, canonicalize) for s in sheets])
        print(f"[index] {coll}: {n} points", flush=True)


if __name__ == "__main__":
    main()
