#!/usr/bin/env python3
"""
evaluation/RQ1/pool_relevance_all.py — TREC-style pooling of the RQ1 ground
truth across every collection of the RQ1 matrix.

Each collection's retrieval can surface relevant (source, page) pairs the
seed ground truth never saw: OCR and Hybrid-engine (other parsers), CTX
(embed_text with situating context), NoEnrich (other chunk boundaries: no
paragraph merge, no low-info filter) and TEXT-8b (another embedder). A
condition scored against a pool that never had a chance to include its
candidates is under-credited.

For every query, retrieve top-k from each pool source via production
retrieval, judge ONLY candidate (source,
page) pairs not already covered by ANY collection judged so far (this run's
new candidates included — later sources in POOL_SOURCES benefit from
earlier ones' finds), append accepted ones with
annotator="llm-auto-pooled-<source_key>" so provenance stays traceable to
which collection surfaced it.

TEXT-8b needs its own embed path (evaluation.RQ1.collections.local_text_embed.local_text_embed_query)
since memento-rag-text was NOT built with the multimodal Qwen3-VL-Embedding-8B —
passing it through embed_query (the VL embedder) would silently embed the query
with the wrong model and produce garbage similarity scores. search()'s `qvec`
parameter exists exactly to support pre-embedded queries for this reason.

Usage:
    python evaluation/RQ1/pool_relevance_all.py --domain <d> \
        --in evaluation/RQ1/ground_truth_auto.json \
        --out evaluation/RQ1/ground_truth_auto_pooled_all.json
"""
import argparse
import os
import threading
from concurrent.futures import ThreadPoolExecutor
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from memento.config.cli_domain import apply_domain_arg, domain_banner  # noqa: E402
apply_domain_arg()
from memento.config.settings import (
    COLLECTION, OCR_COLLECTION, HYBRID_ENGINE_COLLECTION, CONTEXT_COLLECTION,
    NOENRICH_COLLECTION, TEXT_COLLECTION, NOENRICH_TEXT_COLLECTION,
    QDRANT_URL, EMBEDDER_URL, RERANKER_URL, DOMAIN,
)

from qdrant_client import QdrantClient

sys.path.insert(0, str(Path(__file__).parent))
import auto_annotate
from auto_annotate import (
    get_llm, get_judge_llm, judge_hits, QUERY_INSTRUCTION,
    RETRIEVE_K, RERANK_TOP, RERANK_MIN_SCORE, IMAGE_RERANK_RATIO,
    HYBRID_ENABLED, HYBRID_PREFETCH_K,
)
from memento.retrieval.search import search as production_search, embed_query
from evaluation.RQ1.collections.local_text_embed import local_text_embed_query

import json

# (short key for annotator tag, collection, embedder) — embedder is "local"
# (Qwen3-VL-Embedding-8B, embed_query) or "text8b" (Qwen3-Embedding-8B,
# local_text_embed_query). "vlm" (the GT's own seed collection) matters for
# top-up runs after a rebuild, when VLM itself can surface unjudged pages.
POOL_SOURCES = [
    ("vlm",      COLLECTION,               "local"),
    ("ocr",      OCR_COLLECTION,           "local"),
    ("hybrid",   HYBRID_ENGINE_COLLECTION, "local"),
    ("ctx",      CONTEXT_COLLECTION,       "local"),
    ("noenrich", NOENRICH_COLLECTION,      "local"),
    ("text8b",   TEXT_COLLECTION,          "text8b"),
    ("noenrichtxt", NOENRICH_TEXT_COLLECTION, "text8b"),
]


def judged_pages_from_logs(paths: list[str]) -> dict[str, set]:
    """Every (source, page) any earlier judge call saw, per query text —
    relevant AND rejected — rebuilt from --log-calls JSONL files. The judge
    prompt labels each candidate '[i] <source> (page <page+1>, <type>...'."""
    import re
    label = re.compile(r"^\[\d+\] (.+?) \(page (\d+), ", re.M)
    judged: dict[str, set] = {}
    for path in paths:
        for line in open(path):
            d = json.loads(line)
            if d.get("kind") != "judge":
                continue
            user = next(m for m in d["messages"] if m["role"] == "user")["content"]
            texts = [user] if isinstance(user, str) else [c.get("text", "") for c in user if c.get("type") == "text"]
            pairs = judged.setdefault(d["query"], set())
            for t in texts:
                for src, page in label.findall(t.lstrip("\n")):
                    pairs.add((src, int(page) - 1))
    return judged


def _embed(query: str, embedder: str) -> list[float]:
    if embedder == "text8b":
        return local_text_embed_query(query)
    return embed_query(query, EMBEDDER_URL, QUERY_INSTRUCTION)


def pool_query(llm, model, client, collection: str, embedder: str,
               query: str, existing_pages: set, annotator: str) -> list[dict]:
    qvec = _embed(query, embedder)
    hits, _ = production_search(
        client, collection, query,
        embedder_url=EMBEDDER_URL, query_instruction=QUERY_INSTRUCTION,
        retrieve_k=RETRIEVE_K, hybrid=HYBRID_ENABLED, hybrid_prefetch_k=HYBRID_PREFETCH_K,
        rerank_enabled=True, reranker_url=RERANKER_URL, rerank_top=RERANK_TOP,
        rerank_min_score=RERANK_MIN_SCORE, image_rerank_ratio=IMAGE_RERANK_RATIO,
        qvec=qvec,
    )
    new_hits = [h for h in hits if (h["source"], h["page"]) not in existing_pages]
    if not new_hits:
        return [], set()
    judged = {(h["source"], h["page"]) for h in new_hits}
    for attempt in (1, 2):
        try:
            return judge_hits(llm, model, query, new_hits, collection, annotator=annotator), judged
        except Exception as e:
            if attempt == 2:
                # A silently skipped query would leave its new pages unjudged and
                # scored as non-relevant — stop instead (the run is resumable).
                raise RuntimeError(f"judge_hits failed twice for {query[:60]!r}: {e!r}") from e
            print(f"    judge_hits failed ({e!r}), retrying once")


def main():
    # A failed rerank must abort, never degrade silently to vector order.
    os.environ["MEMENTO_STRICT_RERANK"] = "1"
    ap = argparse.ArgumentParser()
    ap.add_argument('--domain', required=True, help='Corpus domain profile (config/domains/<name>.yaml), e.g. ceramics | polymer-engineering')
    ap.add_argument("--in", dest="inp", default=str(Path(__file__).parent / "ground_truth_auto.json"))
    ap.add_argument("--out", default=str(Path(__file__).parent / "ground_truth_auto_pooled_all.json"))
    ap.add_argument("--resume", action="store_true",
                     help="If --out already exists, resume from it (skips sources already marked done per query)")
    ap.add_argument("--sources", nargs="*", default=[k for k, _, _ in POOL_SOURCES],
                     help="Subset of pool source keys to run (default: all)")
    ap.add_argument("--run-tag", default="from",
                     help="Per-run done-flag/annotator tag (pooled_<tag>_<source>); use a new tag "
                          "for a top-up so earlier runs' done flags don't skip every query")
    ap.add_argument("--judged-calls", nargs="*", default=[],
                     help="Earlier judge-call JSONL logs: every (source, page) they judged — "
                          "relevant or not — is skipped instead of re-judged")
    ap.add_argument("--judge-base-url", default=None,
                     help="OpenAI-compatible endpoint for the judge (e.g. http://localhost:8090/v1); default: cloud judge")
    ap.add_argument("--judge-model", default=None, help="Judge model name for --judge-base-url")
    ap.add_argument("--fresh", action="store_true",
                     help="Start from empty labels: drop the input's relevant_chunks/judged_pages and ignore "
                          "--judged-calls, so every pooled page is judged by this run's judge")
    ap.add_argument("--workers", type=int, default=4,
                     help="Queries judged in parallel per pool source")
    ap.add_argument("--log-calls", default=None,
                     help="Append every judge call's exact request+response (JSONL) to this path.")
    args = ap.parse_args()
    domain_banner()

    if args.log_calls:
        auto_annotate.CALL_LOG_PATH = Path(args.log_calls)
        auto_annotate.CALL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"Logging every LLM call to {auto_annotate.CALL_LOG_PATH}")

    active_sources = [s for s in POOL_SOURCES if s[0] in args.sources]
    if not active_sources:
        print(f"No matching pool sources for --sources {args.sources}", file=sys.stderr)
        sys.exit(1)

    gt = json.loads(Path(args.inp).read_text())
    if args.resume and Path(args.out).exists():
        gt = json.loads(Path(args.out).read_text())

    client = QdrantClient(url=QDRANT_URL, timeout=30)
    if args.judge_base_url:
        from openai import OpenAI
        llm, model = OpenAI(base_url=args.judge_base_url, api_key="none", timeout=180.0, max_retries=0), args.judge_model
    else:
        llm, model = get_judge_llm()
    print(f"Judge: {model} @ {llm.base_url}", flush=True)
    if args.fresh and not (args.resume and Path(args.out).exists()):
        for q in gt["queries"]:
            q["relevant_chunks"], q["judged_pages"] = [], []
        args.judged_calls = []

    lock = threading.Lock()
    judged_logs = judged_pages_from_logs(args.judged_calls)
    if args.judged_calls:
        n = sum(len(v) for v in judged_logs.values())
        print(f"Seeded {n} previously judged (source, page) pairs for {len(judged_logs)} queries from logs")

    for key, collection, embedder in active_sources:
        print(f"\n=== Pooling source: {key}  ({collection}, embedder={embedder}) ===")
        n_new_total = 0
        done_flag = f"pooled_{args.run_tag}_{key}"
        annotator = "llm-auto-pooled-" + (key if args.run_tag == "from" else f"{args.run_tag}-{key}")

        def pool_one(i, q):
            # queries are independent; parallel workers only overlap the cloud
            # judge's intermittent stalls — results are identical to a serial run
            judged = {tuple(x) for x in q.get("judged_pages", [])}
            judged |= judged_logs.get(q["query"], set())
            judged |= {(c["paper_id"], c["page"]) for c in q["relevant_chunks"]}
            new_relevant, newly_judged = pool_query(
                llm, model, client, collection, embedder, q["query"], judged,
                annotator=annotator,
            )
            with lock:
                q["relevant_chunks"].extend(new_relevant)
                q["judged_pages"] = sorted([list(x) for x in judged | newly_judged])
                q[done_flag] = True
                print(f"  [{i+1}/{len(gt['queries'])}] {q['query'][:60]!r} -> +{len(new_relevant)} new", flush=True)
                # checkpoint after every query — safe to interrupt/resume
                Path(args.out).write_text(json.dumps(gt, indent=2, ensure_ascii=False))
            return len(new_relevant)

        pending = [(i, q) for i, q in enumerate(gt["queries"]) if not q.get(done_flag)]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(pool_one, i, q) for i, q in pending]
            for f in futures:
                n_new_total += f.result()

        print(f"  {key}: {n_new_total} new relevant_chunks added across {len(gt['queries'])} queries.")

    print(f"\nDone. Wrote {args.out}")


if __name__ == "__main__":
    main()
