#!/usr/bin/env python3
"""
evaluation/RQ1/rq1_eval.py — RQ1 evaluator: parsing (VLM vs OCR) and retrieval-
mode (dense/hybrid/+rerank) comparison, plus the modality-content ablation
all on production retrieval (memento/retrieval/search.py).

Relevance is keyed on (source, page) — parser-independent (chunk_id is not
portable between the VLM and OCR parses). Ground truth must already be
pooled across the collections (pool_relevance_all.py) before these numbers
are treated as final — scoring a condition against a pool it never fed
under-credits it for pages it alone retrieves.

Conditions are explicit (collection, hybrid, rerank, type_filter) tuples —
no bogus cells (e.g. no claim about "OCR + multimodal-ablation", since the
modality-content ablation is specifically a VLM-collection question).

Metrics: Hit@{1,5,10,20}, MRR, Recall@{1,5,10,20}, and NDCG@{1,5,10,20}, computed by deduplicating each ranked hit
list down to unique (source, page) pages first (several chunks can share a
page) — rank is the position of the first such page, not the raw chunk
position.

Usage:
    python evaluation/RQ1/rq1_eval.py --domain <d> --in evaluation/RQ1/ground_truth_auto_pooled_all.json \
        --out evaluation/RQ1/rq1_results.json
"""
import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parents[2]))
from memento.config.cli_domain import apply_domain_arg, domain_banner  # noqa: E402
apply_domain_arg()
from memento.config.settings import COLLECTION, OCR_COLLECTION, TEXT_COLLECTION, HYBRID_ENGINE_COLLECTION, CONTEXT_COLLECTION, NOENRICH_COLLECTION, NOENRICH_TEXT_COLLECTION, QDRANT_URL, EMBEDDER_URL, RERANKER_URL, LOCAL_TEXT_RERANK_URL, LOCAL_TEXT_EMBED_URL, DOMAIN, OUTPUT_DIR

from qdrant_client import QdrantClient

from memento.retrieval.search import search as production_search, embed_query, build_filter, rerank as cross_rerank
from evaluation.RQ1.collections.local_text_embed import local_text_embed_query
from evaluation.RQ1.collections.colqwen_search import colqwen_search
from memento.retrieval.search import rrf_fuse

QUERY_INSTRUCTION = DOMAIN["instruction"]

RETRIEVE_K = 80
RERANK_TOP = 20
RERANK_MIN_SCORE = 0.03
IMAGE_RERANK_RATIO = 0.2
HYBRID_PREFETCH_K = 80
K_VALUES = [1, 5, 10, 20]

# Second VL-Reranker-8B instance (same model, own GPU) so the ~14 "local"-
# reranker conditions per query can be split across two reranker services
# instead of serializing through one — reranker.process() itself is a stable
# ~2.9s/call for 80 candidates (real compute, confirmed via direct timing),
# the actual bottleneck is that CONDITIONS is evaluated strictly
# sequentially. Defaults to RERANKER_URL itself (a no-op — everything routes
# to the one instance, unchanged behavior) when no replica is running.
RERANKER_URL_REPLICA = os.getenv("RERANKER_URL_REPLICA", RERANKER_URL)
# All conditions (36 + a few headroom) submitted at once — real concurrency is
# capped by the backend services themselves (2 reranker replicas + 1
# text-reranker + qdrant), not by an artificially small thread-pool size. A
# low worker count here just re-serializes work into batches instead of
# letting the natural per-service bottleneck set the pace (confirmed
# empirically: max_workers=6 gave almost no speedup over fully sequential).
CONDITION_WORKERS = int(os.getenv("RQ1_CONDITION_WORKERS", "44"))

# (name, collection, hybrid, rerank_enabled, type_filter, embedder, reranker)
# embedder is "local" (Qwen3-VL-Embedding-8B, the production embedder) or
# "text8b" (Qwen3-Embedding-8B, text-only, served locally — the
# embedding-MODEL ablation; same size/dim as the VL model, same instruction
# treatment, isolating only the vision-language component — see
# evaluation/RQ1/collections/local_text_embed.py).
# reranker is "local" (Qwen3-VL-Reranker-8B, the production cross-encoder,
# RERANKER_URL) or "text8b" (Qwen3-Reranker-8B, text-only sibling, served via
# services/text_reranker_api.py at LOCAL_TEXT_RERANK_URL — the
# reranker-MODEL ablation). The reranker reranks raw text content, not
# vectors, so it's independent of which embedder retrieved the candidates —
# text8b-embedded candidates can still be reranked by either reranker, kept
# symmetric with the embedder ablation above.
CONDITIONS = [
    ("VLM-dense",                  COLLECTION,      False, False, None, "local", "local"),
    ("VLM-dense+rerank",           COLLECTION,      False, True,  None, "local", "local"),
    ("VLM-hybrid",                 COLLECTION,      True,  False, None, "local", "local"),
    ("VLM-hybrid+rerank",          COLLECTION,      True,  True,  None, "local", "local"),
    ("OCR-dense",                  OCR_COLLECTION,  False, False, None, "local", "local"),
    ("OCR-dense+rerank",           OCR_COLLECTION,  False, True,  None, "local", "local"),
    ("OCR-hybrid",                 OCR_COLLECTION,  True,  False, None, "local", "local"),
    ("OCR-hybrid+rerank",          OCR_COLLECTION,  True,  True,  None, "local", "local"),
    # Hybrid-engine conditions are not run: the hybrid parser tied VLM and beat OCR on an earlier corpus. The core
    # comparison is VLM vs OCR (baseline). Modality-content ablation: same VLM collection/embedder,
    # retrieval restricted to text-only chunks vs the unrestricted default (VLM-hybrid+rerank above): "does retrievable
    # multimodal content help", especially on visual queries. Full 4-way retrieval-mode grid (dense / dense+rerank /
    # hybrid / hybrid+rerank), so "does reranking compensate for reduced modality access" can be checked as for every
    # other factor.
    ("VLM-dense-textonly",           COLLECTION,      False, False, ["text"], "local", "local"),
    ("VLM-dense+rerank-textonly",    COLLECTION,      False, True,  ["text"], "local", "local"),
    ("VLM-hybrid-textonly",          COLLECTION,      True,  False, ["text"], "local", "local"),
    ("VLM-hybrid+rerank-textonly",   COLLECTION,      True,  True,  ["text"], "local", "local"),
    # Third modality level per the thesis proposal's "text-only, text+table,
    # text+table+figure" ablation — the textonly conditions above are level 1;
    # this is level 2. Level 3 is the unrestricted VLM-* conditions.
    ("VLM-dense-text+table",         COLLECTION,      False, False, ["text", "table"], "local", "local"),
    ("VLM-dense+rerank-text+table",  COLLECTION,      False, True,  ["text", "table"], "local", "local"),
    ("VLM-hybrid-text+table",        COLLECTION,      True,  False, ["text", "table"], "local", "local"),
    ("VLM-hybrid+rerank-text+table", COLLECTION,      True,  True,  ["text", "table"], "local", "local"),
    # Embedding-model ablation: same text-type chunks, same
    # retrieval pipeline, different embedder (local multimodal Qwen3-VL-
    # Embedding-8B vs local text-only Qwen3-Embedding-8B). Only meaningful on
    # target_type == "text" queries, since memento-rag-text contains no
    # non-text content.
    ("TEXT-8b-dense",                 TEXT_COLLECTION, False, False, None, "text8b", "local"),
    ("TEXT-8b-dense+rerank",          TEXT_COLLECTION, False, True,  None, "text8b", "local"),
    ("TEXT-8b-hybrid",                TEXT_COLLECTION, True,  False, None, "text8b", "local"),
    ("TEXT-8b-hybrid+rerank",         TEXT_COLLECTION, True,  True,  None, "text8b", "local"),
    # Reranker-model ablation (mirrors the embedding ablation above, but
    # holds the embedder/collection fixed and swaps the CROSS-ENCODER
    # instead): production Qwen3-VL-Reranker-8B vs its text-only sibling
    # Qwen3-Reranker-8B (services/text_reranker_api.py, LOCAL_TEXT_RERANK_URL).
    ("VLM-dense+textrerank",         COLLECTION,      False, True, None, "local",  "text8b"),
    ("VLM-hybrid+textrerank",        COLLECTION,      True,  True, None, "local",  "text8b"),
    ("TEXT-8b-dense+textrerank",     TEXT_COLLECTION, False, True, None, "text8b", "text8b"),
    ("TEXT-8b-hybrid+textrerank",    TEXT_COLLECTION, True,  True, None, "text8b", "text8b"),
    # Contextual Retrieval ablation (Anthropic situating-context) — same VLM
    # parse + local embedder as the COLLECTION conditions; the ONLY variable is
    # that each chunk's embed_text was prefixed with an LLM-generated sentence
    # situating it in its paper (CONTEXTUALIZE=true at ingest; CONTEXT_BM25
    # stayed false, so the sparse/BM25 signal is identical to VLM's). Contrast
    # CTX-* vs the matching VLM-* condition to isolate the context effect.
    ("CTX-dense",                  CONTEXT_COLLECTION, False, False, None, "local", "local"),
    ("CTX-dense+rerank",           CONTEXT_COLLECTION, False, True,  None, "local", "local"),
    ("CTX-hybrid",                 CONTEXT_COLLECTION, True,  False, None, "local", "local"),
    ("CTX-hybrid+rerank",          CONTEXT_COLLECTION, True,  True,  None, "local", "local"),
    # Enrichment ablation — same VLM parse + local embedder as the COLLECTION
    # conditions; the ONLY variable is that ingestion skipped
    # enrich_content_list() (--no-enrich): no figure/table/equation context
    # grounding, no section assignment, no junk-fragment filtering. Contrast
    # NoEnrich-* vs the matching VLM-* condition to isolate the enrichment
    # effect. Ingested via
    # `run_pipeline.py --no-enrich --collection config.NOENRICH_COLLECTION`.
    ("NoEnrich-dense",             NOENRICH_COLLECTION, False, False, None, "local", "local"),
    ("NoEnrich-dense+rerank",      NOENRICH_COLLECTION, False, True,  None, "local", "local"),
    ("NoEnrich-hybrid",            NOENRICH_COLLECTION, True,  False, None, "local", "local"),
    ("NoEnrich-hybrid+rerank",     NOENRICH_COLLECTION, True,  True,  None, "local", "local"),
    # NoEnrich + text-only embedder — clean VL-vs-text embedder ablation WITHOUT
    # enrichment confound. With enrichment, 100% of image chunks have rich text
    # descriptions (median 496 chars) making VL pixel embedding redundant. Without
    # enrichment, 48% of image chunks have <10 chars text — this is where VL
    # pixels should actually matter. Contrast NoEnrichTXT-* vs NoEnrich-* to
    # isolate the embedder difference on raw (unenriched) content.
    ("NoEnrichTXT-dense",           NOENRICH_TEXT_COLLECTION, False, False, None, "text8b", "local"),
    ("NoEnrichTXT-dense+rerank",    NOENRICH_TEXT_COLLECTION, False, True,  None, "text8b", "local"),
    ("NoEnrichTXT-hybrid",          NOENRICH_TEXT_COLLECTION, True,  False, None, "text8b", "local"),
    ("NoEnrichTXT-hybrid+rerank",   NOENRICH_TEXT_COLLECTION, True,  True,  None, "text8b", "local"),
]

# RQ1 visual-document retrieval sub-study (UniDoc-Bench style) — retrieval architecture
# contrast. Not collection-
# tuple conditions like CONDITIONS above (ColQwen has no dense/sparse named
# vectors, no chunk text) — handled by their own eval functions below, but
# folded into the same per-query result dict / hit@k / mrr shape.
COLQWEN_CONDITION_NAMES = [
    "ColQwen-only", "ColQwen-only+rerank",
    "Fusion-text+colqwen", "Fusion-text+colqwen+rerank",
]
COLQWEN_TOP_K = 20  # per-channel top-k before fusion (plan deviation from UniDoc-Bench's top-5 — see plan)


def dedup_to_pages(hits: list[dict]) -> list[tuple[str, int]]:
    """First-occurrence-wins page list — several chunks can share a page;
    rank should reflect the page's earliest appearance, not chunk count."""
    seen = set()
    pages = []
    for h in hits:
        key = (h["source"], h["page"])
        if key not in seen:
            seen.add(key)
            pages.append(key)
    return pages


def score_pages(pages: list[tuple], relevant_set: set) -> dict:
    """Hit@k / MRR / Recall@k over an already-deduped, ranked (source, page) list.

    Hit@k and MRR both stop at the FIRST relevant page found — they answer
    "did the system find *something* relevant in the top k" and are
    guaranteed to saturate near 1.0 once retrieval is good enough for the
    majority of queries that have only one relevant page (Hit@k==1 for any
    k >= that single page's rank). Recall@k answers a different, still-
    discriminating question for multi-relevant-page queries: of ALL the
    relevant pages, what fraction did the top-k actually surface — so it
    keeps separating conditions even after Hit@k has ceilinged out."""
    result = {f"hit@{k}": 0 for k in K_VALUES}
    result["mrr"] = 0.0
    for rank, page in enumerate(pages):
        if page in relevant_set:
            for k in K_VALUES:
                if rank < k:
                    result[f"hit@{k}"] = 1
            result["mrr"] = 1.0 / (rank + 1)
            break
    n_relevant = len(relevant_set)
    for k in K_VALUES:
        if n_relevant == 0:
            result[f"recall@{k}"] = 0.0
        else:
            found = sum(1 for page in pages[:k] if page in relevant_set)
            result[f"recall@{k}"] = found / n_relevant
    for k in K_VALUES:
        dcg = sum(1.0 / math.log2(r + 2) for r, p in enumerate(pages[:k]) if p in relevant_set)
        idcg = sum(1.0 / math.log2(r + 2) for r in range(min(k, n_relevant)))
        result[f"ndcg@{k}"] = dcg / idcg if idcg > 0 else 0.0
    result["n_results"] = len(pages)
    return result


def eval_one(client, qvec, query: str, relevant_set: set,
             collection: str, hybrid: bool, rerank_enabled: bool,
             type_filter: list | None, reranker_url: str = RERANKER_URL) -> dict:
    query_filter = build_filter(type_filter=type_filter)
    final_hits, _pre = production_search(
        client, collection, query,
        embedder_url=EMBEDDER_URL, query_instruction=QUERY_INSTRUCTION,
        retrieve_k=RETRIEVE_K, hybrid=hybrid, hybrid_prefetch_k=HYBRID_PREFETCH_K,
        rerank_enabled=rerank_enabled, reranker_url=reranker_url, rerank_top=RERANK_TOP,
        rerank_min_score=RERANK_MIN_SCORE, image_rerank_ratio=IMAGE_RERANK_RATIO,
        query_filter=query_filter, qvec=qvec,
        rerank_multimodal=(reranker_url != LOCAL_TEXT_RERANK_URL),
    )
    pages = dedup_to_pages(final_hits)
    out = score_pages(pages, relevant_set)
    # persisted so the same run can be re-scored against another ground truth
    # without repeating retrieval
    out["ranked_pages"] = [list(p) for p in pages[:20]]
    return out


def _page_img_path(source: str, page: int) -> str | None:
    p = OUTPUT_DIR / source / "page_images" / f"page_{page:03d}.png"
    return str(p) if p.exists() else None


def _colqwen_pseudo_hit(source: str, page: int, score: float = 0.0) -> dict:
    """Page-level, no chunk text — type forced to 'image' so cross_rerank
    attaches the rendered page PNG."""
    return {"text": "", "source": source, "page": page, "type": "image",
            "section": None, "img_path": _page_img_path(source, page), "score": score}


def eval_colqwen(client, query: str, relevant_set: set, rerank_enabled: bool) -> dict:
    hits = colqwen_search(client, query, COLQWEN_TOP_K)
    pseudo = [_colqwen_pseudo_hit(h["source"], h["page"], h["score"]) for h in hits]
    if rerank_enabled:
        pseudo = cross_rerank(query, pseudo, RERANKER_URL, RERANK_TOP,
                               RERANK_MIN_SCORE, IMAGE_RERANK_RATIO)
    return score_pages(dedup_to_pages(pseudo), relevant_set)


def eval_fusion(client, query: str, qvec_text8b, relevant_set: set, rerank_enabled: bool) -> dict:
    text_hits, _ = production_search(
        client, TEXT_COLLECTION, query,
        embedder_url=EMBEDDER_URL, query_instruction=QUERY_INSTRUCTION,
        retrieve_k=COLQWEN_TOP_K, hybrid=False, hybrid_prefetch_k=COLQWEN_TOP_K,
        rerank_enabled=False, reranker_url=RERANKER_URL, rerank_top=COLQWEN_TOP_K,
        rerank_min_score=0.0, image_rerank_ratio=0.0, qvec=qvec_text8b,
    )
    colqwen_hits = colqwen_search(client, query, COLQWEN_TOP_K)

    def key(s, p):
        return f"{s}::{p}"

    text_ranked = [key(h["source"], h["page"]) for h in text_hits]
    colqwen_ranked = [key(h["source"], h["page"]) for h in colqwen_hits]
    fused_scores = rrf_fuse([text_ranked, colqwen_ranked])
    fused_order = sorted(fused_scores, key=lambda k: fused_scores[k], reverse=True)

    text_by_key = {key(h["source"], h["page"]): h for h in text_hits}
    colqwen_by_key = {key(h["source"], h["page"]): h for h in colqwen_hits}

    if not rerank_enabled:
        pages = []
        seen = set()
        for k in fused_order:
            source, page = k.rsplit("::", 1)
            pair = (source, int(page))
            if pair not in seen:
                seen.add(pair)
                pages.append(pair)
        return score_pages(pages, relevant_set)

    fused_hits = []
    for k in fused_order:
        source, page = k.rsplit("::", 1)
        page = int(page)
        if k in colqwen_by_key:
            fused_hits.append(_colqwen_pseudo_hit(source, page, colqwen_by_key[k]["score"]))
        else:
            fused_hits.append(text_by_key[k])
    ranked = cross_rerank(query, fused_hits, RERANKER_URL, RERANK_TOP,
                           RERANK_MIN_SCORE, IMAGE_RERANK_RATIO)
    return score_pages(dedup_to_pages(ranked), relevant_set)


def _git_rev() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5,
            cwd=str(Path(__file__).parent),
        ).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _git_worktree_state() -> dict:
    # git_rev alone hides uncommitted changes: hash the working-tree diff and list untracked files so dirty runs are
    # distinguishable.
    def _git(*args):
        return subprocess.run(["git", *args], capture_output=True, text=True, timeout=30,
                              cwd=str(Path(__file__).parent)).stdout
    try:
        diff = _git("diff", "HEAD")
        untracked = _git("ls-files", "--others", "--exclude-standard").split()
        return {"dirty": bool(diff.strip() or untracked),
                "diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
                "untracked": untracked}
    except Exception:
        return {"dirty": "unknown"}


def _served_models() -> dict:
    out = {}
    for name, url in [("embedder", EMBEDDER_URL), ("reranker", RERANKER_URL),
                      ("text_embedder", LOCAL_TEXT_EMBED_URL), ("text_reranker", LOCAL_TEXT_RERANK_URL)]:
        try:
            data = requests.get(f"{url}/v1/models", timeout=5).json()["data"][0]
            out[name] = {"url": url, "model": data.get("root") or data.get("id"),
                         "max_model_len": data.get("max_model_len")}
        except Exception:
            out[name] = {"url": url, "model": None}
    return out


def _provenance(gt_path: str) -> dict:
    """Stamp every results file with the ground truth and code that produced it, so results of different GT
    versions or configurations can be told apart after the fact."""
    gt_bytes = Path(gt_path).read_bytes()
    return {
        "ground_truth_path": str(Path(gt_path).resolve()),
        "ground_truth_sha256": hashlib.sha256(gt_bytes).hexdigest(),
        "git_rev": _git_rev(),
        "worktree": _git_worktree_state(),
        "served_models": _served_models(),
        "domain": os.environ.get("DOMAIN_NAME"),
        "instruction": DOMAIN["instruction"],
        "rerank_instruction": DOMAIN["rerank_instruction"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def main():
    # A failed rerank must abort the run, never degrade silently to vector order.
    os.environ["MEMENTO_STRICT_RERANK"] = "1"
    ap = argparse.ArgumentParser()
    ap.add_argument('--domain', required=True, help='Corpus domain profile (config/domains/<name>.yaml), e.g. ceramics | polymer-engineering')
    ap.add_argument("--in", dest="inp", default=str(Path(__file__).parent / "ground_truth_auto_pooled_all.json"))
    ap.add_argument("--out", default=str(Path(__file__).parent / "rq1_results.json"))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--only", metavar="PREFIX", nargs="+",
                    help="Run only conditions whose name starts with any PREFIX "
                         "(e.g. 'CTX-'); skips the ColQwen/Fusion block unless "
                         "it matches. Use with --merge-into to fold the new "
                         "conditions into an existing full results file.")
    ap.add_argument("--merge-into", metavar="FILE",
                    help="After the run, merge each query's new condition "
                         "scores into FILE (matched by query_id) instead of "
                         "leaving them only in --out.")
    ap.add_argument("--exclude", metavar="SUBSTR", nargs="*", default=[],
                    help="Exclude conditions whose name contains any of the "
                         "given substrings (e.g. --exclude textrerank NoEnrichTXT).")
    ap.add_argument("--no-colqwen", action="store_true",
                    help="Skip the ColQwen/Fusion block (e.g. when the ColQwen "
                         "encoder is unavailable). The core 21 conditions still "
                         "run; carry ColQwen scores from a prior results file.")
    args = ap.parse_args()
    domain_banner()

    conditions = CONDITIONS
    run_colqwen = not args.no_colqwen
    if args.only:
        prefixes = tuple(args.only)
        conditions = [c for c in CONDITIONS if c[0].startswith(prefixes)]
        run_colqwen = any(n.startswith(prefixes) for n in COLQWEN_CONDITION_NAMES)
        if not conditions and not run_colqwen:
            sys.exit(f"--only {args.only!r} matches no conditions")
        print(f"--only {args.only!r}: {[c[0] for c in conditions]}"
              f"{' + ColQwen block' if run_colqwen else ''}")
    if args.exclude:
        before = len(conditions)
        conditions = [c for c in conditions if not any(x in c[0] for x in args.exclude)]
        print(f"--exclude {args.exclude}: dropped {before - len(conditions)} conditions, "
              f"{len(conditions)} remaining")

    gt = json.loads(Path(args.inp).read_text())
    queries = gt["queries"]

    client = QdrantClient(url=QDRANT_URL, timeout=30)

    per_query_results = []
    start_idx = 0
    if args.resume and Path(args.out).exists():
        prior = json.loads(Path(args.out).read_text())
        per_query_results = prior.get("per_query", [])
        start_idx = len(per_query_results)
        print(f"Resuming from query {start_idx}/{len(queries)}")

    for qi in range(start_idx, len(queries)):
        q = queries[qi]
        relevant_set = {(c["paper_id"], c["page"]) for c in q["relevant_chunks"]}
        qvec_local = embed_query(q["query"], EMBEDDER_URL, QUERY_INSTRUCTION)
        # Only compute the text8b query embedding if a condition needs it —
        # cheap either way, but avoids a network call when nobody asked for it.
        qvec_text8b = None
        if any(embedder == "text8b" for *_rest, embedder, _reranker in conditions):
            qvec_text8b = local_text_embed_query(q["query"])

        # Rerank-enabled "local" conditions go to one of two reranker instances at random, so the executor below runs
        # them in parallel (RERANKER_URL_REPLICA is the same instance when no replica is configured). Not a cycle:
        # CONDITIONS orders dense before hybrid per backbone and hybrid is heavier to rerank, so any fixed order would
        # load one instance more. "text8b" conditions use their own service (LOCAL_TEXT_RERANK_URL).
        jobs = []
        for name, collection, hybrid, rerank_enabled, type_filter, embedder, reranker in conditions:
            qvec = qvec_local if embedder == "local" else qvec_text8b
            if reranker == "local":
                reranker_url = random.choice([RERANKER_URL, RERANKER_URL_REPLICA]) if rerank_enabled else RERANKER_URL
            else:
                reranker_url = LOCAL_TEXT_RERANK_URL
            jobs.append((name, collection, hybrid, rerank_enabled, type_filter, qvec, reranker_url))

        conditions_out = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=CONDITION_WORKERS) as pool:
            futures = {
                pool.submit(eval_one, client, qvec, q["query"], relevant_set,
                            collection, hybrid, rerank_enabled, type_filter, reranker_url): name
                for name, collection, hybrid, rerank_enabled, type_filter, qvec, reranker_url in jobs
            }
            for future in concurrent.futures.as_completed(futures):
                conditions_out[futures[future]] = future.result()
        if run_colqwen:
            if qvec_text8b is None:
                qvec_text8b = local_text_embed_query(q["query"])
            conditions_out["ColQwen-only"] = eval_colqwen(client, q["query"], relevant_set, rerank_enabled=False)
            conditions_out["ColQwen-only+rerank"] = eval_colqwen(client, q["query"], relevant_set, rerank_enabled=True)
            conditions_out["Fusion-text+colqwen"] = eval_fusion(client, q["query"], qvec_text8b, relevant_set, rerank_enabled=False)
            conditions_out["Fusion-text+colqwen+rerank"] = eval_fusion(client, q["query"], qvec_text8b, relevant_set, rerank_enabled=True)

        per_query_results.append({
            "query_id": q["query_id"],
            "target_type": q["target_type"],
            "target_paper": q["target_paper"],
            "n_relevant": len(relevant_set),
            "conditions": conditions_out,
        })
        headline = [n for n in ("VLM-hybrid+rerank", "OCR-hybrid+rerank",
                                "Hybrid-hybrid+rerank") if n in conditions_out]
        if not headline:  # --only run — show whatever we're actually computing
            headline = [n for n in conditions_out if n.endswith("+rerank")] or list(conditions_out)
        print(f"[{qi+1}/{len(queries)}] {q['target_type']:<10} "
              f"{q['query'][:60]!r}  "
              + "  ".join(f"{n} hit@5={conditions_out[n]['hit@5']}" for n in headline[:3]))

        Path(args.out).write_text(json.dumps({
            "provenance": _provenance(args.inp),
            "n_queries_total": len(queries),
            "n_queries_done": len(per_query_results),
            "conditions": [c[0] for c in conditions]
                          + (COLQWEN_CONDITION_NAMES if run_colqwen else []),
            "per_query": per_query_results,
        }, indent=2))

    print(f"\nWrote {len(per_query_results)} per-query results to {args.out}")

    if args.merge_into:
        master = json.loads(Path(args.merge_into).read_text())
        by_id = {r["query_id"]: r for r in master["per_query"]}
        merged = 0
        for r in per_query_results:
            tgt = by_id.get(r["query_id"])
            if tgt is None:
                print(f"  merge: query_id {r['query_id']} not in master — skipped")
                continue
            tgt["conditions"].update(r["conditions"])
            merged += 1
        new_names = [c[0] for c in conditions] + (COLQWEN_CONDITION_NAMES if run_colqwen else [])
        master["conditions"] = list(dict.fromkeys(master["conditions"] + new_names))
        # Record what was merged in and under what GT/code state, so a later
        # reader can tell whether every condition in this file was scored
        # against the same ground truth (see _provenance docstring above).
        master.setdefault("merge_history", []).append({
            "merged_conditions": new_names,
            "source_provenance": _provenance(args.inp),
        })
        Path(args.merge_into).write_text(json.dumps(master, indent=2))
        print(f"Merged {merged} queries' new conditions into {args.merge_into}")


if __name__ == "__main__":
    main()
