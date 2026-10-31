#!/usr/bin/env python3
"""Retrieval and reranking for the Memento RAG pipeline: query → embed → Qdrant top-K → reranker → top-N → optional
neighbour expansion → formatted output.

Services: the embedder API (EMBEDDER_URL), the reranker API (RERANKER_URL) and Qdrant (QDRANT_URL).

CLI: python retrieve.py "<query>" [--retrieve-k 80 --rerank-top 20] [--paper <source>]
[--expand-neighbors] [--json results.json] [--for-llm] [--no-rerank]

Importable: from retrieve import retrieve; hits = retrieve("query text", retrieve_k=80, rerank_top=20)"""
import argparse
import json
import sys
import time
import requests
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from memento.config.settings import DOMAIN, COLLECTION, \
    EMBEDDER_URL, RERANKER_URL, QDRANT_URL, HYBRID_ENABLED, HYBRID_PREFETCH_K, \
    GRAPH_RETRIEVAL_ENABLED
from memento.retrieval.search import rrf_fuse

# ── Config ────────────────────────────────────────────────────────────────────
QUERY_INSTRUCTION = DOMAIN["instruction"]

# Defaults — tuned for fine-grained MinerU chunks
DEFAULT_RETRIEVE_K  = 80      # wide vector net from Qdrant
DEFAULT_RERANK_TOP  = 20      # kept after cross-encoder rerank
DEFAULT_NEIGHBORS   = 1       # neighbors per side when --expand-neighbors


# ── Data types ────────────────────────────────────────────────────────────────
@dataclass
class Hit:
    qdrant_id:     str
    paper_id:      str
    type:          str
    page:          int
    bbox:          list
    page_end:      Optional[int] = None
    text:          Optional[str] = None
    caption:       Optional[str] = None
    img_path:      Optional[str] = None
    vector_score:  float = 0.0          # cosine similarity from Qdrant
    rerank_score:  Optional[float] = None
    is_neighbor:   bool = False         # added via parent-doc expansion
    from_graph:    bool = False         # added via graph-traversal


# ── Service health ────────────────────────────────────────────────────────────
def check_service(url: str, name: str, hint: str):
    try:
        r = requests.get(f"{url}/health", timeout=3)
        r.raise_for_status()
    except Exception as e:
        print(f"[ERROR] {name} not reachable at {url}: {e}")
        print(f"        Start it:  {hint}")
        sys.exit(1)


# ── Embedder ──────────────────────────────────────────────────────────────────
def embed_query(query: str, url: str = EMBEDDER_URL) -> list[float]:
    from memento.config.settings import EMBEDDER_MODEL
    from memento.common.model_servers import vl_embed
    return vl_embed(url, EMBEDDER_MODEL, QUERY_INSTRUCTION, text=query, timeout=30)


# ── Reranker ──────────────────────────────────────────────────────────────────
def call_reranker(query: str, hits: list[Hit], top_n: int,
                  url: str = RERANKER_URL) -> tuple[list[Hit], float]:
    """Call the vLLM /rerank endpoint.

    Returns:
        (hits, elapsed_s)
          hits      — list[Hit] sorted descending by rerank_score, truncated
                      to top_n, with .rerank_score attached
          elapsed_s — wall-clock time (vLLM does not report GPU time separately)
    """
    import time as _time
    from memento.config.settings import reranker_model_for
    from memento.common.model_servers import rerank as vllm_rerank

    if not hits:
        return [], 0.0

    t0 = _time.perf_counter()
    docs = []
    for h in hits:
        img = h.img_path if h.type in ("image", "chart") else None
        docs.append({"text": h.text, "image": img if img and Path(img).is_file() else None})
    ranked = vllm_rerank(url, reranker_model_for(url), query, docs, DOMAIN["rerank_instruction"],
                         top_n=top_n)
    elapsed = _time.perf_counter() - t0

    out: list[Hit] = []
    for idx, score in ranked:
        hit = hits[idx]
        hit.rerank_score = score
        out.append(hit)

    return out, elapsed


# ── Qdrant retrieval ──────────────────────────────────────────────────────────
def vector_search(client: QdrantClient, query_vec: list[float],
                  retrieve_k: int, paper_id: Optional[str] = None,
                  query_text: Optional[str] = None,
                  hybrid: bool = HYBRID_ENABLED) -> list[Hit]:
    must = []
    if paper_id:
        must.append(FieldCondition(key="source", match=MatchValue(value=paper_id)))

    # Keywords-section chunks are bare 1-3 word fragments (the individual
    # terms from a paper's "Keywords:" line) with no enriched context —
    # never useful as a cite-able source, so exclude them from candidates.
    filt = Filter(
        must=must,
        must_not=[FieldCondition(key="section", match=MatchValue(value="Keywords"))],
    )

    # Hybrid dense+BM25. Off by default;
    # falls back to dense-only if no query_text is given (e.g. neighbor
    # expansion has no query) or HYBRID_ENABLED is false.
    if hybrid and query_text:
        from qdrant_client.models import Prefetch, FusionQuery, Fusion
        from memento.ingestion.embed import sparse_encode_query
        sparse_vec = sparse_encode_query(query_text)
        points = client.query_points(
            collection_name=COLLECTION,
            prefetch=[
                Prefetch(query=query_vec, using='dense', filter=filt, limit=HYBRID_PREFETCH_K),
                Prefetch(query=sparse_vec, using='sparse', filter=filt, limit=HYBRID_PREFETCH_K),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=retrieve_k,
            with_payload=True,
        ).points
    else:
        points = client.query_points(
            collection_name=COLLECTION,
            query=query_vec,
            using='dense',
            limit=retrieve_k,
            query_filter=filt,
            with_payload=True,
        ).points

    return [point_to_hit(p) for p in points]


def point_to_hit(point) -> Hit:
    """
    Converts a Qdrant result (ScoredPoint or Record) into our internal Hit dataclass.
    
    Handles the case where 'point' is a Record (from retrieve/neighbor expansion)
    which lacks a '.score' attribute, vs a ScoredPoint (from search).
    """
    # 1. Safely extract the vector search score
    # Records (neighbors) have no score, so we default to 0.0
    v_score = getattr(point, "score", 0.0)
    if v_score is None:
        v_score = 0.0

    # 2. Extract the payload (metadata)
    # Qdrant payloads are dicts
    p = point.payload or {}

    # 3. Map to the Hit dataclass
    # We use .get() to avoid KeyErrors if a specific field is missing in a chunk
    return Hit(
        qdrant_id    = str(point.id),  # Qdrant IDs can be int or str; we standardize to str
        paper_id     = p.get("source", "unknown"),  # payload field is "source", not "paper_id"
        page         = p.get("page", 0),
        type         = p.get("type", "text"),
        text         = p.get("text"),
        caption      = p.get("caption"),
        img_path     = p.get("img_path"),
        # bbox is often stored as a list [x0, y0, x1, y1]
        bbox         = p.get("bbox"),
        page_end     = p.get("page_end"),
        vector_score = float(v_score),
        is_neighbor  = False,  # Default to False; expand_neighbors will flip this
        rerank_score = None    # Filled later by the reranker service
    )


# ── Parent-document expansion ─────────────────────────────────────────────────
def expand_neighbors(hits: list[Hit], client: QdrantClient,
                     per_side: int = 1) -> list[Hit]:
    """
    For each hit, add the chunks immediately before/after it on the same
    page of the same paper (by bbox y-coordinate). Equations and figure
    captions in scientific PDFs are much more useful with their surrounding
    paragraph than alone.

    Only adds neighbors that aren't already in the result set.
    """
    if not hits:
        return hits

    # Group hits by (paper_id, page) so we fetch each page only once
    pages_needed: dict[tuple[str, int], list[Hit]] = {}
    for h in hits:
        pages_needed.setdefault((h.paper_id, h.page), []).append(h)

    selected_ids = {h.qdrant_id for h in hits}
    out: list[Hit] = list(hits)

    for (paper_id, page), hits_on_page in pages_needed.items():
        # Pull every chunk on this page
        page_chunks: list[Hit] = []
        offset = None
        while True:
            results, offset = client.scroll(
                collection_name=COLLECTION,
                limit=200,
                offset=offset,
                with_payload=True,
                scroll_filter=Filter(must=[
                    FieldCondition(key="source", match=MatchValue(value=paper_id)),
                    FieldCondition(key="page",   match=MatchValue(value=page)),
                ]),
            )
            for r in results:
                page_chunks.append(point_to_hit(r))
            if offset is None:
                break

        # Sort top-to-bottom by bbox.y0 (falls back to 0 if bbox missing)
        page_chunks.sort(key=lambda h: (h.bbox[1] if h.bbox else 0))

        # Index lookup for this page
        idx_by_id = {h.qdrant_id: i for i, h in enumerate(page_chunks)}

        for hit in hits_on_page:
            i = idx_by_id.get(hit.qdrant_id)
            if i is None:
                continue
            for j in range(max(0, i - per_side), min(len(page_chunks), i + per_side + 1)):
                if j == i:
                    continue
                cand = page_chunks[j]
                if cand.qdrant_id in selected_ids:
                    continue
                cand.is_neighbor = True
                out.append(cand)
                selected_ids.add(cand.qdrant_id)

    return out


# ── Main retrieval function (importable) ──────────────────────────────────────
def retrieve(query: str,
             retrieve_k: int       = DEFAULT_RETRIEVE_K,
             rerank_top: int       = DEFAULT_RERANK_TOP,
             paper_id: Optional[str] = None,
             expand: bool          = False,
             neighbors_per_side: int = DEFAULT_NEIGHBORS,
             use_reranker: bool    = True,
             hybrid: bool          = HYBRID_ENABLED,
             client: Optional[QdrantClient] = None,
             verbose: bool         = False) -> tuple[list[Hit], dict]:
    """
    Returns (hits, timing) where:
      hits   = list[Hit] sorted by rerank_score (or vector_score if no rerank)
               Neighbors (if --expand-neighbors) are appended at the end,
               sorted by (paper_id, page, bbox.y) for natural reading order.
      timing = dict of per-stage seconds
    """
    timing = {}
    client = client or QdrantClient(url=QDRANT_URL)

    # 1. Embed query
    t0 = time.perf_counter()
    qvec = embed_query(query)
    timing["embed_s"] = time.perf_counter() - t0

    # 2. Vector search — wide net
    t0 = time.perf_counter()
    candidates = vector_search(client, qvec, retrieve_k, paper_id,
                                query_text=query, hybrid=hybrid)
    timing["vector_search_s"] = time.perf_counter() - t0
    if verbose:
        mode = "hybrid" if hybrid else "dense"
        print(f"  [retrieve] {len(candidates)} candidates from Qdrant ({mode}) "
              f"in {timing['vector_search_s']:.3f}s")

    if not candidates:
        return [], timing

    # 2b. Graph-traversal candidates (off by default, see config.GRAPH_RETRIEVAL_ENABLED). Best effort: returns [] and
    # leaves `candidates`/timing unchanged if disabled, unreachable or the graph package is missing (imported lazily, so
    # this file does not depend on it).
    t0 = time.perf_counter()
    dense_ids = [h.qdrant_id for h in candidates]
    dense_sources = list({h.paper_id for h in candidates if h.paper_id})
    exclude_ids = {int(qid) for qid in dense_ids if qid.lstrip("-").isdigit()}
    graph_ids = []
    if GRAPH_RETRIEVAL_ENABLED:
        try:
            from memento.graph.query.graph_retrieval import graph_candidate_ids
            graph_ids = graph_candidate_ids(
                query,
                # dense rank order (REFERENCES rows are ranked by it); string form for graph node ids
                dense_chunk_ids=[q for q in dense_ids if q.lstrip("-").isdigit()],
                dense_sources=dense_sources,
                exclude_ids=exclude_ids,
                query_vector=qvec,
                qdrant_client=client,
            )
        except Exception:
            graph_ids = []
    if graph_ids:
        graph_points = client.retrieve(collection_name=COLLECTION, ids=graph_ids, with_payload=True)
        graph_hits = []
        for p in graph_points:
            h = point_to_hit(p)
            h.from_graph = True
            graph_hits.append(h)
        candidates.extend(graph_hits)
    timing["graph_s"] = time.perf_counter() - t0
    if verbose and graph_ids:
        print(f"  [retrieve] +{len(graph_ids)} graph-traversal candidates "
              f"in {timing['graph_s']:.3f}s")

    # 3. Rerank
    if use_reranker and candidates:
        t0 = time.perf_counter()
        kept, gpu_elapsed = call_reranker(query, candidates, top_n=rerank_top)
        timing["rerank_s"]      = time.perf_counter() - t0
        timing["rerank_gpu_s"]  = gpu_elapsed
        if verbose:
            http_overhead = timing["rerank_s"] - gpu_elapsed
            print(f"  [retrieve] reranked {len(candidates)} → top {len(kept)} "
                  f"(GPU {gpu_elapsed:.3f}s, HTTP {http_overhead:.3f}s)")
    else:
        # No rerank: fuse dense + graph rankings via RRF, keep top rerank_top
        fused = rrf_fuse([dense_ids, [str(i) for i in graph_ids]])
        candidates.sort(key=lambda h: fused.get(h.qdrant_id, 0.0), reverse=True)
        kept = candidates[:rerank_top]

    # 4. Optional parent-doc expansion
    if expand:
        t0 = time.perf_counter()
        kept = expand_neighbors(kept, client, per_side=neighbors_per_side)
        timing["expand_s"] = time.perf_counter() - t0
        # Resort: primary hits first (by score), then neighbors in reading order
        primary   = [h for h in kept if not h.is_neighbor]
        neighbors = sorted(
            (h for h in kept if h.is_neighbor),
            key=lambda h: (h.paper_id, h.page, h.bbox[1] if h.bbox else 0),
        )
        kept = primary + neighbors
        if verbose:
            print(f"  [retrieve] expanded with {len(neighbors)} neighbor chunks")

    timing["total_s"] = sum(v for k, v in timing.items() if k.endswith("_s"))
    return kept, timing


# ── Output formatting ─────────────────────────────────────────────────────────
def snippet(h: Hit, n: int = 150) -> str:
    if h.type in ("image", "chart"):
        img_name = Path(h.img_path or "").name
        cap = (h.caption or "").replace("\n", " ")
        return f"[{h.type.upper()} {img_name}]" + (f" — {cap[:n]}" if cap else "")
    if h.type == "table":
        s = (h.caption or h.text or "").replace("\n", " ")
        return f"[TABLE] {s[:n]}"
    return (h.text or "").replace("\n", " ")[:n]


def print_human(hits: list[Hit], timing: dict, query: str):
    print(f'\nQuery: "{query}"')
    print(f"  embed {timing.get('embed_s', 0):.3f}s  |  "
          f"vector {timing.get('vector_search_s', 0):.3f}s  |  "
          f"graph {timing.get('graph_s', 0):.3f}s  |  "
          f"rerank {timing.get('rerank_s', 0):.3f}s  |  "
          f"expand {timing.get('expand_s', 0):.3f}s  |  "
          f"TOTAL {timing.get('total_s', 0):.3f}s")
    print(f"  {len(hits)} chunks returned\n")

    for i, h in enumerate(hits, 1):
        tag    = " [neighbor]" if h.is_neighbor else (" [graph]" if h.from_graph else "")
        vs     = f"v={h.vector_score:.3f}"
        rs     = f" r={h.rerank_score:.3f}" if h.rerank_score is not None else ""
        header = (f"  {i:2d}. {vs}{rs}  type={h.type:<8}  "
                  f"page={h.page:<3}  paper={h.paper_id}{tag}")
        print(header)
        print(f"      {snippet(h, 160)}")
        print()


def print_for_llm(hits: list[Hit], query: str):
    """
    Format suitable for direct injection into an LLM generation prompt.
    Each chunk gets a numbered source header so the model can cite by index.
    """
    print(f"# Query\n{query}\n")
    print(f"# Retrieved sources ({len(hits)} chunks)\n")
    for i, h in enumerate(hits, 1):
        meta_bits = [f"page {h.page}", h.type]
        if h.is_neighbor:
            meta_bits.append("context")
        meta = ", ".join(meta_bits)
        print(f"## [Source {i}] {h.paper_id} ({meta})")
        if h.type in ("image", "chart"):
            print(f"Image: {h.img_path}")
            if h.caption:
                print(f"Caption: {h.caption}")
        elif h.type == "table":
            if h.caption:
                print(f"Caption: {h.caption}")
            print(h.text or "")
        else:
            print(h.text or "")
        print()


def hits_to_dicts(hits: list[Hit]) -> list[dict]:
    return [asdict(h) for h in hits]


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(
        description="Retrieval + reranking for the Memento RAG pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("query", help="The user query")
    p.add_argument("--retrieve-k", type=int, default=DEFAULT_RETRIEVE_K,
                   help=f"Initial Qdrant top-K (default: {DEFAULT_RETRIEVE_K})")
    p.add_argument("--rerank-top", type=int, default=DEFAULT_RERANK_TOP,
                   help=f"Kept after reranking (default: {DEFAULT_RERANK_TOP})")
    p.add_argument("--paper", metavar="PAPER_ID",
                   help="Restrict to a single paper_id")
    p.add_argument("--expand-neighbors", action="store_true",
                   help="Add same-page neighbors of each kept chunk (parent-doc expansion)")
    p.add_argument("--neighbors-per-side", type=int, default=DEFAULT_NEIGHBORS,
                   help=f"Neighbors above/below each hit (default: {DEFAULT_NEIGHBORS})")
    p.add_argument("--no-rerank", action="store_true",
                   help="Skip reranker (vector search only — for ablation)")
    p.add_argument("--hybrid", action="store_true", default=HYBRID_ENABLED,
                   help=f"Use dense+BM25 hybrid search (default: {HYBRID_ENABLED})")
    p.add_argument("--no-hybrid", action="store_false", dest="hybrid",
                   help="Force dense-only search, overriding HYBRID_ENABLED")
    p.add_argument("--for-llm", action="store_true",
                   help="Print output formatted for direct LLM prompt injection")
    p.add_argument("--json", metavar="PATH",
                   help="Save full results to JSON file")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Print per-stage progress")
    args = p.parse_args()

    # Health checks
    check_service(EMBEDDER_URL, "Embedder",
                  "podman-compose --env-file config/.env up -d embedder")
    if not args.no_rerank:
        check_service(RERANKER_URL, "Reranker",
                      "podman-compose --env-file config/.env up -d reranker")

    client = QdrantClient(url=QDRANT_URL)
    try:
        client.get_collection(COLLECTION)
    except Exception:
        print(f"[ERROR] Collection '{COLLECTION}' not found. "
              f"Run ingestion first.")
        sys.exit(1)

    hits, timing = retrieve(
        query              = args.query,
        retrieve_k         = args.retrieve_k,
        rerank_top         = args.rerank_top,
        paper_id           = args.paper,
        expand             = args.expand_neighbors,
        neighbors_per_side = args.neighbors_per_side,
        use_reranker       = not args.no_rerank,
        hybrid             = args.hybrid,
        client             = client,
        verbose            = args.verbose,
    )

    if args.for_llm:
        print_for_llm(hits, args.query)
    else:
        print_human(hits, timing, args.query)

    if args.json:
        out = {
            "query":          args.query,
            "params": {
                "retrieve_k":         args.retrieve_k,
                "rerank_top":         args.rerank_top,
                "paper_id":           args.paper,
                "expand_neighbors":   args.expand_neighbors,
                "neighbors_per_side": args.neighbors_per_side,
                "use_reranker":       not args.no_rerank,
            },
            "timing":         timing,
            "n_hits":         len(hits),
            "hits":           hits_to_dicts(hits),
        }
        Path(args.json).write_text(json.dumps(out, indent=2))
        print(f"\nResults written to {args.json}")


if __name__ == "__main__":
    main()
