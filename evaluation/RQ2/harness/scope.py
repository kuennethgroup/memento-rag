"""Universe-scoped retrieval primitives shared by all arms."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

from evaluation.RQ2.questions.pooling import IMAGE_RERANK_RATIO, RERANK_MIN_SCORE, RERANK_TOP, RETRIEVE_K
from memento.retrieval.search import embed_query, point_to_hit, rerank, search


@dataclass
class ArmContext:
    domain: str
    qdrant: Any
    collection: str
    allowed: set | None
    llm: Any
    model: str
    registry: dict
    store: Any
    k_slots: int = 5
    max_df: int | None = None
    min_hits: int = 1
    reranker_url: str = ""
    stream_pool: int = 40
    chunk_source: dict = field(default_factory=dict)
    stores: dict = field(default_factory=dict)      # G2-lite arm name -> its own graph store
    sheets: str = ""                                # fact-sheets Qdrant collection


_RR = __import__("itertools").count()


def reranker_for(ctx):
    """ctx.reranker_url may list several identical reranker servers (comma-separated: the reranker is
    the arms stage's bottleneck); calls alternate between them."""
    from memento.config.settings import RERANKER_URL
    urls = [u.strip() for u in (ctx.reranker_url or RERANKER_URL).split(",") if u.strip()]
    return urls[next(_RR) % len(urls)]


def source_filter(allowed):
    if allowed is None:
        return None
    return Filter(must=[FieldCondition(key="source", match=MatchAny(any=sorted(allowed)))])


def g0_retrieve(ctx, text, qvec=None):
    from memento.config.settings import DOMAIN, EMBEDDER_URL, HYBRID_ENABLED, HYBRID_PREFETCH_K, RERANKER_URL
    qvec = qvec or embed_query(text, EMBEDDER_URL, DOMAIN["instruction"])
    hits, _ = search(ctx.qdrant, ctx.collection, text, embedder_url=EMBEDDER_URL,
                     query_instruction=DOMAIN["instruction"], retrieve_k=RETRIEVE_K, hybrid=HYBRID_ENABLED,
                     hybrid_prefetch_k=HYBRID_PREFETCH_K, rerank_enabled=True,
                     reranker_url=reranker_for(ctx), rerank_top=RERANK_TOP,
                     rerank_min_score=RERANK_MIN_SCORE, image_rerank_ratio=IMAGE_RERANK_RATIO,
                     query_filter=source_filter(ctx.allowed), qvec=qvec)
    return hits, qvec


def rank_stream(ctx, question, hits):
    from memento.config.settings import RERANKER_URL
    if not hits:
        return []
    return rerank(question, hits, reranker_for(ctx), top_n=len(hits), min_score=0.0,
                  image_rerank_ratio=0.0)


def payload_hits(ctx, ids):
    if not ids:
        return []
    pts = ctx.qdrant.retrieve(ctx.collection, ids=list(ids), with_payload=True)
    out = [point_to_hit(p.payload or {}, 0.0, p.id) for p in pts if (p.payload or {}).get("text", "").strip()]
    order = {str(i): n for n, i in enumerate(ids)}
    out = [h for h in out if ctx.allowed is None or h["source"] in ctx.allowed]
    return sorted(out, key=lambda h: order.get(h["chunk_id"], len(order)))


def best_chunk_per_doc(ctx, docs, qvec):
    out = []
    for d in docs:
        pts = ctx.qdrant.query_points(ctx.collection, query=qvec, using="dense", limit=1, with_payload=True,
                                      query_filter=Filter(must=[FieldCondition(key="source",
                                                                               match=MatchValue(value=d))])).points
        out += [point_to_hit(p.payload or {}, float(p.score), p.id) for p in pts]
    return out
