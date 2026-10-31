"""Hardness label from retrievers independent of G0: BM25-only on the
VLM collection and the text-only embedder on the text collection."""
from __future__ import annotations


def _filter(allowed):
    if not allowed:
        return None
    from qdrant_client.models import FieldCondition, Filter, MatchAny
    return Filter(must=[FieldCondition(key="source", match=MatchAny(any=sorted(allowed)))])


def bm25_top(client, collection, question, k=20, allowed=None):
    from memento.ingestion.embed import sparse_encode_query
    pts = client.query_points(collection, query=sparse_encode_query(question), using="sparse", limit=k,
                              query_filter=_filter(allowed), with_payload=["source", "page"]).points
    return [(p.payload.get("source"), p.payload.get("page", 0)) for p in pts]


def text_dense_top(client, collection, question, embed_fn, k=20, allowed=None):
    pts = client.query_points(collection, query=embed_fn(question), using="dense", limit=k,
                              query_filter=_filter(allowed), with_payload=["source", "page"]).points
    return [(p.payload.get("source"), p.payload.get("page", 0)) for p in pts]


def hardness(b_page, bm25, dense):
    return "easy" if tuple(b_page) in {tuple(x) for x in bm25} | {tuple(x) for x in dense} else "hard"
