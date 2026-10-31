"""Shared retrieval core: the dense/hybrid Qdrant query and the rerank logic (including the RERANK_MIN_SCORE /
IMAGE_RERANK_RATIO floors), used by Library Chat (memento/query/search_step.py) and the evaluation, so the two cannot
drift apart: change it here only.

Scope: the query and rerank core only. Product decisions (named documents, image intent, verbatim sections, in-progress
exclusion) stay in memento/query; this module offers generic `type_filter` / `paper_filter` knobs.

Reciprocal Rank Fusion (rrf_fuse) combines independent rankings (dense, sparse/BM25, graph traversal) into one
score per candidate; it is generic and dependency-free.
"""

import re
from collections import Counter
from pathlib import Path
from typing import Iterable

def build_filter(paper_filter: str | None = None,
                  type_filter: list[str] | None = None,
                  exclude_sections: list[str] | None = None):
    """Compose a Qdrant `Filter` from generic constraints; None if none are given. The constraints Library Chat's
    filters apply (memento/query/search_step.py), for callers without its intent detection."""
    from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny

    must = []
    must_not = []
    if paper_filter:
        must.append(FieldCondition(key="source", match=MatchValue(value=paper_filter)))
    if type_filter:
        must.append(FieldCondition(key="type", match=MatchAny(any=list(type_filter))))
    for sec in (exclude_sections or []):
        must_not.append(FieldCondition(key="section", match=MatchValue(value=sec)))
    if not must and not must_not:
        return None
    return Filter(must=must, must_not=must_not)


def qdrant_query(client, collection: str, qvec: list[float], query_text: str,
                  query_filter, limit: int, hybrid: bool, hybrid_prefetch_k: int):
    """Dense or hybrid (dense + BM25, RRF) Qdrant search, shared with Library Chat: change it here only."""
    if hybrid:
        from qdrant_client.models import Prefetch, FusionQuery, Fusion
        from memento.ingestion.embed import sparse_encode_query
        sparse_vec = sparse_encode_query(query_text)
        return client.query_points(
            collection_name=collection,
            prefetch=[
                Prefetch(query=qvec, using="dense", filter=query_filter, limit=hybrid_prefetch_k),
                Prefetch(query=sparse_vec, using="sparse", filter=query_filter, limit=hybrid_prefetch_k),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
    return client.query_points(
        collection_name=collection,
        query=qvec,
        using="dense",
        query_filter=query_filter,
        limit=limit,
        with_payload=True,
    )


class RelatedHeads(list):
    """The best passage of each document, with what the search reached: `capped` = the grouped search returned as many
    documents as it may (more may exist), `found` = documents with a usable passage (before any reranking cut)."""

    def __init__(self, items=(), capped: bool = False, found: int | None = None):
        super().__init__(items)
        self.capped = capped
        self.found = len(self) if found is None else found


def related_heads(client, collection: str, qvec: list[float], query_filter, groups: int) -> list[dict]:
    """The best passage of each document (dense search grouped by `source`, one call; the `source` payload index keeps
    it fast at corpus scale), best document first, at most `groups` documents; query_pipe reranks the best of them."""
    res = client.query_points_groups(collection_name=collection, query=qvec, using="dense", query_filter=query_filter,
                                     group_by="source", group_size=1, limit=groups, with_payload=True)
    # the cap from the raw response, before groups without text are dropped
    return RelatedHeads([point_to_hit(g.hits[0].payload or {}, g.hits[0].score, getattr(g.hits[0], "id", None))
                         for g in res.groups if g.hits and (g.hits[0].payload or {}).get("text", "").strip()],
                        capped=len(res.groups) >= groups)


# a table of contents, a list of figures or tables, a reference list: such a line names no content, so it must not stand
# for a document
_TOC_SECTION = re.compile(r"verzeichnis|\bcontents\b|\blist of (?:figures|tables|abbreviations|symbols)\b|^references$"
                          r"|^bibliography$|^literatur", re.I)
_TOC_LINE = re.compile(r"(?:\.\s?){4,}\s*\d+\s*$", re.M)          # "….43", ". . . . . 112": a page number after dots


def toc_like(hit: dict) -> bool:
    """A passage from a table of contents, a list of figures or tables, or a reference list: no evidence that the
    document works on a topic."""
    return bool(_TOC_SECTION.search((hit.get("section") or "").strip()) or _TOC_LINE.search(hit.get("text") or ""))


def list_heads(client, collection: str, qvec: list[float], query_filter, groups: int, per_doc: int) -> RelatedHeads:
    """For a list question: up to `per_doc` passages of each of the `groups` documents closest to the topic (dense
    search grouped by `source`, one call), best document first, without table-of-contents or list lines. `capped`: the
    search returned as many documents as it may; `found`: documents with a usable passage."""
    res = client.query_points_groups(collection_name=collection, query=qvec, using="dense", query_filter=query_filter,
                                     group_by="source", group_size=per_doc + 2, limit=groups, with_payload=True)
    out, docs = [], 0
    for g in res.groups:
        kept = [point_to_hit(h.payload or {}, h.score, getattr(h, "id", None)) for h in (g.hits or [])
                if (h.payload or {}).get("text", "").strip()]
        kept = [h for h in kept if not toc_like(h)][:per_doc]
        docs += bool(kept)
        out += kept
    return RelatedHeads(out, capped=len(res.groups) >= groups, found=docs)


def point_to_hit(payload: dict, score: float = 0.0, point_id=None) -> dict:
    """Qdrant payload → the hit dict used by Library Chat and the evaluation. `point_id` is optional; callers that need
    chunk identity (chunk-level scoring, graph joins) use it."""
    return {
        "chunk_id": None if point_id is None else str(point_id),
        "text":     payload.get("text", ""),
        "source":   payload.get("source", "unknown"),
        "page":     payload.get("page", 0),
        "type":     payload.get("type", "text"),
        "section":  payload.get("section"),
        "section_raw": payload.get("section_raw"),
        "bbox":     payload.get("bbox"),
        "page_end": payload.get("page_end"),
        "score":    score,
        "img_path": payload.get("img_path"),
        # the authors' text the answer model reads under a table / an equation (answer_pipeline.source_text)
        "table_context":    payload.get("table_context"),
        "equation_context": payload.get("equation_context"),
    }


def rerank(query: str, hits: list[dict], reranker_url: str, top_n: int,
           min_score: float, image_rerank_ratio: float,
           multimodal: bool = True, instruction: str | None = None) -> list[dict]:
    """Cross-encoder rerank via the vLLM /rerank endpoint. Image/chart hits are scored on image + caption text;
    multimodal=False (text-only reranker) sends the text alone. `instruction`: the caller's domain instruction; None
    keeps the process domain's. On reranker failure, the first `top_n` hits in their incoming (vector-score) order,
    unless MEMENTO_STRICT_RERANK=1 (evaluation), where failure must abort."""
    import os
    from memento.config.settings import DOMAIN, reranker_model_for
    from memento.common.model_servers import rerank as vllm_rerank

    if not hits:
        return []
    try:
        docs = []
        for h in hits:
            img = h.get("img_path") if h["type"] in ("image", "chart") else None
            docs.append({"text": h["text"], "image": img if img and Path(img).is_file() else None})
        ranked = vllm_rerank(reranker_url, reranker_model_for(reranker_url), query, docs,
                             DOMAIN["rerank_instruction"] if instruction is None else instruction, top_n=top_n,
                             multimodal=multimodal, timeout=120)
        out = [{**hits[idx], "rerank_score": score} for idx, score in ranked]

        if min_score > 0:
            filtered = [h for h in out if (h.get("rerank_score") or 0) >= min_score]
            if filtered:
                out = filtered

        if image_rerank_ratio > 0 and out:
            # Same-modality gate: an image/chart is scored against the best image/chart score, not the top (text) score.
            # The cross-encoder scores images on a lower scale than text, so a cross-modal gate deleted answer-bearing
            # figures on visual queries. Image-to-image keeps the strongest figure and any within `image_rerank_ratio`
            # of it, and still drops unrelated figures. MEMENTO_IMAGE_FILTER_LEGACY=1 restores the cross-modal gate,
            # only to reproduce the earlier RQ1 baseline.
            import os as _os
            if _os.getenv("MEMENTO_IMAGE_FILTER_LEGACY") == "1":
                top_ref = max(h.get("rerank_score") or 0 for h in out)
            else:
                img_scores = [h.get("rerank_score") or 0 for h in out
                              if h["type"] in ("image", "chart")]
                top_ref = max(img_scores) if img_scores else 0.0
            threshold = top_ref * image_rerank_ratio
            out = [
                h for h in out
                if h["type"] not in ("image", "chart")
                or (h.get("rerank_score") or 0) >= threshold
            ]
        return out
    except Exception as exc:
        if os.getenv("MEMENTO_STRICT_RERANK") == "1":
            raise
        # falling back to the hybrid order is much worse for German documents (RQ1), so it is logged: a dead reranker
        # shows up in the logs
        import logging
        logging.getLogger("memento.rerank").warning("RERANKER FALLBACK (%s): %r — using retrieval order",
                                                    reranker_url, exc)
        for h in hits:
            h["rerank_fallback"] = True
        return hits[:top_n]


def balance_papers(ranked: list[dict], top_n: int, cap: int, ratio: float) -> list[dict]:
    """The `top_n` passages for the answer model, with a soft per-document cap: one document can fill most slots while
    others have relevant passages. A document above `cap` gives up its weakest passage to the best passage of a document
    still under the cap, while that newcomer scores at least `ratio` times the passage it replaces; the first one that
    falls short ends the swapping, so a document that is the only strong match keeps every slot. `ranked` is the
    reranker's full order (best first). Off with cap <= 0, and without reranker scores (the failure fallback)."""
    chosen, pool = list(ranked[:top_n]), list(ranked[top_n:])
    if cap <= 0 or not pool or any(h.get("rerank_fallback") or h.get("rerank_score") is None for h in ranked):
        return chosen
    while True:
        counts = Counter(h["source"] for h in chosen)
        over = [h for h in chosen if counts[h["source"]] > cap]
        if not over:
            break
        weakest = min(over, key=lambda h: h["rerank_score"])
        newcomer = next((h for h in pool if counts[h["source"]] < cap), None)
        if newcomer is None or newcomer["rerank_score"] < ratio * weakest["rerank_score"]:
            break
        chosen = [h for h in chosen if h is not weakest] + [newcomer]
        pool = [h for h in pool if h is not newcomer]
    return sorted(chosen, key=lambda h: h["rerank_score"], reverse=True)


def share_evenly(ranked: list[dict], top_n: int, sources: list[str]) -> list[dict]:
    """The `top_n` passages with an equal share for every one of `sources` (Document Chat, several files and none named:
    "the attached files" means all of them). Each file gives its best top_n // len(sources) passages in the reranker's
    order; slots a file cannot fill go to the best passages left. Best first."""
    if len(ranked) <= top_n or not sources:
        return list(ranked[:top_n])
    per = max(1, top_n // len(sources))
    chosen = []
    for src in sources:
        chosen += [h for h in ranked if h["source"] == src][:per]
    chosen += [h for h in ranked if not any(h is c for c in chosen)][:top_n - len(chosen)]
    return sorted(chosen, key=lambda h: h.get("rerank_score") or 0.0, reverse=True)


def embed_query(text: str, embedder_url: str, instruction: str) -> list[float]:
    """Embed a single query via the vLLM VL embedder."""
    from memento.config.settings import EMBEDDER_MODEL
    from memento.common.model_servers import vl_embed

    return vl_embed(embedder_url, EMBEDDER_MODEL, instruction, text=text, timeout=30)


def search(client, collection: str, query: str, *,
           embedder_url: str, query_instruction: str,
           retrieve_k: int, hybrid: bool, hybrid_prefetch_k: int,
           rerank_enabled: bool, reranker_url: str, rerank_top: int,
           rerank_min_score: float, image_rerank_ratio: float,
           query_filter=None, qvec: list[float] | None = None,
           rerank_multimodal: bool = True,
           rerank_instruction: str | None = None) -> tuple[list[dict], list[dict]]:
    """End-to-end retrieval: embed (unless `qvec` is given) → Qdrant query → rerank (if enabled). Returns (final_hits,
    pre_rerank_hits): the evaluation needs both, pre_rerank_hits for vector-only Hit@k/MRR, final_hits for what users
    see."""
    if qvec is None:
        qvec = embed_query(query, embedder_url, query_instruction)

    results = qdrant_query(
        client, collection, qvec, query, query_filter,
        retrieve_k, hybrid, hybrid_prefetch_k,
    ).points
    pre_rerank_hits = [
        point_to_hit(r.payload or {}, r.score, getattr(r, "id", None))
        for r in results
        if (r.payload or {}).get("text", "").strip()
    ]

    if rerank_enabled:
        final_hits = rerank(
            query, pre_rerank_hits, reranker_url, rerank_top,
            rerank_min_score, image_rerank_ratio, multimodal=rerank_multimodal,
            instruction=rerank_instruction,
        )
    else:
        final_hits = pre_rerank_hits[:rerank_top]

    return final_hits, pre_rerank_hits


# ── Reciprocal rank fusion ────────────────────────────────────────────────────
def rrf_fuse(ranked_lists: Iterable[Iterable], k: int = 60) -> dict:
    """Combine multiple rankings (iterables of hashable ids, best-first) into
    a single score per id:

        score(id) = sum over lists containing id of 1 / (k + rank)

    where `rank` is 1-based. Higher score = better; ids absent from a given
    list simply don't contribute a term for that list. `k=60` is the
    standard RRF damping constant.
    """
    scores: dict = {}
    for ranked in ranked_lists:
        for rank, item_id in enumerate(ranked, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
    return scores
