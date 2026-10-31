#!/usr/bin/env python3
"""Graph-traversal candidate generator for the dual-graph knowledge graph, shared by memento/retrieval/retrieve.py and
Library Chat's search step.

Query entities (registry match, optionally an LLM pass) seed three candidate streams: (1) entity traversal over RELATION
edges, collecting chunks that MENTION a visited entity (cross-paper); (2) REFERENCES expansion from the dense candidates
to linked figure/table/equation and text chunks; (3) cross-paper expansion over CITES_PAPER to chunks of cited or citing
papers that mention the query entities. graph_candidate_ids() returns their union as Qdrant point ids, without ids dense
search already found. The normal path unions dense and graph candidates and lets the reranker score them;
memento/retrieval/search.py:rrf_fuse() fuses the two rankings when the reranker is skipped.

Gated by config.GRAPH_RETRIEVAL_ENABLED (env GRAPH_RETRIEVAL, default off: vector-only). Best effort: any failure
degrades to no candidates; GRAPH_STRICT=1 raises instead, as evaluation runs need."""
from __future__ import annotations

import logging
from typing import Optional

from memento.config.settings import (  # noqa: E402
    EMBEDDER_URL,
    ENTITY_COLLECTION,
    ENTITY_EMBED_INSTRUCTION,
    GRAPH_FIRST_HOP,
    GRAPH_FIRST_HOP_K,
    GRAPH_LIMIT,
    GRAPH_MAX_DF,
    GRAPH_MAX_HOPS,
    GRAPH_PPR_DAMPING,
    GRAPH_PPR_TOP_ENTITIES,
    GRAPH_QUERY_USE_LLM,
    GRAPH_RETRIEVAL_ENABLED,
    GRAPH_SEED_MIN_SCORE,
    GRAPH_SEED_MODE,
    GRAPH_SEED_TOP_K,
    GRAPH_STREAMS,
    GRAPH_STRICT,
    GRAPH_TRAVERSAL,
    QDRANT_URL,
)
from memento.graph.ppr import personalized_pagerank, score_chunks  # noqa: E402
from memento.graph.store import GraphStore, get_store  # noqa: E402
from memento.graph.build.extract_entities import extract_entities  # noqa: E402

log = logging.getLogger(__name__)


def _degrade(where: str, exc: Exception) -> list:
    """Every graph failure funnels through here so a null is never silent."""
    if GRAPH_STRICT:
        raise
    log.warning("graph retrieval degraded to [] in %s: %s: %s",
                where, type(exc).__name__, exc)
    return []


# ── Query entity extraction ────────────────────────────────────────────────
def extract_query_entities(query: str, *, use_llm: bool = GRAPH_QUERY_USE_LLM) -> list[str]:
    """Entity ids mentioned in `query` (registry canonical_ids for known
    terms, normalized slugs for novel ones), in extraction order, deduped.

    Used as seed nodes for graph traversal — see `graph_candidates()`.
    Best-effort: `extract_entities` already degrades to dictionary-only
    results on LLM failure; an empty/unresolvable query returns [].
    """
    result = extract_entities(query, use_llm=use_llm)
    seen: set[str] = set()
    ids: list[str] = []
    for ent in result.entities:
        if ent.id not in seen:
            seen.add(ent.id)
            ids.append(ent.id)
    return ids


# ── Dense entity seeding (HippoRAG2-style query-to-entity retrieval) ─────────
def dense_seed_entities(
    query: str,
    *,
    top_k: int = GRAPH_SEED_TOP_K,
    min_score: float = GRAPH_SEED_MIN_SCORE,
    client=None,
    embedder_url: str = EMBEDDER_URL,
) -> dict[str, float]:
    """Seed entities by embedding similarity against the `memento-entities` collection (built by
    memento/graph/build/embed_entities.py), instead of or besides the registry match, so a query worded unlike the small registry
    still reaches the entities it is close to. Returns {entity_id: similarity} above `min_score` (cosine). Best effort:
    any failure → {}."""
    try:
        from memento.config.settings import EMBEDDER_MODEL
        from memento.common.model_servers import vl_embed
        qvec = vl_embed(embedder_url, EMBEDDER_MODEL, ENTITY_EMBED_INSTRUCTION,
                        text=query, timeout=30)

        if client is None:
            from qdrant_client import QdrantClient
            client = QdrantClient(url=QDRANT_URL, timeout=10)
        points = client.query_points(
            collection_name=ENTITY_COLLECTION, query=qvec, limit=top_k,
            with_payload=True,
        ).points
        return {
            p.payload["entity_id"]: float(p.score)
            for p in points
            if p.score is not None and p.score >= min_score and p.payload.get("entity_id")
        }
    except Exception as e:
        if GRAPH_STRICT:
            raise
        log.warning("dense entity seeding degraded to {}: %s: %s",
                    type(e).__name__, e)
        return {}


def _build_seeds(query: str, *, seed_mode: str, use_llm: bool,
                 seed_client=None) -> dict[str, float]:
    """Weighted seed map per `seed_mode` (dict | dense | both). Dictionary
    matches get weight 1.0 (exact registry hits are certain); dense seeds get
    their cosine similarity; on overlap the larger weight wins."""
    seeds: dict[str, float] = {}
    if seed_mode in ("dict", "both"):
        for eid in extract_query_entities(query, use_llm=use_llm):
            seeds[eid] = 1.0
    if seed_mode in ("dense", "both"):
        for eid, score in dense_seed_entities(query, client=seed_client).items():
            seeds[eid] = max(seeds.get(eid, 0.0), score)
    return seeds


# ── Structural-stream ranking ────────────────────────────────────────────────
def _neighbor_chunks_by_vector(store, dense_sources, query_vector, qdrant_client, limit):
    """Chunks of citation-neighbor papers, ranked by similarity to the query.
    Searches config.COLLECTION: graph Chunk ids are that collection's point ids."""
    if not dense_sources:
        return []
    if query_vector is None:
        if GRAPH_STRICT:
            raise ValueError("cross-paper structural stream needs query_vector")
        return []
    neighbors = store.citing_neighbors(dense_sources)
    if not neighbors:
        return []
    from qdrant_client import QdrantClient
    from qdrant_client.models import FieldCondition, Filter, MatchAny
    from memento.config.settings import COLLECTION
    client = qdrant_client or QdrantClient(url=QDRANT_URL, timeout=30)
    pts = client.query_points(
        COLLECTION, query=query_vector, using="dense", limit=limit,
        query_filter=Filter(must=[FieldCondition(key="source", match=MatchAny(any=neighbors))]),
        with_payload=["source"]).points
    return [{"chunk_id": str(pt.id), "source": pt.payload.get("source"), "score": float(pt.score)}
            for pt in pts]


def _same_work_chunks_by_vector(store, dense_sources, query_vector, qdrant_client, limit):
    """Chunks of same-work sibling documents (talk ↔ dissertation, chapter ↔ thesis),
    ranked by similarity to the query. Structural (G1): no entity reads."""
    if not dense_sources or query_vector is None or not hasattr(store, "same_work_neighbors"):
        return []          # a store without the method must not empty the other structural streams
    sibs = store.same_work_neighbors(list(dense_sources))
    if not sibs:
        return []
    from qdrant_client import QdrantClient
    from qdrant_client.models import FieldCondition, Filter, MatchAny
    from memento.config.settings import COLLECTION
    client = qdrant_client or QdrantClient(url=QDRANT_URL, timeout=30)
    pts = client.query_points(COLLECTION, query=query_vector, using="dense", limit=limit,
                              query_filter=Filter(must=[FieldCondition(key="source", match=MatchAny(any=sibs))]),
                              with_payload=["source"]).points
    return [{"chunk_id": str(p.id), "source": p.payload.get("source"), "score": float(p.score)} for p in pts]


COPY_MATCH_MIN = 0.95


def _nearest_in_doc(qdrant_client, chunk_id, doc):
    """(point id, cosine) of `doc`'s chunk closest to chunk `chunk_id`'s own dense vector."""
    from qdrant_client.models import FieldCondition, Filter, MatchValue
    from memento.config.settings import COLLECTION
    got = qdrant_client.retrieve(COLLECTION, ids=[int(chunk_id)], with_vectors=["dense"])
    if not got:
        return None, 0.0
    vec = got[0].vector["dense"] if isinstance(got[0].vector, dict) else got[0].vector
    pts = qdrant_client.query_points(COLLECTION, query=vec, using="dense", limit=1, with_payload=False,
                                     query_filter=Filter(must=[FieldCondition(key="source",
                                                                              match=MatchValue(value=doc))])).points
    return (str(pts[0].id), float(pts[0].score)) if pts else (None, 0.0)


def first_hop_chunk_ids(dense_chunk_ids, chunk_source, registry, qdrant_client):
    """Copy documents carry no MENTIONS (chapter copies and duplicates get no entities),
    so a first-hop chunk from a copy is replaced by the parent's nearest chunk (cosine ≥ 0.95)."""
    out = []
    for cid in dense_chunk_ids:
        info = (registry or {}).get(chunk_source.get(cid)) or {}
        parent = info.get("copy_of") or info.get("duplicate_of")
        if parent:
            pid, score = _nearest_in_doc(qdrant_client, cid, parent)
            out.append(pid if pid and score >= COPY_MATCH_MIN else cid)
        else:
            out.append(cid)
    return out


# Candidate pool for entity traversal when a query vector can break score ties
# (a query naming one entity gives every candidate the same IDF score).
GRAPH_TIE_POOL = 2000


def _break_ties_by_vector(rows, query_vector, qdrant_client, limit):
    """Order rows by graph score, ties by dense similarity to the query; top `limit`."""
    if query_vector is None or len(rows) <= 1:
        return rows[:limit]
    ids = [int(r["chunk_id"]) for r in rows if str(r.get("chunk_id", "")).lstrip("-").isdigit()]
    if not ids:
        return rows[:limit]
    from qdrant_client import QdrantClient
    from qdrant_client.models import Filter, HasIdCondition
    from memento.config.settings import COLLECTION
    client = qdrant_client or QdrantClient(url=QDRANT_URL, timeout=30)
    pts = client.query_points(COLLECTION, query=query_vector, using="dense", limit=len(ids),
                              query_filter=Filter(must=[HasIdCondition(has_id=ids)]),
                              with_payload=False).points
    sim = {str(pt.id): float(pt.score) for pt in pts}
    return sorted(rows, key=lambda r: (-round(float(r.get("score") or 0.0), 9),
                                       -sim.get(str(r.get("chunk_id")), float("-inf"))))[:limit]


def _rank_references(rows, dense_chunk_ids):
    """REFERENCES rows ordered by the dense rank of the chunk they hang off."""
    pos = {cid: i for i, cid in enumerate(dense_chunk_ids or [])}
    return sorted(rows, key=lambda r: pos.get(r.get("via_chunk"), len(pos)))


# ── Graph candidates ─────────────────────────────────────────────────────────
def graph_candidates(
    query: str,
    *,
    dense_chunk_ids: Optional[list[str]] = None,
    dense_sources: Optional[list[str]] = None,
    store: Optional[GraphStore] = None,
    max_hops: int = GRAPH_MAX_HOPS,
    limit: int = GRAPH_LIMIT,
    use_llm: bool = GRAPH_QUERY_USE_LLM,
    enabled: bool = GRAPH_RETRIEVAL_ENABLED,
    streams: str = GRAPH_STREAMS,
    traversal: str = GRAPH_TRAVERSAL,
    seed_mode: str = GRAPH_SEED_MODE,
    ppr_damping: float = GRAPH_PPR_DAMPING,
    ppr_top_entities: int = GRAPH_PPR_TOP_ENTITIES,
    seed_client=None,
    query_vector: Optional[list[float]] = None,
    qdrant_client=None,
    max_df: Optional[int] = GRAPH_MAX_DF,
    first_hop: bool = GRAPH_FIRST_HOP,
    first_hop_k: int = GRAPH_FIRST_HOP_K,
    extra_seeds: Optional[dict] = None,
    registry: Optional[dict] = None,
    chunk_source: Optional[dict] = None,
) -> list[dict]:
    """Graph-traversal candidates for `query`, from up to three streams (see the module docstring), in order: 1. entity
    traversal, seeded per `seed_mode` ("dict" registry match, "dense" entity-embedding similarity, "both") and traversed
    per `traversal` ("bfs": subgraph_chunks union; "ppr": Personalized PageRank over RELATION + SIMILAR edges,
    memento/graph/ppr.py, chunks ranked by the PPR mass of the entities they mention); 2. REFERENCES expansion from
    `dense_chunk_ids`; 3. cross-paper chunks from `dense_sources` and the query entities (the entity-free variant ranks
    the citation neighbours' chunks by dense similarity to `query_vector`, and is skipped without one).

    Entity traversal is ranked by IDF (`max_df` caps hub entities; with a `query_vector`, ties are broken by dense
    similarity), REFERENCES rows by the dense rank of their `via_chunk`. Streams 2 and 3 need dense_chunk_ids /
    dense_sources. Returns [] when graph retrieval is off, no entity resolves or the store is unreachable (never
    raises); PPR falls back to BFS on a store without its export methods. `store` may be injected (a fake in tests);
    `seed_client` is an optional Qdrant client for dense seeding."""
    if not enabled:
        return []

    if streams == "structural":
        # B1: no entity seeding, no entity traversal — REFERENCES + CITES_PAPER
        # expansion only. Best-effort, same empty-on-failure contract.
        try:
            owns_store = store is None
            if owns_store:
                store = get_store()
            try:
                results: list[dict] = []
                if dense_chunk_ids:
                    results.extend(_rank_references(
                        store.expand_via_references(dense_chunk_ids, limit=limit), dense_chunk_ids))
                if dense_sources:
                    try:
                        results.extend(_neighbor_chunks_by_vector(
                            store, dense_sources, query_vector, qdrant_client, limit))
                    except NotImplementedError:
                        pass
                    try:
                        results.extend(_same_work_chunks_by_vector(
                            store, dense_sources, query_vector, qdrant_client, limit))
                    except NotImplementedError:
                        pass
                return results
            finally:
                if owns_store:
                    store.close()
        except Exception as e:
            return _degrade("structural streams", e)

    try:
        seeds = _build_seeds(query, seed_mode=seed_mode, use_llm=use_llm,
                             seed_client=seed_client)
        for eid, w in (extra_seeds or {}).items():      # question-term seeds (RQ2 harness), for instance
            seeds[eid] = max(seeds.get(eid, 0.0), w)
        # No entity seeds is normal (a query may name nothing in the registry). For "entity" that means no candidates,
        # but "all" must not stop here: the structural streams do not need seeds, and stopping would make the combined
        # arm smaller than the structural one, breaking the superset the ablation relies on.
        if not seeds and streams == "entity" and not (first_hop and dense_chunk_ids):
            return []

        owns_store = store is None
        if owns_store:
            store = get_store()
        try:
            if first_hop and dense_chunk_ids:
                # multi-hop bridge: the bridge term is absent from the question,
                # so seed from the entities of the top dense chunks (IDF-weighted,
                # hubs capped); query seeds keep weight 1.0 and win on overlap
                hop_ids = list(dense_chunk_ids)[:first_hop_k]
                if registry and qdrant_client is not None:
                    hop_ids = first_hop_chunk_ids(hop_ids, chunk_source or {}, registry, qdrant_client)
                hop = store.chunk_entities(hop_ids, max_df=max_df)
                top = max(hop.values(), default=0.0)
                for eid, w in hop.items():
                    if w > 0 and top > 0:
                        seeds.setdefault(eid, w / top)
            seed_ids = list(seeds)
            if not seed_ids and streams == "entity":
                return []
            results: list[dict] = []
            if not seed_ids:
                pass  # no entity layer for this query; structural streams below still run
            elif traversal == "ppr":
                try:
                    from memento.graph.ppr_cache import load_or_build, ppr_sparse
                    from memento.config.settings import GRAPH_PPR_CACHE_DIR
                    try:
                        scores = ppr_sparse(load_or_build(store, GRAPH_PPR_CACHE_DIR), seeds,
                                            damping=ppr_damping)
                    except NotImplementedError:
                        scores = personalized_pagerank(store.entity_edges(), seeds, damping=ppr_damping)
                    top = sorted((e for e in scores.items() if e[1] > 0),
                                 key=lambda kv: kv[1], reverse=True)[:ppr_top_entities]
                    if top:
                        mentions = store.chunk_mentions_for([eid for eid, _ in top])
                        results = score_chunks(dict(top), mentions)[:limit]
                except NotImplementedError:
                    results = _break_ties_by_vector(
                        store.subgraph_chunks(seed_ids, max_hops=max_hops, max_df=max_df,
                                              limit=GRAPH_TIE_POOL if query_vector else limit),
                        query_vector, qdrant_client, limit)
            else:
                results = _break_ties_by_vector(
                    store.subgraph_chunks(seed_ids, max_hops=max_hops, max_df=max_df,
                                          limit=GRAPH_TIE_POOL if query_vector else limit),
                    query_vector, qdrant_client, limit)

            # "entity" isolates the entity layer for the factorial (structural × entity × rerank): without it the entity
            # layer could only be measured bundled with the structural one.
            entity_only = streams == "entity"

            if dense_chunk_ids and not entity_only:
                results.extend(_rank_references(
                    store.expand_via_references(dense_chunk_ids, limit=limit), dense_chunk_ids))

            if dense_sources and not entity_only:
                # B2 must be a superset of B1 or the three-rung ablation cannot
                # attribute a difference to the entity layer: B1's structural
                # stream is entity-unfiltered, so taking only the filtered
                # variant here would make B2 *drop* candidates B1 returns.
                try:
                    results.extend(_neighbor_chunks_by_vector(
                        store, dense_sources, query_vector, qdrant_client, limit))
                except NotImplementedError:
                    pass
                try:
                    results.extend(_same_work_chunks_by_vector(
                        store, dense_sources, query_vector, qdrant_client, limit))
                except NotImplementedError:
                    pass
                if seed_ids:
                    for row in store.cross_paper_chunks(dense_sources, seed_ids, limit=limit):
                        results.append(row)
            elif dense_sources and entity_only and seed_ids:
                # Entity-filtered cross-paper hop only: this stream IS the
                # entity layer's cross-document contribution.
                for row in store.cross_paper_chunks(dense_sources, seed_ids, limit=limit):
                    results.append(row)

            return results
        finally:
            if owns_store:
                store.close()
    except Exception as e:
        return _degrade("entity traversal", e)


def graph_candidate_ids(
    query: str,
    *,
    dense_chunk_ids: Optional[list[str]] = None,
    dense_sources: Optional[list[str]] = None,
    exclude_ids: Optional[set] = None,
    store: Optional[GraphStore] = None,
    max_hops: int = GRAPH_MAX_HOPS,
    limit: int = GRAPH_LIMIT,
    use_llm: bool = GRAPH_QUERY_USE_LLM,
    enabled: bool = GRAPH_RETRIEVAL_ENABLED,
    traversal: str = GRAPH_TRAVERSAL,
    seed_mode: str = GRAPH_SEED_MODE,
    streams: str = GRAPH_STREAMS,
    query_vector: Optional[list[float]] = None,
    qdrant_client=None,
    max_df: Optional[int] = GRAPH_MAX_DF,
    first_hop: bool = GRAPH_FIRST_HOP,
    first_hop_k: int = GRAPH_FIRST_HOP_K,
    extra_seeds: Optional[dict] = None,
    registry: Optional[dict] = None,
    chunk_source: Optional[dict] = None,
) -> list[int]:
    """Like graph_candidates(), but returns deduped Qdrant point ids (ints) in traversal order, without any id in
    `exclude_ids`. dense_chunk_ids and dense_sources are forwarded for the REFERENCES and cross-paper streams. Chunk
    node ids that are not base-10 integers are skipped (both ingestion paths and build_graph.py use the same
    sha256-derived ids, so this only guards against malformed nodes)."""
    exclude_ids = exclude_ids or set()
    seen: set[int] = set()
    ids: list[int] = []
    for row in graph_candidates(
        query,
        dense_chunk_ids=dense_chunk_ids,
        dense_sources=dense_sources,
        store=store,
        max_hops=max_hops,
        limit=limit,
        use_llm=use_llm,
        enabled=enabled,
        traversal=traversal,
        seed_mode=seed_mode,
        streams=streams,
        query_vector=query_vector,
        qdrant_client=qdrant_client,
        max_df=max_df,
        first_hop=first_hop,
        first_hop_k=first_hop_k,
        extra_seeds=extra_seeds,
        registry=registry,
        chunk_source=chunk_source,
    ):
        cid = row.get("chunk_id")
        try:
            cid_int = int(cid)
        except (TypeError, ValueError):
            continue
        if cid_int in exclude_ids or cid_int in seen:
            continue
        seen.add(cid_int)
        ids.append(cid_int)
    return ids
