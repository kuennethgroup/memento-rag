"""
memento/graph/ppr.py — Personalized PageRank traversal for the Memento knowledge graph.

HippoRAG2-inspired (Gutiérrez et al.; arXiv:2502.14802) replacement for the
unweighted BFS-union traversal in `GraphStore.subgraph_chunks`: instead of
"every chunk that MENTIONS any entity within N hops counts the same, capped by
an arbitrary LIMIT", PPR gives every entity a *score* — probability mass that
starts at the query's seed entities and diffuses along Entity–Entity edges
(RELATION + SIMILAR), decaying with distance and concentrating on
well-connected evidence. Chunks are then ranked by the total mass of the
entities they MENTION, so the graph channel contributes its *best* N
candidates rather than its first N.

Pure Python (dict-based power iteration), no scipy/igraph dependency:
the corpus-scale entity graph (10^3–10^5 edges) converges in well under a
second. If a corpus of 1000+ documents makes this a hot spot, swap the
inner loop for scipy.sparse — the interface is deliberately array-free so
callers won't change.

Used by `memento/graph/query/graph_retrieval.py` when `config.GRAPH_TRAVERSAL == "ppr"`.
"""
from __future__ import annotations

from typing import Iterable


def personalized_pagerank(
    edges: Iterable[tuple[str, str]],
    seeds: dict[str, float],
    damping: float = 0.5,
    max_iter: int = 50,
    tol: float = 1e-9,
) -> dict[str, float]:
    """PPR scores over an undirected graph.

    `edges` — (a, b) node-id pairs; treated as undirected, self-loops ignored,
    duplicates allowed (they just weight the transition, which is harmless).
    `seeds` — {node_id: weight}; the restart distribution (normalized here).
    Seed nodes absent from `edges` still participate (they hold their restart
    mass; isolated non-seeds score 0).
    `damping` — probability of following an edge vs restarting at the seeds.
    HippoRAG uses 0.5: strong locality, mass stays near the query entities.

    Returns {node_id: score} summing to ~1.0. Empty seeds (or all-zero
    weights) → {}.
    """
    total = sum(w for w in seeds.values() if w > 0)
    if total <= 0:
        return {}

    adj: dict[str, list[str]] = {}
    for a, b in edges:
        if a == b or a is None or b is None:
            continue
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)

    nodes = set(adj) | set(seeds)
    restart = {n: seeds.get(n, 0.0) / total if seeds.get(n, 0.0) > 0 else 0.0 for n in nodes}
    score = dict(restart)

    for _ in range(max_iter):
        nxt = {n: (1.0 - damping) * restart[n] for n in nodes}
        dangling = 0.0
        for n in nodes:
            out = adj.get(n)
            if not out:
                dangling += score[n]
                continue
            share = damping * score[n] / len(out)
            for m in out:
                nxt[m] += share
        if dangling:
            # Mass at dangling nodes restarts at the seeds (standard fix —
            # keeps the distribution stochastic instead of leaking mass).
            for n in nodes:
                nxt[n] += damping * dangling * restart[n]
        delta = sum(abs(nxt[n] - score[n]) for n in nodes)
        score = nxt
        if delta < tol:
            break
    return score


def score_chunks(
    entity_scores: dict[str, float],
    mention_rows: Iterable[dict],
) -> list[dict]:
    """Rank chunks by the PPR mass of the entities they mention.

    `mention_rows` — dicts with `chunk_id`, `source`, `entity_id` (the shape
    `GraphStore.chunk_mentions_for` returns). A chunk's score is the SUM of
    its mentioned entities' scores (a chunk touching two high-mass entities
    beats a chunk touching one). Returns rows shaped like `subgraph_chunks`
    output — `chunk_id`, `source`, `via_entity` (top-contributing entity),
    `seed_id` (None — PPR has no single seed attribution), plus `score` —
    sorted descending, deduped by chunk_id.
    """
    acc: dict[str, dict] = {}
    for row in mention_rows:
        cid = row.get("chunk_id")
        eid = row.get("entity_id")
        if cid is None or eid is None:
            continue
        w = entity_scores.get(eid, 0.0)
        if w <= 0:
            continue
        slot = acc.get(cid)
        if slot is None:
            slot = acc[cid] = {"chunk_id": cid, "source": row.get("source"),
                               "via_entity": eid, "seed_id": None,
                               "score": 0.0, "_best": 0.0}
        slot["score"] += w
        if w > slot["_best"]:
            slot["_best"] = w
            slot["via_entity"] = eid
    ranked = sorted(acc.values(), key=lambda r: r["score"], reverse=True)
    for r in ranked:
        r.pop("_best", None)
    return ranked
