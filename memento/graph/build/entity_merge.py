"""Offline alias merging for LLM-discovered entities (the G2 merge step).

plan_merges is pure: name-embedding similarity within one entity type, joined
by union-find under one constraint — a cluster may hold at most one registry
(canonical) entity, so curated distinctions between registry entities are
never collapsed. The mapping is written to JSON for the audit before
apply_merges rewires the graph."""
from __future__ import annotations

import numpy as np


def plan_merges(entities, vectors, threshold):
    m = np.asarray(vectors, dtype=np.float32)
    m /= np.clip(np.linalg.norm(m, axis=1, keepdims=True), 1e-12, None)
    sim = m @ m.T
    n = len(entities)
    parent = list(range(n))
    canon = [bool(e.get("canonical")) for e in entities]
    has_canon = canon[:]

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    pairs = sorted(((float(sim[i, j]), i, j) for i in range(n) for j in range(i + 1, n)
                    if sim[i, j] >= threshold and entities[i]["type"] == entities[j]["type"]),
                   reverse=True)
    used = []
    for cos, i, j in pairs:
        ri, rj = find(i), find(j)
        if ri == rj or (has_canon[ri] and has_canon[rj]):
            continue
        parent[rj] = ri
        has_canon[ri] = has_canon[ri] or has_canon[rj]
        used.append({"a": entities[i]["id"], "b": entities[j]["id"], "cos": round(cos, 4)})
    clusters: dict[int, list[int]] = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)
    mapping = {}
    for members in clusters.values():
        if len(members) < 2:
            continue
        rep = sorted(members, key=lambda i: (not canon[i], -int(entities[i].get("n_mentions") or 0),
                                             len(entities[i]["name"]), entities[i]["id"]))[0]
        for i in members:
            if i != rep:
                mapping[entities[i]["id"]] = entities[rep]["id"]
    return {"mapping": mapping, "pairs": used}


_APPLY = """
UNWIND $pairs AS p
MATCH (s:Entity {id: p.src}), (d:Entity {id: p.dst})
CALL { WITH s, d
  MATCH (c:Chunk)-[m:MENTIONS]->(s)
  MERGE (c)-[m2:MENTIONS]->(d) ON CREATE SET m2.paper_id = m.paper_id
  RETURN count(*) AS n_m }
CALL { WITH s, d
  MATCH (s)-[r:RELATION]->(t:Entity) WHERE t <> d
  MERGE (d)-[r2:RELATION]->(t) ON CREATE SET r2 = properties(r)
  RETURN count(*) AS n_out }
CALL { WITH s, d
  MATCH (t:Entity)-[r:RELATION]->(s) WHERE t <> d
  MERGE (t)-[r2:RELATION]->(d) ON CREATE SET r2 = properties(r)
  RETURN count(*) AS n_in }
SET d.aliases = [x IN coalesce(d.aliases, []) + [s.name] + coalesce(s.aliases, []) WHERE x IS NOT NULL]
DETACH DELETE s
RETURN count(*) AS n_merged, sum(n_m) AS mentions_moved, sum(n_out) + sum(n_in) AS relations_moved
"""


def apply_merges(store, mapping):
    if not mapping:
        return {"n_merged": 0, "mentions_moved": 0, "relations_moved": 0}
    pairs = [{"src": s, "dst": d} for s, d in sorted(mapping.items())]
    out = {"n_merged": 0, "mentions_moved": 0, "relations_moved": 0}
    for i in range(0, len(pairs), 500):
        rec = store._run(_APPLY, pairs=pairs[i:i + 500]).records[0]
        for k in out:
            out[k] += rec[k] or 0
    store.mark_changed()                     # invalidates the PPR matrix cache
    return out
