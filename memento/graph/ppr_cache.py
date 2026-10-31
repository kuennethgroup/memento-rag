"""Entity-graph CSR cache and sparse Personalized PageRank. The column-stochastic transition matrix is built once per
graph state (keyed by the store's identity and its `build_id` meta, which every store write replaces) and saved as .npz;
PPR is then a few sparse matrix-vector products. Same fixed point as memento/graph/ppr.py."""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse


@dataclass
class EntityGraph:
    ids: list[str]
    index: dict[str, int]
    trans: sparse.csr_matrix


def build_entity_graph(edges):
    pairs = [(a, b) for a, b in edges if a is not None and b is not None and a != b]
    ids = sorted({x for p in pairs for x in p})
    index = {x: i for i, x in enumerate(ids)}
    n = len(ids)
    if not pairs:
        return EntityGraph(ids, index, sparse.csr_matrix((n, n)))
    r = [index[a] for a, b in pairs] + [index[b] for a, b in pairs]
    c = [index[b] for a, b in pairs] + [index[a] for a, b in pairs]
    adj = sparse.csr_matrix((np.ones(len(r)), (r, c)), shape=(n, n))   # duplicates add weight
    deg = np.asarray(adj.sum(axis=0)).ravel()
    inv = np.divide(1.0, deg, out=np.zeros_like(deg), where=deg > 0)
    return EntityGraph(ids, index, sparse.csr_matrix(adj @ sparse.diags(inv)))


def ppr_sparse(g, seeds, damping=0.5, max_iter=50, tol=1e-9):
    total = sum(w for w in seeds.values() if w > 0)
    if total <= 0:
        return {}
    extra = [s for s in seeds if s not in g.index and seeds[s] > 0]
    n = len(g.ids)
    restart = np.zeros(n + len(extra))
    for s, w in seeds.items():
        if w > 0:
            restart[g.index[s] if s in g.index else n + extra.index(s)] = w / total
    dangling = np.ones(n + len(extra), dtype=bool)
    if n:
        dangling[:n] = np.asarray(g.trans.sum(axis=0)).ravel() == 0
    x = restart.copy()
    for _ in range(max_iter):
        nxt = (1 - damping) * restart
        if n:
            nxt[:n] += damping * (g.trans @ x[:n])
        nxt += damping * x[dangling].sum() * restart
        done = np.abs(nxt - x).sum() < tol
        x = nxt
        if done:
            break
    names = g.ids + extra
    return {names[i]: float(v) for i, v in enumerate(x) if v > 0}


def load_or_build(store, cache_dir, types="RELATION|SIMILAR"):
    build_id = store.get_meta("build_id")
    if not build_id:
        return build_entity_graph(store.entity_edges(types))
    identity = getattr(store, "identity", "")          # per-corpus Neo4j: same build_id text, other graph
    key = hashlib.sha1(f"{identity}:{build_id}:{types}".encode()).hexdigest()[:16]
    cache_dir = Path(cache_dir)
    npz, meta = cache_dir / f"{key}.npz", cache_dir / f"{key}.json"
    if npz.exists() and meta.exists():
        ids = json.loads(meta.read_text())["ids"]
        return EntityGraph(ids, {x: i for i, x in enumerate(ids)}, sparse.load_npz(npz).tocsr())
    g = build_entity_graph(store.entity_edges(types))
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{os.getpid()}.{uuid.uuid4().hex[:8]}"         # concurrent query processes: write, then rename
    tmp_npz, tmp_meta = cache_dir / f"{key}.{tag}.tmp.npz", cache_dir / f"{key}.{tag}.tmp.json"
    sparse.save_npz(tmp_npz, g.trans)
    tmp_meta.write_text(json.dumps({"identity": identity, "build_id": build_id, "types": types, "ids": g.ids}))
    os.replace(tmp_npz, npz)
    os.replace(tmp_meta, meta)
    return g
