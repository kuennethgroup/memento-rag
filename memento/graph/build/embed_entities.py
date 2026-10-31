#!/usr/bin/env python3
"""
memento/graph/build/embed_entities.py — Entity-name embeddings for dense graph seeding
(+ optional SIMILAR synonymy edges).

The HippoRAG2-style upgrades in memento/graph/query/graph_retrieval.py need two offline
artifacts this script builds from an already-populated knowledge graph:

1. `memento-entities` Qdrant collection (config.ENTITY_COLLECTION): one dense
   vector per Entity node, embedding "name (type): aliases" with the local
   embedder under config.ENTITY_EMBED_INSTRUCTION (the SAME instruction the
   query-side seed lookup uses — keep them single-sourced). Enables
   `dense_seed_entities()`: query → nearest entities → traversal seeds, so
   seeds are not limited to registry entities.

2. `--synonymy T`: SIMILAR edges between entity pairs whose embeddings have
   cosine ≥ T (default 0.85). Bridges aliases the registry doesn't know —
   especially LLM-discovered "novel" entities named differently from their
   registry entity, so PPR mass can flow across naming variants.
   Registry-known aliases already collapse onto one node at build time; this
   catches the rest. Idempotent (MERGE) and cheap to re-run.

Run AFTER `build_graph.py --all` (needs Entity nodes to exist), re-run after
any graph rebuild:

    python memento/graph/build/embed_entities.py                 # embeddings only
    python memento/graph/build/embed_entities.py --synonymy 0.85 # + SIMILAR edges
    python memento/graph/build/embed_entities.py --recreate      # wipe + rebuild collection

Requires: Neo4j up (graph populated), embedder up, Qdrant up.
"""
from __future__ import annotations

import argparse
import hashlib
import sys

from memento.config.settings import (  # noqa: E402
    EMBED_DIM,
    EMBEDDER_URL,
    ENTITY_COLLECTION,
    ENTITY_EMBED_INSTRUCTION,
    QDRANT_URL,
)

EMBED_BATCH = 32


def entity_text(ent: dict) -> str:
    """The string that gets embedded for an entity: name, type, aliases."""
    name = (ent.get("name") or ent.get("id") or "").strip()
    etype = (ent.get("type") or "").strip()
    aliases = [a for a in (ent.get("aliases") or []) if a and a.strip() and a.strip() != name]
    text = f"{name} ({etype})" if etype else name
    if aliases:
        text += ": " + ", ".join(aliases[:8])
    return text


def entity_point_id(entity_id: str) -> int:
    """Deterministic Qdrant point id for an entity (same sha256 % 2^63 scheme
    as chunk point ids — stable across rebuilds)."""
    return int(hashlib.sha256(f"entity::{entity_id}".encode()).hexdigest(), 16) % (2 ** 63)


def embed_texts(texts: list[str], embedder_url: str = EMBEDDER_URL) -> list[list[float]]:
    from memento.config.settings import EMBEDDER_MODEL
    from memento.common.model_servers import vl_embed_many

    out: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i:i + EMBED_BATCH]
        out.extend(vl_embed_many(embedder_url, EMBEDDER_MODEL,
                                 [{"instruction": ENTITY_EMBED_INSTRUCTION, "text": t}
                                  for t in batch]))
    return out


def build_collection(entities: list[dict], vectors: list[list[float]],
                     client, recreate: bool = False) -> int:
    from qdrant_client.models import Distance, PointStruct, VectorParams

    if recreate and client.collection_exists(ENTITY_COLLECTION):
        client.delete_collection(ENTITY_COLLECTION)
    if not client.collection_exists(ENTITY_COLLECTION):
        client.create_collection(
            ENTITY_COLLECTION,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )
    points = [
        PointStruct(
            id=entity_point_id(e["id"]),
            vector=v,
            payload={"entity_id": e["id"], "name": e.get("name"), "type": e.get("type")},
        )
        for e, v in zip(entities, vectors)
    ]
    for i in range(0, len(points), 256):
        client.upsert(ENTITY_COLLECTION, points=points[i:i + 256], wait=True)
    return len(points)


def synonymy_pairs(entities: list[dict], vectors: list[list[float]],
                   threshold: float) -> list[tuple[str, str, float]]:
    """(id_a, id_b, cosine) for all entity pairs with cosine ≥ threshold.
    O(n²) on a few thousand entities via one numpy matmul — fine at corpus
    scale; revisit (FAISS / blocked matmul) if entities reach 10^5."""
    import numpy as np

    m = np.asarray(vectors, dtype=np.float32)
    m /= np.clip(np.linalg.norm(m, axis=1, keepdims=True), 1e-12, None)
    sim = m @ m.T
    ids = [e["id"] for e in entities]
    pairs = []
    n = len(ids)
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= threshold:
                pairs.append((ids[i], ids[j], float(sim[i, j])))
    return pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--synonymy", type=float, metavar="T", default=None,
                    help="also write SIMILAR edges for entity pairs with cosine >= T (e.g. 0.85)")
    ap.add_argument("--recreate", action="store_true",
                    help="drop and rebuild the entity collection")
    args = ap.parse_args()

    from qdrant_client import QdrantClient
    from memento.graph.store import GraphEdge, get_store

    with get_store() as store:
        entities = store.all_entities()
        if not entities:
            sys.exit("No Entity nodes in the graph — run build_graph.py first.")
        print(f"Embedding {len(entities)} entities...")
        vectors = embed_texts([entity_text(e) for e in entities])

        client = QdrantClient(url=QDRANT_URL, timeout=30)
        n = build_collection(entities, vectors, client, recreate=args.recreate)
        print(f"Upserted {n} entity vectors into {ENTITY_COLLECTION}")

        if args.synonymy is not None:
            pairs = synonymy_pairs(entities, vectors, args.synonymy)
            edges = [
                GraphEdge(start_id=a, start_label="Entity",
                          end_id=b, end_label="Entity",
                          type="SIMILAR", props={"score": round(s, 4)})
                for a, b, s in pairs
            ]
            store.upsert_edges(edges)
            print(f"Wrote {len(edges)} SIMILAR edges (cosine >= {args.synonymy})")


if __name__ == "__main__":
    main()
