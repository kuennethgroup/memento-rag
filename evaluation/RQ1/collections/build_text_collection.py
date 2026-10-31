#!/usr/bin/env python3
"""RQ1 embedding-model ablation: re-embed the VLM collection's text chunks with a local text-only model
(Qwen3-Embedding-8B) into memento-rag-text. Qwen3-Embedding-8B is the text-only sibling of the multimodal
Qwen3-VL-Embedding-8B (same size, same 4096 dimensions), so this isolates the vision-language component. The documents
are not re-parsed: existing chunks are re-embedded, so the embedder is the only variable. The schema mirrors the other
collections (named dense + sparse, `source` payload key), so memento/retrieval/search.py runs unchanged.

python evaluation/RQ1/collections/build_text_collection.py --all [--reset]"""
import argparse
import hashlib
import sys
import time
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    SparseVectorParams, SparseIndexParams, Modifier,
    Filter, FieldCondition, MatchValue,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from memento.config.settings import COLLECTION, TEXT_COLLECTION, QDRANT_URL, LOCAL_TEXT_EMBED_DIM
from memento.ingestion.embed import sparse_encode
from evaluation.RQ1.collections.local_text_embed import local_text_embed

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
SCROLL_BATCH = 500
EMBED_BATCH = 100


def fmt(s: float) -> str:
    if s < 60:
        return f"{s:.1f}s"
    m, sec = divmod(int(s), 60)
    return f"{m}m {sec}s"


def scroll_text_chunks(client: QdrantClient) -> list[dict]:
    """All `type == 'text'` points from the VLM collection, with the payload
    fields needed to re-embed and re-store them."""
    chunks = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION,
            scroll_filter=Filter(must=[FieldCondition(key="type", match=MatchValue(value="text"))]),
            with_payload=True, with_vectors=False,
            limit=SCROLL_BATCH, offset=offset,
        )
        for p in points:
            pl = p.payload or {}
            if (pl.get("text") or "").strip():
                chunks.append(pl)
        if offset is None:
            break
    return chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", required=True)
    ap.add_argument("--reset", action="store_true", help="Wipe the collection and rebuild from scratch")
    args = ap.parse_args()

    client = QdrantClient(url=QDRANT_URL, timeout=30)

    if args.reset and client.collection_exists(TEXT_COLLECTION):
        client.delete_collection(TEXT_COLLECTION)
        print(f"[Qdrant] Collection '{TEXT_COLLECTION}' wiped")

    if not client.collection_exists(TEXT_COLLECTION):
        client.create_collection(
            collection_name=TEXT_COLLECTION,
            vectors_config={DENSE_VECTOR_NAME: VectorParams(size=LOCAL_TEXT_EMBED_DIM, distance=Distance.COSINE)},
            # modifier=IDF: see memento/ingestion/run_pipeline.py's create_collection
            # for why this is required for BM25 to actually rescue rare terms
            # under hybrid (RRF) search, not just fixed to TF-only scoring.
            sparse_vectors_config={SPARSE_VECTOR_NAME: SparseVectorParams(index=SparseIndexParams(), modifier=Modifier.IDF)},
        )
        print(f"[Qdrant] Created collection '{TEXT_COLLECTION}' (named dense[{LOCAL_TEXT_EMBED_DIM}] + sparse)")
    else:
        n = client.count(collection_name=TEXT_COLLECTION).count
        print(f"[Qdrant] Collection '{TEXT_COLLECTION}' already exists — {n} vectors; re-running will upsert "
              f"(same deterministic ids) — use --reset for a clean rebuild")

    print("Scrolling text-type chunks from", COLLECTION, "...")
    t0 = time.perf_counter()
    chunks = scroll_text_chunks(client)
    print(f"  {len(chunks)} text chunks  ({fmt(time.perf_counter() - t0)})")

    t0 = time.perf_counter()
    n_done = 0
    for i in range(0, len(chunks), EMBED_BATCH):
        batch = chunks[i:i + EMBED_BATCH]
        texts = [c["text"] for c in batch]

        dense_vecs = local_text_embed(texts)
        sparse_vecs = sparse_encode(texts)

        points = []
        for j, (meta, dvec, svec) in enumerate(zip(batch, dense_vecs, sparse_vecs)):
            payload = {
                "source":      meta.get("source"),
                "type":        "text",
                "text":        meta.get("text"),
                "page":        meta.get("page", 0),
                "section":     meta.get("section"),
                "section_raw": meta.get("section_raw"),
                "bbox":        meta.get("bbox"),
                "page_end":    meta.get("page_end"),
                "img_path":    None,
            }
            done = i + j
            point_id = int(hashlib.sha256(
                f"{meta.get('source')}_{meta.get('page', 0)}_text_{done}".encode()
            ).hexdigest(), 16) % (2 ** 63)
            vector = {DENSE_VECTOR_NAME: dvec}
            if svec.indices:
                vector[SPARSE_VECTOR_NAME] = svec
            points.append(PointStruct(id=point_id, vector=vector, payload=payload))

        client.upsert(collection_name=TEXT_COLLECTION, points=points)
        n_done += len(points)
        elapsed = time.perf_counter() - t0
        rate = n_done / elapsed if elapsed > 0 else 0
        print(f"  [{n_done}/{len(chunks)}]  {fmt(elapsed)}  ({rate:.1f} vec/s)")

    total = time.perf_counter() - t0
    print(f"\nDone. {n_done} vectors embedded+upserted in {fmt(total)}.")
    print(f"Collection total: {client.count(collection_name=TEXT_COLLECTION).count}")


if __name__ == "__main__":
    main()
