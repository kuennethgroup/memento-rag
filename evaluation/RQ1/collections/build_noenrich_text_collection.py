#!/usr/bin/env python3
"""
evaluation/RQ1/collections/build_noenrich_text_collection.py — Re-embed all NoEnrich chunks
with the text-only Qwen3-Embedding-8B, for the clean VL-vs-text embedder
ablation without enrichment confound.

Scrolls memento-*-rag-noenrich, re-embeds each chunk's text payload with the
local text embedder, and stores in memento-*-rag-noenrich-text. Image/chart
chunks that have no text (<10 chars) are skipped — text embedder can't embed
empty strings, and this mirrors the text_only fallback in run_pipeline.py.

Uses adaptive batching: long chunks get smaller batches to prevent GPU OOM.

Usage:
    DOMAIN_NAME=ceramics COLLECTION_PREFIX=memento-ceramics \
        python evaluation/RQ1/collections/build_noenrich_text_collection.py --all
    # Add --reset to wipe the destination collection first
    # Add --pilot to process only first 500 chunks as a test
"""
import argparse
import sys
import time
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    SparseVectorParams, SparseIndexParams, Modifier,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from memento.config.cli_domain import apply_domain_arg, domain_banner  # noqa: E402
apply_domain_arg()
from memento.config.settings import NOENRICH_COLLECTION, QDRANT_URL, LOCAL_TEXT_EMBED_DIM
from memento.ingestion.embed import sparse_encode
from evaluation.RQ1.collections.local_text_embed import local_text_embed

DEST_COLLECTION = NOENRICH_COLLECTION.replace("-rag-noenrich", "-rag-noenrich-text")
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
SCROLL_BATCH = 500
MIN_TEXT_LEN = 10

# Adaptive batch sizes by text length to prevent GPU OOM
# (77K-char outliers need batch=1, short chunks can do 64)
BATCH_TIERS = [
    (5000, 1),    # >5000 chars: one at a time
    (2000, 16),   # 2000-5000 chars: small batch
    (0, 64),      # <2000 chars: large batch
]


def batch_size_for(text_len: int) -> int:
    for threshold, bs in BATCH_TIERS:
        if text_len > threshold:
            return bs
    return 64


def fmt(s: float) -> str:
    if s < 60:
        return f"{s:.1f}s"
    m, sec = divmod(int(s), 60)
    return f"{m}m {sec}s"


def embed_and_upsert(client, batch, n_done, total, t0):
    texts = [c["text"] for c in batch]
    dense_vecs = local_text_embed(texts)
    sparse_vecs = sparse_encode(texts)

    points = []
    for c, dvec, svec in zip(batch, dense_vecs, sparse_vecs):
        vector = {DENSE_VECTOR_NAME: dvec}
        if svec.indices:
            vector[SPARSE_VECTOR_NAME] = svec
        points.append(PointStruct(id=c["id"], vector=vector, payload=c["payload"]))

    client.upsert(collection_name=DEST_COLLECTION, points=points)
    n_done += len(points)
    elapsed = time.perf_counter() - t0
    rate = n_done / elapsed if elapsed > 0 else 0
    max_len = max(len(c["text"]) for c in batch)
    print(f"  [{n_done}/{total}]  {fmt(elapsed)}  ({rate:.1f} vec/s)  batch={len(batch)} max_chars={max_len}")
    return n_done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--domain', required=True, help='Corpus domain profile (config/domains/<name>.yaml), e.g. ceramics | polymer-engineering')
    ap.add_argument("--all", action="store_true", required=True)
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--pilot", action="store_true", help="Only process first 500 chunks as a test")
    args = ap.parse_args()
    domain_banner()

    client = QdrantClient(url=QDRANT_URL, timeout=30)
    print(f"Source: {NOENRICH_COLLECTION}")
    print(f"Dest:   {DEST_COLLECTION}")

    if args.reset and client.collection_exists(DEST_COLLECTION):
        client.delete_collection(DEST_COLLECTION)
        print(f"[Qdrant] Wiped '{DEST_COLLECTION}'")

    if not client.collection_exists(DEST_COLLECTION):
        client.create_collection(
            collection_name=DEST_COLLECTION,
            vectors_config={DENSE_VECTOR_NAME: VectorParams(size=LOCAL_TEXT_EMBED_DIM, distance=Distance.COSINE)},
            sparse_vectors_config={SPARSE_VECTOR_NAME: SparseVectorParams(
                index=SparseIndexParams(), modifier=Modifier.IDF)},
        )
        print(f"[Qdrant] Created '{DEST_COLLECTION}' (dense[{LOCAL_TEXT_EMBED_DIM}] + sparse)")

    # Scroll ALL chunks from NoEnrich
    print(f"Scrolling all chunks from {NOENRICH_COLLECTION}...")
    t0 = time.perf_counter()
    chunks = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=NOENRICH_COLLECTION,
            with_payload=True, with_vectors=False,
            limit=SCROLL_BATCH, offset=offset,
        )
        for p in points:
            pl = p.payload or {}
            text = (pl.get("text") or "").strip()
            if len(text) < MIN_TEXT_LEN:
                continue
            chunks.append({"id": p.id, "payload": pl, "text": text})
        if offset is None:
            break
    print(f"  {len(chunks)} chunks with text >= {MIN_TEXT_LEN} chars ({fmt(time.perf_counter() - t0)})")

    if args.pilot:
        chunks = chunks[:500]
        print(f"  [PILOT MODE] Processing only first {len(chunks)} chunks")

    # Sort by text length descending so long chunks (with batch=1) go first
    # while GPU memory is clean
    chunks.sort(key=lambda c: len(c["text"]), reverse=True)

    # Build adaptive batches
    batches = []
    i = 0
    while i < len(chunks):
        bs = batch_size_for(len(chunks[i]["text"]))
        batches.append(chunks[i:i + bs])
        i += bs

    tier_counts = {}
    for b in batches:
        max_len = max(len(c["text"]) for c in b)
        bs = batch_size_for(max_len)
        tier_counts[bs] = tier_counts.get(bs, 0) + len(b)
    print(f"  Adaptive batching: {len(batches)} batches ({tier_counts})")

    # Re-embed
    t0 = time.perf_counter()
    n_done = 0
    for batch in batches:
        n_done = embed_and_upsert(client, batch, n_done, len(chunks), t0)

    total = time.perf_counter() - t0
    final_count = client.count(collection_name=DEST_COLLECTION).count
    print(f"\nDone. {n_done} vectors embedded+upserted in {fmt(total)}.")
    print(f"Collection total: {final_count}")


if __name__ == "__main__":
    main()
