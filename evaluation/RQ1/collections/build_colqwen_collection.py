#!/usr/bin/env python3
"""RQ1 UniDoc-Bench sub-study: embed every rendered page image (evaluation/RQ1/collections/render_pages.py) with a late-interaction
visual document retrieval model (TomoroAI/tomoro-colqwen3-embed-4b) into a Qdrant multi-vector (MaxSim) collection, the
"Image-Only" channel of UniDoc-Bench's architecture (no text extraction; the whole page is embedded as an image).

Runs under .venv-colvision/, not the main .venv (see memento.config.settings's COLQWEN_* constants: the main venv's transformers
is too new for this model's remote code). One page at a time, on purpose: the processor's batching of variable patch
counts is not verified, and mishandling it would corrupt embeddings.

.venv-colvision/bin/python3 evaluation/RQ1/collections/build_colqwen_collection.py --all [--reset]"""
import argparse
import hashlib
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, MultiVectorConfig, MultiVectorComparator, PointStruct,
)
from transformers import AutoModel, AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from memento.config.settings import INPUT_DIR, OUTPUT_DIR, QDRANT_URL, COLQWEN_COLLECTION, \
    COLQWEN_MODEL, COLQWEN_DIM, COLQWEN_DEVICE


def fmt(s: float) -> str:
    if s < 60:
        return f"{s:.1f}s"
    m, sec = divmod(int(s), 60)
    return f"{m}m {sec}s"


def load_model():
    processor = AutoProcessor.from_pretrained(
        COLQWEN_MODEL, trust_remote_code=True, max_num_visual_tokens=1280,
    )
    model = AutoModel.from_pretrained(
        COLQWEN_MODEL, dtype=torch.bfloat16, trust_remote_code=True,
        device_map=COLQWEN_DEVICE,
    ).eval()
    return model, processor


def embed_page_image(model, processor, image_path: Path) -> list[list[float]]:
    img = Image.open(image_path).convert("RGB")
    features = processor.process_images(images=[img])
    features = {k: v.to(COLQWEN_DEVICE) if isinstance(v, torch.Tensor) else v
                for k, v in features.items()}
    with torch.inference_mode():
        emb = model(**features).embeddings  # shape (1, n_patches, dim)
    return emb[0].float().cpu().numpy().tolist()


def iter_page_images():
    """Yield (source, page, path) for every rendered page image, source ids
    matching the `source` payload field used throughout the corpus (the PDF
    stem)."""
    for pdf_path in sorted(INPUT_DIR.glob("*.pdf")):
        source = pdf_path.stem
        page_dir = OUTPUT_DIR / source / "page_images"
        if not page_dir.exists():
            continue
        for img_path in sorted(page_dir.glob("page_*.png")):
            page = int(img_path.stem.split("_")[1])
            yield source, page, img_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", required=True)
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()

    client = QdrantClient(url=QDRANT_URL, timeout=60)

    if args.reset and client.collection_exists(COLQWEN_COLLECTION):
        client.delete_collection(COLQWEN_COLLECTION)
        print(f"[Qdrant] Collection '{COLQWEN_COLLECTION}' wiped")

    if not client.collection_exists(COLQWEN_COLLECTION):
        client.create_collection(
            collection_name=COLQWEN_COLLECTION,
            vectors_config=VectorParams(
                size=COLQWEN_DIM, distance=Distance.COSINE,
                multivector_config=MultiVectorConfig(comparator=MultiVectorComparator.MAX_SIM),
            ),
        )
        print(f"[Qdrant] Created collection '{COLQWEN_COLLECTION}' "
              f"(multi-vector[{COLQWEN_DIM}], MaxSim)")
    else:
        n = client.count(collection_name=COLQWEN_COLLECTION).count
        print(f"[Qdrant] Collection '{COLQWEN_COLLECTION}' exists — {n} points "
              f"(re-running upserts idempotently; use --reset for a clean rebuild)")

    print(f"Loading {COLQWEN_MODEL} on {COLQWEN_DEVICE}...")
    model, processor = load_model()
    print("Model loaded.")

    pages = list(iter_page_images())
    print(f"{len(pages)} page images to embed.")

    t0 = time.perf_counter()
    n_done = 0
    UPSERT_BATCH = 2  # each point's multi-vector JSON can be several MB (up to ~1280 patches x 320 dims) — Qdrant's REST payload limit is 32MB
    buffer = []
    for source, page, img_path in pages:
        vec = embed_page_image(model, processor, img_path)
        point_id = int(hashlib.sha256(f"{source}_{page}_colqwen".encode()).hexdigest(), 16) % (2 ** 63)
        buffer.append(PointStruct(id=point_id, vector=vec, payload={"source": source, "page": page}))
        n_done += 1
        if len(buffer) >= UPSERT_BATCH:
            client.upsert(collection_name=COLQWEN_COLLECTION, points=buffer)
            buffer = []
        if n_done % 100 == 0 or n_done == len(pages):
            elapsed = time.perf_counter() - t0
            rate = n_done / elapsed if elapsed > 0 else 0
            eta = (len(pages) - n_done) / rate if rate > 0 else 0
            print(f"  [{n_done}/{len(pages)}]  {fmt(elapsed)} elapsed  "
                  f"({rate:.2f} pages/s, ETA {fmt(eta)})")
    if buffer:
        client.upsert(collection_name=COLQWEN_COLLECTION, points=buffer)

    total = time.perf_counter() - t0
    print(f"\nDone. {n_done} pages embedded+upserted in {fmt(total)}.")
    print(f"Collection total: {client.count(collection_name=COLQWEN_COLLECTION).count}")


if __name__ == "__main__":
    main()
