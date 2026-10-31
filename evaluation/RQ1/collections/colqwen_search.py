"""Query-side helper for the RQ1 UniDoc-Bench sub-study's "Image-Only" channel (collection built by
evaluation/RQ1/collections/build_colqwen_collection.py). Runs under .venv-colvision/, not the main .venv (see
memento.config.settings.COLQWEN_MODEL: its custom model classes need another transformers version). Supports the ColQwen-only /
Fusion conditions in evaluation/RQ1/rq1_eval.py, which assume the collection exists."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from memento.config.settings import COLQWEN_COLLECTION, COLQWEN_MODEL, COLQWEN_DEVICE

_model = None
_processor = None


def _load():
    global _model, _processor
    if _model is None:
        from transformers import AutoModel, AutoProcessor
        _processor = AutoProcessor.from_pretrained(
            COLQWEN_MODEL, trust_remote_code=True, max_num_visual_tokens=1280,
        )
        _model = AutoModel.from_pretrained(
            COLQWEN_MODEL, dtype=torch.bfloat16, trust_remote_code=True,
            device_map=COLQWEN_DEVICE,
        ).eval()
    return _model, _processor


def colqwen_embed_query(text: str) -> list[list[float]]:
    """Multi-vector query embedding (load-once model)."""
    model, processor = _load()
    features = processor.process_texts(texts=[text])
    features = {k: v.to(COLQWEN_DEVICE) if isinstance(v, torch.Tensor) else v
                for k, v in features.items()}
    with torch.inference_mode():
        emb = model(**features).embeddings  # shape (1, n_tokens, dim)
    return emb[0].float().cpu().numpy().tolist()


def colqwen_search(client, query_text: str, top_k: int) -> list[dict]:
    """Plain MaxSim search against memento-rag-colqwen. Returns ranked
    {source, page, score} dicts — no chunked text, ColQwen never sees text."""
    qvec = colqwen_embed_query(query_text)
    results = client.query_points(
        collection_name=COLQWEN_COLLECTION,
        query=qvec,
        limit=top_k,
        with_payload=True,
    ).points
    return [
        {"source": r.payload["source"], "page": r.payload["page"], "score": r.score}
        for r in results
    ]
