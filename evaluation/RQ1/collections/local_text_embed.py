"""
evaluation/RQ1/collections/local_text_embed.py — Qwen3-Embedding-8B (text-only) client, for
RQ1 embedding-model ablation.

Talks to a vLLM embedding server (--runner pooling) serving Qwen3-Embedding-8B
on LOCAL_TEXT_EMBED_URL, using the model's documented raw
"Instruct: ...\nQuery: ..." format (see memento/common/model_servers.text_embed).
The SAME instruction (DOMAIN["instruction"]) as the VL embedder is applied to
documents and queries alike, so this ablation isolates only the choice of model.
"""
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from memento.config.settings import LOCAL_TEXT_EMBED_URL, TEXT_EMBEDDER_MODEL, DOMAIN
from memento.common.model_servers import text_embed

QUERY_INSTRUCTION = DOMAIN["instruction"]

_MAX_WORKERS = 16


def local_text_embed(texts: list[str], instruction: str = QUERY_INSTRUCTION,
                     batch_size: int = 100) -> list[list[float]]:
    """Batch-embed a list of texts. Order-preserving, concurrent."""
    out = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        with ThreadPoolExecutor(max_workers=min(_MAX_WORKERS, len(batch))) as pool:
            futures = [pool.submit(text_embed, LOCAL_TEXT_EMBED_URL, TEXT_EMBEDDER_MODEL,
                                   instruction, t) for t in batch]
            out.extend(f.result() for f in futures)
    return out


def local_text_embed_query(text: str, instruction: str = QUERY_INSTRUCTION) -> list[float]:
    """Single-text embed — query-side call, same model/instruction as
    ingestion so query and corpus vectors live in the same space."""
    return text_embed(LOCAL_TEXT_EMBED_URL, TEXT_EMBEDDER_MODEL, instruction, text)
