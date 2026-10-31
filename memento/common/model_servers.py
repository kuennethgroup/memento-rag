"""
memento/common/model_servers.py — the one place that formats requests for the
vLLM-served Qwen3 embedders and rerankers.

Request formatting mirrors Qwen's reference implementations:
  - VL embedder: Qwen3VLEmbedder.format_model_input + apply_chat_template(
    add_generation_prompt=True), tokenized with special tokens (appends the
    pooled <|endoftext|>). vLLM's chat-embedding defaults for both flags are
    False, which pools a different token than the reference — never drop them.
  - VL reranker: Qwen3VLReranker.format_mm_instruction (instruction, image
    before text in the document).
  - Text embedder: Qwen3-Embedding's documented raw "Instruct: ...\nQuery: ..."
    string (no chat template).

vLLM rejects inputs longer than --max-model-len instead of truncating, so
long text is cut client-side by token count before the template is applied —
the template's closing tokens (the pooled / scored position) stay intact.
"""
import base64
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

EMBED_MAX_TOKENS = 8192     # Qwen3VLEmbedder MAX_LENGTH; vLLM --max-model-len must match
RERANK_MAX_TOKENS = 10240   # Qwen3VLReranker MAX_LENGTH; vLLM --max-model-len must match
IMAGE_TOKEN_RESERVE = 1850  # reference max_pixels = 1800 visual tokens (+ vision start/end)
TEMPLATE_RESERVE = 128      # system prompt, instruction, role tags
QUERY_RESERVE = 512         # reranker: query text + instruction

_MIME = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
         "gif": "image/gif", "webp": "image/webp", "bmp": "image/bmp"}


def image_data_url(path: str) -> str:
    ext = path.rsplit(".", 1)[-1].lower()
    if ext == "svg":
        # vLLM's image loader can't decode SVG (400). Rasterize so the model
        # sees the figure — the reference Qwen code silently embedded a batch
        # containing an SVG as the text "NULL" instead.
        import fitz
        from memento.common.pdf import PYMUPDF
        with PYMUPDF, fitz.open(path) as doc:
            data = doc[0].get_pixmap(dpi=150).tobytes("png")
        return f"data:image/png;base64,{base64.b64encode(data).decode()}"
    b64 = base64.b64encode(Path(path).read_bytes()).decode()
    return f"data:{_MIME.get(ext, 'image/png')};base64,{b64}"


def _raise_for_status(resp: requests.Response):
    if resp.status_code >= 400:
        raise requests.HTTPError(f"{resp.status_code} from {resp.url}: {resp.text[:500]}", response=resp)


def reference_instruction(instruction: str) -> str:
    # Qwen3VLEmbedder appends '.' when the instruction doesn't end in punctuation
    instruction = instruction.strip()
    if instruction and not unicodedata.category(instruction[-1]).startswith("P"):
        instruction += "."
    return instruction


def truncate_to_tokens(base_url: str, model: str, text: str, max_tokens: int) -> str:
    # Byte-level BPE: every token covers >= 1 byte, so this skips the round
    # trip for everything that cannot possibly exceed the budget.
    if len(text.encode("utf-8")) <= max_tokens:
        return text
    r = requests.post(f"{base_url}/tokenize",
                      json={"model": model, "prompt": text, "add_special_tokens": False},
                      timeout=60)
    r.raise_for_status()
    tokens = r.json()["tokens"]
    if len(tokens) <= max_tokens:
        return text
    r = requests.post(f"{base_url}/detokenize",
                      json={"model": model, "tokens": tokens[:max_tokens]}, timeout=60)
    r.raise_for_status()
    return r.json()["prompt"]


# ── VL embedder ────────────────────────────────────────────────────────────────
def vl_embed(base_url: str, model: str, instruction: str,
             text: str | None = None, image: str | None = None,
             timeout: int = 300) -> list[float]:
    content: list[dict] = []
    if image:
        content.append({"type": "image_url", "image_url": {"url": image_data_url(image)}})
    if text:
        budget = EMBED_MAX_TOKENS - TEMPLATE_RESERVE - (IMAGE_TOKEN_RESERVE if image else 0)
        content.append({"type": "text", "text": truncate_to_tokens(base_url, model, text, budget)})
    if not content:
        raise ValueError("vl_embed needs text or image")
    resp = requests.post(
        f"{base_url}/v1/embeddings",
        json={
            "model": model,
            "messages": [
                {"role": "system",
                 "content": [{"type": "text", "text": reference_instruction(instruction)}]},
                {"role": "user", "content": content},
            ],
            "add_generation_prompt": True,
            # The reference tokenizes with special tokens on, which appends
            # <|endoftext|> — the token that is pooled. vLLM's chat default
            # (False) pools one position earlier: cosine ~0.5 to the reference.
            "add_special_tokens": True,
        },
        timeout=timeout,
    )
    _raise_for_status(resp)
    return resp.json()["data"][0]["embedding"]


def vl_embed_many(base_url: str, model: str, inputs: list[dict],
                  workers: int = 16) -> list[list[float]]:
    """inputs: [{"instruction", "text"?, "image"?}] -> vectors, order-preserving."""
    if not inputs:
        return []
    with ThreadPoolExecutor(max_workers=min(workers, len(inputs))) as pool:
        futures = [pool.submit(vl_embed, base_url, model, i["instruction"],
                               i.get("text"), i.get("image")) for i in inputs]
        return [f.result() for f in futures]


# ── Text embedder (Qwen3-Embedding-8B) ─────────────────────────────────────────
def text_embed(base_url: str, model: str, instruction: str, text: str,
               timeout: int = 120) -> list[float]:
    prefix = f"Instruct: {instruction}\nQuery: "
    budget = EMBED_MAX_TOKENS - TEMPLATE_RESERVE
    resp = requests.post(
        f"{base_url}/v1/embeddings",
        json={"model": model,
              "input": prefix + truncate_to_tokens(base_url, model, text, budget)},
        timeout=timeout,
    )
    _raise_for_status(resp)
    return resp.json()["data"][0]["embedding"]


# ── Rerankers ──────────────────────────────────────────────────────────────────
def rerank(base_url: str, model: str, query: str, docs: list[dict],
           instruction: str, top_n: int | None = None, multimodal: bool = True,
           timeout: int = 300) -> list[tuple[int, float]]:
    """docs: [{"text", "image"?}] -> [(doc_index, score)] sorted by score desc.

    multimodal=False sends text only (the text-only Qwen3-Reranker-8B service,
    which has no image channel)."""
    if not docs:
        return []
    documents = []
    for d in docs:
        image = d.get("image") if multimodal else None
        budget = (RERANK_MAX_TOKENS - TEMPLATE_RESERVE - QUERY_RESERVE
                  - (IMAGE_TOKEN_RESERVE if image else 0))
        text = truncate_to_tokens(base_url, model, d.get("text") or "", budget)
        if image:
            parts = [{"type": "image_url", "image_url": {"url": image_data_url(image)}}]
            if text:
                parts.append({"type": "text", "text": text})
            documents.append({"content": parts})
        else:
            documents.append(text)
    resp = requests.post(
        f"{base_url}/rerank",
        json={"model": model, "query": query, "documents": documents,
              "instruction": instruction, "top_n": top_n or len(documents)},
        timeout=timeout,
    )
    _raise_for_status(resp)
    return [(r["index"], float(r["relevance_score"])) for r in resp.json()["results"]]
