#!/usr/bin/env python3
"""End-to-end RAG for the evaluation: retrieve (hybrid dense + BM25, reranked) → reason → answer, with a vision-capable
model through vLLM's OpenAI-compatible API (image and chart sources are sent as images).

Services: the embedder, the reranker, Qdrant and the answer model (OPENAI_BASE_URL, or the optional local container in
docker-compose.yml).

The command line runs the Library Chat pipe itself (memento/retrieval/pipe_cli.py): python -m memento.retrieval.generate "<question>"
[--model <variant id>] [--thorough] [--json result.json] [--show-thinking]

Importable, the standalone retrieve-and-answer path (memento/retrieval/retrieve.py search + this module's prompt):
from memento.retrieval.generate import generate"""
import argparse
import base64
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# ── vLLM is an OpenAI-compatible server — use the official client ──────────────
try:
    from openai import OpenAI
except ImportError:
    print("[ERROR] openai package not installed.  pip install openai")
    sys.exit(1)

import sys
from pathlib import Path
from memento.config.settings import (DOMAIN, VALIDATION_ENABLED, VALIDATION_MAX_ROUNDS, VALIDATION_STRATEGY, VERIFIER_MODEL,
                           VERIFIER_URL)

try:
    from retrieve import retrieve, Hit
except ImportError:
    from memento.retrieval.retrieve import retrieve, Hit


# ── Config ────────────────────────────────────────────────────────────────────
REASONING_URL  = os.getenv("REASONING_URL",       "http://localhost:8023")
REASONING_PORT = int(os.getenv("REASONING_PORT",  "8023"))
MODEL_NAME     = os.getenv("REASONING_MODEL_NAME", "/models/reasoning")
REASONING_API_KEY = os.getenv("REASONING_API_KEY", "none")

# cloud fallback when the local model is unavailable: REASONING_PROVIDER=cloud (config/.env)
REASONING_PROVIDER = os.getenv("REASONING_PROVIDER", "local")   # "local" | "cloud"
CLOUD_BASE_URL     = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
CLOUD_MODEL        = os.getenv("CLOUD_REASONING_MODEL", "qwen3.8-27b")

DEFAULT_RETRIEVE_K  = 80
DEFAULT_RERANK_TOP  = 20
DEFAULT_MAX_TOKENS  = 16384     # don't go below 8192, hidden <think>
                                # reasoning can burn the whole budget and return an empty answer
DEFAULT_TEMPERATURE = 0.15      # low temp for factual RAG
DEFAULT_TOP_P       = 0.9
MAX_IMAGES_TO_LLM  = int(os.getenv("MAX_IMAGES_TO_LLM", "6"))  # cost/latency cap


# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = DOMAIN["system_prompt"]


# ── Prompt assembly ───────────────────────────────────────────────────────────
def format_hit(h: Hit, idx: int) -> str:
    """Render a single retrieved chunk as a numbered source block."""
    page_str = f"{h.page}-{h.page_end}" if h.page_end is not None and h.page_end > h.page else str(h.page)
    header = f"[Source {idx}]  paper={h.paper_id}  page={page_str}  type={h.type}"
    if h.is_neighbor:
        header += "  (context)"

    if h.type in ("image", "chart"):
        img_name = Path(h.img_path or "").name
        body = f"[Figure: {img_name}]"
        if h.text:
            body += f"\n{h.text}"
        elif h.caption:
            body += f"\nCaption: {h.caption}"
    elif h.type == "table":
        body = h.text or h.caption or "[empty table]"
    else:
        body = h.text or "[empty]"

    return f"{header}\n{body}"


def build_user_message(query: str, hits: list[Hit]) -> str:
    sources = "\n\n".join(format_hit(h, i + 1) for i, h in enumerate(hits))
    return (
        f"Question: {query}\n\n"
        f"{'─' * 60}\n"
        f"Sources ({len(hits)} chunks retrieved and reranked):\n\n"
        f"{sources}\n"
        f"{'─' * 60}\n\n"
        "Please answer the question based strictly on the sources above."
    )


# ── Result type ───────────────────────────────────────────────────────────────
@dataclass
class GenerateResult:
    query:          str
    answer:         str
    thinking:       Optional[str]   # None unless thinking mode was on
    hits:           list[Hit]
    timing:         dict            # {retrieve_s, generate_s, total_s}
    model:          str
    usage:          dict            # {prompt_tokens, completion_tokens, total_tokens}


# ── Core generate function (importable) ───────────────────────────────────────
def generate(
    query:          str,
    think:          bool         = False,
    retrieve_k:     int          = DEFAULT_RETRIEVE_K,
    rerank_top:     int          = DEFAULT_RERANK_TOP,
    paper_id:       Optional[str] = None,
    use_reranker:   bool         = True,
    expand:         bool         = True,
    max_tokens:     int          = DEFAULT_MAX_TOKENS,
    temperature:    float        = DEFAULT_TEMPERATURE,
    reasoning_url:  str          = REASONING_URL,
    validate:       bool         = VALIDATION_ENABLED,
    strategy:       str          = VALIDATION_STRATEGY,
    verbose:        bool         = False,
) -> GenerateResult:
    """
    Full RAG pipeline: retrieve → format prompt → call vLLM → return result.

    Args:
        think:      enable Qwen's extended thinking mode (slower, more thorough)
        expand:     include same-page neighbours of top chunks (recommended)
        use_hybrid: dense + BM25 retrieval; set False for dense-only ablation
    """
    t_start = time.perf_counter()

    # ── Step 1: Retrieve ───────────────────────────────────────────────────────
    if verbose:
        print(f"[generate] Retrieving, rerank={use_reranker})...")

    hits, retrieval_timing = retrieve(
        query          = query,
        retrieve_k     = retrieve_k,
        rerank_top     = rerank_top,
        paper_id       = paper_id,
        expand         = expand,
        use_reranker   = use_reranker,
        verbose        = verbose,
    )
    t_retrieved = time.perf_counter()

    # Pick the reasoning provider — local vLLM (default) or the cloud
    # vision+text model (REASONING_PROVIDER=cloud)
    if REASONING_PROVIDER == "cloud":
        client_base_url, client_api_key, model_name = (
            CLOUD_BASE_URL, os.getenv("OPENAI_API_KEY", ""), CLOUD_MODEL,
        )
    else:
        client_base_url, client_api_key, model_name = (
            f"{reasoning_url}/v1", REASONING_API_KEY, MODEL_NAME,
        )

    if not hits:
        # Nothing retrieved — return a well-formed "no results" answer
        return GenerateResult(
            query    = query,
            answer   = "No relevant sources were found in the knowledge base for this query.",
            thinking = None,
            hits     = [],
            timing   = {"retrieve_s": t_retrieved - t_start,
                        "generate_s": 0.0, "total_s": t_retrieved - t_start},
            model    = model_name,
            usage    = {},
        )

    if verbose:
        print(f"[generate] {len(hits)} chunks retrieved in "
              f"{t_retrieved - t_start:.2f}s — calling reasoning model...")

    # ── Step 2: Build prompt ───────────────────────────────────────────────────
    user_message = build_user_message(query, hits)

    # image/chart sources: the image with its caption, at most MAX_IMAGES_TO_LLM
    user_content: list | str = user_message
    image_parts = []
    for i, h in enumerate(hits):
        if len(image_parts) >= MAX_IMAGES_TO_LLM:
            break
        if h.type in ("image", "chart") and h.img_path:
            try:
                from memento.common.model_servers import image_data_url   # SVG rasterized, correct image type
                url = image_data_url(h.img_path)
                image_parts.append({"type": "text", "text": f"Image for Source {i + 1}:"})
                image_parts.append({"type": "image_url", "image_url": {"url": url}})
            except Exception:
                pass  # image missing on disk — degrade gracefully to caption-only
    if image_parts:
        user_content = [{"type": "text", "text": user_message}] + image_parts

    # ── Step 3: Call the reasoning model (local vLLM or cloud) ────────
    client = OpenAI(base_url=client_base_url, api_key=client_api_key)

    # Qwen3's thinking follows `think` (reasoning_effort and enable_thinking)

    # Ensure the model is encouraged to think deeply for scientific accuracy
    extra_body = {
        "reasoning_effort": "high" if think else "low",
        "enable_thinking": True if think else False  # Explicit flag for Qwen3 series
    }

    MODEL_CTX = 32768
    if isinstance(user_content, str):
        prompt_chars = len(SYSTEM_PROMPT) + len(user_content)
    else:
        prompt_chars = len(SYSTEM_PROMPT) + sum(
            len(p.get("text", "")) for p in user_content if isinstance(p, dict) and "text" in p
        )
    est_prompt_tokens = prompt_chars // 3
    safe_max = max(2048, MODEL_CTX - est_prompt_tokens - 256)
    effective_max_tokens = min(max_tokens, safe_max)

    response = client.chat.completions.create(
        model       = model_name,
        messages    = [
            {"role": "system",  "content": SYSTEM_PROMPT},
            {"role": "user",    "content": user_content},
        ],
        temperature = temperature,
        top_p       = DEFAULT_TOP_P,
        max_tokens  = effective_max_tokens,
        extra_body  = extra_body,
    )
    t_generated = time.perf_counter()

    msg     = response.choices[0].message
    answer  = msg.content or ""
    # vLLM with --reasoning-parser qwen3 puts the <think>…</think> block in
    # reasoning_content; the cloud API exposes the same trace as
    # plain "reasoning" — check both.
    thinking = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)

    usage = {}
    if response.usage:
        usage = {
            "prompt_tokens":     response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens":      response.usage.total_tokens,
        }

    validation_result = None
    if validate and hits:
        # RQ3 verification loop (memento/agentic_validation/strategies.py), the same components as the live pipe
        from memento.agentic_validation.strategies import ArmContext, run_strategy
        from memento.agentic_validation.verifiers import LLMJudgeVerifier, VerdictCache, VLMFigureVerifier

        def as_dict(h) -> dict:
            return {"text": h.text or h.caption or "", "source": h.paper_id, "page": h.page, "type": h.type,
                    "img_path": h.img_path, "section": None, "section_raw": None, "bbox": h.bbox}

        def retrieve_fn(q: str) -> list[dict]:
            found, _ = retrieve(query=q, retrieve_k=retrieve_k, rerank_top=rerank_top, use_reranker=use_reranker)
            return [as_dict(h) for h in found]
        ver_client = OpenAI(base_url=VERIFIER_URL, api_key=client_api_key) if VERIFIER_URL else client
        ver_model = VERIFIER_MODEL or model_name
        ctx = ArmContext(client, model_name, SYSTEM_PROMPT, query, [as_dict(h) for h in hits],
                         LLMJudgeVerifier(ver_client, ver_model), VLMFigureVerifier(ver_client, ver_model),
                         VerdictCache(), retrieve_fn=retrieve_fn, enable_thinking=think)
        validation_result = run_strategy(strategy, ctx, p1_answer=answer, max_rounds=VALIDATION_MAX_ROUNDS)
        answer = validation_result.answer
        if verbose:
            print(f"[generate] verification {validation_result.arm}: {len(validation_result.claims)} claims, "
                  f"{validation_result.rounds} rounds, {validation_result.llm_calls} calls")

    t_final = time.perf_counter()

    return GenerateResult(
        query    = query,
        answer   = answer,
        thinking = thinking,
        hits     = hits,
        timing   = {
            "retrieve_s": round(t_retrieved - t_start, 3),
            "generate_s": round(t_generated - t_retrieved, 3),
            "validate_s": round(t_final - t_generated, 3) if validation_result else 0.0,
            "total_s":    round(t_final - t_start, 3),
        },
        model = response.model,
        usage = usage,
    )


# ── Pretty printer ────────────────────────────────────────────────────────────
def print_result(result: GenerateResult, show_thinking: bool = False,
                 show_sources: bool = True):
    print(f'\n{"═" * 72}')
    print(f'Query: "{result.query}"')
    print(f'Model: {result.model}  |  '
          f'retrieve {result.timing["retrieve_s"]:.2f}s  |  '
          f'generate {result.timing["generate_s"]:.2f}s  |  '
          f'total {result.timing["total_s"]:.2f}s')
    if result.usage:
        u = result.usage
        print(f'Tokens: {u.get("prompt_tokens", "?")} prompt + '
              f'{u.get("completion_tokens", "?")} completion = '
              f'{u.get("total_tokens", "?")} total')
    print(f'Sources: {len(result.hits)} chunks')
    print(f'{"─" * 72}')

    if show_thinking and result.thinking:
        print("[Thinking trace]")
        print(result.thinking)
        print(f'{"─" * 72}')

    print(result.answer)

    if show_sources and result.hits:
        print(f'\n{"─" * 72}')
        print("Retrieved sources:")
        for i, h in enumerate(result.hits, 1):
            tag = " [ctx]" if h.is_neighbor else ""
            score = (f" r={h.rerank_score:.3f}"
                     if h.rerank_score is not None else f" v={h.vector_score:.3f}")
            page_str = f"p{h.page}-{h.page_end}" if h.page_end is not None and h.page_end > h.page else f"p{h.page}"
            print(f"  [{i}] {h.paper_id} {page_str} {h.type}{score}{tag}")
            if h.type in ("image", "chart"):
                cap = (h.caption or "")[:80]
                print(f"       {Path(h.img_path or '').name}" + (f" — {cap}" if cap else ""))
            elif h.text:
                print(f"       {h.text[:120].replace(chr(10), ' ')}")
    print(f'{"═" * 72}')


def result_to_dict(r: GenerateResult) -> dict:
    from dataclasses import asdict as dc_asdict
    return {
        "query":    r.query,
        "answer":   r.answer,
        "thinking": r.thinking,
        "model":    r.model,
        "timing":   r.timing,
        "usage":    r.usage,
        "hits":     [dc_asdict(h) for h in r.hits],
    }


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    """The command line is the Library Chat pipe itself (memento/retrieval/pipe_cli.py), so both stay in step."""
    from memento.retrieval.pipe_cli import main as pipe_main
    return pipe_main()


if __name__ == "__main__":
    main()