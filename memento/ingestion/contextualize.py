"""Contextual Retrieval (Anthropic) situating contexts at ingestion: a short context that places each content-list item
in its document. run_pipeline.py:build_embed_input prepends it to the embedded text; the displayed `payload['text']`
stays the original chunk. A semantic enrichment, after the structural one in enrich_parsed_content_list.py (its
`section` / `enriched_context` reach this step through each item's snippet).

Off by default (config.CONTEXTUALIZE_ENABLED). Best effort: any LLM or parse failure leaves `chunk_context` unset (no
context); never raises, never blocks ingestion.

Granularity (config.CONTEXT_GRANULARITY): "per_doc" (default) batches chunks (CONTEXT_BATCH_SIZE per call), each call
scoped to one document with a truncated overview, about ceil(n_chunks / batch) calls per document, cheap where prompt
caching is not discounted; "per_chunk": one call per chunk with the overview, higher quality, sensible only with prompt
caching.

The LLM call is isolated behind _call_llm; prompt building and response parsing are pure functions, unit-tested
directly."""
from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from memento.config.settings import (  # noqa: E402
    CONTEXT_BATCH_SIZE,
    CONTEXT_GRANULARITY,
    CONTEXT_MAX_OVERVIEW_CHARS,
    CONTEXT_MODEL,
    CONTEXT_WORKERS,
    CONTEXTUALIZE_ENABLED,
)

log = logging.getLogger(__name__)

_SNIPPET_CHARS = 320
_SKIP_TYPES = {"page_number", "header", "footer", "aside_text", "page_footnote", "ref_text"}

_SYSTEM = (
    "You situate short excerpts within a scientific paper to improve search "
    "retrieval. You never answer questions or add facts not present in the text."
)


# ── Pure helpers (no network — unit-tested) ──────────────────────────────────
def item_snippet(item: dict, max_chars: int = _SNIPPET_CHARS) -> str:
    """Representative text for an item, for use in the context prompt. Prefers
    the chunk text, then caption fields, then VLM content. Truncated to
    `max_chars` (default 320 — the combined enrichment pass in llm_enrich.py
    passes a larger cap because entity extraction needs more of the chunk)."""
    for key in ("text", "table_body", "content"):
        v = item.get(key)
        if v and str(v).strip():
            return str(v).strip()[:max_chars]
    for key in ("table_caption", "image_caption", "chart_caption"):
        v = item.get(key)
        if v:
            joined = " ".join(v) if isinstance(v, list) else str(v)
            if joined.strip():
                return joined.strip()[:max_chars]
    return ""


def embeddable_indices(items: list[dict]) -> list[int]:
    """Indices of items that will be embedded and have text worth situating —
    a light mirror of build_embed_input's skip logic (section headings, watermark
    types, low-info, empty). Over-inclusion is harmless: an unused context is
    just ignored by build_embed_input."""
    out = []
    for i, it in enumerate(items):
        if it.get("text_level") or it.get("low_info"):
            continue
        if it.get("type") in _SKIP_TYPES:
            continue
        if item_snippet(it):
            out.append(i)
    return out


def document_overview(items: list[dict], max_chars: int = CONTEXT_MAX_OVERVIEW_CHARS) -> str:
    """Compact plain-text overview of the paper (front matter first — title /
    abstract / intro usually lead the content list), truncated to `max_chars`."""
    parts = []
    total = 0
    for it in items:
        if it.get("type") not in ("text", "list"):
            continue
        t = (it.get("text") or "").strip()
        if not t:
            continue
        parts.append(t)
        total += len(t)
        if total >= max_chars:
            break
    return "\n".join(parts)[:max_chars]


def build_batch_prompt(overview: str, chunks: list[tuple[int, str]]) -> str:
    """User prompt: a document overview + numbered excerpts, asking for one
    situating sentence per excerpt, returned as a JSON object keyed by number."""
    lines = [
        "<document>",
        overview,
        "</document>",
        "",
        "For each excerpt below, write ONE short sentence (max 25 words) that "
        "situates it within the document above — what it is about and how it "
        "relates to the paper's overall topic. Use specific entities (materials, "
        "methods, quantities) from the document. Do not add facts not present.",
        "",
        'Return ONLY a JSON object mapping each excerpt number (as a string) to '
        'its one-sentence context, e.g. {"1": "...", "2": "..."}.',
        "",
        "Excerpts:",
    ]
    for idx, snippet in chunks:
        lines.append(f"[{idx}] {snippet}")
    return "\n".join(lines)


def parse_context_response(text: str, ids: list[int]) -> dict[int, str]:
    """Extract {id: context} from an LLM response. Robust to code fences and
    surrounding prose: pulls the first {...} JSON object. Only keys in `ids` are
    kept; non-string / empty values are dropped. Returns {} on any parse failure
    (best-effort — the caller degrades to no context)."""
    if not text:
        return {}
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {}
    try:
        raw = json.loads(m.group(0))
    except Exception:
        return {}
    valid = set(ids)
    out: dict[int, str] = {}
    for k, v in raw.items():
        try:
            ki = int(k)
        except (TypeError, ValueError):
            continue
        if ki in valid and isinstance(v, str) and v.strip():
            out[ki] = v.strip()
    return out


def _chunks(lst: list, n: int):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


# ── LLM call (network — isolated, best-effort) ───────────────────────────────
def _default_client_and_model(fallback_model: str) -> tuple:
    """REASONING_PROVIDER=cloud (default "local") routes this to the cloud API, as evaluation/RQ1/auto_annotate.py's
    get_llm() does."""
    import os
    from openai import OpenAI
    if os.getenv("REASONING_PROVIDER", "local") == "cloud":
        return (
            OpenAI(
                base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                api_key=os.getenv("OPENAI_API_KEY", ""),
            ),
            fallback_model,
        )
    reasoning_url = os.getenv("REASONING_URL", "http://localhost:8023")
    return (
        OpenAI(base_url=f"{reasoning_url}/v1",
               api_key=os.getenv("REASONING_API_KEY", os.getenv("OPENAI_API_KEY", "none"))),
        os.getenv("REASONING_MODEL_NAME", "/models/reasoning"),
    )


def _call_llm(client, model: str, user: str) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}],
        temperature=0.1,
        max_tokens=4096,  # 20 sentences × ~25 words + JSON scaffolding; default caps can truncate mid-object
        # Without this, thinking-enabled deployments spend the whole token budget on hidden reasoning and return empty
        # content (finish_reason=length). Both spellings are needed: some servers honour top-level enable_thinking, vLLM
        # only chat_template_kwargs (as llm_enrich / extract_entities).
        extra_body={
            "enable_thinking": False,
            "reasoning_effort": "low",
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    return resp.choices[0].message.content or ""


def _run_batch(client, model: str, overview: str, items: list[dict], batch: list[int]) -> dict[int, str]:
    """One batch → {idx: context}. Retries ONCE, re-requesting only the ids the
    first response missed (partial JSON, truncation, or a full parse failure).
    Never raises — returns whatever was recovered."""
    ctx: dict[int, str] = {}
    want = list(batch)
    for attempt in (0, 1):
        try:
            chunks = [(i, item_snippet(items[i])) for i in want]
            resp = _call_llm(client, model, build_batch_prompt(overview, chunks))
            got = parse_context_response(resp, want)
            if not got:
                log.warning(f"[contextualize] batch of {len(want)} items — 0 usable context(s) "
                            f"parsed from response (attempt {attempt + 1}/2): {resp[:200]!r}")
            ctx.update(got)
        except Exception as e:
            log.warning(f"[contextualize] batch of {len(want)} items failed "
                        f"(attempt {attempt + 1}/2): {type(e).__name__}: {e}")
        want = [i for i in batch if i not in ctx]
        if not want:
            break
    if want:
        log.warning(f"[contextualize] giving up on {len(want)} item(s) after 2 attempts")
    return ctx


# ── Orchestrator ─────────────────────────────────────────────────────────────
def contextualize_items(
    items: list[dict],
    *,
    enabled: bool = CONTEXTUALIZE_ENABLED,
    model: str = CONTEXT_MODEL,
    granularity: str = CONTEXT_GRANULARITY,
    batch_size: int = CONTEXT_BATCH_SIZE,
    max_overview_chars: int = CONTEXT_MAX_OVERVIEW_CHARS,
    workers: int = CONTEXT_WORKERS,
    client=None,
) -> list[dict]:
    """Mutate `items` in place, setting `item['chunk_context']` for embeddable
    text-bearing items. Returns `items`. No-op (and no LLM call) when disabled or
    when nothing is embeddable. Best-effort: a failed batch simply leaves those
    items without a context.

    `client` may be injected (e.g. a fake in tests); if omitted, a real
    OpenAI-compatible client for the configured provider is built lazily.
    """
    if not enabled:
        return items

    idxs = embeddable_indices(items)
    if not idxs:
        return items

    overview = document_overview(items, max_overview_chars)

    try:
        if client is None:
            client, model = _default_client_and_model(model)
    except Exception as e:
        log.warning(f"[contextualize] could not build LLM client — skipping "
                    f"contextualization entirely for this paper: {type(e).__name__}: {e}")
        return items

    batches = list(_chunks(idxs, batch_size)) if granularity == "per_doc" else [[i] for i in idxs]
    # Batches are independent and network-bound (~60s/call on the academic
    # cloud), so run them concurrently. Assignment happens on the main thread.
    n_workers = max(1, min(workers, len(batches)))
    if n_workers == 1:
        results = [_run_batch(client, model, overview, items, b) for b in batches]
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = [pool.submit(_run_batch, client, model, overview, items, b) for b in batches]
            results = [f.result() for f in as_completed(futures)]
    for ctx in results:
        for i, c in ctx.items():
            items[i]["chunk_context"] = c
    return items
