"""One batched LLM pass writing both the Contextual-Retrieval situating context and the knowledge-graph entity/relation
extraction per chunk. Run separately, ingestion would pay a few batched context calls per document plus one extraction
call per prose chunk; combined, both share the batched calls and the extraction sees the document overview.

Per embeddable item: item['chunk_context'] (the situating sentence contextualize.py also sets;
embed._fold_context reads it) and item['llm_extraction'] ({"entities", "relations"} in the shape
extract_entities._fuse validates, read by build_graph through extract_entities(precomputed=...)). Registry resolution,
slugging and schema validation still happen in the fusion step; this module never writes to the graph.

run_pipeline.py picks the combined pass when CONTEXTUALIZE and GRAPH_BUILD are both on (and GRAPH_ENRICH_COMBINED,
default true), the plain contextualize_items with CONTEXTUALIZE only (its prompt stays byte-stable for the ablation),
the entities-only combined pass with GRAPH_BUILD only. Uses config.ENRICH_MODEL. Off the hot path and best effort: a
failed batch leaves its items unenriched (extraction then falls back to the per-chunk path or dictionary-only); batches
run concurrently, with one retry for the ids the first response missed.

Vision-conditioned extraction (config.GRAPH_VISION_EXTRACTION, off by default): with an `images_dir`, image/chart items
get their own extraction call with the image attached instead of caption text only, since caption-only extraction on
thin captions missed entities and sometimes invented ones from the document context. One call per image; only the
command line's combined pass passes `images_dir`; the per-chunk path and Library Upload stay caption-only."""
from __future__ import annotations

import base64
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

from memento.config.settings import (  # noqa: E402
    CONTEXT_BATCH_SIZE,
    CONTEXT_MAX_OVERVIEW_CHARS,
    CONTEXT_WORKERS,
    ENRICH_MODEL,
    GRAPH_VISION_EXTRACTION,
    REASONING_URL,
)
from memento.graph import ENTITY_TYPES  # noqa: E402
from memento.ingestion.contextualize import (  # noqa: E402
    _chunks,
    document_overview,
    embeddable_indices,
    item_snippet,
)

# Local/cloud switch (as extract_entities.py's ENRICH_PROVIDER), apart from memento/retrieval/generate.py's REASONING_PROVIDER.
# ENRICH_PROVIDER=local (default) targets the local vLLM reasoning service (REASONING_URL); "cloud" uses the cloud API
# with ENRICH_MODEL.
import os as _os  # noqa: E402

ENRICH_PROVIDER       = _os.getenv("ENRICH_PROVIDER", "local")
REASONING_MODEL_NAME  = _os.getenv("REASONING_MODEL_NAME", "/models/reasoning")


def _default_client_and_model(model: str) -> tuple:
    from openai import OpenAI
    if ENRICH_PROVIDER == "cloud":
        return (
            OpenAI(base_url=_os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                   api_key=_os.getenv("OPENAI_API_KEY", "")),
            model,
        )
    return OpenAI(base_url=f"{REASONING_URL}/v1",
                  api_key=_os.getenv("REASONING_API_KEY", _os.getenv("OPENAI_API_KEY", "none"))), REASONING_MODEL_NAME

# Entity extraction needs to see more of the chunk than the context sentence
# does — 320 chars would silently halve entity recall on long paragraphs.
_ENTITY_SNIPPET_CHARS = 900

_SYSTEM = (
    "You enrich scientific-paper excerpts for a retrieval system and a knowledge "
    "graph. You never add facts not present in the given text."
)


def build_enrich_prompt(overview: str, chunks: list[tuple[int, str]],
                        want_context: bool = True, want_entities: bool = True,
                        want_relations: bool = True) -> str:
    """User prompt: document overview + numbered excerpts → per-excerpt JSON
    object with the requested fields ("context" and/or "entities"+"relations")."""
    fields = []
    if want_context:
        fields.append(
            '"context": ONE sentence (max 25 words) situating the excerpt within '
            "the document — what it is about and how it relates to the paper's "
            "overall topic, using specific entities (materials, methods, "
            "quantities) from the document"
        )
    if want_entities:
        entity_types_str = ", ".join(ENTITY_TYPES)
        fields.append(
            '"entities": [{"name": str, "type": <one of: ' + entity_types_str + ">}] — "
            "specific scientific entities explicitly present in the excerpt (materials, processes, "
            "properties, conditions, characterization techniques). Do NOT list person names, "
            "organisations or affiliations, funding or grant numbers, software or instrument brand "
            "names, or generic words and phrases (e.g. 'sample', 'results', 'experimental plan')"
        )
        if want_relations:
            # Open relations (as in LightRAG / HippoRAG 2): a free-text description instead of a per-domain type list,
            # so any domain works without a relation vocabulary.
            fields.append(
                '"relations": [{"source": str, "target": str, "description": str}] — '
                "relationships explicitly stated in the excerpt between two names from your "
                "\"entities\" list; \"description\" is a short phrase (at most 8 words) saying "
                "how the source relates to the target, e.g. \"toughened with\", \"measured by\", "
                "\"transforms into\". Only relationships the excerpt itself states."
            )
    field_block = ",\n".join(f"    {f}" for f in fields)

    lines = [
        "<document>",
        overview,
        "</document>",
        "",
        "For each numbered excerpt below, produce a JSON object with:",
        field_block,
        "",
        "Do not add facts not present in the text. Empty lists are fine.",
        "",
        "Return ONLY a JSON object mapping each excerpt number (as a string) to "
        'its object, e.g. {"1": {...}, "2": {...}}. No prose, no markdown fences.',
        "",
        "Excerpts:",
    ]
    for idx, snippet in chunks:
        lines.append(f"[{idx}] {snippet}")
    return "\n".join(lines)


def parse_enrich_response(text: str, ids: list[int],
                          want_context: bool = True,
                          want_entities: bool = True,
                          want_relations: bool = True) -> dict[int, dict]:
    """Extract {id: {context?, entities?, relations?}} from an LLM response.
    Robust to code fences / surrounding prose (first {...} span). Only keys in
    `ids` with dict values are kept; inside each, `context` must be a
    non-empty str, `entities`/`relations` must be lists (of dicts — deeper
    validation happens in extract_entities._fuse). Returns {} on parse
    failure (best-effort)."""
    if not text:
        return {}
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {}
    try:
        raw = json.loads(m.group(0))
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    if not raw and want_entities and not want_context:
        # an explicit empty object is the model's answer "nothing to extract" (a batch of affiliation lines, say), not a
        # malformed response
        return {i: {"entities": [], "relations": []} for i in ids}
    valid = set(ids)
    out: dict[int, dict] = {}
    for k, v in raw.items():
        try:
            ki = int(k)
        except (TypeError, ValueError):
            continue
        if ki not in valid or not isinstance(v, dict):
            continue
        row: dict = {}
        if want_context:
            ctx = v.get("context")
            if isinstance(ctx, str) and ctx.strip():
                row["context"] = ctx.strip()
        if want_entities:
            ents = v.get("entities")
            rels = v.get("relations")
            row["entities"] = [e for e in ents if isinstance(e, dict)] if isinstance(ents, list) else []
            row["relations"] = ([r for r in rels if isinstance(r, dict)]
                                if want_relations and isinstance(rels, list) else [])
        if row:
            out[ki] = row
    return out


def _call_llm(client, model: str, user: str, image_path: str | None = None) -> str:
    if image_path:
        try:
            data = base64.b64encode(Path(image_path).read_bytes()).decode()
            content = [
                {"type": "text", "text": user},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}},
            ]
        except Exception:
            content = user  # image unreadable — degrade to text-only, never raise
    else:
        content = user
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": _SYSTEM}, {"role": "user", "content": content}],
        temperature=0.1,
        max_tokens=16384,
        # Without this, thinking-enabled deployments spend the whole token budget on hidden reasoning and return empty
        # content (finish_reason=length). Both spellings are needed: some servers honour top-level enable_thinking, vLLM
        # only chat_template_kwargs (as in extract_entities._llm_pass).
        extra_body={
            "enable_thinking": False,
            "reasoning_effort": "low",
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    return resp.choices[0].message.content or ""


def _image_sha1(image_path: str | None) -> str:
    if not image_path:
        return ""
    try:
        import hashlib
        return hashlib.sha1(Path(image_path).read_bytes()).hexdigest()
    except OSError:
        return ""


def _call_cached(cache, client, model: str, user: str, image_path: str | None = None):
    """(raw, key, fresh): served from `cache` when present (fresh=False)."""
    if cache is None:
        return _call_llm(client, model, user, image_path=image_path), None, True
    key = cache.key(model, _SYSTEM, user, _image_sha1(image_path))
    raw = cache.get(key)
    if raw is not None:
        return raw, key, False
    return _call_llm(client, model, user, image_path=image_path), key, True


def _snippet(items: list[dict], i: int, snippet_chars: int,
             snippets: dict[int, str] | None) -> str:
    """The excerpt text for item i: the caller's override when given (the
    graph build sends each chunk's own text, never MinerU's `content`), else
    the generic item_snippet."""
    if snippets is not None and i in snippets:
        return snippets[i][:snippet_chars]
    return item_snippet(items[i], max_chars=snippet_chars)


def _run_batch(client, model: str, overview: str, items: list[dict], batch: list[int],
               want_context: bool, want_entities: bool, snippet_chars: int,
               label: str = "", snippets: dict[int, str] | None = None,
               cache=None, want_relations: bool = True) -> dict[int, dict]:
    """One batch → {idx: row}. Retries ONCE, re-requesting only missing ids.
    Never raises."""
    out: dict[int, dict] = {}
    want = list(batch)
    last_exc: Optional[BaseException] = None
    for _attempt in (0, 1):
        try:
            chunks = [(i, _snippet(items, i, snippet_chars, snippets)) for i in want]
            prompt = build_enrich_prompt(overview, chunks, want_context=want_context,
                                         want_entities=want_entities, want_relations=want_relations)
            resp, key, fresh = _call_cached(cache, client, model, prompt)
            parsed = parse_enrich_response(resp, want, want_context=want_context,
                                           want_entities=want_entities, want_relations=want_relations)
            if parsed and fresh and key is not None:
                cache.put(key, resp)          # only responses that parsed are cached
            out.update(parsed)
            last_exc = None
        except Exception as exc:
            last_exc = exc
        want = [i for i in batch if i not in out]
        if not want:
            break
    if want:
        # Silent by design elsewhere in this module (best-effort), but a
        # batch that never fully resolves is exactly the failure class that
        # otherwise vanishes without a trace — surface it.
        reason = f"exception: {last_exc!r}" if last_exc is not None else "malformed/incomplete response"
        logger.warning(
            "llm_enrich[%s]: batch of %d item(s) left unenriched after retry (%s) — idxs=%s",
            label, len(want), reason, want,
        )
    return out


def _resolve_img_path(item: dict, images_dir: Path | None) -> str | None:
    """`images_dir / item['img_path']` if both are set and the file exists,
    else None (caller falls back to caption-only text extraction)."""
    p = item.get("img_path")
    if not p or images_dir is None:
        return None
    full = Path(images_dir) / p
    return str(full) if full.exists() else None


def _run_vision_item(client, model: str, overview: str, items: list[dict], i: int,
                     img_path: str, want_context: bool, want_entities: bool,
                     snippet_chars: int, label: str = "",
                     snippets: dict[int, str] | None = None,
                     cache=None, want_relations: bool = True) -> dict[int, dict]:
    """One image/chart item, one call, actual pixels attached alongside its
    caption. Can't batch (each item needs its own image). Same one-retry,
    never-raise contract as _run_batch."""
    out: dict[int, dict] = {}
    last_exc: Optional[BaseException] = None
    for _attempt in (0, 1):
        try:
            snippet = _snippet(items, i, snippet_chars, snippets)
            prompt = build_enrich_prompt(overview, [(i, snippet)], want_context=want_context,
                                         want_entities=want_entities, want_relations=want_relations)
            resp, key, fresh = _call_cached(cache, client, model, prompt, image_path=img_path)
            parsed = parse_enrich_response(resp, [i], want_context=want_context,
                                           want_entities=want_entities, want_relations=want_relations)
            if parsed and fresh and key is not None:
                cache.put(key, resp)
            out.update(parsed)
            last_exc = None
        except Exception as exc:
            last_exc = exc
        if i in out:
            break
    if i not in out:
        reason = f"exception: {last_exc!r}" if last_exc is not None else "malformed/incomplete response"
        logger.warning("llm_enrich[%s]: vision item idx=%d left unenriched after retry (%s)", label, i, reason)
    return out


def llm_enrich_items(
    items: list[dict],
    *,
    want_context: bool = True,
    want_entities: bool = True,
    model: str = ENRICH_MODEL,
    batch_size: int = CONTEXT_BATCH_SIZE,
    workers: int = CONTEXT_WORKERS,
    max_overview_chars: int = CONTEXT_MAX_OVERVIEW_CHARS,
    client=None,
    images_dir: Path | None = None,
    vision_enabled: bool = GRAPH_VISION_EXTRACTION,
    label: str = "",
    snippets: dict[int, str] | None = None,
    cache=None,
    want_relations: bool = True,
) -> list[dict]:
    """Mutate `items` in place: set `chunk_context` (if want_context) and
    `llm_extraction` (if want_entities) on embeddable items. Returns `items`.
    No-op when both wants are False or nothing is embeddable. Best-effort —
    a failed batch simply leaves those items unenriched.

    `label` (e.g. a paper_id) is purely for log messages — identifies which
    document a batch/vision failure belongs to, and tags the aggregate
    completeness summary logged at the end of this call.

    When `vision_enabled` and `images_dir` is given, image/chart items with a
    resolvable image file are pulled out of the normal batched text pass and
    each gets its own call with the actual pixels attached (see module
    docstring). Items whose image can't be resolved (missing path, file not
    on disk) fall back to the ordinary caption-only batched path."""
    if not (want_context or want_entities):
        return items
    idxs = sorted(snippets) if snippets is not None else embeddable_indices(items)
    if not idxs:
        return items

    overview = document_overview(items, max_overview_chars)
    try:
        if client is None:
            client, model = _default_client_and_model(model)
    except Exception:
        return items

    snippet_chars = _ENTITY_SNIPPET_CHARS if want_entities else 320

    vision_idxs: list[int] = []
    if vision_enabled and images_dir is not None:
        vision_idxs = [i for i in idxs if items[i].get("type") in ("image", "chart")
                       and _resolve_img_path(items[i], images_dir)]
    text_idxs = [i for i in idxs if i not in set(vision_idxs)]

    batches = list(_chunks(text_idxs, batch_size)) if text_idxs else []
    tasks = [("text", b) for b in batches] + [("vision", [i]) for i in vision_idxs]
    if not tasks:
        return items
    n_workers = max(1, min(workers, len(tasks)))

    def _run_task(kind: str, idx_list: list[int]) -> dict[int, dict]:
        if kind == "text":
            return _run_batch(client, model, overview, items, idx_list,
                              want_context=want_context, want_entities=want_entities,
                              snippet_chars=snippet_chars, label=label, snippets=snippets,
                              cache=cache, want_relations=want_relations)
        i = idx_list[0]
        img_path = _resolve_img_path(items[i], images_dir)
        return _run_vision_item(client, model, overview, items, i, img_path,
                                want_context=want_context, want_entities=want_entities,
                                snippet_chars=snippet_chars, label=label, snippets=snippets,
                              cache=cache, want_relations=want_relations)

    if n_workers == 1:
        results = [_run_task(kind, b) for kind, b in tasks]
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = [pool.submit(_run_task, kind, b) for kind, b in tasks]
            results = [f.result() for f in as_completed(futures)]

    n_enriched = 0
    for rows in results:
        for i, row in rows.items():
            n_enriched += 1
            if want_context and row.get("context"):
                items[i]["chunk_context"] = row["context"]
            if want_entities:
                items[i]["llm_extraction"] = {
                    "entities": row.get("entities", []),
                    "relations": row.get("relations", []),
                }

    n_requested = len(text_idxs) + len(vision_idxs)
    if n_enriched < n_requested:
        logger.warning(
            "llm_enrich[%s]: %d/%d item(s) enriched (%d text, %d vision requested) — "
            "%d fell back to dictionary-only/no extraction",
            label, n_enriched, n_requested, len(text_idxs), len(vision_idxs), n_requested - n_enriched,
        )
    else:
        logger.info(
            "llm_enrich[%s]: %d/%d item(s) enriched (%d text, %d vision)",
            label, n_enriched, n_requested, len(text_idxs), len(vision_idxs),
        )
    return items
