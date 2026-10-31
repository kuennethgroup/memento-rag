#!/usr/bin/env python3
"""
memento/graph/build/extract_entities.py — Hybrid entity & relation extraction (RQ2 dual-graph KG).

For each text chunk, two passes are combined:

  1. **Dictionary match** — `graph.find_entities_in_text()` against the
     configured domain registry (graph/entity_registry_<domain>.py). High
     precision, cross-lingual/cross-notation (a formula and a German name
     resolve to the same registry canonical_id).

  2. **LLM extraction** — one Qwen3 call constrained to the domain schema
     (`memento.graph.ENTITY_TYPES` / `RELATION_TYPES`), to recover relations
     between entities and entities not in the registry (recall). The
     dictionary-matched entities are given to the LLM as "known entities" so
     it can form relations against them by name.

The two passes are fused into a single `ExtractionResult`: entities keyed by
id (registry `canonical_id` for known entities, a normalized slug for novel
ones) + relations between entity ids.

Best-effort: any failure in the LLM pass (network, timeout, malformed JSON)
degrades to dictionary-only results (entities, no relations) — this function
never raises and never blocks ingestion, mirroring the
`enrich_parsed_content_list._detect_paper` philosophy elsewhere in the
pipeline. 429s from the reasoning endpoint are retried with backoff
(`RATE_LIMIT_MAX_RETRIES`) before falling back, and logged so a throttled
cloud endpoint doesn't silently and invisibly thin out the graph.

Usage:
    from memento.graph.build.extract_entities import extract_entities

    result = extract_entities(chunk_text)
    result.entities    # list[ExtractedEntity]
    result.relations   # list[ExtractedRelation]
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from memento.config.settings import REASONING_URL  # noqa: E402
from pydantic import BaseModel  # noqa: E402


class EntityExtractionResponse(BaseModel):
    entities: list[Any] = []
    relations: list[Any] = []
from memento.graph import ENTITY_TYPES, RELATION_TYPES, canonicalize, find_entities_in_text, normalize  # noqa: E402

try:
    from openai import OpenAI, RateLimitError
except ImportError:  # pragma: no cover — exercised only when openai isn't installed
    OpenAI = None  # type: ignore[assignment]
    RateLimitError = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)


# ── Config ──
# Its own switch, apart from memento/retrieval/generate.py's REASONING_PROVIDER: graph extraction has its own local/cloud needs.
REASONING_MODEL_NAME = os.getenv("REASONING_MODEL_NAME", "/models/reasoning")
ENRICH_PROVIDER      = os.getenv("ENRICH_PROVIDER", "local")   # "local" | "cloud"
CLOUD_BASE_URL       = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
CLOUD_MODEL          = os.getenv("CLOUD_REASONING_MODEL", "qwen3.8-27b")

# Bulk per-chunk extraction — keep it cheap. Override via extract_entities(...)
# if a corpus benefits from deeper reasoning (slower, more relations).
DEFAULT_REASONING_EFFORT = "low"
DEFAULT_ENABLE_THINKING  = False
DEFAULT_TEMPERATURE      = 0.0
DEFAULT_MAX_TOKENS       = 1024

# Chunks shorter than this rarely contain a relation worth an LLM call
# (table fragments, short captions, headings) — dictionary match still runs.
MIN_CHARS_FOR_LLM = 40

# Retry policy for transient 429s from the reasoning endpoint (the cloud
# provider enforces per-minute/hour/day quotas). Exponential backoff, capped
# so a sustained outage still degrades to dictionary-only within a bounded
# time per chunk rather than stalling the whole ingestion run.
RATE_LIMIT_MAX_RETRIES     = 3
RATE_LIMIT_BACKOFF_SECONDS = 5.0


# ── Data model ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ExtractedEntity:
    id:        str   # registry canonical_id, or a normalized slug for novel entities
    name:      str   # display name
    type:      str   # one of memento.graph.ENTITY_TYPES
    canonical: bool  # True if resolved against the domain registry


@dataclass(frozen=True)
class ExtractedRelation:
    source:   str  # ExtractedEntity.id
    target:   str  # ExtractedEntity.id
    rel_type: str  # one of memento.graph.RELATION_TYPES keys


@dataclass
class ExtractionResult:
    entities:  list[ExtractedEntity]  = field(default_factory=list)
    relations: list[ExtractedRelation] = field(default_factory=list)


# ── Novel-entity junk filter + id slugging ────────────────────────────────
_SLUG_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")

_ELEMENTS = frozenset(e.lower() for e in
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co "
    "Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb "
    "Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re "
    "Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm".split())

_VALUE_RE = re.compile(
    r"^[\d.,]+\s*(?:°?[ckf]|℃|k|mpa|gpa|kn|wt\.?%|mol\.?%|vol\.?%|at\.?%|%|"
    r"h|hr|hrs|min|s|ms|nm|[μu]m|mm|cm|m|kg|g|mg|hz|khz|mhz|ev|kv|v|a|ma)?$",
    re.I,
)


_PERSON_RE = re.compile(
    r"^(?:[A-Z]\.\s?){1,3}[A-Z][a-z\u00C0-\u017F'-]+$")      # D.C. Sinclair, J. Müller
# Initials alone (C.M.R.) are not filtered: they look like dotted technique or phase abbreviations (B.E.T., C.S.H.).
_TITLE_RE = re.compile(r"^(?:prof|dr|mr|mrs|ms)\.?\s+\S", re.I)


def _is_grant_id(s: str) -> bool:
    """Funding/grant numbers (DE-AC02-05CH11231, EP/K008749/1): at least two
    '-' or '/' separators, six or more digits including a run of four, no
    spaces or parentheses. Fractional formulas ((Hf1/3Zr1/3Ce1/3)O2) and sample
    names (3Y-TZP, EP1-3_80/20) fail the digit-run or parenthesis test."""
    return ((s.count("-") + s.count("/")) >= 2 and sum(ch.isdigit() for ch in s) >= 6
            and re.search(r"\d{4,}", s) is not None and not any(c in s for c in " ()"))


def _is_junk(name: str) -> bool:
    """True for non-entities the LLM extracts: bare measurements (`1450°C`), stray single-character tokens (`c`, `M*`),
    person names and grant numbers (acknowledgement and author text). Element symbols and simple diatomics are kept."""
    s = (name or "").strip()
    if not s:
        return True
    if _VALUE_RE.match(s):
        return True
    if _TITLE_RE.match(s) or _PERSON_RE.match(s) or _is_grant_id(s):
        return True
    base = re.sub(r"[\d₀-₉]+$", "", s).lower()
    if base in _ELEMENTS:
        return False
    return len(s) <= 2


def _slugify(name: str) -> str:
    """Turn an arbitrary entity name into a stable snake_case id, using the
    same normalization (Greek/sub-superscript folding, lowercasing) as
    registry canonical_ids so a novel entity discovered first in one chunk
    and re-discovered in another fuses to the same id."""
    slug = _SLUG_NON_ALNUM_RE.sub("_", normalize(name)).strip("_")
    return slug


def _resolve_entity(name: str, type_hint: Optional[str] = None) -> Optional[ExtractedEntity]:
    """Resolve an entity name (from the LLM or a relation endpoint) to an
    ExtractedEntity — a registry entity if `canonicalize` recognizes it
    (cross-lingual/alias fusion), otherwise a novel entity keyed by a
    normalized slug. Returns None if the name is empty/unusable or the type
    cannot be placed in the schema."""
    name = (name or "").strip()
    if not name:
        return None

    known = canonicalize(name)
    if known is not None:
        return ExtractedEntity(id=known.canonical_id, name=known.name, type=known.type, canonical=True)

    if _is_junk(name):
        return None

    slug = _slugify(name)
    if not slug:
        return None

    entity_type = type_hint if type_hint in ENTITY_TYPES else None
    if entity_type is None:
        return None

    return ExtractedEntity(id=slug, name=name, type=entity_type, canonical=False)


# ── Dictionary pass ────────────────────────────────────────────────────────
def _dictionary_pass(text: str) -> list[ExtractedEntity]:
    return [
        ExtractedEntity(id=e.canonical_id, name=e.name, type=e.type, canonical=True)
        for e in find_entities_in_text(text)
    ]


# ── LLM pass ───────────────────────────────────────────────────────────────
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def _build_prompt(text: str, known_entities: list[ExtractedEntity]) -> tuple[str, str]:
    entity_types_str = ", ".join(ENTITY_TYPES)
    relation_lines = "\n".join(f"  - {name}: {desc}" for name, desc in RELATION_TYPES.items())
    known_str = (
        ", ".join(f'"{e.name}" ({e.type})' for e in known_entities)
        if known_entities else "(none)"
    )

    system = (
        "You extract entities and relations from scientific text for a knowledge graph. "
        "Respond with a single JSON object and nothing else — no prose, no markdown fences.\n\n"
        f"Allowed entity types: {entity_types_str}\n\n"
        f"Allowed relation types:\n{relation_lines}\n\n"
        "Output schema:\n"
        '{"entities": [{"name": str, "type": <one of the allowed entity types>}], '
        '"relations": [{"source": str, "target": str, "rel_type": <one of the allowed relation types>}]}\n\n'
        "Rules:\n"
        "- Only extract entities/relations explicitly supported by the passage.\n"
        "- `source` and `target` must be entity names that appear in your \"entities\" list "
        "or in the known-entities list below.\n"
        "- Do not invent relation types outside the allowed list.\n"
        "- If nothing applies, return {\"entities\": [], \"relations\": []}."
    )
    user = (
        f"Known entities already identified in this passage: {known_str}\n\n"
        f"Passage:\n{text}"
    )
    return system, user


def _get_client_and_model() -> tuple["OpenAI", str]:
    if OpenAI is None:
        raise RuntimeError("openai package not installed")
    if ENRICH_PROVIDER == "cloud":
        return (
            OpenAI(base_url=CLOUD_BASE_URL, api_key=os.getenv("OPENAI_API_KEY", "")),
            CLOUD_MODEL,
        )
    return OpenAI(base_url=f"{REASONING_URL}/v1",
                  api_key=os.getenv("REASONING_API_KEY", os.getenv("OPENAI_API_KEY", "none"))), REASONING_MODEL_NAME


def _parse_llm_json(content: str) -> Optional[EntityExtractionResponse]:
    content = _THINK_RE.sub("", content or "").strip()
    m = _FENCE_RE.search(content)
    if m:
        content = m.group(1).strip()
    if not content:
        return None
    if not content.startswith("{"):
        start, end = content.find("{"), content.rfind("}")
        if start == -1 or end == -1 or end < start:
            return None
        content = content[start:end + 1]
    try:
        return EntityExtractionResponse.model_validate(json.loads(content))
    except Exception:
        return None


def _rate_limit_delay(exc: "RateLimitError", attempt: int) -> float:
    """Honor the API's Retry-After header if present, else exponential backoff."""
    response = getattr(exc, "response", None)
    retry_after = response.headers.get("retry-after") if response is not None else None
    if retry_after is not None:
        try:
            return max(float(retry_after), 0.0)
        except ValueError:
            pass
    return RATE_LIMIT_BACKOFF_SECONDS * (2 ** attempt)


def _llm_pass(
    text: str,
    known_entities: list[ExtractedEntity],
    client: Optional["OpenAI"],
    model_name: Optional[str],
    reasoning_effort: str,
    enable_thinking: bool,
    temperature: float,
    max_tokens: int,
) -> Optional[dict]:
    """Returns the parsed {"entities": [...], "relations": [...]} dict, or
    None on any failure (caller falls back to dictionary-only results)."""
    if client is None:
        client, model_name = _get_client_and_model()
    if model_name is None:
        _, model_name = _get_client_and_model()

    system, user = _build_prompt(text, known_entities)
    kwargs = dict(
        model=model_name,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body={
            "reasoning_effort": reasoning_effort,
            "enable_thinking": enable_thinking,
            # qwen3.6 (vLLM) deployments ignore top-level enable_thinking and
            # only honor chat_template_kwargs — send both spellings.
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        },
    )

    for attempt in range(RATE_LIMIT_MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(**kwargs)
            break
        except RateLimitError as exc:
            if attempt >= RATE_LIMIT_MAX_RETRIES:
                logger.warning(
                    "Entity extraction: rate-limited %d times in a row, giving up on "
                    "this chunk (falling back to dictionary-only): %s",
                    attempt + 1, exc,
                )
                raise
            delay = _rate_limit_delay(exc, attempt)
            logger.warning(
                "Entity extraction: rate-limited by reasoning endpoint "
                "(attempt %d/%d), retrying in %.1fs",
                attempt + 1, RATE_LIMIT_MAX_RETRIES, delay,
            )
            time.sleep(delay)

    content = response.choices[0].message.content
    return _parse_llm_json(content)


# ── Fusion ─────────────────────────────────────────────────────────────────
def _fuse(
    dict_entities: list[ExtractedEntity],
    llm_data: Optional[EntityExtractionResponse],
) -> ExtractionResult:
    entities: dict[str, ExtractedEntity] = {e.id: e for e in dict_entities}
    relations: list[ExtractedRelation] = []

    if not llm_data:
        return ExtractionResult(entities=list(entities.values()), relations=relations)

    llm_types: dict[str, str] = {}
    for raw in llm_data.entities:
        if not isinstance(raw, dict):
            continue
        name = (raw.get("name") or "").strip()
        if name:
            llm_types.setdefault(normalize(name), raw.get("type"))
        ent = _resolve_entity(name, raw.get("type"))
        if ent is None:
            continue
        entities.setdefault(ent.id, ent)

    for raw in llm_data.relations:
        if not isinstance(raw, dict):
            continue
        desc = raw.get("description")
        if isinstance(desc, str):
            # open relation: the normalised free-text description is the label
            rel_type = " ".join(desc.split()).lower()[:80]
            if not rel_type:
                continue
        else:
            # legacy typed relation (per-domain vocabulary)
            rel_type = raw.get("rel_type")
            if rel_type not in RELATION_TYPES:
                continue
        src_name = raw.get("source", "")
        tgt_name = raw.get("target", "")
        src = _resolve_entity(src_name, llm_types.get(normalize(src_name)))
        tgt = _resolve_entity(tgt_name, llm_types.get(normalize(tgt_name)))
        if src is None or tgt is None or src.id == tgt.id:
            continue
        entities.setdefault(src.id, src)
        entities.setdefault(tgt.id, tgt)
        relations.append(ExtractedRelation(source=src.id, target=tgt.id, rel_type=rel_type))

    return ExtractionResult(entities=list(entities.values()), relations=relations)


# ── Public entry point ────────────────────────────────────────────────────
def extract_entities(
    text: str,
    *,
    use_llm: bool = True,
    client: Optional["OpenAI"] = None,
    model_name: Optional[str] = None,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    enable_thinking: bool = DEFAULT_ENABLE_THINKING,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    precomputed: Optional[dict] = None,
) -> ExtractionResult:
    """
    Extract entities and relations from one text chunk.

    Args:
        text:    the chunk's text (or enriched_context + text for figures/
                 equations — caller's choice of what to feed in).
        use_llm: set False for the "structure-reuse" ablation (dictionary
                 match only, no LLM call, no relations).
        client/model_name: inject an OpenAI-compatible client for testing or
                 to reuse a client across many calls. If omitted, one is
                 built from config (ENRICH_PROVIDER local/cloud switch).
        reasoning_effort/enable_thinking/temperature/max_tokens: passed
                 through to the LLM call — see module constants for defaults.
        precomputed: an already-obtained {"entities": [...], "relations":
                 [...]} dict (the combined batched pass in llm_enrich.py
                 stashes one per chunk as item['llm_extraction']). Used in
                 place of the per-chunk LLM call; the same _fuse validation
                 (registry resolution, slugging, schema checks) still applies.
                 Ignored when use_llm=False, so ablation D (dictionary-only,
                 zero relations) stays pure.

    Never raises: LLM failures degrade to dictionary-only results.
    """
    text = (text or "").strip()
    if not text:
        return ExtractionResult()

    dict_entities = _dictionary_pass(text)

    llm_data = None
    if use_llm and len(text) >= MIN_CHARS_FOR_LLM:
        if precomputed is not None:
            # llm_enrich stashes a plain dict; _fuse reads attributes
            llm_data = (EntityExtractionResponse(**precomputed)
                        if isinstance(precomputed, dict) else precomputed)
        else:
            try:
                llm_data = _llm_pass(
                    text, dict_entities, client, model_name,
                    reasoning_effort, enable_thinking, temperature, max_tokens,
                )
            except Exception as exc:
                logger.debug("Entity extraction LLM pass failed, falling back to dictionary-only: %s", exc)
                llm_data = None  # best-effort — never block ingestion

    return _fuse(dict_entities, llm_data)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Extract entities/relations from a text chunk")
    p.add_argument("text", help="Chunk text (or '-' to read from stdin)")
    p.add_argument("--no-llm", action="store_true", help="Dictionary match only")
    args = p.parse_args()

    chunk_text = sys.stdin.read() if args.text == "-" else args.text
    result = extract_entities(chunk_text, use_llm=not args.no_llm)
    print(json.dumps({
        "entities":  [e.__dict__ for e in result.entities],
        "relations": [r.__dict__ for r in result.relations],
    }, indent=2, ensure_ascii=False))
