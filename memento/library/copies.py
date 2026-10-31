"""Is a copy of the Memento pipes ready to work?

One Open WebUI can serve two libraries: the default copy (functions `memento_query` / `memento_ingestion`) and a second
copy (`memento_query_<name>` / `memento_ingestion_<name>`), the same code with different saved Valves. Open WebUI saves
only the Valves a form sets explicitly, and an unsaved Valve takes its default from config/.env, which names the default
library. So a second copy works only when its function row exists and its saved Valves hold every setting that decides
the library, the models and the answer-cache key ("configured"), and when its function is active ("enabled"). The
default ids are always configured and enabled, without reading anything.

A query copy also needs its linked ingestion copy (Valve INGESTION_FUNCTION_ID): Document Chat reads its settings.

It also rewrites reply links to the copy's own models: the pipes' fixed texts link Library Chat, Library Upload and
Document Chat by the default model ids, and a copy's output rewrites each link to the ids its Valves name
(rewriting twice changes nothing more).
"""

from __future__ import annotations

import json
from typing import NamedTuple

import re


DEFAULT_IDS = frozenset({"memento_query", "memento_ingestion"})

# The settings a second copy must have saved (exact Valve names). Query: the library, the answer model and its key, the
# warm-up's model server and slot rule, the reply links' model ids, the linked ingestion copy, the picker's model list,
# the deployment behaviour, and every Valve that feeds the answer-cache key (Open WebUI's process gets only part of
# config/.env, so an unsaved Valve could default differently there and in a script, and the two would build different
# keys). The Valves the answer-cache key is built from (library, model, answer prompt, prompt versions, thinking) are
# checked as well.
CACHE_KEY_VALVES = (
    "DOMAIN_NAME", "COLLECTION", "LLM_MODEL", "VARIANTS_FILE",
    "ENABLE_THINKING", "TEMPERATURE", "MAX_TOKENS", "SUMMARY_MAX_TOKENS", "GAP_BUDGET_TOKENS", "GAP_THINKING",
    "GAP_THINKING_BUDGET",
)
REQUIRED_QUERY_VALVES = (
    "LLM_BASE_URL", "LLM_API_KEY_ENV", "METRICS_URL", "WARMUP_MAX_RUNNING",
    "QUERY_MODEL_ID", "UPLOAD_MODEL_ID", "DOC_CHAT_MODEL_ID", "INGESTION_FUNCTION_ID",
    "THINKING_AUTO", "VALIDATION_ENABLED", "VALIDATION_STRATEGY", "VALIDATION_MAX_ROUNDS", "HYBRID_ENABLED",
    "GRAPH_RETRIEVAL",
) + CACHE_KEY_VALVES
# Ingestion: the library, the caption/tag model and its key, the reply links, who may upload, the parser, the
# research-gap tags, and the graph switch (off).
REQUIRED_INGESTION_VALVES = (
    "DOMAIN_NAME", "COLLECTION", "LLM_BASE_URL", "LLM_MODEL", "LLM_API_KEY_ENV",
    "QUERY_MODEL_ID", "UPLOAD_MODEL_ID", "DOC_CHAT_MODEL_ID",
    "CONTRIBUTOR_GROUP", "BACKEND", "RESEARCH_GAP_TAG", "GRAPH_BUILD",
)
_ROLES = {"query": REQUIRED_QUERY_VALVES, "ingestion": REQUIRED_INGESTION_VALVES}


class CopyState(NamedTuple):
    configured: bool
    enabled: bool
    reason: str = ""


def _is_missing(value) -> bool:
    """Absent, None, or text that is empty or only whitespace. False and 0 are real values."""
    return value is None or (isinstance(value, str) and not value.strip())


def missing(valves: dict, role: str) -> list[str]:
    """The required settings of `role` that `valves` lacks, in the list's order."""
    return [k for k in _ROLES[role] if _is_missing((valves or {}).get(k))]


def _decode(raw):
    """The saved Valves as a dict, or None when they cannot be read (e.g. encrypted Valves: a plain string)."""
    for _ in range(2):                      # a JSON text, possibly wrapped once more in JSON
        if isinstance(raw, dict):
            return raw
        if not isinstance(raw, str):
            return None
        try:
            raw = json.loads(raw)
        except ValueError:
            return None
    return raw if isinstance(raw, dict) else None


def read_function(engine, function_id: str):
    """(row exists, saved Valves dict or None, is_active) of OpenWebUI's `function` row."""
    from sqlalchemy import text
    with engine.connect() as c:
        row = c.execute(text("SELECT valves, is_active FROM function WHERE id = :id"), {"id": function_id}).first()
    if row is None:
        return False, None, False
    return True, _decode(row[0]), bool(row[1])


def copy_state(engine, function_id: str, role: str) -> CopyState:
    """(configured, enabled, reason) of one copy. Fails closed: anything unreadable is "not configured"."""
    if role not in _ROLES:
        raise ValueError(f"unknown role {role!r}")
    if function_id in DEFAULT_IDS:
        return CopyState(True, True, "")
    try:
        exists, valves, active = read_function(engine, function_id)
    except Exception as e:                  # the database cannot be read: never fall back to today's settings
        return CopyState(False, False, f"{function_id}: database not readable ({type(e).__name__})")
    if not exists:
        return CopyState(False, False, f"{function_id}: not installed")
    if valves is None:
        return CopyState(False, False, f"{function_id}: saved valves not readable")
    lacking = missing(valves, role)
    if lacking:
        return CopyState(False, False, f"{function_id}: settings not saved: {', '.join(lacking)}")
    configured, enabled, reasons = True, active, [] if active else [f"{function_id}: inactive"]
    if role == "query":
        linked = copy_state(engine, valves["INGESTION_FUNCTION_ID"].strip(), "ingestion")
        if not linked.configured:
            return CopyState(False, False, f"linked ingestion copy {linked.reason}")
        if not linked.enabled:
            enabled = False
            reasons.append(f"linked ingestion copy {linked.reason}")
    return CopyState(configured, enabled, "; ".join(reasons))


def find_copy(engine, collection: str, function_id: str | None = None) -> str | None:
    """The query copy that serves `collection` (its saved COLLECTION), for the command-line tools: `function_id` when
    given and it serves the collection, else the one copy among memento_query and memento_query_* whose saved Valves
    name it; None when none does. Several copies naming it, or a candidate's Valves that cannot be read (encrypted),
    raise ValueError: never a guess, never defaults."""
    from sqlalchemy import text
    with engine.connect() as c:
        rows = c.execute(text("SELECT id, valves FROM function")).all()
    if function_id is not None:
        rows = [r for r in rows if r[0] == function_id]
    else:
        rows = [r for r in rows if r[0] == "memento_query" or str(r[0]).startswith("memento_query_")]
    found = []
    for fid, raw in rows:
        valves = _decode(raw)
        if valves is None:
            raise ValueError(f"{fid}: saved valves not readable (encrypted?)")
        if valves.get("COLLECTION") == collection:
            found.append(fid)
    if len(found) > 1:
        raise ValueError(f"several copies serve {collection}: {', '.join(sorted(found))}")
    return found[0] if found else None


# ── Links to the copy's own models ────────────────────────────────────────────
TODAY_IDS = ("memento-query", "memento-ingestion-prod", "memento-document-chat")
_LINK_RE = re.compile(r"(/\?model=)(" + "|".join(re.escape(i) for i in TODAY_IDS) + r")(?=[&)\s\"'\]<>]|$)")


def link_ids(valves) -> tuple:
    """(Library Chat, Library Upload, Document Chat) model ids of a copy's Valves; today's for a missing field."""
    return tuple(getattr(valves, name, None) or default
                 for name, default in zip(("QUERY_MODEL_ID", "UPLOAD_MODEL_ID", "DOC_CHAT_MODEL_ID"), TODAY_IDS))


def relink(text, ids) -> str:
    """`text` with every link to the three default models pointing at `ids` instead (unchanged for the default ids)."""
    if not isinstance(text, str) or not ids or tuple(ids) == TODAY_IDS or "/?model=" not in text:
        return text
    to = dict(zip(TODAY_IDS, ids))
    return _LINK_RE.sub(lambda m: m.group(1) + to[m.group(2)], text)
