# memento/library/answers.py: stored answers for the standard per-document requests
"""Summaries and research gaps of one document, stored so a repeat of the same request is answered at once. The key
holds everything that changes the answer: the document and its content fingerprint, the action, the language, the model
(dropdown entry and actual model name), the mode and the prompt version. SQLAlchemy Core: Open WebUI's Postgres live,
SQLite in tests."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

from sqlalchemy import (Boolean, Column, DateTime, Integer, MetaData, String, Table, Text, delete, insert, inspect,
                        select, text, update)

METADATA = MetaData()
ANSWERS = Table(
    "memento_answer_cache", METADATA,
    Column("key", String(64), primary_key=True),
    Column("collection", String(200), nullable=False),
    Column("source", String(500), nullable=False),
    Column("content_hash", String(128), nullable=False),
    Column("action", String(20), nullable=False),
    Column("language", String(10), nullable=False),
    Column("model", String(200), nullable=False),
    Column("mode", String(20), nullable=False),
    Column("prompt_version", String(32), nullable=False),
    Column("generated_by", String(200), nullable=False),
    Column("answer", Text, nullable=False),
    Column("citations", Text, nullable=False),
    Column("status", String(10), nullable=False, default="ready"),
    Column("created_at", DateTime(timezone=True)),
    Column("hits", Integer, nullable=False, default=0),
    Column("last_hit_at", DateTime(timezone=True)),
    # the answer went through "Thorough check" to the end: the warm-up writes checked answers, and the switch replays
    # only a checked one
    Column("checked", Boolean, nullable=False, default=False),
    # which check a checked answer passed: "checked" counts only under the current contract
    Column("check_contract", String(32), nullable=True),
)
# the current meaning of "checked": at least one statement extracted, every statement judged and none still failing
# (verdicts and scientific checks).
# Answers checked under an older meaning replay as unchecked; change it when the meaning changes again.
CHECK_CONTRACT = "2026-10-31"
KEY_FIELDS = ("collection", "source", "content_hash", "action", "language", "model", "mode", "prompt_version")


_ADDED_COLUMNS = (("checked", "BOOLEAN NOT NULL DEFAULT FALSE"),     # existing answers count as unchecked
                  ("check_contract", "VARCHAR(32)"))                    # existing ones have none


def _columns(engine) -> set:
    return {c["name"] for c in inspect(engine).get_columns(ANSWERS.name)}


def _alter(engine, sql: str) -> None:
    with engine.begin() as c:
        c.execute(text(sql))


def ensure_table(engine) -> None:
    """The table, with the columns added since it was first created, in place. Safe when two first callers race:
    PostgreSQL adds a column only if it is missing; SQLite's "duplicate column" from the slower one is ignored (any
    other error is raised)."""
    METADATA.create_all(engine, tables=[ANSWERS])
    have = _columns(engine)
    for name, ddl in _ADDED_COLUMNS:
        if name in have:
            continue
        if engine.dialect.name == "postgresql":
            _alter(engine, f"ALTER TABLE {ANSWERS.name} ADD COLUMN IF NOT EXISTS {name} {ddl}")
            continue
        try:
            _alter(engine, f"ALTER TABLE {ANSWERS.name} ADD COLUMN {name} {ddl}")
        except Exception as e:
            if "duplicate column" not in str(e).lower():
                raise


def _effective(checked, contract) -> bool:
    """Checked under the current meaning of "checked" (CHECK_CONTRACT)."""
    return bool(checked) and contract == CHECK_CONTRACT


def cache_key(**fields) -> str:
    return hashlib.sha256("\x1f".join(str(fields[f]) for f in KEY_FIELDS).encode()).hexdigest()


def prompt_version(*parts) -> str:
    return hashlib.sha256("\x1f".join(str(p) for p in parts).encode()).hexdigest()[:16]


def get(engine, key: str, count_hit: bool = True) -> dict | None:
    """The stored answer ({answer, citations, generated_by, hits, …}) when ready, else None; counts the hit unless
    `count_hit` is False (the pre-warm status reads without changing the statistics)."""
    with engine.begin() as c:
        row = c.execute(select(ANSWERS).where(ANSWERS.c.key == key, ANSWERS.c.status == "ready")).mappings().first()
        if row is None:
            return None
        if count_hit:
            c.execute(update(ANSWERS).where(ANSWERS.c.key == key)
                      .values(hits=ANSWERS.c.hits + 1, last_hit_at=datetime.now(timezone.utc)))
    out = dict(row)
    out["citations"] = json.loads(out["citations"])
    out["checked"] = bool(out.get("checked"))
    out["effective_checked"] = _effective(out["checked"], out.get("check_contract"))
    out["hits"] += 1 if count_hit else 0
    return out


def put(engine, key: str, *, fields: dict, answer: str, citations: list, generated_by: str,
        replace: bool = True, checked: bool = False, keep_if=None) -> str:
    """Store (or replace) the answer under `key`. replace=False (the upload warm-up) keeps an answer already stored (a
    live answer or a Regenerate may have come first), except that a checked answer replaces an unchecked one, and only
    while that answer is still usable: ready and accepted by `keep_if` (the replay check of its kind); one no click
    would replay is replaced. Returns what happened, so a warm-up never counts an answer that was not stored: "written",
    "kept_existing", "document_gone" or "document_changed" (nothing stored)."""
    row = {f: fields[f] for f in KEY_FIELDS}
    row.update(key=key, answer=answer, citations=json.dumps(citations, ensure_ascii=False), status="ready",
               generated_by=generated_by, created_at=datetime.now(timezone.utc), hits=0, last_hit_at=None,
               checked=bool(checked), check_contract=CHECK_CONTRACT if checked else None)
    from memento.library import locks as db_locks
    library = inspect(engine).has_table("memento_documents")          # none in a test or before the first setup
    with engine.begin() as c:
        # the document's lock, then its library row: an answer is stored only for the version in the library, and a
        # removal cannot come between the check and the write
        db_locks.document(c, fields["collection"], fields["source"])
        if library:
            have = c.execute(text("SELECT content_hash FROM memento_documents WHERE collection = :c AND source = :s"),
                             {"c": fields["collection"], "s": fields["source"]}).first()
            if have is None:
                return "document_gone"
            if have[0] and have[0] != fields["content_hash"]:
                return "document_changed"
        old = c.execute(select(ANSWERS.c.checked, ANSWERS.c.check_contract, ANSWERS.c.status, ANSWERS.c.answer)
                        .where(ANSWERS.c.key == key)).first()
        usable = old is not None and old[2] == "ready" and (keep_if is None or keep_if(old[3]))
        # a checked answer under the current contract stays; an older "checked" one is upgraded like an unchecked one
        if not replace and usable and (_effective(old[0], old[1]) or not checked):
            return "kept_existing"
        c.execute(delete(ANSWERS).where(ANSWERS.c.key == key))
        c.execute(insert(ANSWERS).values(**row))
    return "written"


def delete_for_source(engine, collection: str, source: str, conn=None) -> int:
    """Remove every stored answer of one document (it was deleted or reset); returns how many. `conn`: inside the
    caller's transaction."""
    if conn is not None:
        return conn.execute(delete(ANSWERS).where(ANSWERS.c.collection == collection,
                                                  ANSWERS.c.source == source)).rowcount
    with engine.begin() as c:
        return c.execute(delete(ANSWERS).where(ANSWERS.c.collection == collection,
                                               ANSWERS.c.source == source)).rowcount
