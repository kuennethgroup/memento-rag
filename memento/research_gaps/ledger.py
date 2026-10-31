# memento/research_gaps/ledger.py: the gap ledger's storage
"""Open research questions of one library, each with the documents and passages that state it, and the evidence of other
documents and the literature. Also what was not taken in, with the reason, so the extraction can be audited: statements
that are only a study's own scope (`scope_only`) and questions the judge did not find in their passage (`dropped`). One
status row per processed document (content fingerprint, counts) makes the build resumable. SQLAlchemy Core: Open WebUI's
Postgres live, SQLite in tests."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (Column, DateTime, Integer, MetaData, String, Table, Text, delete, func, insert, inspect,
                        select, text)

METADATA = MetaData()
GAPS = Table(
    "memento_gaps", METADATA,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("collection", String(200), nullable=False, index=True),
    Column("statement", Text, nullable=False),          # the open question, general form, English
    Column("kind", String(30), nullable=False),         # limitation | future_work | untested_condition | uncertainty
    Column("created_at", DateTime(timezone=True)),
    # step 2 on (library check) and step 3 (literature check)
    Column("library_status", String(20)),                # open | partly | addressed | contested
    Column("library_checked_at", DateTime(timezone=True)),
    Column("library_docs_checked", Integer),
    Column("web_status", String(30)),                    # nothing_found | partly | possibly_addressed | contested
    Column("web_checked_at", DateTime(timezone=True)),
    Column("web_query", Text),                           # the search phrase sent (step 3)
    Column("web_sources", String(100)),                  # the services searched, comma-separated
)
MENTIONS = Table(
    "memento_gap_mentions", METADATA,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("gap_id", Integer, nullable=False, index=True),
    Column("collection", String(200), nullable=False),
    Column("source", String(500), nullable=False, index=True),
    Column("point_id", String(64), nullable=False),      # the passage (Qdrant point) that states it
    Column("page", Integer),
    Column("signal", Text),                               # the ingestion tag's one-line statement
    Column("question", Text),                             # this mention's own question, before merging (audit)
    Column("claim", Text, nullable=False),                # what the passage says is open, as judged
    Column("passage", Text, nullable=False),              # the passage text as judged (the evidence)
    Column("judge", String(20), nullable=False),          # supported (only these are stored as mentions)
    Column("judge_reason", Text),
    Column("created_at", DateTime(timezone=True)),
)
SCOPE_ONLY = Table(
    "memento_gap_scope_only", METADATA,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("collection", String(200), nullable=False),
    Column("source", String(500), nullable=False, index=True),
    Column("point_id", String(64)),
    Column("page", Integer),
    Column("signal", Text),
    Column("claim", Text),
    Column("reason", Text),
    Column("created_at", DateTime(timezone=True)),
)
DROPPED = Table(
    "memento_gap_dropped", METADATA,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("collection", String(200), nullable=False),
    Column("source", String(500), nullable=False, index=True),
    Column("point_id", String(64)),
    Column("page", Integer),
    Column("question", Text),
    Column("claim", Text),
    Column("reason", Text),                               # the judge's label and rationale
    Column("created_at", DateTime(timezone=True)),
)
EVIDENCE = Table(                                         # filled from step 2 on
    "memento_gap_evidence", METADATA,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("gap_id", Integer, nullable=False, index=True),
    Column("collection", String(200), nullable=False),
    Column("origin", String(20), nullable=False),          # library | openalex | semantic_scholar
    Column("source", String(500)),                         # library document, or the outside work's id
    Column("point_id", String(64)),
    Column("page", Integer),
    Column("title", Text),
    Column("year", Integer),
    Column("doi", String(300)),
    Column("url", Text),                                    # outside works: the link shown
    Column("text", Text),                                   # outside works: the abstract the verdict was made from
    Column("relation", String(20), nullable=False),        # addresses | partly | disagrees | near_miss
    Column("reason", Text),
    Column("created_at", DateTime(timezone=True)),
)
RELATIONS = Table(                                        # related (not the same) open questions: a theme
    "memento_gap_relations", METADATA,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("collection", String(200), nullable=False, index=True),
    Column("a", Integer, nullable=False),                  # the smaller gap id
    Column("b", Integer, nullable=False),
    Column("created_at", DateTime(timezone=True)),
)
DOCUMENTS = Table(
    "memento_gap_documents", METADATA,
    Column("collection", String(200), primary_key=True),
    Column("source", String(500), primary_key=True),
    Column("content_hash", String(128)),
    Column("status", String(10), nullable=False),          # done | failed
    Column("n_questions", Integer, default=0),
    Column("n_scope", Integer, default=0),
    Column("n_dropped", Integer, default=0),
    Column("error", Text),
    Column("finished_at", DateTime(timezone=True)),
)
TABLES = [GAPS, MENTIONS, SCOPE_ONLY, DROPPED, EVIDENCE, RELATIONS, DOCUMENTS]


def _now():
    return datetime.now(timezone.utc)


# columns added after the tables were first created: added in place (name, table, SQL type)
_ADDED = [("question", MENTIONS, "TEXT"), ("web_query", GAPS, "TEXT"), ("web_sources", GAPS, "VARCHAR(100)"),
          ("url", EVIDENCE, "TEXT"), ("text", EVIDENCE, "TEXT")]


def ensure_tables(engine) -> None:
    METADATA.create_all(engine, tables=TABLES)
    for col, table, sql_type in _ADDED:
        if col not in {c["name"] for c in inspect(engine).get_columns(table.name)}:
            with engine.begin() as c:
                c.execute(text(f"ALTER TABLE {table.name} ADD COLUMN {col} {sql_type}"))


def add_gap(engine, collection: str, statement: str, kind: str) -> int:
    with engine.begin() as c:
        return c.execute(insert(GAPS).values(collection=collection, statement=statement, kind=kind,
                                             created_at=_now())).inserted_primary_key[0]


def add_mention(engine, gap_id: int, collection: str, *, source: str, point_id: str, page, signal: str, claim: str,
                passage: str, judge: str, judge_reason: str = "", question: str = "") -> None:
    with engine.begin() as c:
        c.execute(insert(MENTIONS).values(gap_id=gap_id, collection=collection, source=source, point_id=str(point_id),
                                          page=page, signal=signal, question=question or None, claim=claim,
                                          passage=passage, judge=judge, judge_reason=judge_reason, created_at=_now()))


def add_scope_only(engine, collection: str, source: str, *, point_id, page, signal: str, claim: str,
                   reason: str) -> None:
    with engine.begin() as c:
        c.execute(insert(SCOPE_ONLY).values(collection=collection, source=source, point_id=str(point_id), page=page,
                                            signal=signal, claim=claim, reason=reason, created_at=_now()))


def add_dropped(engine, collection: str, source: str, *, point_id, page, question: str, claim: str,
                reason: str) -> None:
    with engine.begin() as c:
        c.execute(insert(DROPPED).values(collection=collection, source=source, point_id=str(point_id), page=page,
                                         question=question, claim=claim, reason=reason, created_at=_now()))


def add_relation(engine, collection: str, a: int, b: int) -> None:
    """Link two RELATED (not the same) open questions; stored once per pair, whatever the direction."""
    a, b = sorted((int(a), int(b)))
    if a == b:
        return
    with engine.begin() as c:
        if c.execute(select(RELATIONS.c.id).where(RELATIONS.c.collection == collection, RELATIONS.c.a == a,
                                                  RELATIONS.c.b == b)).first() is None:
            c.execute(insert(RELATIONS).values(collection=collection, a=a, b=b, created_at=_now()))


def related(engine, collection: str, gap_id: int) -> list[int]:
    with engine.connect() as c:
        rows = c.execute(select(RELATIONS.c.a, RELATIONS.c.b).where(
            RELATIONS.c.collection == collection, (RELATIONS.c.a == gap_id) | (RELATIONS.c.b == gap_id))).all()
    return sorted({b if a == gap_id else a for a, b in rows})


def related_counts(engine, collection: str) -> dict:
    """{gap id: number of related questions}: a question linked to many is central to the library (report order)."""
    out: dict[int, int] = {}
    with engine.connect() as c:
        for x, y in c.execute(select(RELATIONS.c.a, RELATIONS.c.b).where(RELATIONS.c.collection == collection)):
            out[x] = out.get(x, 0) + 1
            out[y] = out.get(y, 0) + 1
    return out


def themes(engine, collection: str, within: set | None = None) -> list[list[int]]:
    """Groups of related open questions (connected through "related" links), largest first; single gaps excluded. Use
    `within` (the questions one report covers): over the whole library a few broad hub questions chain everything into a
    few huge groups, which is no theme."""
    with engine.connect() as c:
        rows = c.execute(select(RELATIONS.c.a, RELATIONS.c.b).where(RELATIONS.c.collection == collection)).all()
    if within is not None:
        rows = [(a, b) for a, b in rows if a in within and b in within]
    parent: dict[int, int] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in rows:
        parent[find(a)] = find(b)
    groups: dict[int, list[int]] = {}
    for x in list(parent):
        groups.setdefault(find(x), []).append(x)
    return sorted((sorted(g) for g in groups.values() if len(g) > 1), key=lambda g: (-len(g), g))


def set_library_status(engine, gap_id: int, *, status: str, docs_checked: int, evidence: list[dict]) -> None:
    """Store a library check: the status, when, against how many documents, and its evidence (replacing a previous
    check's library evidence). Evidence: dicts with source, point_id, page, relation, reason."""
    with engine.begin() as c:
        col = c.execute(select(GAPS.c.collection).where(GAPS.c.id == gap_id)).scalar()
        c.execute(delete(EVIDENCE).where(EVIDENCE.c.gap_id == gap_id, EVIDENCE.c.origin == "library"))
        for e in evidence:
            c.execute(insert(EVIDENCE).values(gap_id=gap_id, collection=col, origin="library", source=e["source"],
                                              point_id=str(e.get("point_id") or ""), page=e.get("page"),
                                              relation=e["relation"], reason=e.get("reason") or "",
                                              created_at=_now()))
        from sqlalchemy import update
        c.execute(update(GAPS).where(GAPS.c.id == gap_id).values(library_status=status, library_checked_at=_now(),
                                                                  library_docs_checked=docs_checked))


def add_library_evidence(engine, gap_id: int, *, status: str, new_docs: int, evidence: list[dict]) -> None:
    """A new document checked against a question already checked (step 5): its evidence added, the status updated,
    the documents checked counted up. A question no longer open loses its near-misses (they explained "open")."""
    from sqlalchemy import update
    with engine.begin() as c:
        g = c.execute(select(GAPS.c.collection, GAPS.c.library_docs_checked).where(GAPS.c.id == gap_id)).first()
        if g is None:
            return
        if status != "open":
            c.execute(delete(EVIDENCE).where(EVIDENCE.c.gap_id == gap_id, EVIDENCE.c.origin == "library",
                                             EVIDENCE.c.relation == "near_miss"))
        for e in evidence:
            c.execute(insert(EVIDENCE).values(gap_id=gap_id, collection=g[0], origin="library", source=e["source"],
                                              point_id=str(e.get("point_id") or ""), page=e.get("page"),
                                              relation=e["relation"], reason=e.get("reason") or "",
                                              created_at=_now()))
        c.execute(update(GAPS).where(GAPS.c.id == gap_id).values(
            library_status=status, library_checked_at=_now(), library_docs_checked=(g[1] or 0) + new_docs))


WEB_ORIGINS = ("openalex", "semantic_scholar", "alexandria")   # alexandria: an offline index of abstracts
                                                               # (memento/research_gaps/literature.py)


def set_web_status(engine, gap_id: int, *, status: str, query: str, sources: list[str], evidence: list[dict]) -> None:
    """Store a literature check: the status, the phrase searched, the services searched, and the papers that address,
    partly address or contradict the question (replacing a previous literature check's papers). Evidence: dicts with
    origin (the service), source (its work id), title, year, doi, url, relation, reason."""
    from sqlalchemy import update
    with engine.begin() as c:
        col = c.execute(select(GAPS.c.collection).where(GAPS.c.id == gap_id)).scalar()
        c.execute(delete(EVIDENCE).where(EVIDENCE.c.gap_id == gap_id, EVIDENCE.c.origin.in_(WEB_ORIGINS)))
        for e in evidence:
            c.execute(insert(EVIDENCE).values(gap_id=gap_id, collection=col, origin=e["origin"],
                                              source=e.get("source") or "", title=e.get("title"), year=e.get("year"),
                                              doi=e.get("doi") or None, url=e.get("url") or None,
                                              text=e.get("text") or None,
                                              relation=e["relation"], reason=e.get("reason") or "",
                                              created_at=_now()))
        c.execute(update(GAPS).where(GAPS.c.id == gap_id).values(
            web_status=status, web_checked_at=_now(), web_query=query, web_sources=",".join(sorted(set(sources)))))


def fill_web_text(engine, abstracts: dict) -> int:
    """Outside papers stored without their abstract: filled in by the service's work id from `abstracts` ({work id:
    abstract}). Returns the number of rows filled."""
    from sqlalchemy import update
    n = 0
    with engine.begin() as c:
        rows = c.execute(select(EVIDENCE.c.id, EVIDENCE.c.source).where(
            EVIDENCE.c.origin.in_(WEB_ORIGINS), EVIDENCE.c.text.is_(None))).all()
        for rid, src in rows:
            if abstracts.get(src):
                c.execute(update(EVIDENCE).where(EVIDENCE.c.id == rid).values(text=abstracts[src]))
                n += 1
    return n


def evidence(engine, gap_id: int, origin: str | None = None) -> list[dict]:
    with engine.connect() as c:
        q = select(EVIDENCE).where(EVIDENCE.c.gap_id == gap_id)
        if origin:
            q = q.where(EVIDENCE.c.origin == origin)
        return [dict(r) for r in c.execute(q.order_by(EVIDENCE.c.id)).mappings()]


def evidence_many(engine, gap_ids: list[int]) -> dict:
    """{gap id: its evidence rows} for many questions in one query; every id present, [] when it has none."""
    ids = list(dict.fromkeys(gap_ids))
    out: dict = {i: [] for i in ids}
    if not ids:
        return out
    with engine.connect() as c:
        for r in c.execute(select(EVIDENCE).where(EVIDENCE.c.gap_id.in_(ids)).order_by(EVIDENCE.c.id)).mappings():
            out[r["gap_id"]].append(dict(r))
    return out


def list_gaps(engine, collection: str) -> list[dict]:
    """Every gap of the library with its mentions; `documents` = the distinct stating documents, in order."""
    with engine.connect() as c:
        gaps = [dict(r) for r in c.execute(select(GAPS).where(GAPS.c.collection == collection)
                                           .order_by(GAPS.c.id)).mappings()]
        ms = [dict(r) for r in c.execute(select(MENTIONS).where(MENTIONS.c.collection == collection)
                                         .order_by(MENTIONS.c.id)).mappings()]
    by_gap: dict[int, list] = {}
    for m in ms:
        by_gap.setdefault(m["gap_id"], []).append(m)
    for g in gaps:
        g["mentions"] = by_gap.get(g["id"], [])
        g["documents"] = list(dict.fromkeys(m["source"] for m in g["mentions"]))
    return gaps


def get_gap(engine, gap_id: int) -> dict | None:
    with engine.connect() as c:
        row = c.execute(select(GAPS).where(GAPS.c.id == gap_id)).mappings().first()
    return dict(row) if row else None


def scope_only(engine, collection: str) -> list[dict]:
    with engine.connect() as c:
        return [dict(r) for r in c.execute(select(SCOPE_ONLY).where(SCOPE_ONLY.c.collection == collection)
                                           .order_by(SCOPE_ONLY.c.id)).mappings()]


def dropped(engine, collection: str) -> list[dict]:
    with engine.connect() as c:
        return [dict(r) for r in c.execute(select(DROPPED).where(DROPPED.c.collection == collection)
                                           .order_by(DROPPED.c.id)).mappings()]


def set_document(engine, collection: str, source: str, *, content_hash: str | None, status: str, n_questions: int = 0,
                 n_scope: int = 0, n_dropped: int = 0, error: str = "") -> None:
    row = dict(collection=collection, source=source, content_hash=content_hash, status=status,
               n_questions=n_questions, n_scope=n_scope, n_dropped=n_dropped, error=error or None,
               finished_at=_now())
    with engine.begin() as c:
        c.execute(delete(DOCUMENTS).where(DOCUMENTS.c.collection == collection, DOCUMENTS.c.source == source))
        c.execute(insert(DOCUMENTS).values(**row))


def processed_sources(engine, collection: str) -> set[str]:
    """The documents the ledger has processed (status done): the ones every existing question has been checked
    against or counted for."""
    with engine.connect() as c:
        return {r[0] for r in c.execute(select(DOCUMENTS.c.source).where(DOCUMENTS.c.collection == collection,
                                                                          DOCUMENTS.c.status == "done"))}


def get_document(engine, collection: str, source: str) -> dict | None:
    with engine.connect() as c:
        row = c.execute(select(DOCUMENTS).where(DOCUMENTS.c.collection == collection,
                                                DOCUMENTS.c.source == source)).mappings().first()
    return dict(row) if row else None


def delete_for_source(engine, collection: str, source: str) -> list[int]:
    """Remove one document's mentions, scope-only and dropped rows, status and evidence; return the ids of gaps that
    no document states any more (deleted here; their vectors in the companion collection must go too)."""
    from sqlalchemy import update
    with engine.begin() as c:
        touched = {r[0] for r in c.execute(select(MENTIONS.c.gap_id).where(MENTIONS.c.collection == collection,
                                                                            MENTIONS.c.source == source))}
        # counted among the documents checked only once the ledger had processed it (a new document is cleared before
        # its first extraction: nothing to count off)
        was_counted = c.execute(select(DOCUMENTS.c.source).where(DOCUMENTS.c.collection == collection,
                                                                  DOCUMENTS.c.source == source)).first() is not None
        # questions whose library check used this document's passages: their status is taken back below
        evidenced = {r[0] for r in c.execute(select(EVIDENCE.c.gap_id).where(
            EVIDENCE.c.collection == collection, EVIDENCE.c.origin == "library", EVIDENCE.c.source == source,
            EVIDENCE.c.relation != "near_miss"))}
        c.execute(delete(MENTIONS).where(MENTIONS.c.collection == collection, MENTIONS.c.source == source))
        c.execute(delete(SCOPE_ONLY).where(SCOPE_ONLY.c.collection == collection, SCOPE_ONLY.c.source == source))
        c.execute(delete(DROPPED).where(DROPPED.c.collection == collection, DROPPED.c.source == source))
        c.execute(delete(DOCUMENTS).where(DOCUMENTS.c.collection == collection, DOCUMENTS.c.source == source))
        c.execute(delete(EVIDENCE).where(EVIDENCE.c.collection == collection, EVIDENCE.c.source == source))
        orphans = []
        for gid in sorted(touched):
            left = c.execute(select(func.count()).select_from(MENTIONS).where(MENTIONS.c.gap_id == gid)).scalar()
            if not left:
                orphans.append(gid)
        # every checked question this document did not raise counted it among the documents checked: one fewer
        if was_counted:
            c.execute(update(GAPS).where(GAPS.c.collection == collection, GAPS.c.library_checked_at.is_not(None),
                                         GAPS.c.library_docs_checked > 0, GAPS.c.id.not_in(touched or {-1}))
                      .values(library_docs_checked=GAPS.c.library_docs_checked - 1))
        # the status of a question its passages supported follows from the evidence that is left (same rules)
        for gid in sorted(evidenced - set(orphans)):
            rels = {r[0] for r in c.execute(select(EVIDENCE.c.relation).where(
                EVIDENCE.c.gap_id == gid, EVIDENCE.c.origin == "library", EVIDENCE.c.relation != "near_miss"))}
            st = ("contested" if "contradicts" in rels else "addressed" if "addresses" in rels
                  else "partly" if "partly" in rels else "open")
            c.execute(update(GAPS).where(GAPS.c.id == gid).values(library_status=st))
        if orphans:
            c.execute(delete(EVIDENCE).where(EVIDENCE.c.gap_id.in_(orphans)))
            c.execute(delete(RELATIONS).where(RELATIONS.c.a.in_(orphans) | RELATIONS.c.b.in_(orphans)))
            c.execute(delete(GAPS).where(GAPS.c.id.in_(orphans)))
    return orphans


def ledger_documents(engine, collection: str) -> list[dict]:
    with engine.connect() as c:
        return [dict(r) for r in c.execute(select(DOCUMENTS).where(DOCUMENTS.c.collection == collection)).mappings()]


def pending_documents(library: list[dict], ledger: list[dict]) -> dict:
    """What the updater has to do (step 5): documents of the library that are new or changed since the ledger read
    them (a document that failed with the same content is not retried by the updater), and documents the ledger still
    has that the library no longer has."""
    seen = {d["source"]: d for d in ledger}
    new = [d["source"] for d in library
           if d["source"] not in seen or seen[d["source"]].get("content_hash") != d.get("content_hash")]
    lib = {d["source"] for d in library}
    return {"new": new, "removed": [s for s in seen if s not in lib]}


def clear_library(engine, collection: str) -> int:
    """Everything of one library (a library reset); returns the number of gaps removed."""
    with engine.begin() as c:
        n = c.execute(delete(GAPS).where(GAPS.c.collection == collection)).rowcount
        for t in (MENTIONS, SCOPE_ONLY, DROPPED, EVIDENCE, RELATIONS, DOCUMENTS):
            c.execute(delete(t).where(t.c.collection == collection))
    return n
