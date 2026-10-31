# memento/library/warmup_queue.py: jobs for the upload warm-up of stored answers
"""After every stored upload, the document's summary and research gaps are written in the background while the local
model is idle, so the first person asking gets them at once. One row per collection, document and action. Questions
never wait for this queue: a job still running, crashed or failed means the question is answered and stored the normal
way. A job running longer than `stale_after_s` (a crash or a restart) is claimed again; after `max_attempts` it is
`failed`."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import Column, DateTime, Integer, MetaData, String, Table, Text, and_, delete, insert, or_, select, update

METADATA = MetaData()
JOBS = Table(
    "memento_warmup_queue", METADATA,
    Column("collection", String(200), primary_key=True),
    Column("source", String(500), primary_key=True),
    Column("action", String(20), primary_key=True),
    Column("status", String(10), nullable=False),          # queued | running | done | failed | ineligible | skipped
    Column("enqueued_at", DateTime(timezone=True), nullable=False),
    Column("started_at", DateTime(timezone=True)),
    Column("attempts", Integer, nullable=False, default=0),
    Column("last_error", Text),
    Column("cache_key", String(64)),                        # the answer a done job stored (a record)
)


def _now():
    return datetime.now(timezone.utc)


def ensure_table(engine) -> None:
    """The table, and the cache_key column on an older table."""
    from sqlalchemy import inspect, text
    METADATA.create_all(engine, tables=[JOBS])
    if "cache_key" not in {c["name"] for c in inspect(engine).get_columns(JOBS.name)}:
        with engine.begin() as c:
            c.execute(text(f"ALTER TABLE {JOBS.name} ADD COLUMN cache_key VARCHAR(64)"))


def _where(collection, source, action):
    return and_(JOBS.c.collection == collection, JOBS.c.source == source, JOBS.c.action == action)


def enqueue(engine, collection: str, source: str, actions) -> None:
    """Queue (again) each action for this document; an upload of a changed file queues it afresh."""
    with engine.begin() as c:
        for action in actions:
            c.execute(delete(JOBS).where(_where(collection, source, action)))
            c.execute(insert(JOBS).values(collection=collection, source=source, action=action, status="queued",
                                          enqueued_at=_now(), started_at=None, attempts=0, last_error=None))


def claim_next(engine, stale_after_s: int = 600, max_attempts: int = 3, collection: str | None = None) -> dict | None:
    """The next job, marked running: the oldest queued one, or one left running longer than `stale_after_s`. A job that
    already used `max_attempts` is marked failed instead. Claiming is conditional on the row being unchanged, so two
    workers never run the same job. `collection`: only that library's jobs (each copy's worker its own); None: any."""
    cutoff = _now() - timedelta(seconds=stale_after_s)
    while True:
        with engine.begin() as c:
            due = or_(JOBS.c.status == "queued", and_(JOBS.c.status == "running", JOBS.c.started_at < cutoff))
            if collection is not None:
                due = and_(due, JOBS.c.collection == collection)
            row = c.execute(select(JOBS).where(due)
                            .order_by(JOBS.c.enqueued_at, JOBS.c.action.desc()).limit(1)).mappings().first()
            if row is None:
                return None
            same = and_(_where(row["collection"], row["source"], row["action"]), JOBS.c.status == row["status"],
                        JOBS.c.attempts == row["attempts"])
            if row["attempts"] >= max_attempts:
                c.execute(update(JOBS).where(same).values(status="failed"))
                continue
            claimed = c.execute(update(JOBS).where(same).values(status="running", started_at=_now(),
                                                                attempts=row["attempts"] + 1)).rowcount
        if claimed:
            return {**dict(row), "status": "running", "attempts": row["attempts"] + 1}


def finish(engine, job: dict, ok: bool, error: str = "", max_attempts: int = 3, *, outcome: str | None = None,
           cache_key: str | None = None) -> None:
    """Done, or back to queued for another attempt (failed once the attempts are used up). `outcome`: "done" with the
    stored answer's `cache_key`, "ineligible" (an answer that may not be stored: final, never retried) or "skipped"
    (another library's job, or the document is gone)."""
    status = outcome or ("done" if ok else ("failed" if job["attempts"] >= max_attempts else "queued"))
    with engine.begin() as c:
        c.execute(update(JOBS).where(_where(job["collection"], job["source"], job["action"]))
                  .values(status=status, last_error=None if status == "done" else (error or "")[:500],
                          cache_key=cache_key if status == "done" else None))


def delete_for_source(engine, collection: str, source: str, conn=None) -> int:
    """A document's jobs out; `conn`: inside the caller's transaction (a removal is one transaction)."""
    if conn is not None:
        return conn.execute(delete(JOBS).where(JOBS.c.collection == collection, JOBS.c.source == source)).rowcount
    with engine.begin() as c:
        return c.execute(delete(JOBS).where(JOBS.c.collection == collection, JOBS.c.source == source)).rowcount
