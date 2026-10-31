# The upload warm-up's job queue.
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from sqlalchemy import create_engine, update
from sqlalchemy.pool import StaticPool
from memento.config.paths import PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT))
from memento.library import documents as db  # noqa: E402
from memento.library import warmup_queue as wq  # noqa: E402


@pytest.fixture
def engine():
    e = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    db.ensure_tables(e)
    return e


def test_enqueue_and_claim_in_order(engine):
    wq.enqueue(engine, "col", "a", ["summary", "gaps"])
    wq.enqueue(engine, "col", "b", ["summary"])
    got = [wq.claim_next(engine) for _ in range(4)]
    assert [(j["source"], j["action"]) for j in got[:3]] == [("a", "summary"), ("a", "gaps"), ("b", "summary")]
    assert got[3] is None and all(j["attempts"] == 1 for j in got[:3])


def test_finish_and_re_enqueue(engine):
    wq.enqueue(engine, "col", "a", ["summary"])
    job = wq.claim_next(engine)
    wq.finish(engine, job, ok=True)
    assert wq.claim_next(engine) is None
    wq.enqueue(engine, "col", "a", ["summary"])                    # uploaded again: queued afresh
    assert wq.claim_next(engine)["source"] == "a"


def test_stale_running_reclaimed(engine):
    wq.enqueue(engine, "col", "a", ["summary"])
    wq.claim_next(engine)                                          # running, then the process died
    assert wq.claim_next(engine) is None                           # not stale yet
    with engine.begin() as c:
        c.execute(update(wq.JOBS).values(started_at=datetime.now(timezone.utc) - timedelta(minutes=11)))
    job = wq.claim_next(engine)
    assert job["source"] == "a" and job["attempts"] == 2


def test_attempts_cap(engine):
    wq.enqueue(engine, "col", "a", ["summary"])
    for _ in range(3):
        wq.finish(engine, wq.claim_next(engine), ok=False, error="boom")
    assert wq.claim_next(engine) is None
    with engine.connect() as c:
        row = c.execute(wq.JOBS.select()).mappings().one()
    assert row["status"] == "failed" and row["attempts"] == 3 and row["last_error"] == "boom"


def test_claim_is_exclusive(engine):
    wq.enqueue(engine, "col", "a", ["summary"])
    first, second = wq.claim_next(engine), wq.claim_next(engine)
    assert first is not None and second is None


def test_delete_for_source_removes_jobs(engine, tmp_path):
    db.add_document(engine, collection="col", entry={"source": "a", "title": "a"}, content_hash="h", uploaded_by="u",
                    output_root=None)
    wq.enqueue(engine, "col", "a", ["summary", "gaps"])
    wq.enqueue(engine, "col", "b", ["summary"])
    db.remove_document(engine, collection="col", source="a", output_root=None)
    assert [wq.claim_next(engine)["source"], wq.claim_next(engine)] == ["b", None]
