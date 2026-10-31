# The two-library lock rules on PostgreSQL. A second copy's Document Chat reads and its cleaner hold the library's
# lifecycle lock shared; removal (archive) and restore take it exclusive, waiting for every holder, and a holder that
# waited rechecks that the copy is still enabled. The default library takes no lock. Concurrent first files of one chat
# in two libraries are serialised by a per-chat lock.
import asyncio
import logging
import sys
import threading
import time
import types
from pathlib import Path

import pytest
from sqlalchemy import text
from memento.config.paths import PROJECT_ROOT
from memento.query import library_state
from memento.document_chat import chat, cleanup

ROOT = PROJECT_ROOT
sys.path.insert(0, str(ROOT))
from _pg_scratch import scratch_engine  # noqa: E402
from memento.library import copies as cs  # noqa: E402
from memento.document_chat import store as td  # noqa: E402

COMP, POLY = td.DEFAULT_TEMP_COLLECTION, "memento-polymer-temp"
CHAT, CHAT2 = "3f2b8c1e-6a4d-4e0b-9c7a-1d2e3f4a5b6c", "9a8b7c6d-5e4f-4a3b-8c2d-1e0f9a8b7c6d"


@pytest.fixture(scope="module")
def pg():
    yield from scratch_engine()


@pytest.fixture
def db(pg):
    td.ensure_tables(pg)
    with pg.begin() as c:
        c.execute(text("DELETE FROM memento_temp_documents"))
        c.execute(text("DELETE FROM memento_temp_documents_study"))
        c.execute(text("CREATE TABLE IF NOT EXISTS chat (id TEXT PRIMARY KEY)"))
        c.execute(text("DELETE FROM chat"))
    return pg


def _advisory(db, granted=None) -> list:
    with db.connect() as c:
        q = "SELECT mode, granted FROM pg_locks WHERE locktype = 'advisory' AND database = " \
            "(SELECT oid FROM pg_database WHERE datname = current_database())"
        rows = c.execute(text(q)).all()
    return [r for r in rows if granted is None or r[1] is granted]


def _wait_for(cond, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.05)
    return False


def _claim(db, coll, chat=CHAT, fid="f1", status="ready"):
    src = td.claim(db, chat_id=chat, user_id="u1", file_id=fid, name=f"{fid}.pdf", collection=coll)
    assert isinstance(src, str)
    td.set_status(db, chat, fid, status, pages=2)
    return src


def _holder(db, coll):
    """A thread holding the shared lock until released."""
    held, release = threading.Event(), threading.Event()

    def run():
        with td.lifecycle(db, coll):
            held.set()
            release.wait(20)
    t = threading.Thread(target=run, daemon=True)
    t.start()
    assert held.wait(5)
    return t, release


# ── the lock itself ──────────────────────────────────────────────────────────

def test_a_shared_holder_makes_removal_wait_and_time_out(db):
    _claim(db, POLY)
    t, release = _holder(db, POLY)
    t0 = time.monotonic()
    with pytest.raises(td.LockTimeout):
        td.archive(db, POLY, timeout_s=1)
    assert 0.8 < time.monotonic() - t0 < 5
    assert [r["file_id"] for r in td.chat_documents(db, CHAT, "u1", collection=POLY)] == ["f1"]   # nothing moved
    release.set()
    t.join(5)
    assert td.archive(db, POLY, timeout_s=5) == {"moved": 1, "identical": 0}


def test_a_waiting_removal_lets_no_new_read_start_before_it(db):
    _claim(db, POLY)
    t, release = _holder(db, POLY)
    done = {}

    def remove():
        td.archive(db, POLY, timeout_s=15)
        done["archive"] = time.monotonic()

    def read():
        with td.lifecycle(db, POLY):
            done["read"] = time.monotonic()
    r = threading.Thread(target=remove, daemon=True)
    r.start()
    assert _wait_for(lambda: _advisory(db, granted=False))                    # removal is queued
    n = threading.Thread(target=read, daemon=True)
    n.start()
    time.sleep(0.5)
    assert "read" not in done                                                # the new read waits behind removal
    release.set()
    t.join(5), r.join(10), n.join(10)
    assert done["archive"] <= done["read"]


def test_the_shared_lock_is_released_when_the_holder_fails(db):
    with pytest.raises(RuntimeError):
        with td.lifecycle(db, POLY):
            raise RuntimeError("read failed")
    assert _advisory(db) == []                                               # not left on a pooled connection
    assert td.archive(db, POLY, timeout_s=1) == {"moved": 0, "identical": 0}


def test_todays_library_takes_no_lock(db):
    with td.lifecycle(db, COMP):
        assert _advisory(db) == []


def test_concurrent_first_files_in_two_libraries_leave_the_chat_in_one(db):
    for i in range(20):
        chat = f"{i:08d}-6a4d-4e0b-9c7a-1d2e3f4a5b6c"
        start, out = threading.Barrier(2), {}

        def go(coll, fid):
            start.wait(5)
            out[coll] = td.claim(db, chat_id=chat, user_id="u1", file_id=fid, name="x.pdf", collection=coll)
        ts = [threading.Thread(target=go, args=a) for a in ((COMP, "f1"), (POLY, "f2"))]
        [t.start() for t in ts]
        [t.join(10) for t in ts]
        assert sorted(isinstance(v, str) for v in out.values()) == [False, True]
        assert [v for v in out.values() if not isinstance(v, str)] == [td.OTHER_LIBRARY]
        assert len(td.chat_collections(db, chat)) == 1


# ── the second copy's read and cleaner against removal ──

@pytest.fixture
def study(db, monkeypatch, tmp_path):
    import memento.document_chat.intake as ti
    import qdrant_client
    from test_two_libraries_worker import QUERY, load_copy
    mod = load_copy(QUERY, "memento_query_polymer", monkeypatch)
    monkeypatch.setattr(mod, "_db_engine", lambda: db)
    state = {"fresh": True}
    monkeypatch.setattr(library_state, "copy_state", lambda role="query", *, cached=True, function_id=None, db_engine=None:
                        cs.CopyState(True, state["fresh"], ""))
    gate = {"record": None, "read": None, "entered": threading.Event()}

    async def record(fid):
        if gate["record"] is not None:
            gate["entered"].set()
            await asyncio.to_thread(gate["record"].wait, 20)
        return types.SimpleNamespace(user_id="u1", path=f"/uploads/{fid}.pdf")
    monkeypatch.setattr(chat, "file_record", record)

    def read_one(ti_, ing, path, src, max_pages, progress=None):
        if gate["read"] is not None:
            gate["entered"].set()
            gate["read"].wait(20)
        return {"ok": True, "pages": 4, "vectors": 3}
    monkeypatch.setattr(chat, "doc_chat_read_one", read_one)
    monkeypatch.setattr(library_state, "saved_valves", lambda *a, **k: {})
    monkeypatch.setattr(ti, "make_ingester", lambda *a, **k: types.SimpleNamespace())
    monkeypatch.setattr(qdrant_client, "QdrantClient", lambda *a, **k: SlowQdrant())
    monkeypatch.setattr(cleanup, "remove_owui_upload", lambda *a, **k: True)
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "out"))
    monkeypatch.setenv("INPUT_ROOT", str(tmp_path / "in"))
    pipe = mod.Pipe()
    pipe.valves = pipe.Valves(DOMAIN_NAME="polymer-engineering", COLLECTION="memento-polymer-rag",
                              QDRANT_URL="http://127.0.0.1:9")
    return types.SimpleNamespace(mod=mod, pipe=pipe, state=state, gate=gate, db=db)


class SlowQdrant:
    """Point deletes that can be held up (a cleaner that already read its rows)."""
    hold: threading.Event | None = None
    entered = threading.Event()
    deleted: list = []

    def delete(self, collection_name, points_selector):
        SlowQdrant.entered.set()
        if SlowQdrant.hold is not None:
            SlowQdrant.hold.wait(20)
        SlowQdrant.deleted.append(points_selector.filter.must[0].match.value)


def _read_in_thread(s, files):
    out = {}

    async def status(*a, **k):
        return None

    def run():
        out["result"] = asyncio.run(chat.doc_chat_read(s.pipe.valves, s.db, td, CHAT, "u1", files, status, logging.getLogger("t"),
                                                       function_id=s.mod._OWN_ID, db_engine=s.mod._db_engine,
                                                       saved_valves=lambda fid: library_state.saved_valves(fid, s.mod._db_engine)))
    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t, out


def _archive_in_thread(db, **kw):
    out = {}

    def run():
        try:
            out["result"] = td.archive(db, POLY, timeout_s=15, **kw)
        except Exception as e:          # noqa: BLE001
            out["error"] = e
        out["at"] = time.monotonic()
    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t, out


def test_removal_waits_for_a_cleaner_that_already_read_its_rows(study):
    db = study.db
    _claim(db, POLY, chat=CHAT, fid="f1")                  # chat deleted → the cleaner removes it
    _claim(db, POLY, chat=CHAT2, fid="f2")                 # chat still there → archived
    with db.begin() as c:
        c.execute(text("INSERT INTO chat (id) VALUES (:i)"), {"i": CHAT2})
    SlowQdrant.hold, SlowQdrant.deleted = threading.Event(), []
    SlowQdrant.entered.clear()
    cleaner = threading.Thread(target=cleanup.doc_chat_cleanup, args=(study.pipe.valves, db), daemon=True,
                               kwargs=dict(function_id=study.pipe.function_id, db_engine=study.pipe.engine))
    cleaner.start()
    assert SlowQdrant.entered.wait(5)                       # mid-run, its snapshot read
    a, out = _archive_in_thread(db)
    time.sleep(0.7)
    assert "at" not in out                                  # removal waits
    SlowQdrant.hold.set()
    cleaner.join(10), a.join(10)
    SlowQdrant.hold = None
    assert out.get("result") == {"moved": 1, "identical": 0} and SlowQdrant.deleted == [CHAT]
    with db.connect() as c:
        live = c.execute(text("SELECT file_id FROM memento_temp_documents")).all()
        arch = c.execute(text("SELECT file_id, status FROM memento_temp_documents_study")).all()
    assert live == [] and arch == [("f2", "ready")]


def test_a_cleaner_waiting_behind_removal_finds_the_copy_off_and_does_nothing(study, monkeypatch):
    db = study.db
    _claim(db, POLY, chat=CHAT, fid="f1")
    calls = []
    monkeypatch.setattr(td, "cleanup", lambda *a, **k: calls.append(a) or 0)
    study.state["fresh"] = False                            # removal switched the copy off first
    holding, release = threading.Event(), threading.Event()
    a, out = _archive_in_thread(db, before_move=lambda rows: (holding.set(), release.wait(20)))
    assert holding.wait(5)
    cleaner = threading.Thread(target=cleanup.doc_chat_cleanup, args=(study.pipe.valves, db), daemon=True,
                               kwargs=dict(function_id=study.pipe.function_id, db_engine=study.pipe.engine))
    cleaner.start()
    time.sleep(0.5)
    release.set()
    a.join(10), cleaner.join(10)
    assert out.get("result") == {"moved": 1, "identical": 0} and calls == []


def test_a_read_admitted_before_removal_waits_then_stops_without_a_row(study):
    db = study.db
    holding, release = threading.Event(), threading.Event()
    study.state["fresh"] = False                            # removal switched the copy off, then took the lock
    a, out = _archive_in_thread(db, before_move=lambda rows: (holding.set(), release.wait(20)))
    assert holding.wait(5)
    r, got = _read_in_thread(study, [{"id": "f1", "name": "f1.pdf"}])
    time.sleep(0.5)
    assert "result" not in got                              # waiting for the lock
    release.set()
    a.join(10), r.join(10)
    assert got["result"][1:3] == (0, "not_ready")
    with db.connect() as c:
        assert c.execute(text("SELECT count(*) FROM memento_temp_documents")).scalar() == 0


@pytest.mark.parametrize("pause", ["record", "read"])      # before its claim / after its claim, before its status
def test_removal_waits_for_a_read_and_archives_its_finished_row(study, pause):
    db = study.db
    study.gate[pause] = threading.Event()
    study.gate["entered"].clear()
    r, got = _read_in_thread(study, [{"id": "f1", "name": "f1.pdf"}])
    assert study.gate["entered"].wait(5)
    a, out = _archive_in_thread(db)
    time.sleep(0.7)
    assert "at" not in out                                  # removal waits for the read
    study.gate[pause].set()
    r.join(10), a.join(10)
    assert got["result"][1:3] == (1, "ok") and out.get("result") == {"moved": 1, "identical": 0}
    with db.connect() as c:
        live = c.execute(text("SELECT count(*) FROM memento_temp_documents")).scalar()
        arch = c.execute(text("SELECT status, pages, collection FROM memento_temp_documents_study")).all()
    assert live == 0 and arch == [("ready", 4, POLY)]


def test_a_cancelled_request_keeps_the_lock_until_its_last_write(study):
    db = study.db
    study.gate["read"] = threading.Event()
    study.gate["entered"].clear()
    loop, box = asyncio.new_event_loop(), {}

    async def status(*a, **k):
        return None

    def run():
        asyncio.set_event_loop(loop)
        box["task"] = loop.create_task(chat.doc_chat_read(study.pipe.valves, db, td, CHAT, "u1", [{"id": "f1", "name": "f1.pdf"}],
                                                          status, logging.getLogger("t"), function_id=study.mod._OWN_ID,
                                                          db_engine=study.mod._db_engine, saved_valves=lambda fid: library_state.saved_valves(fid, study.mod._db_engine)))
        try:
            loop.run_until_complete(box["task"])
        except asyncio.CancelledError:
            box["cancelled"] = True
    t = threading.Thread(target=run, daemon=True)
    t.start()
    assert study.gate["entered"].wait(5)                    # the read thread is inside the file read
    loop.call_soon_threadsafe(box["task"].cancel)
    t.join(5)
    assert box.get("cancelled")
    with pytest.raises(td.LockTimeout):                     # the read thread still holds the lock
        td.archive(db, POLY, timeout_s=1)
    study.gate["read"].set()
    assert _wait_for(lambda: [r["status"] for r in td.chat_documents(db, CHAT, "u1", collection=POLY)] == ["ready"])
    assert _wait_for(lambda: _advisory(db) == [])
    assert td.archive(db, POLY, timeout_s=5) == {"moved": 1, "identical": 0}
    loop.close()


# ── the transfer rule on PostgreSQL ──

def _all(db, table):
    from sqlalchemy import select
    with db.connect() as c:
        return sorted((dict(r) for r in c.execute(select(table)).mappings()), key=lambda r: r["file_id"])


def test_a_repeated_archive_is_idempotent_on_postgres(db):
    _claim(db, POLY, fid="f1")
    live = _all(db, td.TEMP_DOCS)
    with db.begin() as c:
        c.execute(td.ARCHIVE.insert().values(**live[0]))
    assert td.archive(db, POLY, timeout_s=5) == {"moved": 0, "identical": 1}
    assert _all(db, td.TEMP_DOCS) == [] and _all(db, td.ARCHIVE) == live


@pytest.mark.parametrize("column, value", [("source", "Other-0123456789ab"), ("status", "failed")])
def test_a_conflicting_archive_row_keeps_both_on_postgres(db, column, value):
    _claim(db, POLY, fid="f1")
    live = _all(db, td.TEMP_DOCS)
    with db.begin() as c:
        c.execute(td.ARCHIVE.insert().values(**{**live[0], column: value}))
    archived = _all(db, td.ARCHIVE)
    with pytest.raises(td.TransferConflict):
        td.archive(db, POLY, timeout_s=5)
    assert _all(db, td.TEMP_DOCS) == live and _all(db, td.ARCHIVE) == archived


def test_an_interrupted_transfer_moves_nothing_on_postgres(db, monkeypatch):
    _claim(db, POLY, fid="f1")
    _claim(db, POLY, chat=CHAT2, fid="f2")
    live = _all(db, td.TEMP_DOCS)
    real, n = td._delete_row, []

    def failing(conn, table, row):
        n.append(1)
        if len(n) == 2:
            raise RuntimeError("connection lost")
        return real(conn, table, row)
    monkeypatch.setattr(td, "_delete_row", failing)
    with pytest.raises(RuntimeError):
        td.archive(db, POLY, timeout_s=5)
    assert _all(db, td.TEMP_DOCS) == live and _all(db, td.ARCHIVE) == []
