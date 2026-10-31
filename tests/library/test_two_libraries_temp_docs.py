# Document Chat's files belong to one library. Two copies of the pipes share the table `memento_temp_documents`; each
# row records its library's temporary collection (`<prefix>-temp`), so each copy's reads and its cleaner see only its
# own rows. Rows without a collection (NULL) are the default copy's.
import sys
import threading
from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect, select, text, update
from sqlalchemy.pool import StaticPool
from memento.config.paths import PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT))
from memento.document_chat import store as td  # noqa: E402

CHAT_A, CHAT_B = "3f2b8c1e-6a4d-4e0b-9c7a-1d2e3f4a5b6c", "9a8b7c6d-5e4f-4a3b-8c2d-1e0f9a8b7c6d"
COMP, POLY = td.DEFAULT_TEMP_COLLECTION, "memento-polymer-temp"


@pytest.fixture
def engine():
    e = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    td.ensure_tables(e)
    return e


def claim(engine, chat, fid, coll, user="u1", status="ready"):
    src = td.claim(engine, chat_id=chat, user_id=user, file_id=fid, name=f"{fid}.pdf", collection=coll)
    assert isinstance(src, str), src
    td.set_status(engine, chat, fid, status)
    return src


def test_the_default_is_todays_comp_mat_collection():
    assert td.DEFAULT_TEMP_COLLECTION == "memento-comp-mat-temp"


# ── schema ───────────────────────────────────────────────────────────────────

def test_an_existing_table_gets_the_collection_column_and_keeps_its_rows():
    e = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    with e.begin() as c:                      # today's table, without the column
        c.execute(text("CREATE TABLE memento_temp_documents (chat_id VARCHAR(64) NOT NULL, file_id VARCHAR(128) NOT "
                       "NULL, user_id VARCHAR(128) NOT NULL, source VARCHAR(120) NOT NULL UNIQUE, name TEXT NOT NULL, "
                       "pages INTEGER, status VARCHAR(10) NOT NULL, error TEXT, added_at DATETIME, "
                       "PRIMARY KEY (chat_id, file_id))"))
        c.execute(text("INSERT INTO memento_temp_documents (chat_id, file_id, user_id, source, name, status) "
                       "VALUES (:c, 'f1', 'u1', 'Old-0123456789ab', 'Old.pdf', 'ready')"), {"c": CHAT_A})
    td.ensure_tables(e)
    td.ensure_tables(e)                       # twice: no error
    cols = {c["name"] for c in inspect(e).get_columns("memento_temp_documents")}
    assert "collection" in cols and inspect(e).has_table("memento_temp_documents_study")
    rows = td.chat_documents(e, CHAT_A, "u1", collection=COMP)       # a NULL row is today's
    assert [r["source"] for r in rows] == ["Old-0123456789ab"] and rows[0]["collection"] is None
    assert td.chat_documents(e, CHAT_A, "u1", collection=POLY) == []


def test_the_archive_table_has_the_same_columns(engine):
    live = [c.name for c in td.TEMP_DOCS.columns]
    assert [c.name for c in td.ARCHIVE.columns] == live and "collection" in live


# ── scoping ──────────────────────────────────────────────────────────────────

def test_each_library_reads_only_its_own_rows(engine):
    a = claim(engine, CHAT_A, "f1", COMP)
    b = claim(engine, CHAT_B, "f2", POLY)
    assert [r["source"] for r in td.chat_documents(engine, CHAT_A, "u1", collection=COMP)] == [a]
    assert td.chat_documents(engine, CHAT_A, "u1", collection=POLY) == []
    assert [r["source"] for r in td.chat_documents(engine, CHAT_B, "u1", collection=POLY)] == [b]
    with engine.connect() as c:
        stored = dict(c.execute(select(td.TEMP_DOCS.c.file_id, td.TEMP_DOCS.c.collection)).all())
    assert stored == {"f1": COMP, "f2": POLY}


def test_calls_without_a_collection_are_todays_library(engine):
    # today's callers and tests: the default copy, unchanged
    src = td.claim(engine, chat_id=CHAT_A, user_id="u1", file_id="f1", name="A.pdf")
    assert [r["source"] for r in td.chat_documents(engine, CHAT_A, "u1")] == [src]
    assert [r["source"] for r in td.chat_documents(engine, CHAT_A, "u1", collection=COMP)] == [src]


def test_a_null_row_counts_only_for_todays_library(engine):
    src = claim(engine, CHAT_A, "f1", COMP)
    with engine.begin() as c:
        c.execute(update(td.TEMP_DOCS).values(collection=None))
    assert [r["source"] for r in td.chat_documents(engine, CHAT_A, "u1", collection=COMP)] == [src]
    assert td.chat_documents(engine, CHAT_A, "u1", collection=POLY) == []
    assert [r["chat_id"] for r in td.orphans(engine, set(), collection=COMP)] == [CHAT_A]
    assert td.orphans(engine, set(), collection=POLY) == []
    td.remove_rows(engine, CHAT_A, collection=POLY)
    assert td.chat_documents(engine, CHAT_A, "u1", collection=COMP)        # the second copy cannot remove it
    td.remove_rows(engine, CHAT_A, collection=COMP)
    assert td.chat_documents(engine, CHAT_A, "u1", collection=COMP) == []


# ── one library per chat (R2) ─────────────────────────────────────────────────

def test_a_chat_knows_which_libraries_it_has_files_in(engine):
    assert td.chat_collections(engine, CHAT_A) == set()
    claim(engine, CHAT_A, "f1", COMP, user="u1")
    claim(engine, CHAT_A, "f2", COMP, user="u2")                       # any user's rows count
    assert td.chat_collections(engine, CHAT_A) == {COMP}
    with engine.begin() as c:
        c.execute(update(td.TEMP_DOCS).where(td.TEMP_DOCS.c.file_id == "f2").values(collection=None))
    assert td.chat_collections(engine, CHAT_A) == {COMP}                # NULL is today's library


@pytest.mark.parametrize("fid", ["f1", "f2"])                          # the same file again, or another file
def test_a_chat_with_files_in_one_library_cannot_claim_in_the_other(engine, fid):
    claim(engine, CHAT_A, "f1", COMP)
    assert td.claim(engine, chat_id=CHAT_A, user_id="u1", file_id=fid, name="x.pdf",
                    collection=POLY) is td.OTHER_LIBRARY
    assert [r["collection"] for r in td.chat_documents(engine, CHAT_A, "u1", collection=COMP)] == [COMP]
    with engine.connect() as c:
        assert c.execute(select(td.TEMP_DOCS)).mappings().all().__len__() == 1        # nothing inserted
    # the same library goes on as before
    assert isinstance(td.claim(engine, chat_id=CHAT_A, user_id="u1", file_id="f3", name="y.pdf", collection=COMP),
                      str)


def test_other_library_is_a_distinct_marker(engine):
    claim(engine, CHAT_A, "f1", COMP)
    assert td.claim(engine, chat_id=CHAT_A, user_id="u1", file_id="f1", name="x.pdf", collection=COMP) is None
    assert td.OTHER_LIBRARY is not None and not isinstance(td.OTHER_LIBRARY, str)


# ── the cleaner: only its own library's rows ──────────────────────────────────

class FakeQdrant:
    """Point deletes by chat: records (collection, chat id); `fail` makes every delete fail."""
    def __init__(self, fail=False):
        self.deleted, self.fail = [], fail

    def delete(self, collection_name, points_selector):
        if self.fail:
            raise RuntimeError("qdrant down")
        chat = points_selector.filter.must[0].match.value
        self.deleted.append((collection_name, chat))


def _files(tmp_path, chat, sources):
    inp, out = tmp_path / "in", tmp_path / "out"
    (inp / "_temp" / chat).mkdir(parents=True, exist_ok=True)
    (inp / "_temp" / chat / "f.pdf").write_text("x")
    for s in sources:
        (out / "_temp" / s).mkdir(parents=True, exist_ok=True)
    return inp, out


def test_each_cleaner_removes_only_its_own_librarys_deleted_chats(engine, tmp_path):
    a = claim(engine, CHAT_A, "f1", COMP)
    b = claim(engine, CHAT_B, "f2", POLY)
    inp, out = _files(tmp_path, CHAT_A, [a])
    _files(tmp_path, CHAT_B, [b])
    q = FakeQdrant()
    assert td.cleanup(engine, q, POLY, inp, out, existing_chat_ids=set()) == 1       # both chats deleted
    assert q.deleted == [(POLY, CHAT_B)]
    assert td.chat_documents(engine, CHAT_A, "u1", collection=COMP)                  # comp-mat's row untouched
    assert (inp / "_temp" / CHAT_A).exists() and (out / "_temp" / a).exists()
    assert not (inp / "_temp" / CHAT_B).exists() and not (out / "_temp" / b).exists()
    assert td.cleanup(engine, q, COMP, inp, out, existing_chat_ids=set()) == 1
    assert q.deleted[-1] == (COMP, CHAT_A) and td.chat_documents(engine, CHAT_A, "u1", collection=COMP) == []
    assert not (inp / "_temp" / CHAT_A).exists() and not (out / "_temp" / a).exists()


@pytest.mark.parametrize("order", [(COMP, POLY), (POLY, COMP)])
def test_the_cleaners_in_either_order_leave_both_libraries_consistent(engine, tmp_path, order):
    a = claim(engine, CHAT_A, "f1", COMP)
    b = claim(engine, CHAT_B, "f2", POLY)
    inp, out = _files(tmp_path, CHAT_A, [a])
    _files(tmp_path, CHAT_B, [b])
    q = FakeQdrant()
    for coll in order:
        td.cleanup(engine, q, coll, inp, out, existing_chat_ids=set())
    assert sorted(q.deleted) == sorted([(COMP, CHAT_A), (POLY, CHAT_B)])
    with engine.connect() as c:
        assert c.execute(select(td.TEMP_DOCS)).all() == []
    assert not any((inp / "_temp").iterdir()) and not any((out / "_temp").iterdir())


def test_two_cleaners_at_once_leave_both_libraries_consistent(tmp_path):
    # a database file, one connection per thread (an in-memory SQLite connection cannot be shared by two threads)
    engine = create_engine(f"sqlite:///{tmp_path / 'db.sqlite'}", connect_args={"timeout": 30})
    td.ensure_tables(engine)
    srcs = {}
    for i in range(6):
        chat = f"{i:08d}-6a4d-4e0b-9c7a-1d2e3f4a5b6c"
        coll = COMP if i % 2 else POLY
        srcs[chat] = (coll, claim(engine, chat, f"f{i}", coll))
        _files(tmp_path, chat, [srcs[chat][1]])
    inp, out = tmp_path / "in", tmp_path / "out"
    q = FakeQdrant()
    ts = [threading.Thread(target=td.cleanup, args=(engine, q, coll, inp, out, set())) for coll in (COMP, POLY)]
    [t.start() for t in ts]
    [t.join(30) for t in ts]
    assert sorted(q.deleted) == sorted((coll, chat) for chat, (coll, _) in srcs.items())
    with engine.connect() as c:
        assert c.execute(select(td.TEMP_DOCS)).all() == []
    assert not any((inp / "_temp").iterdir()) and not any((out / "_temp").iterdir())


def test_a_failed_vector_delete_keeps_the_rows_and_the_next_run_removes_them(engine, tmp_path):
    b = claim(engine, CHAT_B, "f2", POLY)
    inp, out = _files(tmp_path, CHAT_B, [b])
    assert td.cleanup(engine, FakeQdrant(fail=True), POLY, inp, out, existing_chat_ids=set()) == 0
    assert td.chat_documents(engine, CHAT_B, "u1", collection=POLY) and (inp / "_temp" / CHAT_B).exists()
    assert td.cleanup(engine, FakeQdrant(), POLY, inp, out, existing_chat_ids=set()) == 1
    assert td.chat_documents(engine, CHAT_B, "u1", collection=POLY) == []


def test_a_chat_with_rows_in_both_libraries_keeps_its_folder_until_the_last_row_is_gone(engine, tmp_path):
    # legacy data only (a chat can no longer get rows in two libraries): written directly
    a = claim(engine, CHAT_A, "f1", COMP)
    with engine.begin() as c:
        c.execute(td.TEMP_DOCS.insert().values(chat_id=CHAT_A, file_id="f2", user_id="u1", source="B-0123456789ab",
                                               name="B.pdf", status="ready", collection=POLY))
    inp, out = _files(tmp_path, CHAT_A, [a, "B-0123456789ab"])
    td.cleanup(engine, FakeQdrant(), POLY, inp, out, existing_chat_ids=set())
    assert (inp / "_temp" / CHAT_A).exists() and (out / "_temp" / a).exists()        # comp-mat's row remains
    assert not (out / "_temp" / "B-0123456789ab").exists()
    td.cleanup(engine, FakeQdrant(), COMP, inp, out, existing_chat_ids=set())
    assert not (inp / "_temp" / CHAT_A).exists()


def test_a_null_row_is_cleaned_only_by_todays_copy(engine, tmp_path):
    a = claim(engine, CHAT_A, "f1", COMP)
    with engine.begin() as c:
        c.execute(update(td.TEMP_DOCS).values(collection=None))
    inp, out = _files(tmp_path, CHAT_A, [a])
    assert td.cleanup(engine, FakeQdrant(), POLY, inp, out, existing_chat_ids=set()) == 0
    assert td.chat_documents(engine, CHAT_A, "u1", collection=COMP)
    assert td.cleanup(engine, FakeQdrant(), COMP, inp, out, existing_chat_ids=set()) == 1


# ── the lifecycle lock: SQLite and today's library take none ──────────────────

def test_the_lifecycle_lock_is_a_no_op_on_sqlite_and_for_todays_library(engine):
    with td.lifecycle(engine, POLY):
        pass
    with td.lifecycle(engine, COMP):
        pass


# ── archive and restore: the transfer rule (V3) ───────────────────────────────

def _rows(engine, table):
    with engine.connect() as c:
        return sorted((dict(r) for r in c.execute(select(table)).mappings()), key=lambda r: r["file_id"])


def test_archive_moves_a_librarys_rows_and_restore_brings_them_back(engine):
    claim(engine, CHAT_A, "f1", COMP)
    claim(engine, CHAT_B, "f2", POLY)
    claim(engine, CHAT_B, "f3", POLY, status="failed")
    before = _rows(engine, td.TEMP_DOCS)
    r = td.archive(engine, POLY, timeout_s=5)
    assert r == {"moved": 2, "identical": 0}
    assert [x["file_id"] for x in _rows(engine, td.TEMP_DOCS)] == ["f1"]          # the default library stays live
    assert [x["file_id"] for x in _rows(engine, td.ARCHIVE)] == ["f2", "f3"]
    assert td.restore(engine, POLY, timeout_s=5) == {"moved": 2, "identical": 0}
    assert _rows(engine, td.TEMP_DOCS) == before and _rows(engine, td.ARCHIVE) == []


def test_restore_with_an_empty_archive_takes_no_lock_and_changes_nothing(engine, monkeypatch):
    from memento.library import locks as db_locks
    monkeypatch.setattr(db_locks, "temp_lifecycle_exclusive", lambda *a, **k: pytest.fail("lock taken"))
    claim(engine, CHAT_B, "f2", POLY)
    assert td.restore(engine, POLY, timeout_s=5) == {"moved": 0, "identical": 0}


def test_an_identical_row_in_the_archive_makes_a_repeated_archive_idempotent(engine):
    claim(engine, CHAT_B, "f2", POLY)
    live = _rows(engine, td.TEMP_DOCS)
    with engine.begin() as c:
        c.execute(td.ARCHIVE.insert().values(**live[0]))
    assert td.archive(engine, POLY, timeout_s=5) == {"moved": 0, "identical": 1}
    assert _rows(engine, td.TEMP_DOCS) == [] and _rows(engine, td.ARCHIVE) == live


@pytest.mark.parametrize("column, value", [("source", "Other-0123456789ab"), ("status", "failed"), ("pages", 99)])
def test_a_different_row_under_the_same_key_stops_the_archive_with_both_intact(engine, column, value):
    claim(engine, CHAT_B, "f2", POLY)
    live = _rows(engine, td.TEMP_DOCS)
    with engine.begin() as c:
        c.execute(td.ARCHIVE.insert().values(**{**live[0], column: value}))
    archived = _rows(engine, td.ARCHIVE)
    with pytest.raises(td.TransferConflict) as e:
        td.archive(engine, POLY, timeout_s=5)
    assert _rows(engine, td.TEMP_DOCS) == live and _rows(engine, td.ARCHIVE) == archived      # nothing deleted
    assert e.value.conflicts and e.value.conflicts[0][0][column] != e.value.conflicts[0][1][column]


def test_a_conflict_rolls_back_rows_moved_earlier_in_the_same_run(engine):
    claim(engine, CHAT_A, "f1", POLY)
    claim(engine, CHAT_B, "f2", POLY)
    live = _rows(engine, td.TEMP_DOCS)
    with engine.begin() as c:
        c.execute(td.ARCHIVE.insert().values(**{**live[1], "status": "failed"}))
    with pytest.raises(td.TransferConflict):
        td.archive(engine, POLY, timeout_s=5)
    assert _rows(engine, td.TEMP_DOCS) == live                                      # f1 not moved either


def test_a_restore_conflict_keeps_both_versions_and_a_later_archive_keeps_them_too(engine):
    claim(engine, CHAT_B, "f2", POLY)
    old = _rows(engine, td.TEMP_DOCS)
    td.archive(engine, POLY, timeout_s=5)
    claim(engine, CHAT_B, "f2", POLY)                       # the same key claimed live again, other source
    new = _rows(engine, td.TEMP_DOCS)
    assert new[0]["source"] != old[0]["source"]
    with pytest.raises(td.TransferConflict):
        td.restore(engine, POLY, timeout_s=5)
    with pytest.raises(td.TransferConflict):
        td.archive(engine, POLY, timeout_s=5)
    assert _rows(engine, td.TEMP_DOCS) == new and _rows(engine, td.ARCHIVE) == old


def test_a_transfer_interrupted_midway_moves_nothing(engine, monkeypatch):
    claim(engine, CHAT_A, "f1", POLY)
    claim(engine, CHAT_B, "f2", POLY)
    live = _rows(engine, td.TEMP_DOCS)
    real, calls = td._delete_row, []

    def failing(conn, table, row):
        calls.append(row["file_id"])
        if len(calls) == 2:
            raise RuntimeError("connection lost")
        return real(conn, table, row)
    monkeypatch.setattr(td, "_delete_row", failing)
    with pytest.raises(RuntimeError):
        td.archive(engine, POLY, timeout_s=5)
    assert _rows(engine, td.TEMP_DOCS) == live and _rows(engine, td.ARCHIVE) == []


def test_the_archive_hands_its_rows_to_the_caller_inside_the_lock(engine):
    claim(engine, CHAT_B, "f2", POLY)
    seen = []
    td.archive(engine, POLY, timeout_s=5, before_move=lambda rows: seen.extend(rows))
    assert [r["file_id"] for r in seen] == ["f2"] and seen[0]["collection"] == POLY


def test_todays_library_is_never_archived(engine):
    claim(engine, CHAT_A, "f1", COMP)
    with pytest.raises(ValueError):
        td.archive(engine, COMP, timeout_s=5)


# ── backfill (X4) ─────────────────────────────────────────────────────────────

class CountingQdrant:
    """{collection: {chat: points}}; count() by chat id."""
    def __init__(self, points):
        self.points = points

    def get_collections(self):
        from types import SimpleNamespace
        return SimpleNamespace(collections=[SimpleNamespace(name=n) for n in self.points])

    def count(self, collection_name, count_filter=None, exact=True):
        from types import SimpleNamespace
        chat = count_filter.must[0].match.value
        return SimpleNamespace(count=self.points.get(collection_name, {}).get(chat, 0))
