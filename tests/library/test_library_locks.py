# Storing and removing coordinated: transaction-scoped advisory locks (shared collection lock, then the document's), one
# transaction for a removal, the collection lock for reset / rebuild, and stores that check the library row under the
# lock, so no answer is stored for a document being removed and a rebuild never publishes a stale list. Logic on SQLite
# (the locks are no-ops there); the locking itself on a throwaway PostgreSQL when MEMENTO_TEST_PG_URL is set (never a
# live database).
import os
import threading
import time

import pytest
from sqlalchemy import create_engine, text

from memento.library import answers as ac
from memento.library import locks as db_locks, front_matter as doc_authors, documents as ldb, warmup_queue
from memento.config.paths import PROJECT_ROOT

F = lambda source="Doc", h="h1": dict(collection="c", source=source, content_hash=h, action="summary", language="en",
                                      model="m", mode="whole", prompt_version="p")


def _add(engine, source="Doc", h="h1"):
    ldb.add_document(engine, collection="c", entry={"source": source, "title": source}, content_hash=h,
                     uploaded_by="u", output_root=None)


def _store(engine, source="Doc", h="h1", answer="a"):
    ac.put(engine, ac.cache_key(**F(source, h)), fields=F(source, h), answer=answer, citations=[], generated_by="m")


def _answers(engine, source="Doc"):
    with engine.connect() as c:
        return c.execute(text("select count(*) from memento_answer_cache where source = :s"), {"s": source}).scalar()


@pytest.fixture
def lite():
    e = create_engine("sqlite://")
    ldb.ensure_tables(e)
    doc_authors.ensure_table(e)
    return e


# ── logic (SQLite) ───────────────────────────────────────────────────────────────────────────────────────────────────

def test_keys_are_stable_32_bit_numbers():
    k = db_locks.key("c", "Doc")
    assert k == db_locks.key("c", "Doc") and -2 ** 31 <= k < 2 ** 31 and k != db_locks.key("c", "Doc2")


def test_an_answer_is_stored_only_for_the_version_in_the_library(lite):
    _add(lite, h="h1")
    _store(lite, h="h1")
    assert _answers(lite) == 1
    _store(lite, h="h0", answer="for an older version")             # the library has another version
    _store(lite, source="Gone", h="h1")                             # not in the library
    with lite.connect() as c:
        assert c.execute(text("select answer from memento_answer_cache")).scalars().all() == ["a"]


def test_removal_takes_the_row_its_answers_and_its_jobs_in_one_go(lite):
    _add(lite)
    _store(lite)
    warmup_queue.enqueue(lite, "c", "Doc", ["summary"])
    ldb.remove_document(lite, collection="c", source="Doc", output_root=None)
    assert ldb.get_document(lite, "c", "Doc") is None and _answers(lite) == 0
    with lite.connect() as c:
        assert c.execute(text("select count(*) from memento_warmup_queue where source = 'Doc'")).scalar() == 0


def test_a_rebuild_keeps_a_row_added_meanwhile(lite):
    _add(lite, source="New", h="hn")                                 # an upload finished before the rebuild ran
    ldb.rebuild_documents(lite, collection="c", sources={"Old"}, in_dir=None, out_dir=None, section_hints=None,
                          manifest={}, output_root=None)
    assert {r["source"] for r in ldb.list_documents(lite, "c")} == {"New", "Old"}


def test_a_rebuild_reads_the_sources_again_under_the_lock_and_skips_a_removed_document(lite, monkeypatch):
    # a document removed between the first read of the source list and the rebuild's collection lock: the second read
    # under the lock leaves it out
    _add(lite)
    state = {"docs": {"Doc"}}
    original = ldb.document_entry

    def entry(*a, **kw):
        if state["docs"]:
            state["docs"] = set()                                    # the vectors are gone …
            ldb.remove_document(lite, collection="c", source="Doc", output_root=None)      # … then the row
        return original(*a, **kw)
    monkeypatch.setattr(ldb, "document_entry", entry)
    ldb.rebuild_documents(lite, collection="c", sources=lambda: set(state["docs"]), in_dir=None, out_dir=None,
                          section_hints=None, manifest={"documents": {"Doc": {"content_hash": "h1"}}}, output_root=None)
    assert ldb.get_document(lite, "c", "Doc") is None


def test_a_rebuild_adds_a_document_that_appeared_before_the_lock(lite):
    reads = iter([{"Old"}, {"Old", "New"}])
    n = ldb.rebuild_documents(lite, collection="c", sources=lambda: next(reads), in_dir=None, out_dir=None,
                              section_hints=None, manifest={}, output_root=None)
    assert n == 2 and {r["source"] for r in ldb.list_documents(lite, "c")} == {"New", "Old"}


def test_a_refresh_replaces_the_rows_with_fresh_fingerprints(lite):
    _add(lite, source="Doc", h="h1")
    _add(lite, source="Ghost", h="hg")                               # in the table, no longer in the collection
    ldb.rebuild_documents(lite, collection="c", sources=lambda: {"Doc"}, in_dir=None, out_dir=None, section_hints=None,
                          manifest={"documents": {"Doc": {"content_hash": "h2"}}}, output_root=None, refresh=True)
    assert [(r["source"], r["content_hash"]) for r in ldb.list_documents(lite, "c")] == [("Doc", "h2")]


def test_the_production_callers_pass_a_source_function():
    # the rebuild can only see a removal through a fresh read: every live caller hands it a function
    import inspect
    from pathlib import Path
    root = PROJECT_ROOT
    for f in ("memento/query/library_state.py", "memento/ingestion/upload/document.py", "memento/ingestion/run_pipeline.py"):
        src = (root / f).read_text()
        i = next(src.index(p) for p in ("rebuild_documents(engine", "rebuild_documents(_db_engine()",
                                        "rebuild_documents(db_engine()") if p in src)
        call = src[i:src.index(")", src.index("output_root", i))]
        assert "sources=read_sources" in call or "sources=lambda" in call, (f, call)


# ── the locks (throwaway PostgreSQL) ─────────────────────────────────────────────────────────────────────────────────

PG = os.environ.get("MEMENTO_TEST_PG_URL")
pg_only = pytest.mark.skipif(not PG, reason="needs a throwaway PostgreSQL (MEMENTO_TEST_PG_URL)")


@pytest.fixture
def pg():
    e = create_engine(PG)
    with e.begin() as c:
        for t in ("memento_documents", "memento_answer_cache", "memento_warmup_queue", "memento_front_matter"):
            c.execute(text(f"DROP TABLE IF EXISTS {t}"))
    ldb.ensure_tables(e)
    doc_authors.ensure_table(e)
    yield e
    e.dispose()


def _hold(engine, take, seconds, started):
    """A transaction that takes locks (`take(conn)`) and keeps them for `seconds`."""
    with engine.begin() as c:
        take(c)
        started.set()
        time.sleep(seconds)


def _after(engine, take, op, seconds=0.6):
    """Run `op` while another transaction holds `take`'s locks: (seconds op waited, its result)."""
    started = threading.Event()
    t = threading.Thread(target=_hold, args=(engine, take, seconds, started))
    t.start()
    started.wait(5)
    t0 = time.time()
    out = op()
    waited = time.time() - t0
    t.join()
    return waited, out


@pg_only
def test_a_store_waits_for_a_removal_and_then_skips(pg):
    _add(pg)

    def removing(c):                                  # the removal's transaction, paused after taking its locks
        db_locks.document(c, "c", "Doc")
        c.execute(text("DELETE FROM memento_documents WHERE collection = 'c' AND source = 'Doc'"))
    waited, _ = _after(pg, removing, lambda: _store(pg))
    assert waited >= 0.4 and _answers(pg) == 0        # it waited, then found the document gone


@pg_only
def test_a_removal_waits_for_a_store_and_then_removes_it(pg):
    _add(pg)

    def storing(c):
        db_locks.document(c, "c", "Doc")
        c.execute(text("INSERT INTO memento_answer_cache (key, collection, source, content_hash, action, language, "
                       "model, mode, prompt_version, generated_by, answer, citations, status, hits, checked) VALUES "
                       "('k','c','Doc','h1','summary','en','m','whole','p','m','a','[]','ready',0,false)"))
    waited, _ = _after(pg, storing, lambda: ldb.remove_document(pg, collection="c", source="Doc", output_root=None))
    assert waited >= 0.4 and _answers(pg) == 0


@pg_only
def test_a_reset_waits_for_an_add_and_an_add_for_a_reset(pg):
    waited, _ = _after(pg, lambda c: db_locks.document(c, "c", "Doc"),
                       lambda: ldb.reset_library(pg, collection="c", output_root=None))
    assert waited >= 0.4
    waited, _ = _after(pg, lambda c: db_locks.collection(c, "c"), lambda: _add(pg))
    assert waited >= 0.4 and ldb.get_document(pg, "c", "Doc") is not None


@pg_only
def test_a_rebuild_waits_for_a_removal_and_leaves_the_removed_document_out(pg):
    _add(pg, source="Old")
    vectors = {"Old", "Other"}

    def removing(c):                                   # the removal: vectors gone, its row deleted, locks still held
        vectors.discard("Old")
        db_locks.document(c, "c", "Old")
        c.execute(text("DELETE FROM memento_documents WHERE collection = 'c' AND source = 'Old'"))
    waited, _ = _after(pg, removing,
                       lambda: ldb.rebuild_documents(pg, collection="c", sources=lambda: set(vectors), in_dir=None,
                                                     out_dir=None, section_hints=None, manifest={}, output_root=None))
    assert waited >= 0.4
    assert ldb.get_document(pg, "c", "Old") is None and ldb.get_document(pg, "c", "Other") is not None


@pg_only
def test_a_front_matter_write_waits_for_a_reset_and_a_replacement(pg):
    _add(pg, h="h2")
    waited, ok = _after(pg, lambda c: db_locks.collection(c, "c"),
                        lambda: doc_authors.put(pg, collection="c", source="Doc", content_hash="h2", front="Bob"))
    assert waited >= 0.4 and ok is True
    waited, ok = _after(pg, lambda c: db_locks.document(c, "c", "Doc"),
                        lambda: doc_authors.put(pg, collection="c", source="Doc", content_hash="h1", front="Old job"))
    assert waited >= 0.4 and ok is False                               # the old version's job loses
    assert doc_authors.load(pg, "c", with_hash=True)["Doc"] == ("Bob", "h2")


@pg_only
def test_a_rebuild_that_writes_front_matter_does_not_wait_on_itself(pg, tmp_path):
    (tmp_path / "Doc" / "vlm").mkdir(parents=True)
    (tmp_path / "Doc" / "vlm" / "Doc.md").write_text("# T\n\nAlice Jones\n")
    t0 = time.time()
    n = ldb.rebuild_documents(pg, collection="c", sources={"Doc"}, in_dir=None, out_dir=tmp_path, section_hints=None,
                              manifest={"documents": {"Doc": {"content_hash": "h1"}}}, output_root=None)
    assert n == 1 and time.time() - t0 < 5
    assert doc_authors.load(pg, "c", with_hash=True)["Doc"][1] == "h1"
