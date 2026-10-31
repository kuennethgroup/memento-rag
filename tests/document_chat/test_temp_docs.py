# Memento Document Chat's storage and scoped search client.
import re
import sys
from pathlib import Path

import pytest
from qdrant_client.models import FieldCondition, Filter, MatchValue, Prefetch
from sqlalchemy import create_engine
from memento.config.paths import PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT))
from memento.document_chat import store as td  # noqa: E402

CHAT_A, CHAT_B = "3f2b8c1e-6a4d-4e0b-9c7a-1d2e3f4a5b6c", "9a8b7c6d-5e4f-4a3b-8c2d-1e0f9a8b7c6d"


@pytest.fixture
def engine():
    e = create_engine("sqlite://")
    td.ensure_tables(e)
    return e


# ── table ─────────────────────────────────────────────────────────────────────

def test_a_file_is_claimed_once_per_chat(engine):
    src = td.claim(engine, chat_id=CHAT_A, user_id="u1", file_id="f1", name="Paper.pdf")
    assert src and td.claim(engine, chat_id=CHAT_A, user_id="u1", file_id="f1", name="Paper.pdf") is None
    rows = td.chat_documents(engine, CHAT_A, "u1")
    assert [(r["source"], r["name"], r["status"]) for r in rows] == [(src, "Paper.pdf", "reading")]


def test_chats_and_users_are_kept_apart(engine):
    a = td.claim(engine, chat_id=CHAT_A, user_id="u1", file_id="f1", name="A.pdf")
    b = td.claim(engine, chat_id=CHAT_B, user_id="u1", file_id="f1", name="A.pdf")    # same file, other chat
    assert a and b and a != b
    td.claim(engine, chat_id=CHAT_A, user_id="u2", file_id="f9", name="X.pdf")       # (cannot happen; still apart)
    assert [r["source"] for r in td.chat_documents(engine, CHAT_A, "u1")] == [a]
    assert [r["source"] for r in td.chat_documents(engine, CHAT_B, "u1")] == [b]
    assert td.chat_documents(engine, CHAT_B, "u2") == []


def test_status_pages_and_error_are_recorded(engine):
    td.claim(engine, chat_id=CHAT_A, user_id="u1", file_id="f1", name="A.pdf")
    td.claim(engine, chat_id=CHAT_A, user_id="u1", file_id="f2", name="B.pdf")
    td.set_status(engine, CHAT_A, "f1", "ready", pages=12)
    td.set_status(engine, CHAT_A, "f2", "failed", error="too many pages")
    rows = {r["file_id"]: r for r in td.chat_documents(engine, CHAT_A, "u1")}
    assert (rows["f1"]["status"], rows["f1"]["pages"]) == ("ready", 12)
    assert (rows["f2"]["status"], rows["f2"]["error"]) == ("failed", "too many pages")


def test_rows_of_chats_that_no_longer_exist_are_found_and_removed(engine):
    td.claim(engine, chat_id=CHAT_A, user_id="u1", file_id="f1", name="A.pdf")
    td.claim(engine, chat_id=CHAT_B, user_id="u2", file_id="f2", name="B.pdf")
    gone = td.orphans(engine, existing_chat_ids={CHAT_B})
    assert [r["chat_id"] for r in gone] == [CHAT_A]
    td.remove_rows(engine, CHAT_A)
    assert td.chat_documents(engine, CHAT_A, "u1") == [] and len(td.chat_documents(engine, CHAT_B, "u2")) == 1


# ── names and ids ─────────────────────────────────────────────────────────────

def test_source_names_are_safe_and_unique():
    a, b = td.safe_source("../Mein Bericht (v2).PDF"), td.safe_source("../Mein Bericht (v2).PDF")
    assert a != b
    for s in (a, b):
        assert "/" not in s and ".." not in s and " " not in s
        assert re.fullmatch(td.SOURCE_RE, s)
        assert re.search(r"-[0-9a-f]{12}$", s)
    assert td.safe_source("").startswith("file-")
    assert len(td.safe_source("x" * 400)) <= 100


@pytest.mark.parametrize("cid, ok", [(CHAT_A, True), (None, False), ("", False), ("local", False),
                                     ("local:abc", False), ("../x", False), (CHAT_A + "/..", False)])
def test_only_saved_chats_have_valid_ids(cid, ok):
    assert td.valid_chat_id(cid) is ok


# ── scoped client ─────────────────────────────────────────────────────────────

class FakeQdrant:
    def __init__(self):
        self.calls = []

    def _rec(self, name, kwargs):
        self.calls.append((name, kwargs))
        return name

    def scroll(self, collection_name, scroll_filter=None, limit=10, offset=None, with_payload=True, with_vectors=False,
               **kw):
        return self._rec("scroll", dict(collection_name=collection_name, scroll_filter=scroll_filter, limit=limit))

    def count(self, collection_name, count_filter=None, exact=True, **kw):
        return self._rec("count", dict(collection_name=collection_name, count_filter=count_filter))

    def query_points(self, collection_name, query=None, using=None, prefetch=None, query_filter=None, **kw):
        return self._rec("query_points", dict(prefetch=prefetch, query_filter=query_filter))

    def query_points_groups(self, collection_name, group_by, query=None, using=None, prefetch=None, query_filter=None,
                            **kw):
        return self._rec("query_points_groups", dict(prefetch=prefetch, query_filter=query_filter))

    def collection_exists(self, collection_name, **kw):
        return True

    def delete(self, *a, **kw):
        raise AssertionError("must not be reached")


def _scoped(f: Filter | None, chat=CHAT_A, user="u1") -> bool:
    """The filter requires this chat AND this user (top-level must)."""
    if f is None:
        return False
    keys = {(c.key, c.match.value) for c in (f.must or []) if isinstance(c, FieldCondition)}
    return ("chat_id", chat) in keys and ("user_id", user) in keys


def test_every_search_gets_the_chat_and_user_filter():
    fake = FakeQdrant()
    q = td.ScopedQdrant(fake, CHAT_A, "u1")
    own = Filter(must=[FieldCondition(key="source", match=MatchValue(value="S"))])
    q.scroll(collection_name="c", scroll_filter=own, limit=5)
    q.scroll("c")                                                   # positional, no filter
    q.count("c", exact=False)
    q.count(collection_name="c", count_filter=own)
    q.query_points_groups(collection_name="c", group_by="source", query=[0.1], query_filter=None)
    for name, kw in fake.calls:
        f = kw.get("scroll_filter") or kw.get("count_filter") or kw.get("query_filter")
        assert _scoped(f), name
    # the caller's own condition is kept, nested
    kept = fake.calls[0][1]["scroll_filter"]
    assert any(isinstance(c, Filter) and c.must[0].key == "source" for c in kept.must)


def test_every_prefetch_is_scoped_too():
    fake = FakeQdrant()
    q = td.ScopedQdrant(fake, CHAT_A, "u1")
    inner = Prefetch(query=[0.2], using="dense", limit=5)
    q.query_points(collection_name="c", prefetch=[Prefetch(query=[0.1], using="dense", filter=None, limit=10),
                                                  Prefetch(query=[0.1], using="sparse", prefetch=[inner], limit=10)])
    q.query_points(collection_name="c", prefetch=Prefetch(query=[0.1], using="dense", limit=3))
    for _name, kw in fake.calls:
        assert _scoped(kw["query_filter"])
        pfs = kw["prefetch"] if isinstance(kw["prefetch"], list) else [kw["prefetch"]]
        stack = list(pfs)
        while stack:
            p = stack.pop()
            assert _scoped(p.filter)
            stack += p.prefetch if isinstance(p.prefetch, list) else ([p.prefetch] if p.prefetch else [])


@pytest.mark.parametrize("method", ["delete", "upsert", "retrieve", "search", "set_payload", "query_batch_points",
                                    "recommend", "discover", "facet", "get_collections"])
def test_anything_else_is_refused(method):
    q = td.ScopedQdrant(FakeQdrant(), CHAT_A, "u1")
    with pytest.raises(PermissionError):
        getattr(q, method)


def test_a_scope_needs_both_ids():
    with pytest.raises(ValueError):
        td.ScopedQdrant(FakeQdrant(), "", "u1")
    with pytest.raises(ValueError):
        td.ScopedQdrant(FakeQdrant(), CHAT_A, "")


# ── clean-up ─────────────────────────────────────────────────────────

class DeletingQdrant:
    def __init__(self, fail=False):
        self.deleted, self.fail = [], fail

    def delete(self, collection_name, points_selector, **kw):
        if self.fail:
            raise RuntimeError("qdrant down")
        self.deleted.append((collection_name, points_selector))


# the rows below are claimed without a collection: the default library, so its cleaner is the one that runs (a cleaner
# removes only its own library's rows)
def _files(tmp_path, chat, source):
    inp, out = tmp_path / "in", tmp_path / "out"
    (inp / "_temp" / chat).mkdir(parents=True, exist_ok=True)
    (inp / "_temp" / chat / f"{source}.pdf").write_bytes(b"%PDF")
    (out / "_temp" / source / "vlm").mkdir(parents=True, exist_ok=True)
    (out / "_temp" / source / "vlm" / "x.json").write_text("[]")
    return inp, out


def test_a_deleted_chat_loses_its_passages_folders_and_rows(engine, tmp_path):
    a = td.claim(engine, chat_id=CHAT_A, user_id="u1", file_id="f1", name="A.pdf")
    b = td.claim(engine, chat_id=CHAT_B, user_id="u2", file_id="f2", name="B.pdf")
    inp, out = _files(tmp_path, CHAT_A, a)
    _files(tmp_path, CHAT_B, b)
    q = DeletingQdrant()
    n = td.cleanup(engine, q, td.DEFAULT_TEMP_COLLECTION, inp, out, existing_chat_ids={CHAT_B})
    assert n == 1
    (coll, sel), = q.deleted
    keys = {(c.key, c.match.value) for c in sel.filter.must}
    assert coll == td.DEFAULT_TEMP_COLLECTION and keys == {("chat_id", CHAT_A)}
    assert not (inp / "_temp" / CHAT_A).exists() and not (out / "_temp" / a).exists()
    assert (inp / "_temp" / CHAT_B).exists() and (out / "_temp" / b).exists()            # the other chat stays
    assert td.chat_documents(engine, CHAT_A, "u1") == [] and len(td.chat_documents(engine, CHAT_B, "u2")) == 1


def test_nothing_is_deleted_when_the_chat_list_is_unknown(engine, tmp_path):
    a = td.claim(engine, chat_id=CHAT_A, user_id="u1", file_id="f1", name="A.pdf")
    inp, out = _files(tmp_path, CHAT_A, a)
    q = DeletingQdrant()
    assert td.cleanup(engine, q, td.DEFAULT_TEMP_COLLECTION, inp, out, existing_chat_ids=None) == 0
    assert not q.deleted and (inp / "_temp" / CHAT_A).exists() and td.chat_documents(engine, CHAT_A, "u1")


def test_rows_stay_when_the_passages_cannot_be_deleted(engine, tmp_path):
    a = td.claim(engine, chat_id=CHAT_A, user_id="u1", file_id="f1", name="A.pdf")
    inp, out = _files(tmp_path, CHAT_A, a)
    assert td.cleanup(engine, DeletingQdrant(fail=True), td.DEFAULT_TEMP_COLLECTION, inp, out,
                      existing_chat_ids=set()) == 0
    assert td.chat_documents(engine, CHAT_A, "u1") and (inp / "_temp" / CHAT_A).exists()   # tried again next time


def test_a_path_outside_the_temporary_folders_is_never_removed(engine, tmp_path):
    from sqlalchemy import update
    td.claim(engine, chat_id=CHAT_A, user_id="u1", file_id="f1", name="A.pdf")
    precious = tmp_path / "out" / "library-doc"
    precious.mkdir(parents=True)
    with engine.begin() as c:                                   # a corrupted row pointing outside _temp
        c.execute(update(td.TEMP_DOCS).values(source="../library-doc"))
    td.cleanup(engine, DeletingQdrant(), td.DEFAULT_TEMP_COLLECTION, tmp_path / "in", tmp_path / "out",
               existing_chat_ids=set())
    assert precious.exists()
    assert td._under(tmp_path / "out" / "_temp", tmp_path / "out" / "_temp" / "x") is True
    assert td._under(tmp_path / "out" / "_temp", tmp_path / "out" / "_temp" / ".." / "x") is False
    assert td._under(tmp_path / "out" / "_temp", tmp_path / "out" / "_temp") is False


def test_only_the_ready_files_are_searched():
    # a file still being read, or one that failed half-way, may have passages stored: never searched
    fake = FakeQdrant()
    q = td.ScopedQdrant(fake, CHAT_A, "u1", sources=["A-0123456789ab", "B-0123456789ab"])
    q.scroll(collection_name="c")
    q.query_points(collection_name="c", prefetch=[Prefetch(query=[0.1], using="dense", limit=3)])
    for _name, kw in fake.calls:
        for f in [kw.get("scroll_filter") or kw.get("query_filter")] + [p.filter for p in kw.get("prefetch") or []]:
            src = [c for c in f.must if isinstance(c, FieldCondition) and c.key == "source"]
            assert src and set(src[0].match.any) == {"A-0123456789ab", "B-0123456789ab"}
    with pytest.raises(ValueError):
        td.ScopedQdrant(FakeQdrant(), CHAT_A, "u1", sources=[])


def test_the_upload_callback_runs_for_each_file_of_a_removed_chat(engine, tmp_path):
    a = td.claim(engine, chat_id=CHAT_A, user_id="u1", file_id="f1", name="A.pdf")
    td.claim(engine, chat_id=CHAT_B, user_id="u2", file_id="f2", name="B.pdf")
    inp, out = _files(tmp_path, CHAT_A, a)
    seen = []
    td.cleanup(engine, DeletingQdrant(), td.DEFAULT_TEMP_COLLECTION, inp, out, existing_chat_ids={CHAT_B},
               remove_upload=lambda fid, uid: seen.append((fid, uid)))
    assert seen == [("f1", "u1")]
    seen.clear()                                                  # passages not deleted → nothing else either
    td.claim(engine, chat_id=CHAT_A, user_id="u1", file_id="f3", name="C.pdf")
    td.cleanup(engine, DeletingQdrant(fail=True), td.DEFAULT_TEMP_COLLECTION, inp, out, existing_chat_ids={CHAT_B},
               remove_upload=lambda fid, uid: seen.append(fid))
    assert seen == []
