# Document Chat with two libraries in one Open WebUI: a chat belongs to one library (a chat with files in the other
# library's Document Chat is refused before anything is read or searched), a second copy's file read runs in one thread
# holding its library's lifecycle lock and rechecks that the copy is still enabled, and every storage call names the
# copy's own temporary collection.
import ast
import asyncio
import sys
import threading
import types
from pathlib import Path

import pytest
from memento.config.paths import PROJECT_ROOT
from memento.query import library_state, texts
from memento.document_chat import chat, cleanup

ROOT = PROJECT_ROOT
sys.path.insert(0, str(ROOT))
from memento.library import copies as cs  # noqa: E402
from memento.document_chat import store as td  # noqa: E402
from test_query_pipe_document_chat import CHAT, FakeLLM, FakeQdrant, PASSAGES, _drive  # noqa: E402
from test_two_libraries_worker import QUERY, load_copy  # noqa: E402

COMP, POLY = td.DEFAULT_TEMP_COLLECTION, "memento-polymer-temp"
STUDY_DC = "memento_query_polymer.document-chat"


@pytest.fixture
def study(monkeypatch):
    """The second copy (module function_memento_query_polymer) with fake services; enabled unless the test says not."""
    import openai
    import qdrant_client
    import memento.retrieval.search as search
    import memento.common.model_servers as vc
    import memento.document_chat.intake as ti
    mod = load_copy(QUERY, "memento_query_polymer", monkeypatch)
    FakeQdrant.calls, FakeLLM.calls, FakeLLM.intent = [], [], "ranked_retrieval"
    monkeypatch.setattr(qdrant_client, "QdrantClient", FakeQdrant)
    monkeypatch.setattr(openai, "AsyncOpenAI", FakeLLM)
    monkeypatch.setattr(vc, "vl_embed", lambda *a, **k: [0.0] * 8)
    monkeypatch.setattr(search, "rerank", lambda q, hits, *a, **k: hits)
    monkeypatch.setattr(ti, "make_ingester", lambda *a, **k: types.SimpleNamespace())
    state = {"admitted": True, "fresh": True}

    def copy_state(role="query", *, cached=True, function_id=None, db_engine=None):
        return cs.CopyState(True, state["admitted"] if cached else state["fresh"], "" if state["fresh"] else "off")
    monkeypatch.setattr(library_state, "copy_state", copy_state)

    async def record(fid):
        return types.SimpleNamespace(user_id="u1", path=f"/uploads/{fid}.pdf")
    monkeypatch.setattr(chat, "file_record", record)
    pipe = mod.Pipe()
    pipe.valves = pipe.Valves(DOMAIN_NAME="polymer-engineering", COLLECTION="memento-polymer-rag", HYBRID_ENABLED=False,
                              GRAPH_RETRIEVAL=False, VALIDATION_ENABLED=False, ENABLE_QUERY_REWRITE=True,
                              THINKING_AUTO=False, RELATED_DOCS=False, RERANK_ENABLED=True, LLM_MODEL="m",
                              QDRANT_URL="http://127.0.0.1:9")
    engine = mod._db_engine()
    td.ensure_tables(engine)
    return types.SimpleNamespace(mod=mod, pipe=pipe, state=state, engine=engine)


def _claim(engine, coll, fid="f1", chat=CHAT, status="ready"):
    src = td.claim(engine, chat_id=chat, user_id="u1", file_id=fid, name=f"{fid}.pdf", collection=coll)
    td.set_status(engine, chat, fid, status, pages=3)
    return src


def _ask(s, files=None, content="What does the file say?", model=STUDY_DC, pipe=None):
    body = {"model": model, "messages": [{"role": "user", "content": content}]}
    return _drive(pipe or s.pipe, body, {"id": "u1", "role": "user"}, {"chat_id": CHAT}, files=files)[0]


def _nothing_done(s):
    assert not [c for c in FakeQdrant.calls if c[0] in ("query_points", "query_points_groups", "scroll", "count")]
    assert not [c for c in FakeLLM.calls if c.get("stream")]


# ── one library per chat (R2) ─────────────────────────────────────────────────

@pytest.mark.parametrize("files", [[{"id": "f1", "name": "f1.pdf"}],              # the same attachment again
                                   [{"id": "f2", "name": "f2.pdf"}],              # another attachment
                                   None])                                         # no attachment
def test_a_chat_with_comp_mat_files_is_refused_by_the_study_document_chat(study, monkeypatch, files):
    _claim(study.engine, COMP)
    monkeypatch.setattr(chat, "doc_chat_read_one", lambda *a, **k: pytest.fail("a file was read"))
    out = _ask(study, files)
    assert out.strip() == texts.DOC_CHAT_OTHER_LIBRARY_MSG
    _nothing_done(study)
    assert td.chat_documents(study.engine, CHAT, "u1", collection=POLY) == []
    assert td.chat_collections(study.engine, CHAT) == {COMP}


def test_the_refusal_has_a_formal_german_twin(study):
    _claim(study.engine, COMP)
    out = _ask(study, content="Was steht in der Datei über die Dichte?")
    de = texts.DE_FIXED[texts.DOC_CHAT_OTHER_LIBRARY_MSG]
    assert out.strip() == de and "Sie" in de
    assert "—" not in de and "—" not in texts.DOC_CHAT_OTHER_LIBRARY_MSG


def test_todays_document_chat_refuses_a_chat_with_study_files(monkeypatch):
    from test_query_pipe_document_chat import qp
    import openai
    import qdrant_client
    FakeQdrant.calls, FakeLLM.calls = [], []
    monkeypatch.setattr(qdrant_client, "QdrantClient", FakeQdrant)
    monkeypatch.setattr(openai, "AsyncOpenAI", FakeLLM)
    pipe = qp.Pipe()
    pipe.valves = pipe.Valves(DOMAIN_NAME="generic", COLLECTION="memento-comp-mat-rag", QDRANT_URL="http://127.0.0.1:9")
    engine = qp._db_engine()
    td.ensure_tables(engine)
    _claim(engine, POLY)
    monkeypatch.setattr(chat, "doc_chat_read_one", lambda *a, **k: pytest.fail("a file was read"))
    out = _drive(pipe, {"model": "memento_query.document-chat", "messages": [{"role": "user", "content": "q?"}]},
                 {"id": "u1"}, {"chat_id": CHAT}, files=[{"id": "f2", "name": "f2.pdf"}])[0]
    assert out.strip() == texts.DOC_CHAT_OTHER_LIBRARY_MSG
    assert td.chat_documents(engine, CHAT, "u1", collection=COMP) == []


def test_a_study_chat_goes_on_in_its_own_library(study, monkeypatch):
    src = _claim(study.engine, POLY)
    PASSAGES[0]["source"] = PASSAGES[1]["source"] = src
    out = _ask(study)
    assert "98 % [1]." in out
    searched = {c[1]["collection_name"] for c in FakeQdrant.calls if c[0] == "query_points"}
    assert searched == {POLY}


def test_a_late_claim_in_the_other_library_is_refused_too(study, monkeypatch):
    # the entry check saw no rows, then the other library's first file arrived before this one's claim (a race)
    real = td.chat_collections
    monkeypatch.setattr(td, "chat_collections", lambda e, chat: set())            # the entry check passes
    _claim(study.engine, COMP, fid="f9")                                          # … but comp-mat claimed first
    monkeypatch.setattr(chat, "doc_chat_read_one", lambda *a, **k: pytest.fail("a file was read"))
    out = _ask(study, [{"id": "f1", "name": "f1.pdf"}])
    assert out.strip() == texts.DOC_CHAT_OTHER_LIBRARY_MSG
    _nothing_done(study)
    assert real(study.engine, CHAT) == {COMP}


# ── the read: one thread, the lock, the recheck ───────────────────────────────

def test_the_whole_read_runs_in_one_thread_outside_the_event_loop(study, monkeypatch):
    seen = {}
    real_claim, real_status = td.claim, td.set_status
    monkeypatch.setattr(td, "claim", lambda *a, **k: seen.setdefault("claim", threading.get_ident()) and None
                        or real_claim(*a, **k))
    monkeypatch.setattr(td, "set_status", lambda *a, **k: seen.setdefault("status", threading.get_ident()) and None
                        or real_status(*a, **k))

    def read_one(ti, ing, path, src, max_pages, progress=None):
        seen["read"] = threading.get_ident()
        PASSAGES[0]["source"] = PASSAGES[1]["source"] = src
        return {"ok": True, "pages": 3, "vectors": 2}
    monkeypatch.setattr(chat, "doc_chat_read_one", read_one)
    loop_thread = {}

    async def emitter(ev):
        loop_thread.setdefault("id", threading.get_ident())
    body = {"model": STUDY_DC, "messages": [{"role": "user", "content": "What does the file say?"}]}

    async def go():
        return [c async for c in study.pipe.pipe(body, {"id": "u1", "role": "user"}, emitter,
                                                 [{"id": "f1", "name": "f1.pdf"}], {"chat_id": CHAT})]
    out = "".join(c for c in asyncio.run(go()) if isinstance(c, str))
    assert "98 % [1]." in out
    assert seen["claim"] == seen["read"] == seen["status"] != loop_thread["id"]
    assert [r["status"] for r in td.chat_documents(study.engine, CHAT, "u1", collection=POLY)] == ["ready"]


def test_the_study_read_holds_its_librarys_lifecycle_lock(study, monkeypatch):
    held = []
    real = td.lifecycle

    def lifecycle(engine, collection):
        held.append(collection)
        return real(engine, collection)
    monkeypatch.setattr(td, "lifecycle", lifecycle)
    monkeypatch.setattr(chat, "doc_chat_read_one", lambda *a, **k: {"ok": True, "pages": 1, "vectors": 1})
    _ask(study, [{"id": "f1", "name": "f1.pdf"}])
    assert held == [POLY]


def test_a_copy_switched_off_while_the_request_waited_reads_nothing(study, monkeypatch):
    # admitted with the cached state; the read's own fresh check after taking the lock finds the copy off
    study.state["fresh"] = False
    monkeypatch.setattr(chat, "doc_chat_read_one", lambda *a, **k: pytest.fail("a file was read"))
    out = _ask(study, [{"id": "f1", "name": "f1.pdf"}])
    assert out.strip() == texts.STUDY_NOT_READY_MSG
    assert td.chat_documents(study.engine, CHAT, "u1", collection=POLY) == []
    _nothing_done(study)


def test_todays_read_takes_no_lock_and_no_recheck(monkeypatch):
    from memento.library import locks as db_locks
    from test_query_pipe_document_chat import qp
    import memento.document_chat.intake as ti
    monkeypatch.setattr(db_locks, "temp_lifecycle_shared", lambda *a, **k: pytest.fail("today's read took a lock"))
    monkeypatch.setattr(library_state, "copy_state", lambda *a, **k: pytest.fail("today's read rechecked"))
    monkeypatch.setattr(ti, "make_ingester", lambda *a, **k: types.SimpleNamespace())
    monkeypatch.setattr(chat, "doc_chat_read_one", lambda *a, **k: {"ok": True, "pages": 1, "vectors": 1})

    async def record(fid):
        return types.SimpleNamespace(user_id="u1", path=f"/uploads/{fid}.pdf")
    monkeypatch.setattr(chat, "file_record", record)
    pipe = qp.Pipe()
    pipe.valves = pipe.Valves(COLLECTION="memento-comp-mat-rag", QDRANT_URL="http://127.0.0.1:9")
    engine = qp._db_engine()
    td.ensure_tables(engine)

    async def status(*a, **k):
        return None
    notes, read, outcome, _ = asyncio.run(chat.doc_chat_read(pipe.valves, engine, td, CHAT, "u1", [{"id": "f1", "name": "a.pdf"}],
                                                           status, __import__("logging").getLogger("t"),
                                                           function_id=qp._OWN_ID, db_engine=qp._db_engine,
                                                           saved_valves=lambda fid: library_state.saved_valves(fid, qp._db_engine)))
    assert (read, outcome) == (1, "ok")
    assert [r["collection"] for r in td.chat_documents(engine, CHAT, "u1")] == [COMP]


# ── the cleaner: the lock, the recheck, its own library ───────────────────────

def test_the_study_cleaner_rechecks_after_the_lock_and_skips_when_switched_off(study, monkeypatch):
    calls = []
    monkeypatch.setattr(td, "cleanup", lambda engine, q, coll, *a, **k: calls.append(coll) or 0)
    monkeypatch.setattr(library_state, "saved_valves", lambda *a, **k: {})
    clean = lambda: cleanup.doc_chat_cleanup(study.pipe.valves, study.engine, function_id=study.pipe.function_id,
                                             db_engine=study.pipe.engine)
    study.state["fresh"] = False
    assert clean() == 0 and calls == []
    study.state["fresh"] = True
    clean()
    assert calls == [POLY]


# ── every storage call names the copy's collection ────────────────────────────

def test_every_document_chat_storage_call_in_the_pipe_names_the_collection():
    trees = [ast.parse((ROOT / path).read_text()) for path in ("openwebui/functions/query_pipe.py",
                                                               "memento/document_chat/chat.py")]
    names = {"claim", "chat_documents", "orphans", "remove_rows"}
    calls = []
    for n in (n for tree in trees for n in ast.walk(tree)):
        if isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Attribute) and f.attr in names and getattr(f.value, "id", None) == "td":
                calls.append(n)
            # asyncio.to_thread(td.chat_documents, …)
            if isinstance(f, ast.Attribute) and f.attr == "to_thread" and n.args \
                    and isinstance(n.args[0], ast.Attribute) and n.args[0].attr in names:
                calls.append(n)
    assert len(calls) >= 3
    for c in calls:
        assert "collection" in {k.arg for k in c.keywords}, ast.unparse(c)[:160]
