# Memento Document Chat in the Query pipe. The safety properties: every search is limited to the chat and the user;
# nothing of the library is read or written (document list, picker, answer cache, "being processed", related documents);
# Memento Query never takes this path; background tasks read no file and search nothing; the local model answers.
import asyncio
import sys
import time
import types
from pathlib import Path

import pytest
from memento.config.paths import PROJECT_ROOT
from _owui_loader import load_function
from memento.query import cache, checking, generation, library_state, presentation, request, service, texts
from memento.query.scope import documents
from memento.query.flows import answer as answer_flow, summary
from memento.document_chat import chat, cleanup

ROOT = PROJECT_ROOT
sys.path.insert(0, str(ROOT))
qp = load_function(ROOT / "openwebui/functions/query_pipe.py", "memento_query")
from memento.document_chat import store as td  # noqa: E402
from memento.ingestion.upload import messages

CHAT, OTHER_CHAT = "3f2b8c1e-6a4d-4e0b-9c7a-1d2e3f4a5b6c", "9a8b7c6d-5e4f-4a3b-8c2d-1e0f9a8b7c6d"
DC_MODEL = "memento_query.document-chat"


@pytest.mark.parametrize("mid, yes", [(DC_MODEL, True), ("memento-document-chat", True), ("document-chat", True),
                                      ("memento_query.local", False), ("memento-query", False),
                                      ("memento_query.qwen35", False), ("memento-query-mistral", False), (None, False)])
def test_document_chat_model_ids(mid, yes):
    assert chat.is_doc_chat(mid) is yes


def _drive(pipe, body, user, metadata, files=None, task=None):
    events = []

    async def emitter(ev):
        events.append(ev)

    async def go():
        return [c async for c in pipe.pipe(body, user, emitter, files, metadata, task)]
    return "".join(asyncio.run(go())), events


@pytest.mark.parametrize("model", ["memento-query", "memento_query.local", "memento_query.mistral", None])
def test_memento_query_never_takes_the_document_chat_path(monkeypatch, model):
    pipe = qp.Pipe()
    seen = []

    async def fake_run(turn):
        seen.append(turn.doc_chat)
        yield "library answer"

    async def no_doc_chat(*a, **k):
        pytest.fail("Memento Query reached Document Chat")
        yield ""
    monkeypatch.setattr(service, "run", fake_run)
    monkeypatch.setattr(chat, "answer", no_doc_chat)
    text, _ = _drive(pipe, {"model": model, "messages": [{"role": "user", "content": "q"}]}, {"id": "u1"},
                     {"chat_id": CHAT}, files=[{"id": "f1"}])
    assert text == "library answer" and seen == [None]                   # no Document Chat scope: the library's paths


# ── a full Document Chat answer with fake services ─────────────────────────────

class FakeQdrant:
    """Records every call of every instance; returns two passages of the chat's file for a search."""
    calls: list = []

    def __init__(self, *a, **kw):
        pass

    def _rec(self, name, kw):
        FakeQdrant.calls.append((name, kw))

    def query_points(self, collection_name, query=None, using=None, prefetch=None, query_filter=None, **kw):
        self._rec("query_points", dict(collection_name=collection_name, query_filter=query_filter, prefetch=prefetch))
        wanted = _source_values(query_filter)              # a search of one file
        pts = [types.SimpleNamespace(id=i, score=0.9 - i / 10, payload=p) for i, p in enumerate(PASSAGES)
               if not wanted or p["source"] in wanted]
        return types.SimpleNamespace(points=pts)

    def query_points_groups(self, collection_name, group_by, query=None, using=None, prefetch=None, query_filter=None,
                            **kw):
        self._rec("query_points_groups", dict(collection_name=collection_name, query_filter=query_filter))
        return types.SimpleNamespace(groups=[])

    def scroll(self, collection_name, scroll_filter=None, limit=10, offset=None, **kw):
        self._rec("scroll", dict(collection_name=collection_name, scroll_filter=scroll_filter))
        wanted = _source_values(scroll_filter)            # a whole-document read asks for one document
        return [types.SimpleNamespace(id=i, payload=p) for i, p in enumerate(PASSAGES)
                if not wanted or p["source"] in wanted], None

    def count(self, collection_name, count_filter=None, exact=True, **kw):
        self._rec("count", dict(collection_name=collection_name, count_filter=count_filter))
        return types.SimpleNamespace(count=len(PASSAGES))

    def collection_exists(self, collection_name, **kw):
        return True


def _source_values(f) -> set:
    """The documents a filter pins with source == X (MatchValue), at any depth."""
    out = set()
    for c in (getattr(f, "must", None) or []):
        if getattr(c, "key", None) == "source" and hasattr(getattr(c, "match", None), "value"):
            out.add(c.match.value)
        elif hasattr(c, "must"):
            out |= _source_values(c)
    return out


SRC = "Report-0123456789ab"
PASSAGES = [{"text": "Silicon carbide reached 98 % density at 1900 °C.", "source": SRC, "page": 2, "type": "text",
             "section_raw": "3 Results", "chat_id": CHAT, "user_id": "u1"},
            {"text": "Grain size was 2 µm.", "source": SRC, "page": 3, "type": "text", "chat_id": CHAT,
             "user_id": "u1"}]


class FakeLLM:
    """Router call → a fixed route; streamed call → records the request and streams an answer citing [1]."""
    calls: list = []
    intent = "ranked_retrieval"
    query = "density of SiC in Report.pdf"          # the router's rewrite of the question

    def __init__(self, *a, **k):
        self.chat = self
        self.completions = self

    async def create(self, **kw):
        FakeLLM.calls.append(kw)
        if kw.get("stream"):
            text = "A title" if str(kw["messages"][-1].get("content", "")).startswith("### Task:") else "98 % [1]."

            async def gen():
                delta = types.SimpleNamespace(content=text, reasoning_content=None, reasoning=None)
                yield types.SimpleNamespace(choices=[types.SimpleNamespace(delta=delta)])
            return gen()
        msg = types.SimpleNamespace(content='{"intent":"%s","query":"%s","reasoning":"x"}'
                                    % (FakeLLM.intent, FakeLLM.query))
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


def _library_untouched(monkeypatch):
    import memento.library.answers as ac
    for module, name in ((library_state, "load_documents"), (library_state, "rebuild_documents"),
                         (documents, "publish_library"), (library_state, "in_progress_sources"), (library_state, "collection_status"),
                         (cache, "cache_lookup")):
        monkeypatch.setattr(module, name, lambda *a, _n=name, **k: pytest.fail(f"library touched: {_n}"))
    monkeypatch.setattr(ac, "put", lambda *a, **k: pytest.fail("answer stored"))
    monkeypatch.setattr(ac, "get", lambda *a, **k: pytest.fail("answer cache read"))


@pytest.fixture
def services(monkeypatch):
    import openai
    import qdrant_client
    import memento.retrieval.search as search
    import memento.common.model_servers as vc
    FakeQdrant.calls, FakeLLM.calls, FakeLLM.intent = [], [], "ranked_retrieval"
    FakeLLM.query = "density of SiC in Report.pdf"
    monkeypatch.setattr(qdrant_client, "QdrantClient", FakeQdrant)
    monkeypatch.setattr(openai, "AsyncOpenAI", FakeLLM)
    monkeypatch.setattr(vc, "vl_embed", lambda *a, **k: [0.0] * 8)
    monkeypatch.setattr(search, "rerank", lambda q, hits, *a, **k: hits)
    _library_untouched(monkeypatch)
    pipe = qp.Pipe()
    # the default copy: its rows are the ones claimed without a collection below
    pipe.valves = pipe.Valves(DOMAIN_NAME="generic", COLLECTION="memento-comp-mat-rag", HYBRID_ENABLED=False,
                              GRAPH_RETRIEVAL=False, VALIDATION_ENABLED=False, ENABLE_QUERY_REWRITE=True,
                              THINKING_AUTO=False, RELATED_DOCS=True, RERANK_ENABLED=True, LLM_MODEL="local-27b")
    monkeypatch.setattr(answer_flow, "related_heads", lambda *a, **k: pytest.fail("related documents searched"))
    OWNERS.clear()

    async def record(fid):                   # OpenWebUI's file table: every test file belongs to u1 unless set
        return types.SimpleNamespace(user_id=OWNERS.get(fid, "u1"), path=f"/uploads/{fid}.pdf")
    monkeypatch.setattr(chat, "file_record", record)
    return pipe


OWNERS: dict = {}


def _ready(chat=CHAT, user="u1", fid="f1", name="Report.pdf"):
    engine = qp._db_engine()
    td.ensure_tables(engine)
    src = td.claim(engine, chat_id=chat, user_id=user, file_id=fid, name=name)
    td.set_status(engine, chat, fid, "ready", pages=12)
    return src


def _scoped(f, chat=CHAT, user="u1") -> bool:
    from qdrant_client.models import FieldCondition
    keys = {(c.key, c.match.value) for c in (getattr(f, "must", None) or [])
            if isinstance(c, FieldCondition) and hasattr(c.match, "value")}
    ready = [c for c in (getattr(f, "must", None) or []) if isinstance(c, FieldCondition) and c.key == "source"
             and hasattr(c.match, "any")]
    return ("chat_id", chat) in keys and ("user_id", user) in keys and bool(ready)


def test_a_document_chat_answer_searches_only_this_chat_and_user(services, monkeypatch):
    src = _ready()
    PASSAGES[0]["source"] = PASSAGES[1]["source"] = src
    text, events = _drive(services, {"model": DC_MODEL, "messages": [{"role": "user", "content": "SiC density?"}]},
                          {"id": "u1"}, {"chat_id": CHAT}, files=[{"id": "f1", "name": "Report.pdf"}])
    assert "98 % [1]." in text
    assert FakeQdrant.calls, "no search was made"
    for name, kw in FakeQdrant.calls:                                  # nothing bypasses the scoped client
        assert kw["collection_name"] == "memento-comp-mat-temp", name
        f = kw.get("query_filter") or kw.get("scroll_filter") or kw.get("count_filter")
        assert _scoped(f), name
    answer = [c for c in FakeLLM.calls if c.get("stream")]
    assert answer and answer[-1]["model"] == "local-27b"               # the local model writes the answer
    names = [m["name"] for e in events if e.get("type") == "citation" for m in e["data"]["metadata"]]
    assert names and all("Report.pdf" in n and src not in n for n in names)   # the file's name, not its token
    assert request.SHOWN_NAMES.get() is None                                # the names do not outlive the answer


def test_the_cloud_model_is_never_used(services, monkeypatch):
    # the fake store's passages under the name this chat's file was claimed with: a search of the chat's one file stays
    # in it
    _ready()
    src = next(r["source"] for r in td.chat_documents(qp._db_engine(), CHAT, "u1") if r["file_id"] == "f1")
    monkeypatch.setitem(globals(), "PASSAGES", [dict(p, source=src) for p in PASSAGES])
    monkeypatch.setattr(generation, "client", lambda base, key, cloud: pytest.fail("cloud client") if cloud
                        else FakeLLM())
    body = {"model": DC_MODEL, "messages": [{"role": "user", "content": "SiC density?"}]}
    text, _ = _drive(services, body, {"id": "u1"}, {"chat_id": CHAT}, files=[{"id": "f1"}])
    assert "98 % [1]." in text


def test_another_user_or_chat_sees_nothing(services):
    _ready(chat=CHAT, user="u1")
    body = {"model": DC_MODEL, "messages": [{"role": "user", "content": "SiC density?"}]}
    text, _ = _drive(services, body, {"id": "u2"}, {"chat_id": CHAT}, files=[{"id": "f1"}])
    # the attach hint, with "ask it in Library Chat" for the question
    assert text.strip().endswith(chat.doc_chat_no_file("SiC density?")) or "was not added" in text
    assert not FakeQdrant.calls and "98 %" not in text
    text, _ = _drive(services, body, {"id": "u1"}, {"chat_id": OTHER_CHAT}, files=[])
    assert texts.DOC_CHAT_NO_FILE_MSG in text and not FakeQdrant.calls


@pytest.mark.parametrize("intent", ["chitchat", "general", "ranked_retrieval"])
def test_without_a_file_the_router_still_decides(services, intent):
    # chitchat in a Document Chat without a file is answered (greeting, or directly by the local model); only a question
    # that needs a file gets the attach instructions; nothing is searched
    FakeLLM.intent = intent
    body = {"model": DC_MODEL, "messages": [{"role": "user", "content": "hello"}]}
    text, _ = _drive(services, body, {"id": "u1"}, {"chat_id": CHAT}, files=[])
    assert not FakeQdrant.calls
    if intent in ("chitchat", "general"):   # greetings too go to the model directly
        answer = [c for c in FakeLLM.calls if c.get("stream")]
        assert len(answer) == 1 and answer[0]["model"] == "local-27b"         # Document Chat: the local model only
        system = answer[0]["messages"][0]["content"]
        assert "Document Chat" in system and "Upload Files" in system and "No file is attached" in system
        assert text.startswith("98 %") and texts.DOC_CHAT_NO_FILE_MSG not in text and "library" not in text
        assert (texts.DOC_CHAT_GENERAL_NOTE in text) == (intent == "general")    # no "no sources" note under a greeting
    else:                                   # the attach hint + "ask it in Library Chat"
        assert text.strip() == chat.doc_chat_no_file("hello")


def test_a_background_task_reads_no_file_and_searches_nothing(services, monkeypatch):
    monkeypatch.setattr(td, "claim", lambda *a, **k: pytest.fail("a file was claimed"))
    monkeypatch.setattr(chat, "doc_chat_read_one", lambda *a, **k: pytest.fail("a file was read"))
    body = {"model": DC_MODEL, "messages": [{"role": "user", "content": "### Task:\nGenerate a concise title."}]}
    text, _ = _drive(services, body, {"id": "u1"}, {"chat_id": CHAT}, files=[{"id": "f9", "name": "New.pdf"}],
                     task="title_generation")
    assert text == "A title" and not FakeQdrant.calls


def test_a_temporary_chat_is_refused(services, monkeypatch):
    monkeypatch.setattr(td, "claim", lambda *a, **k: pytest.fail("a file was claimed"))
    for cid in (None, "local", "local:1234"):
        text, _ = _drive(services, {"model": DC_MODEL, "messages": [{"role": "user", "content": "q"}]},
                         {"id": "u1"}, {"chat_id": cid}, files=[{"id": "f1"}])
        assert text == texts.DOC_CHAT_SAVED_CHAT_MSG


# ── reading the chat's files ──────────────────────────────────────────────────

def _fake_ingester(monkeypatch):
    import memento.document_chat.intake as ti
    monkeypatch.setattr(ti, "make_ingester", lambda *a, **k: types.SimpleNamespace())


def test_a_new_file_is_read_once_with_the_privacy_note(services, monkeypatch, tmp_path):
    _fake_ingester(monkeypatch)
    reads = []

    def read_one(ti, ing, path, source, max_pages, progress=None):
        reads.append(source)
        PASSAGES[0]["source"] = PASSAGES[1]["source"] = source
        return {"ok": True, "pages": 3, "vectors": 2}
    monkeypatch.setattr(chat, "doc_chat_read_one", read_one)
    upload = tmp_path / "up.pdf"
    upload.write_bytes(b"%PDF")
    files = [{"id": "f1", "name": "Report.pdf", "file": {"path": str(upload)}}]
    body = {"model": DC_MODEL, "messages": [{"role": "user", "content": "SiC density?"}]}
    text, _ = _drive(services, body, {"id": "u1"}, {"chat_id": CHAT}, files=files)
    assert text.startswith(f"_{texts.DOC_CHAT_PRIVACY}_") and "98 % [1]." in text and len(reads) == 1
    text, _ = _drive(services, body, {"id": "u1"}, {"chat_id": CHAT}, files=files)     # resent with the next message
    assert len(reads) == 1 and texts.DOC_CHAT_PRIVACY not in text and "98 % [1]." in text


@pytest.mark.parametrize("n", [1, 2])
def test_a_file_sent_without_a_question_ends_with_the_finished_bar(services, monkeypatch, tmp_path, n):
    # as after Library Upload, the finished bar stays above "Your file is ready."
    _fake_ingester(monkeypatch)
    monkeypatch.setattr(chat, "doc_chat_read_one", lambda *a, **k: {"ok": True, "pages": 3, "vectors": 2})
    files = [{"id": f"f{i}", "name": f"Report{i}.pdf"} for i in range(n)]
    body = {"model": DC_MODEL, "messages": [{"role": "user", "content": ""}]}
    text, events = _drive(services, body, {"id": "u1"}, {"chat_id": CHAT}, files=files)
    assert texts.DOC_CHAT_READY_MSG in text
    assert events[-1] == {"type": "status", "data": {"description": f"Done · {n} of {n} read · 0 unchanged · 0 failed "
                                                                   "· 0m elapsed", "done": True}}


def _statuses(events) -> list:
    return [e["data"] for e in events if e.get("type") == "status"]


def test_a_file_sent_with_a_question_shows_the_bar_then_the_answer_steps(services, monkeypatch):
    _fake_ingester(monkeypatch)
    monkeypatch.setattr(chat, "DOC_CHAT_STATUS_POLL_S", 0.01)

    def read_one(ti, ing, path, source, max_pages, progress=None):
        PASSAGES[0]["source"] = PASSAGES[1]["source"] = source
        progress("parse", pages=3)
        time.sleep(0.1)
        return {"ok": True, "pages": 3, "vectors": 2}
    monkeypatch.setattr(chat, "doc_chat_read_one", read_one)
    body = {"model": DC_MODEL, "messages": [{"role": "user", "content": "SiC density?"}]}
    text, events = _drive(services, body, {"id": "u1"}, {"chat_id": CHAT}, files=[{"id": "f1", "name": "Report.pdf"}])
    shown = _statuses(events)
    run = next(k for k, s in enumerate(shown)
               if s["description"].startswith("Uploading... 0 of 1 ready (1 parsing, 0 storing) · 0 read") and not s["done"])
    done = shown.index({"description": "Done · 1 of 1 read · 0 unchanged · 0 failed · 0m elapsed", "done": False})
    assert run < done < len(shown) - 1 and shown[-1].get("hidden") is True and "98 % [1]." in text
    assert not any(s["description"].startswith("Reading your file") for s in shown)


def test_the_bar_moves_while_a_long_file_is_read(services, monkeypatch):
    from memento.document_chat import intake as ti
    _fake_ingester(monkeypatch)
    monkeypatch.setattr(chat, "DOC_CHAT_STATUS_POLL_S", 0.01)
    monkeypatch.setattr(messages, "STATUS_EVERY_S", 0.05)

    def read_one(ti_, ing, path, source, max_pages, progress=None):
        progress("parse", pages=3)
        time.sleep(0.3)
        progress("embed", pages=3, pct=50)
        time.sleep(0.3)
        return {"ok": True, "pages": 3, "vectors": 2}
    monkeypatch.setattr(chat, "doc_chat_read_one", read_one)
    body = {"model": DC_MODEL, "messages": [{"role": "user", "content": ""}]}
    _, events = _drive(services, body, {"id": "u1"}, {"chat_id": CHAT}, files=[{"id": "f1", "name": "Report.pdf"}])
    lines = [s["description"] for s in _statuses(events) if s["description"].startswith("Uploading...")]
    assert sum("(1 parsing, 0 storing)" in x for x in lines) >= 2
    assert any("(0 parsing, 1 storing)" in x and x.endswith(" · 80% done") for x in lines)


@pytest.mark.parametrize("results, line", [
    ([{"ok": True, "pages": 3, "vectors": 2}, {"ok": False, "reason": "too_many_pages", "pages": 300}],
     "Done · 1 of 2 read · 0 unchanged · 1 failed · 0m elapsed"),
    ([{"ok": False, "reason": "too_many_pages", "pages": 300}], "Done · 0 of 1 read · 0 unchanged · 1 failed · 0m elapsed")])
def test_a_failed_file_is_counted_on_the_finished_bar(services, monkeypatch, results, line):
    _fake_ingester(monkeypatch)
    left = iter(results)
    monkeypatch.setattr(chat, "doc_chat_read_one", lambda *a, **k: next(left))
    files = [{"id": f"f{i}", "name": n} for i, n in enumerate(["A.pdf", "Big.pdf"][-len(results):])]
    body = {"model": DC_MODEL, "messages": [{"role": "user", "content": ""}]}
    text, events = _drive(services, body, {"id": "u1"}, {"chat_id": CHAT}, files=files)
    assert "**Big.pdf** was not added" in text
    assert events[-1] == {"type": "status", "data": {"description": line, "done": True}}


def test_the_bar_is_not_translated_for_a_german_question():
    from memento.document_chat import intake as ti
    bar = ti.ReadProgress(1)
    bar.update("parse", pages=3)
    running = bar.line()
    bar.finish(True)
    token = request.LANG.set("de")
    try:
        for line in (running, bar.line()):
            assert presentation.status_event(line, False)["data"]["description"] == line
    finally:
        request.LANG.reset(token)


def test_a_failed_file_is_reported_once(services, monkeypatch):
    _fake_ingester(monkeypatch)
    services.valves.DOC_CHAT_MAX_PAGES = 150                    # a limit an admin set
    monkeypatch.setattr(chat, "doc_chat_read_one",
                        lambda *a, **k: {"ok": False, "reason": "too_many_pages", "pages": 300})
    body = {"model": DC_MODEL, "messages": [{"role": "user", "content": "summary?"}]}
    files = [{"id": "f1", "name": "Big.pdf"}]
    text, _ = _drive(services, body, {"id": "u1"}, {"chat_id": CHAT}, files=files)
    assert "**Big.pdf** was not added" in text and "300 pages" in text and "150 pages" in text
    monkeypatch.setattr(chat, "doc_chat_read_one", lambda *a, **k: pytest.fail("read again"))
    text, _ = _drive(services, body, {"id": "u1"}, {"chat_id": CHAT}, files=files)
    assert "was not added" not in text and texts.DOC_CHAT_NO_FILE_MSG in text


def test_files_over_the_limit_are_not_read(services, monkeypatch):
    _fake_ingester(monkeypatch)
    services.valves.DOC_CHAT_MAX_FILES = 2
    reads = []
    monkeypatch.setattr(chat, "doc_chat_read_one", lambda ti, ing, path, src, mp, **k: reads.append(src) or
                        {"ok": True, "pages": 1})
    files = [{"id": f"f{k}", "name": f"D{k}.pdf"} for k in range(3)]
    body = {"model": DC_MODEL, "messages": [{"role": "user", "content": "q"}]}
    text, _ = _drive(services, body, {"id": "u1"}, {"chat_id": CHAT}, files=files)
    assert len(reads) == 2 and "**D2.pdf** was not added" in text and "up to 2 files" in text


def test_a_file_interrupted_while_reading_is_reported(services, monkeypatch):
    from datetime import datetime, timedelta, timezone
    from sqlalchemy import update
    engine = qp._db_engine()
    td.ensure_tables(engine)
    td.claim(engine, chat_id=CHAT, user_id="u1", file_id="f1", name="Old.pdf")
    with engine.begin() as c:
        c.execute(update(td.TEMP_DOCS).values(added_at=datetime.now(timezone.utc) - timedelta(hours=2)))
    body = {"model": DC_MODEL, "messages": [{"role": "user", "content": "q"}]}
    text, _ = _drive(services, body, {"id": "u1"}, {"chat_id": CHAT}, files=[{"id": "f1"}])
    assert "**Old.pdf** was not added" in text and "interrupted" in text
    td.claim(engine, chat_id=CHAT, user_id="u1", file_id="f2", name="Now.pdf")         # just started elsewhere
    text, _ = _drive(services, body, {"id": "u1"}, {"chat_id": CHAT}, files=[{"id": "f1"}, {"id": "f2"}])
    assert texts.DOC_CHAT_READING_MSG in text


@pytest.mark.parametrize("intent, question", [("summary", "Summarize Report.pdf"),
                                              ("gap_query", "What research gaps does Report.pdf name?")])
def test_summaries_and_research_gaps_are_never_stored_or_replayed(services, monkeypatch, intent, question):
    src = _ready()
    PASSAGES[0]["source"] = PASSAGES[1]["source"] = src
    FakeLLM.intent = intent
    monkeypatch.setattr(cache, "prepare", lambda *a, **k: pytest.fail("answer cache used"))
    body = {"model": DC_MODEL, "messages": [{"role": "user", "content": question}]}
    text, _ = _drive(services, body, {"id": "u1"}, {"chat_id": CHAT}, files=[{"id": "f1"}])
    assert [c for c in FakeLLM.calls if c.get("stream")], "no answer was written"
    assert FakeQdrant.calls and all(_scoped(kw.get("query_filter") or kw.get("scroll_filter")
                                            or kw.get("count_filter")) for _n, kw in FakeQdrant.calls)
    assert any(n == "scroll" for n, _kw in FakeQdrant.calls)          # the document's own passages were read


# ── clean-up from the worker ─────────────────────────────────────────

def test_the_worker_cleans_up_chats_that_are_gone(monkeypatch, tmp_path):
    from sqlalchemy import text
    engine = qp._db_engine()
    with engine.begin() as c:
        c.execute(text("CREATE TABLE chat (id TEXT PRIMARY KEY)"))
        c.execute(text("INSERT INTO chat (id) VALUES (:i)"), {"i": OTHER_CHAT})
    td.ensure_tables(engine)
    td.claim(engine, chat_id=CHAT, user_id="u1", file_id="f1", name="A.pdf")
    td.claim(engine, chat_id=OTHER_CHAT, user_id="u1", file_id="f2", name="B.pdf")
    deleted = []
    import qdrant_client
    monkeypatch.setattr(qdrant_client, "QdrantClient", lambda *a, **k: types.SimpleNamespace(
        delete=lambda collection_name, points_selector: deleted.append(collection_name)))
    monkeypatch.setenv("INPUT_ROOT", str(tmp_path / "in"))
    pipe = qp.Pipe()
    pipe.valves.COLLECTION = "memento-comp-mat-rag"              # today's copy: the rows claimed above are its own
    assert cleanup.doc_chat_cleanup(pipe.valves, engine, function_id=pipe.function_id, db_engine=pipe.engine) == 1 \
        and deleted == ["memento-comp-mat-temp"]
    assert td.chat_documents(engine, CHAT, "u1") == [] and td.chat_documents(engine, OTHER_CHAT, "u1")


def test_no_clean_up_without_the_chat_table(monkeypatch):
    engine = qp._db_engine()
    td.ensure_tables(engine)
    td.claim(engine, chat_id=CHAT, user_id="u1", file_id="f1", name="A.pdf")
    import qdrant_client
    monkeypatch.setattr(qdrant_client, "QdrantClient", lambda *a, **k: types.SimpleNamespace(
        delete=lambda **kw: pytest.fail("deleted without knowing the chats")))
    assert cleanup.existing_chat_ids(engine) is None and cleanup.doc_chat_cleanup(
        qp.Pipe().valves, engine, function_id=qp._OWN_ID, db_engine=qp._db_engine) == 0
    assert td.chat_documents(engine, CHAT, "u1")


def test_a_file_of_another_user_is_never_read(services, monkeypatch):
    # a crafted request naming someone else's file id: not claimed, not read, no row
    _fake_ingester(monkeypatch)
    OWNERS["f1"] = "u2"
    monkeypatch.setattr(chat, "doc_chat_read_one", lambda *a, **k: pytest.fail("someone else's file was read"))
    body = {"model": DC_MODEL, "messages": [{"role": "user", "content": "q"}]}
    text, _ = _drive(services, body, {"id": "u1"}, {"chat_id": CHAT},
                     files=[{"id": "f1", "name": "Theirs.pdf", "file": {"path": "/elsewhere/x.pdf"}}])
    assert texts.DOC_CHAT_NO_FILE_MSG in text and td.chat_documents(qp._db_engine(), CHAT, "u1") == []


def test_the_stored_path_is_read_not_the_one_in_the_request(services, monkeypatch):
    _fake_ingester(monkeypatch)
    paths = []
    monkeypatch.setattr(chat, "doc_chat_read_one", lambda ti, ing, path, src, mp, **k: paths.append(path) or
                        {"ok": True, "pages": 1})
    body = {"model": DC_MODEL, "messages": [{"role": "user", "content": "q"}]}
    _drive(services, body, {"id": "u1"}, {"chat_id": CHAT},
           files=[{"id": "f1", "name": "Mine.pdf", "file": {"path": "/etc/passwd"}}])
    assert paths == ["/uploads/f1.pdf"]


def test_a_file_still_being_read_is_not_searched(services):
    # the chat has a ready file and one still being read (a parallel request): only the ready one is searched
    ready = _ready()
    td.claim(qp._db_engine(), chat_id=CHAT, user_id="u1", file_id="f2", name="Half.pdf")
    PASSAGES[0]["source"] = PASSAGES[1]["source"] = ready
    body = {"model": DC_MODEL, "messages": [{"role": "user", "content": "SiC density?"}]}
    _drive(services, body, {"id": "u1"}, {"chat_id": CHAT}, files=[{"id": "f1"}, {"id": "f2"}])
    from qdrant_client.models import FieldCondition
    for _n, kw in FakeQdrant.calls:
        f = kw.get("query_filter") or kw.get("scroll_filter") or kw.get("count_filter")
        src = [c for c in f.must if isinstance(c, FieldCondition) and c.key == "source" and hasattr(c.match, "any")]
        assert src[0].match.any == [ready]


# ── OpenWebUI's own copy of the upload (OpenWebUI keeps file rows and uploads when a chat is deleted) ─────────────

def _owui_files(engine, tmp_path, rows, links=(), knowledge=()):
    from sqlalchemy import text
    up = tmp_path / "uploads"
    up.mkdir(exist_ok=True)
    with engine.begin() as c:
        c.execute(text("CREATE TABLE file (id TEXT PRIMARY KEY, user_id TEXT, path TEXT)"))
        c.execute(text("CREATE TABLE chat_file (chat_id TEXT, file_id TEXT)"))
        c.execute(text("CREATE TABLE knowledge_file (knowledge_id TEXT, file_id TEXT)"))
        for fid, uid, path in rows:
            c.execute(text("INSERT INTO file VALUES (:i, :u, :p)"), {"i": fid, "u": uid, "p": str(path)})
        for cid, fid in links:
            c.execute(text("INSERT INTO chat_file VALUES (:c, :f)"), {"c": cid, "f": fid})
        for kid, fid in knowledge:
            c.execute(text("INSERT INTO knowledge_file VALUES (:k, :f)"), {"k": kid, "f": fid})
    return up


def _file_ids(engine):
    from sqlalchemy import text
    with engine.connect() as c:
        return {r[0] for r in c.execute(text("SELECT id FROM file"))}


def test_the_upload_is_removed_with_its_chat(tmp_path):
    engine = qp._db_engine()
    up = tmp_path / "uploads"
    up.mkdir()
    (up / "f1_A.pdf").write_bytes(b"%PDF")
    _owui_files(engine, tmp_path, [("f1", "u1", up / "f1_A.pdf")])
    assert cleanup.remove_owui_upload(engine, "f1", "u1", upload_dir=up) is True
    assert _file_ids(engine) == set() and not (up / "f1_A.pdf").exists()


@pytest.mark.parametrize("case", ["other owner", "still in a chat", "in a knowledge base", "outside uploads"])
def test_an_upload_still_in_use_or_not_ours_is_kept(tmp_path, case):
    engine = qp._db_engine()
    up = tmp_path / "uploads"
    up.mkdir()
    path = (tmp_path / "elsewhere.pdf") if case == "outside uploads" else (up / "f1_A.pdf")
    path.write_bytes(b"%PDF")
    _owui_files(engine, tmp_path, [("f1", "u2" if case == "other owner" else "u1", path)],
                links=[(OTHER_CHAT, "f1")] if case == "still in a chat" else (),
                knowledge=[("k1", "f1")] if case == "in a knowledge base" else ())
    cleanup.remove_owui_upload(engine, "f1", "u1", upload_dir=up)
    assert path.exists()
    if case != "outside uploads":
        assert _file_ids(engine) == {"f1"}


def test_thorough_check_searches_only_this_chat_too(services, monkeypatch):
    # "Thorough check" re-searches while it checks (memento.retrieval.search.search): through the same scoped client
    from memento.agentic_validation import strategies
    from memento.agentic_validation.strategies import ArmResult
    src = _ready()
    PASSAGES[0]["source"] = PASSAGES[1]["source"] = src
    services.valves.VALIDATION_ENABLED = True
    found = {}

    def fake(name, ctx, *, p1_answer=None, max_rounds=2):
        found["hits"] = ctx.retrieve_fn("grain size of SiC")
        return ArmResult("A", "Checked 98 % [1].", hits=ctx.hits, trace=[{"round": 0, "failing": {}}], rounds=0)
    monkeypatch.setattr(strategies, "run_strategy", fake)
    body = {"model": DC_MODEL, "messages": [{"role": "user", "content": "SiC density?"}]}
    text, _ = _drive(services, body, {"id": "u1"}, {"chat_id": CHAT, "filter_ids": ["memento_thorough"]},
                     files=[{"id": "f1"}])
    assert "Checked 98 % [1]." in text and found["hits"]
    assert len(FakeQdrant.calls) >= 2 and all(
        _scoped(kw.get("query_filter") or kw.get("scroll_filter") or kw.get("count_filter"))
        and kw["collection_name"] == "memento-comp-mat-temp" for _n, kw in FakeQdrant.calls)


def test_only_document_chat_chats_are_looked_up():
    from sqlalchemy import text
    engine = qp._db_engine()
    with engine.begin() as c:
        c.execute(text("CREATE TABLE chat (id TEXT PRIMARY KEY)"))
        for i in (CHAT, "unrelated-chat"):
            c.execute(text("INSERT INTO chat (id) VALUES (:i)"), {"i": i})
    assert cleanup.existing_chat_ids(engine, {CHAT, OTHER_CHAT}) == {CHAT}
    assert cleanup.existing_chat_ids(engine, set()) == set()


def test_the_only_file_counts_as_named(services, monkeypatch):
    # with one ready file, that file is the named document, so the whole-document summary reads all of it (not only the
    # best passages)
    src = _ready()
    PASSAGES[0]["source"] = PASSAGES[1]["source"] = src
    FakeLLM.intent, FakeLLM.query = "summary", "Summarize the attached file."   # as the router keeps it
    seen = {}
    real = summary.summary_hits

    def spy(passages, *a, **k):
        seen["n"] = len(passages)
        return real(passages, *a, **k)
    monkeypatch.setattr(summary, "summary_hits", spy)
    body = {"model": DC_MODEL, "messages": [{"role": "user", "content": "Summarize the attached file."}]}
    _drive(services, body, {"id": "u1"}, {"chat_id": CHAT}, files=[{"id": "f1", "name": "Report.pdf"}])
    assert seen.get("n"), "the whole-document summary was not used"



def test_several_files_are_summarized_one_by_one(services, monkeypatch):
    # "Summarize the file" with several files: each file from all its passages, under its own heading; one list of
    # sources with global numbers, so [N] stays right and a follow-up (a rework) has every source
    a = _ready(fid="f1", name="Report.pdf")
    b = _ready(fid="f2", name="Notes.pdf")
    saved = list(PASSAGES)
    PASSAGES[:] = [{"text": "Silicon carbide reached 98 % density.", "source": a, "page": 2, "type": "text",
                    "chat_id": CHAT, "user_id": "u1"},
                   {"text": "Grain size was 2 µm.", "source": a, "page": 3, "type": "text", "chat_id": CHAT,
                    "user_id": "u1"},
                   {"text": "The notes list three furnaces.", "source": b, "page": 0, "type": "text",
                    "chat_id": CHAT, "user_id": "u1"}]
    try:
        FakeLLM.intent, FakeLLM.query = "summary", "Summarize the file"
        text, events = _drive(services, {"model": DC_MODEL, "messages": [{"role": "user", "content": "Summarize the file"}]},
                              {"id": "u1"}, {"chat_id": CHAT}, files=[{"id": "f1"}, {"id": "f2"}])
    finally:
        PASSAGES[:] = saved
    answers = [c for c in FakeLLM.calls if c.get("stream")]
    assert len(answers) == 2                                          # one summary per file
    first, second = (c["messages"][-1]["content"] for c in answers)
    first, second = (x if isinstance(x, str) else x[0]["text"] for x in (first, second))
    assert "[1]" in first and "[2]" in first and "furnaces" not in first
    assert "[3]" in second and "furnaces" in second and "[1]" not in second    # global numbers, its own passages
    cards = [e for e in events if e.get("type") == "citation"]
    assert len(cards) == 3 and cards[2]["data"]["metadata"][0]["name"].startswith("[3]")
    assert text.index("Report.pdf") < text.index("Notes.pdf")          # a heading per file, in the chat's order
    assert text.count("Answered by") == 1                              # one footer for the whole answer


def test_thorough_check_checks_each_file_summary(services, monkeypatch):
    a = _ready(fid="f1", name="Report.pdf")
    b = _ready(fid="f2", name="Notes.pdf")
    saved = list(PASSAGES)
    PASSAGES[:] = [{"text": "Silicon carbide reached 98 % density.", "source": a, "page": 2, "type": "text",
                    "chat_id": CHAT, "user_id": "u1"},
                   {"text": "The notes list three furnaces.", "source": b, "page": 0, "type": "text",
                    "chat_id": CHAT, "user_id": "u1"}]
    checked = []

    async def fake_check(*, answer, hits, **kw):
        checked.append((answer, len(hits)))
        return answer + " (checked)", None, None, False
    monkeypatch.setattr(checking, "thorough_check", lambda valves, *a, **k: fake_check(*a, **k))
    services.valves.VALIDATION_ENABLED = True
    try:
        FakeLLM.intent, FakeLLM.query = "summary", "Summarize the file"
        text, _ = _drive(services, {"model": DC_MODEL, "messages": [{"role": "user", "content": "Summarize the file"}]},
                         {"id": "u1"}, {"chat_id": CHAT, "filter_ids": ["memento_thorough"]},
                         files=[{"id": "f1"}, {"id": "f2"}])
    finally:
        PASSAGES[:] = saved
    assert len(checked) == 2 and all(n == 2 for _, n in checked)       # each summary, against all numbered sources
    assert text.count("(checked)") == 2 and text.count("Answered by") == 1



def test_a_question_about_several_files_covers_each_of_them(services, monkeypatch):
    # the buttons are shortcuts for what people type; a question about "the attached file" with several files means all
    # of them: each file is searched on its own, gets an equal share of the passages, and the model is told to cover
    # each file
    a = _ready(fid="f1", name="Report.pdf")
    b = _ready(fid="f2", name="Notes.pdf")
    saved = list(PASSAGES)
    PASSAGES[:] = [{"text": f"Report finding {i}.", "source": a, "page": i, "type": "text", "chat_id": CHAT,
                    "user_id": "u1"} for i in range(30)] \
        + [{"text": "Notes finding.", "source": b, "page": 0, "type": "text", "chat_id": CHAT, "user_id": "u1"}]
    try:
        FakeLLM.intent, FakeLLM.query = "ranked_retrieval", "What are the main findings of the attached file?"
        text, events = _drive(services, {"model": DC_MODEL, "messages": [
            {"role": "user", "content": "What are the main findings of the attached file?"}]},
            {"id": "u1"}, {"chat_id": CHAT}, files=[{"id": "f1"}, {"id": "f2"}])
    finally:
        PASSAGES[:] = saved
    searched = [_source_values(kw["query_filter"]) for name, kw in FakeQdrant.calls if name == "query_points"]
    assert {a} in searched and {b} in searched                         # each file searched on its own
    answer = [c for c in FakeLLM.calls if c.get("stream")][-1]
    user = answer["messages"][-1]["content"]
    user = user if isinstance(user, str) else user[0]["text"]
    assert "Notes finding." in user and "Report finding 0." in user   # both files reach the model
    assert "all 2 attached files" in user and "Report.pdf" in user and "Notes.pdf" in user



def test_each_files_best_passages_count_and_summaries_do_not_think(services, monkeypatch):
    # each file keeps its share even when the reranker's minimum score would drop all its passages; and the per-file
    # summaries of "main findings" run without deep reasoning
    import memento.retrieval.search as search
    floors = []

    def rec(q, hits, url, top, min_score, *a, **k):
        floors.append(min_score)
        return hits
    monkeypatch.setattr(search, "rerank", rec)
    a = _ready(fid="f1", name="Report.pdf")
    b = _ready(fid="f2", name="Notes.pdf")
    saved = list(PASSAGES)
    PASSAGES[:] = [{"text": "Report text.", "source": a, "page": 0, "type": "text", "chat_id": CHAT, "user_id": "u1"},
                   {"text": "Notes text.", "source": b, "page": 0, "type": "text", "chat_id": CHAT, "user_id": "u1"}]
    services.valves.ENABLE_THINKING = True
    try:
        FakeLLM.intent, FakeLLM.query = "ranked_retrieval", "Which methods does the attached file use?"
        _drive(services, {"model": DC_MODEL, "messages": [{"role": "user", "content": "Which methods?"}]},
               {"id": "u1"}, {"chat_id": CHAT}, files=[{"id": "f1"}, {"id": "f2"}])
        assert floors and floors[-1] == 0.0                              # every file keeps its best passages
        FakeLLM.calls.clear()
        FakeLLM.intent, FakeLLM.query = "summary", "Summarize the file"
        _drive(services, {"model": DC_MODEL, "messages": [{"role": "user", "content": "Summarize the file"}]},
               {"id": "u1"}, {"chat_id": CHAT}, files=[{"id": "f1"}, {"id": "f2"}])
    finally:
        PASSAGES[:] = saved
    answers = [c for c in FakeLLM.calls if c.get("stream")]
    assert len(answers) == 2
    for c in answers:                                                    # summaries: no thinking
        assert c["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False


def test_document_chat_has_no_page_limit_by_default(monkeypatch):
    # 0 = no limit; a read still ends at MinerU's 30 minutes
    monkeypatch.delenv("DOC_CHAT_MAX_PAGES", raising=False)
    assert qp.Pipe.Valves.model_fields["DOC_CHAT_MAX_PAGES"].default == 0
