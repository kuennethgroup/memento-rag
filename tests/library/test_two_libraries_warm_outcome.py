# A warm-up reports what really happened to its answer. The pipe writes the outcome into metadata["warmup_result"]:
# "hit", "stored" (only when the answer cache really wrote or kept it), "ineligible" with a reason (never retried), or
# "error" (retried). The live picker and the pre-warm status share one key builder and one replay decision, each with
# its own request (model, Thorough check, Regenerate).
import asyncio
import sys
from pathlib import Path

import pytest
from sqlalchemy import text
from memento.config.paths import PROJECT_ROOT
from memento.query import cache, library_state, service, texts, workers

ROOT = PROJECT_ROOT
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "openwebui/functions"))
from memento.library import answers as ac  # noqa: E402
from memento.library import documents as ldb  # noqa: E402
from memento.library import warmup_queue as wq  # noqa: E402

COLL = "memento-polymer-rag"


@pytest.fixture
def qp():
    import query_pipe
    return query_pipe


@pytest.fixture
def engine(qp):
    e = qp._db_engine()
    ldb.ensure_tables(e)
    ac.ensure_table(e)
    ldb.add_document(e, collection=COLL, entry={"source": "doc", "title": "Doc"}, content_hash="h1",
                     uploaded_by="u", output_root=None)
    return e


def _fields(**over):
    f = dict(collection=COLL, source="doc", content_hash="h1", action="summary", language="en", model="local:m",
             mode="whole", prompt_version="p")
    f.update(over)
    return f


# ── answer_cache.put says what it did ─────────────────────────────────────────

def test_put_reports_whether_it_wrote(engine):
    f = _fields()
    k = ac.cache_key(**f)
    assert ac.put(engine, k, fields=f, answer="a", citations=[], generated_by="m") == "written"
    assert ac.put(engine, k, fields=f, answer="b", citations=[], generated_by="m", replace=False) == "kept_existing"
    g = _fields(source="gone")
    assert ac.put(engine, ac.cache_key(**g), fields=g, answer="x", citations=[], generated_by="m") == "document_gone"
    c = _fields(content_hash="old")
    assert ac.put(engine, ac.cache_key(**c), fields=c, answer="x", citations=[], generated_by="m") \
        == "document_changed"


def test_a_status_read_does_not_count_as_a_hit(engine):
    f = _fields()
    k = ac.cache_key(**f)
    ac.put(engine, k, fields=f, answer="a", citations=[], generated_by="m")
    ac.get(engine, k, count_hit=False)
    with engine.connect() as c:
        assert c.execute(text("SELECT hits FROM memento_answer_cache")).scalar() == 0
    ac.get(engine, k)
    with engine.connect() as c:
        assert c.execute(text("SELECT hits FROM memento_answer_cache")).scalar() == 1


# ── the outcome of a warm-up, from the pipe ───────────────────────────────────

def _warm(qp, monkeypatch, behave):
    """pipe() with a stand-in service.run that does `behave(rec)` and yields its text; the warm-up's result dict."""
    p = qp.Pipe()
    p.valves = p.Valves(COLLECTION=COLL)

    async def fake(turn):
        for chunk in behave(turn.rec):
            yield chunk
    monkeypatch.setattr(service, "run", fake)
    result = {}

    async def go():
        return [c async for c in p.pipe({"messages": [{"role": "user", "content": 'Summarize the document "Doc"'}]},
                                        {"id": "w"}, None, None, {"warmup": True, "warmup_result": result})]
    asyncio.run(go())
    return result


def _keyed(rec, clean=True):
    f = _fields()
    rec.key, rec.fields, rec.clean, rec.generated_by = ac.cache_key(**f), f, clean, "m"


def test_a_written_answer_is_stored(qp, engine, monkeypatch):
    r = _warm(qp, monkeypatch, lambda rec: (_keyed(rec), ["This document studies foams."])[1])
    assert r["result"] == "stored" and r["key"] == ac.cache_key(**_fields())


def test_a_hit_is_reported_with_its_key(qp, engine, monkeypatch):
    def behave(rec):
        rec.hit, rec.hit_key = True, "k-hit"
        return ["stored text"]
    assert _warm(qp, monkeypatch, behave) == {"result": "hit", "reason": "", "key": "k-hit"}


def test_an_answer_with_a_note_is_ineligible(qp, engine, monkeypatch):
    r = _warm(qp, monkeypatch, lambda rec: (_keyed(rec), ["Half an answer.\n\n" + texts.TIMEOUT_NOTE])[1])
    assert r["result"] == "ineligible" and "note" in r["reason"]


def test_an_answer_that_did_not_finish_is_an_error(qp, engine, monkeypatch):
    r = _warm(qp, monkeypatch, lambda rec: (_keyed(rec, clean=False), ["⚠ the library is not set up"])[1])
    assert r["result"] == "error"


def test_an_answer_that_is_no_stored_answer_is_ineligible(qp, engine, monkeypatch):
    def behave(rec):
        rec.skip_reason = "not a picker question"
        return ["an answer"]
    assert _warm(qp, monkeypatch, behave)["result"] == "ineligible"


def test_a_graceful_error_reports_nothing_so_the_worker_retries_it(qp, engine, monkeypatch):
    # a polite error text is no warmed answer, and no final verdict either
    r = _warm(qp, monkeypatch, lambda rec: ["This library is not set up yet. Please tell the administrator."])
    assert r == {}


@pytest.mark.parametrize("why", ["library changing", "no fingerprint"])
def test_a_passing_state_is_not_final(qp, engine, monkeypatch, why):
    p = qp.Pipe()
    p.valves = p.Valves(COLLECTION=COLL)
    rec = cache.Recorder()
    docs = {"doc": {"content_hash": None if why == "no fingerprint" else "h1"}}
    asyncio.run(cache.prepare(p.valves, rec, picked=("doc", "summary"), history=[], docs=docs,
                                 in_progress={"other"} if why == "library changing" else set(), source="doc",
                                 action="summary", mode="whole", prompt_version="p", variant=None, metadata={},
                              db_engine=qp._db_engine))
    assert rec.key is None and rec.skip_reason == ""                 # reported as nothing: an error, retried


def test_a_document_changed_before_the_write_is_ineligible(qp, engine, monkeypatch):
    def behave(rec):
        _keyed(rec)
        with engine.begin() as c:                     # replaced between the pipe's check and the write
            c.execute(text("UPDATE memento_documents SET content_hash = 'h2'"))
        return ["This document studies foams."]
    r = _warm(qp, monkeypatch, behave)
    assert r["result"] == "ineligible" and "changed" in r["reason"]


# ── the warm-up worker records it ─────────────────────────────────────────────

class _Stub:
    """The copy's answer path (service.handle) as its worker calls it."""
    def __init__(self, result=None, raise_=False):
        self.result, self.raise_ = result, raise_
        self.valves = None

    async def handle(self, body, user, emit, files, metadata, task, **k):
        assert metadata["warmup"] is True                         # every warm-up behaviour stays switched on
        if self.raise_:
            raise RuntimeError("model down")
        if self.result:
            metadata["warmup_result"].update(self.result)
        yield "x"


def _step(qp, engine, monkeypatch, stub):
    monkeypatch.setattr(workers, "vllm_metrics", lambda url=None, key_env=None: None)
    monkeypatch.setattr(library_state, "in_progress_sources", lambda *a: set())
    stub.valves = qp.Pipe.Valves(COLLECTION=COLL)
    monkeypatch.setattr(service, "handle", stub.handle)
    wq.enqueue(engine, COLL, "doc", ["summary"])
    out = asyncio.run(workers.warmup_step(stub.valves, engine, function_id=qp._OWN_ID, db_engine=lambda: engine))
    with engine.connect() as c:
        row = c.execute(wq.JOBS.select()).mappings().one()
    return out, dict(row)


def test_stored_and_hit_are_done_with_their_key(qp, engine, monkeypatch):
    out, row = _step(qp, engine, monkeypatch, _Stub({"result": "stored", "reason": "", "key": "k1"}))
    assert out == "done" and row["status"] == "done" and row["cache_key"] == "k1"


def test_ineligible_is_final(qp, engine, monkeypatch):
    out, row = _step(qp, engine, monkeypatch, _Stub({"result": "ineligible", "reason": "note in the answer"}))
    assert row["status"] == "ineligible" and "note" in row["last_error"]


@pytest.mark.parametrize("stub", [_Stub(None), _Stub({"result": "error", "reason": "x"}), _Stub(raise_=True)])
def test_no_result_an_error_or_an_exception_is_retried(qp, engine, monkeypatch, stub):
    out, row = _step(qp, engine, monkeypatch, stub)
    assert out == "failed" and row["status"] == "queued" and row["attempts"] == 1


def test_a_job_whose_document_is_gone_is_skipped_not_done(qp, engine, monkeypatch):
    monkeypatch.setattr(workers, "vllm_metrics", lambda url=None, key_env=None: None)
    monkeypatch.setattr(library_state, "in_progress_sources", lambda *a: set())
    wq.enqueue(engine, COLL, "gone", ["summary"])
    stub = _Stub({"result": "stored", "key": "k"})
    stub.valves = qp.Pipe.Valves(COLLECTION=COLL)
    monkeypatch.setattr(service, "handle", stub.handle)
    assert asyncio.run(workers.warmup_step(stub.valves, engine, function_id=qp._OWN_ID, db_engine=lambda: engine)) == "skipped"
    with engine.connect() as c:
        assert c.execute(wq.JOBS.select()).mappings().one()["status"] == "skipped"


# ── a warm-up repairs an answer no click would replay ──

def test_put_keeps_an_existing_answer_only_when_it_is_still_usable(engine):
    from memento.research_gaps import passages as gr
    f = _fields(action="gaps", mode="capped")
    k = ac.cache_key(**f)
    leaked = "### Limitations\n1. One material [1].\nLet's think about whether [2] fits.\n"
    good = "### Limitations\n1. Only one material was tested [1].\n"
    assert not gr.stored_gap_ok(leaked) and gr.stored_gap_ok(good)
    ac.put(engine, k, fields=f, answer=leaked, citations=[], generated_by="old")
    assert ac.put(engine, k, fields=f, answer=good, citations=[], generated_by="warm", replace=False,
                  keep_if=gr.stored_gap_ok) == "written"                    # the unusable row is replaced
    assert ac.get(engine, k, count_hit=False)["answer"] == good
    assert ac.put(engine, k, fields=f, answer=good + "\n", citations=[], generated_by="warm2", replace=False,
                  keep_if=gr.stored_gap_ok) == "kept_existing"              # a usable one (a live answer) stays
    assert ac.get(engine, k, count_hit=False)["generated_by"] == "warm"
