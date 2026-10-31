# Each copy of the pipes knows its own function id, runs its own warm-up worker and works only when it is set up. Open
# WebUI loads a function's code as the module `function_<id>`; the second copies are `memento_query_polymer` /
# `memento_ingestion_polymer`, the default ones `memento_query` / `memento_ingestion` (always ready).
import asyncio
import builtins
import json
import sys
import threading
import types
from pathlib import Path

import pytest
from sqlalchemy import text
from memento.config.paths import PROJECT_ROOT
from memento.query import library_state, service, texts, workers
from memento.ingestion.upload import messages

ROOT = PROJECT_ROOT
sys.path.insert(0, str(ROOT))
from memento.library import copies as cs  # noqa: E402
from memento.library import documents as db  # noqa: E402
from memento.library import warmup_queue as wq  # noqa: E402
from memento.document_chat import chat, cleanup
from memento.ingestion.upload import intake
from memento.ingestion.upload import workflow

QUERY = ROOT / "openwebui/functions/query_pipe.py"
INGEST = ROOT / "openwebui/functions/ingestion_pipe.py"


@pytest.fixture(autouse=True)
def _own_generations(monkeypatch):
    # loading a `function_…` module starts a worker generation: keep the test's out of the process-wide counter
    monkeypatch.setattr(builtins, "_memento_warmup_gen", {}, raising=False)


def load_copy(path: Path, function_id: str, monkeypatch):
    """The pipe file loaded the way OpenWebUI loads it (module `function_<id>`), without starting threads."""
    from _owui_loader import load_function
    with monkeypatch.context() as m:
        m.setattr(threading.Thread, "start", lambda self: None)
        return load_function(path, function_id, start_workers=True)


def _cycle(mod, st):
    """One round of the worker of the copy loaded as `mod` (its id, its Valves class, its database)."""
    return workers.warmup_cycle(st, function_id=mod._OWN_ID, valves_class=mod.Pipe.Valves, db_engine=mod._db_engine)


def engine():
    from open_webui.internal.db import engine as e       # the test's private database (conftest)
    with e.begin() as c:
        c.execute(text("CREATE TABLE IF NOT EXISTS function (id TEXT PRIMARY KEY, valves TEXT, is_active BOOLEAN)"))
    return e


def put(fid, valves, active=True):
    with engine().begin() as c:
        c.execute(text("DELETE FROM function WHERE id = :i"), {"i": fid})
        c.execute(text("INSERT INTO function (id, valves, is_active) VALUES (:i, :v, :a)"),
                  {"i": fid, "v": json.dumps(valves), "a": active})


def study_valves(mod, role: str, **over) -> dict:
    """Every required setting, valid for the pipe's Valves (defaults where the pipe has the field)."""
    names = cs.REQUIRED_QUERY_VALVES if role == "query" else cs.REQUIRED_INGESTION_VALVES
    have = mod.Pipe.Valves().model_dump()
    v = {k: have.get(k, f"{k.lower()}-value") for k in names}
    v.update(DOMAIN_NAME="polymer-engineering", COLLECTION="memento-polymer-rag")
    if role == "query":
        v["INGESTION_FUNCTION_ID"] = "memento_ingestion_polymer"
    v.update(over)
    return v


# ── own id and worker generations ─────────────────────────────────────────────

def test_each_copy_knows_its_own_function_id(monkeypatch):
    q = load_copy(QUERY, "memento_query_polymer", monkeypatch)
    i = load_copy(INGEST, "memento_ingestion_polymer", monkeypatch)
    assert q._OWN_ID == "memento_query_polymer" and i._OWN_ID == "memento_ingestion_polymer"
    assert q._own_id("qp_test") == "memento_query" and i._own_id("ip_test") == "memento_ingestion"   # tests, CLI
    assert q._own_id("function_memento_query") == "memento_query"


def test_saved_valves_default_to_the_copys_own_row(monkeypatch):
    q = load_copy(QUERY, "memento_query_polymer", monkeypatch)
    put("memento_query", {"COLLECTION": "memento-comp-mat-rag"})
    put("memento_query_polymer", {"COLLECTION": "memento-polymer-rag"})
    assert library_state.saved_valves(q._OWN_ID, q._db_engine)["COLLECTION"] == "memento-polymer-rag"   # its worker's
    assert library_state.saved_valves("memento_query", q._db_engine)["COLLECTION"] == "memento-comp-mat-rag"


def test_the_request_path_reads_saved_valves_through_library_state(monkeypatch):
    from memento.query import library_state
    q = load_copy(QUERY, "memento_query_polymer", monkeypatch)
    put("memento_query_polymer", {"COLLECTION": "memento-polymer-rag"})
    assert library_state.saved_valves("memento_query_polymer", q._db_engine)["COLLECTION"] == "memento-polymer-rag"
    assert library_state.saved_valves("memento_no_such_copy", q._db_engine) is None


def test_two_copies_keep_their_own_worker_generation(monkeypatch):
    q = load_copy(QUERY, "memento_query", monkeypatch)
    a, b = workers.next_gen("memento_query"), workers.next_gen("memento_query_polymer")
    assert workers.gen_current("memento_query", a) and workers.gen_current("memento_query_polymer", b)
    newer = workers.next_gen("memento_query")                     # a code update of one copy
    assert not workers.gen_current("memento_query", a) and workers.gen_current("memento_query", newer)
    assert workers.gen_current("memento_query_polymer", b)        # the other copy's worker keeps running


def test_a_worker_of_the_old_code_stops_when_the_counter_becomes_per_copy(monkeypatch):
    # the old code kept one int in builtins; its loop runs while that int equals its own generation
    monkeypatch.setattr(builtins, "_memento_warmup_gen", 7)
    q = load_copy(QUERY, "memento_query", monkeypatch)
    workers.next_gen("memento_query")
    assert getattr(builtins, "_memento_warmup_gen") != 7     # the earlier loop's condition is now false


# ── the queue per collection ─────────────────────────────────────────────────

def test_a_worker_claims_only_its_own_collections_jobs():
    e = engine()
    db.ensure_tables(e)
    wq.enqueue(e, "memento-comp-mat-rag", "a", ["summary"])
    wq.enqueue(e, "memento-polymer-rag", "b", ["summary"])
    job = wq.claim_next(e, collection="memento-polymer-rag")
    assert (job["collection"], job["source"]) == ("memento-polymer-rag", "b")
    assert wq.claim_next(e, collection="memento-polymer-rag") is None        # comp-mat's job is not handed out
    assert wq.claim_next(e)["source"] == "a"                                  # no collection: as before


# ── the copy's own metrics URL and slot rule ──────────────────────────────────

METRICS = """vllm:num_requests_running{engine="0",model_name="m"} %s
vllm:num_requests_waiting{engine="0",model_name="m"} 0.0
"""


def test_the_warmup_reads_the_copys_own_model_server(monkeypatch):
    q = load_copy(QUERY, "memento_query_polymer", monkeypatch)
    asked = []
    monkeypatch.setattr(workers, "vllm_metrics", lambda url=None, *a, **k: asked.append(url) or METRICS % "3.0")
    monkeypatch.setattr(library_state, "in_progress_sources", lambda *a: set())
    pipe = types.SimpleNamespace(valves=types.SimpleNamespace(COLLECTION="memento-polymer-rag",
                                                              METRICS_URL="http://llm.test:8001/metrics",
                                                              WARMUP_MAX_RUNNING=2))
    assert asyncio.run(workers.warmup_step(pipe.valves, engine(), function_id="memento_query_polymer",
                                       db_engine=engine)) == "busy"           # 3 running > 2
    assert asked == ["http://llm.test:8001/metrics"]
    pipe.valves.WARMUP_MAX_RUNNING = 3
    db.ensure_tables(engine())
    assert asyncio.run(workers.warmup_step(pipe.valves, engine(), function_id="memento_query_polymer",
                                       db_engine=engine)) == "empty"          # 3 running ≤ 3: room, nothing queued


def test_the_valves_default_to_todays_environment_values(monkeypatch):
    monkeypatch.setenv("WARMUP_METRICS_URL", "http://127.0.0.1:8090/metrics")
    monkeypatch.setenv("WARMUP_MAX_RUNNING", "2")
    q = load_copy(QUERY, "memento_query", monkeypatch)
    v = q.Pipe.Valves()
    assert v.METRICS_URL == "http://127.0.0.1:8090/metrics" and v.WARMUP_MAX_RUNNING == 2


def test_the_warmup_job_asks_the_copys_own_model(monkeypatch):
    # "memento_query.local" would route the second copy's warm-ups through the default copy
    q = load_copy(QUERY, "memento_query_polymer", monkeypatch)
    seen = []

    async def handle(body, *a, **k):              # the copy's own answer path
        seen.append(body["model"])
        yield "an answer"
    monkeypatch.setattr(service, "handle", handle)
    # the warm-up reads the copy's model list from its Valves
    asyncio.run(workers.run_warmup_job(q.Pipe.Valves(), {"action": "summary"}, "T", function_id=q._OWN_ID,
                                       db_engine=q._db_engine))
    assert seen and seen[0].startswith("memento_query_polymer.")


# ── the worker's round: nothing until the copy is enabled ─────────────────────

@pytest.fixture
def recorded(monkeypatch):
    def attach(mod):
        calls = []

        async def step(valves, e, **k):
            calls.append(("warmup", valves.COLLECTION))
            return "empty"
        monkeypatch.setattr(workers, "warmup_step", step)
        monkeypatch.setattr(cleanup, "doc_chat_cleanup", lambda valves, e, **k: calls.append(("cleanup", valves.COLLECTION)))
        monkeypatch.setattr(cleanup, "remove_old_reports", lambda root, *a: calls.append(("reports",)))
        monkeypatch.setattr(workers, "publish_model_colours", lambda *a, **k: calls.append(("publish",)))
        return calls
    return attach


def test_a_study_copy_does_nothing_until_it_is_enabled_and_starts_without_a_restart(monkeypatch, recorded):
    q = load_copy(QUERY, "memento_query_polymer", monkeypatch)
    calls, st = recorded(q), {}
    assert _cycle(q, st) == 15 and calls == []                       # not installed
    put("memento_query_polymer", study_valves(q, "query"), active=False)
    assert _cycle(q, st) == 15 and calls == []                       # configured, inactive, no ingestion copy
    i = load_copy(INGEST, "memento_ingestion_polymer", monkeypatch)
    put("memento_ingestion_polymer", study_valves(i, "ingestion"), active=False)
    put("memento_query_polymer", study_valves(q, "query"), active=True)
    assert _cycle(q, st) == 15 and calls == []                       # the linked ingestion copy is inactive
    put("memento_ingestion_polymer", study_valves(i, "ingestion"), active=True)
    _cycle(q, st)                                                    # the next round: works, no restart
    assert ("publish",) in calls and ("cleanup", "memento-polymer-rag") in calls
    assert ("warmup", "memento-polymer-rag") in calls


def test_a_deactivated_copy_stops_within_one_round(monkeypatch, recorded):
    q = load_copy(QUERY, "memento_query_polymer", monkeypatch)
    i = load_copy(INGEST, "memento_ingestion_polymer", monkeypatch)
    calls, st = recorded(q), {}
    put("memento_ingestion_polymer", study_valves(i, "ingestion"))
    put("memento_query_polymer", study_valves(q, "query"))
    _cycle(q, st)
    assert ("warmup", "memento-polymer-rag") in calls
    calls.clear()
    put("memento_query_polymer", study_valves(q, "query"), active=False)
    assert _cycle(q, st) == 15 and calls == []


def test_an_incomplete_study_copy_never_runs_with_todays_settings(monkeypatch, recorded):
    # config/.env names the default library; a second copy missing COLLECTION must not fall back to it
    q = load_copy(QUERY, "memento_query_polymer", monkeypatch)
    i = load_copy(INGEST, "memento_ingestion_polymer", monkeypatch)
    calls, st = recorded(q), {}
    put("memento_ingestion_polymer", study_valves(i, "ingestion"))
    v = study_valves(q, "query")
    v.pop("COLLECTION")
    put("memento_query_polymer", v)
    assert _cycle(q, st) == 15 and calls == []


def test_todays_copy_works_as_before(monkeypatch, recorded):
    q = load_copy(QUERY, "memento_query", monkeypatch)
    calls, st = recorded(q), {}
    assert _cycle(q, st) == 15 and calls == []                       # no saved Valves yet: waits (as today)
    put("memento_query", {"COLLECTION": "memento-comp-mat-rag"}, active=True)
    _cycle(q, st)
    assert ("cleanup", "memento-comp-mat-rag") in calls and ("warmup", "memento-comp-mat-rag") in calls
    assert ("publish",) not in calls                                       # published at import, as today


def test_only_todays_copy_publishes_when_its_code_is_loaded(monkeypatch, tmp_path):
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path))
    load_copy(QUERY, "memento_query_polymer", monkeypatch)
    assert not (tmp_path / "_runtime" / "query_models.json").exists()
    load_copy(QUERY, "memento_query", monkeypatch)
    assert (tmp_path / "_runtime" / "query_models.json").exists()


# ── requests: a copy that is not set up answers with one plain line ───────────

def _ask(mod, content="What is a foam?", model="memento_query_polymer.local", user=None, **kw):
    async def run():
        return [c async for c in mod.Pipe().pipe({"model": model, "messages": [{"role": "user", "content": content}]},
                                                 user or {"id": "u", "role": "user"}, **kw)]
    return "".join(c for c in asyncio.run(run()) if isinstance(c, str))


def _no_answer(mod, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("a copy that is not set up must not search or answer")
    monkeypatch.setattr(service, "run", boom)
    monkeypatch.setattr(chat, "answer", boom)


def test_a_study_copy_that_is_not_set_up_answers_with_the_not_ready_line(monkeypatch):
    q = load_copy(QUERY, "memento_query_polymer", monkeypatch)
    _no_answer(q, monkeypatch)
    assert _ask(q).strip() == texts.STUDY_NOT_READY_MSG
    assert _ask(q, model="memento_query_polymer.document-chat").strip() == texts.STUDY_NOT_READY_MSG
    de = _ask(q, "Welche Methoden werden zur Charakterisierung von Schaumstoffen eingesetzt?").strip()
    assert de == texts.DE_FIXED[texts.STUDY_NOT_READY_MSG] and "Sie" in de
    assert "—" not in texts.STUDY_NOT_READY_MSG and "—" not in de


def test_a_configured_but_inactive_study_copy_answers_with_the_not_ready_line(monkeypatch):
    q = load_copy(QUERY, "memento_query_polymer", monkeypatch)
    i = load_copy(INGEST, "memento_ingestion_polymer", monkeypatch)
    put("memento_ingestion_polymer", study_valves(i, "ingestion"), active=False)
    put("memento_query_polymer", study_valves(q, "query"))
    _no_answer(q, monkeypatch)
    assert _ask(q).strip() == texts.STUDY_NOT_READY_MSG


def test_todays_copy_never_checks(monkeypatch):
    q = load_copy(QUERY, "memento_query", monkeypatch)
    monkeypatch.setattr(cs, "copy_state", lambda *a, **k: pytest.fail("today's copy reads no state"))

    async def fake(turn):
        yield "answered"
    monkeypatch.setattr(service, "run", fake)
    assert _ask(q, model="memento_query.local") == "answered"


def test_the_state_is_read_once_per_ten_seconds(monkeypatch):
    q = load_copy(QUERY, "memento_query_polymer", monkeypatch)
    _no_answer(q, monkeypatch)
    reads = []
    real = cs.copy_state
    monkeypatch.setattr(cs, "copy_state", lambda *a, **k: reads.append(a[1]) or real(*a, **k))
    _ask(q)
    _ask(q)
    assert reads == ["memento_query_polymer"]
    later = library_state._state_clock() + 11
    monkeypatch.setattr(library_state, "_state_clock", lambda: later)
    _ask(q)
    assert len(reads) == 2


def test_an_admin_upload_to_an_ingestion_copy_that_is_not_set_up_ingests_nothing(monkeypatch):
    i = load_copy(INGEST, "memento_ingestion_polymer", monkeypatch)
    monkeypatch.setattr(workflow, "pipe_sync", lambda *a, **k: pytest.fail("nothing may be ingested"))

    async def run():
        return [c async for c in i.Pipe().pipe({"model": "memento_ingestion_polymer",
                                                "messages": [{"role": "user", "content": ""}]},
                                               {"id": "admin", "role": "admin"},
                                               [{"id": "f1", "name": "a.pdf"}])]
    out = "".join(c for c in asyncio.run(run()) if isinstance(c, str))
    assert out.strip() == messages.STUDY_NOT_READY_MSG


def test_todays_ingestion_copy_never_checks(monkeypatch):
    i = load_copy(INGEST, "memento_ingestion", monkeypatch)
    monkeypatch.setattr(cs, "copy_state", lambda *a, **k: pytest.fail("today's copy reads no state"))
    monkeypatch.setattr(intake, "may_ingest", lambda valves, user: _false())

    async def run():
        return [c async for c in i.Pipe().pipe({"model": "memento_ingestion",
                                                "messages": [{"role": "user", "content": "hi"}]},
                                               {"id": "u", "role": "user"}, [])]
    out = "".join(c for c in asyncio.run(run()) if isinstance(c, str))
    assert "contributors only" in out                                       # today's next check ran


async def _false():
    return False
