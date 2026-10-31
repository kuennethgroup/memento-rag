# The upload warm-up worker in the Query pipe.
import asyncio
import sys
from pathlib import Path
from memento.config.paths import PROJECT_ROOT
from _owui_loader import load_function
from memento.query import checking, library_state, service, workers

ROOT = PROJECT_ROOT
sys.path.insert(0, str(ROOT))
qp = load_function(ROOT / "openwebui/functions/query_pipe.py", "memento_query")
from memento.library import answers as ac  # noqa: E402
from memento.library import documents as db  # noqa: E402
from memento.library import warmup_queue as wq  # noqa: E402

METRICS = """# HELP x
vllm:num_requests_running{engine="0",model_name="qwen3.8-27b"} %s
vllm:num_requests_waiting{engine="0",model_name="qwen3.8-27b"} %s
"""


def test_idle_rules():
    # the model takes 4 requests at once; a warm-up runs while at least 2 slots are free
    assert workers.warmup_idle(METRICS % ("0.0", "0.0"), set()) is True
    assert workers.warmup_idle(METRICS % ("2.0", "0.0"), set()) is True           # 2 answering: 2 slots free
    assert workers.warmup_idle(METRICS % ("3.0", "0.0"), set()) is False          # only 1 slot left: keep it for questions
    assert workers.warmup_idle(METRICS % ("0.0", "1.0"), set()) is False          # something is waiting
    assert workers.warmup_idle(METRICS % ("0.0", "0.0"), {"doc"}) is False        # an upload is running
    assert workers.warmup_idle(None, set()) is True                               # no metrics: the upload rule decides


def test_warmup_questions_are_the_picker_texts():
    assert workers.warmup_question("summary", 'A "b" title') == "Summarize the document \"A 'b' title\""
    assert workers.warmup_question("gaps", "T") == 'What research gaps does the document "T" name?'


def test_worker_starts_only_inside_openwebui():
    assert qp._should_start_worker("function_memento_query") is True
    assert qp._should_start_worker("qp_warm") is False and qp._should_start_worker("openwebui.functions.query_pipe") is False


class StubPipe:
    """The copy's answer path (service.handle) as its worker calls it."""
    def __init__(self, fail=False):
        self.calls, self.fail = [], fail

    async def handle(self, body, user, emit, files, metadata, task, **k):
        self.calls.append((body, metadata))
        if self.fail:
            raise RuntimeError("model down")
        await emit({"type": "status", "data": {}})
        # the pipe reports what happened to the answer; without a report it is an error
        metadata["warmup_result"].update(result="stored", reason="", key="k")
        yield "an answer"


def _engine_with_job():
    e = qp._db_engine()
    db.ensure_tables(e)
    db.add_document(e, collection="col", entry={"source": "doc", "title": "Doc title"}, content_hash="h",
                    uploaded_by="u", output_root=None)
    wq.enqueue(e, "col", "doc", ["summary"])
    return e


def test_step_runs_a_job_through_the_pipe_and_marks_it_done(monkeypatch):
    e = _engine_with_job()
    monkeypatch.setattr(workers, "vllm_metrics", lambda url=None, key_env=None: METRICS % ("0.0", "0.0"))
    monkeypatch.setattr(library_state, "in_progress_sources", lambda *a: set())
    pipe = StubPipe()
    pipe.valves = qp.Pipe.Valves(COLLECTION="col")
    monkeypatch.setattr(service, "handle", pipe.handle)
    assert asyncio.run(workers.warmup_step(pipe.valves, e, function_id="memento_query", db_engine=qp._db_engine)) == "done"
    body, metadata = pipe.calls[0]
    assert body["messages"][-1]["content"] == 'Summarize the document "Doc title"' and metadata["warmup"] is True and set(metadata) == {"warmup", "warmup_result"}   # unchecked; checked on demand
    assert body["model"].startswith("memento_query.")
    with e.connect() as c:
        assert c.execute(wq.JOBS.select()).mappings().one()["status"] == "done"
    assert asyncio.run(workers.warmup_step(pipe.valves, e, function_id="memento_query", db_engine=qp._db_engine)) == "empty"


def test_step_waits_while_busy_and_retries_failures(monkeypatch):
    e = _engine_with_job()
    monkeypatch.setattr(library_state, "in_progress_sources", lambda *a: set())
    monkeypatch.setattr(workers, "vllm_metrics", lambda url=None, key_env=None: METRICS % ("3.0", "0.0"))
    pipe = StubPipe(fail=True)
    pipe.valves = qp.Pipe.Valves(COLLECTION="col")
    monkeypatch.setattr(service, "handle", pipe.handle)
    assert asyncio.run(workers.warmup_step(pipe.valves, e, function_id="memento_query", db_engine=qp._db_engine)) == "busy" and pipe.calls == []
    monkeypatch.setattr(workers, "vllm_metrics", lambda url=None, key_env=None: METRICS % ("0.0", "0.0"))
    assert asyncio.run(workers.warmup_step(pipe.valves, e, function_id="memento_query", db_engine=qp._db_engine)) == "failed"
    with e.connect() as c:
        row = c.execute(wq.JOBS.select()).mappings().one()
    assert row["status"] == "queued" and row["attempts"] == 1 and "model down" in row["last_error"]


def test_warmup_answers_never_replace_a_stored_one(monkeypatch):
    fields = dict(collection="col", source="doc", content_hash="h", action="summary", language="en",
                  model="local:m", mode="whole", prompt_version="p")
    e = qp._db_engine()
    ac.ensure_table(e)
    key = ac.cache_key(**fields)
    ac.put(e, key, fields=fields, answer="live answer", citations=[], generated_by="m")
    pipe = qp.Pipe()

    async def fake(turn):
        turn.rec.key, turn.rec.fields, turn.rec.clean, turn.rec.generated_by = key, fields, True, "m"
        yield "warm-up answer"
    monkeypatch.setattr(service, "run", fake)

    async def go():
        return [c async for c in pipe.pipe({}, {}, None, None, {"warmup": True})]
    asyncio.run(go())
    assert ac.get(e, key)["answer"] == "live answer"


def test_model_colours_are_published_for_the_loader(tmp_path, monkeypatch):
    # written when Open WebUI loads the pipe, next to the quota and library files
    import json
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path))
    workers.publish_model_colours(None, qp._OWN_ID, valves_class=qp.Pipe.Valves, db_engine=qp._db_engine)
    models = json.loads((tmp_path / "_runtime" / "query_models.json").read_text())["models"]
    assert models[0]["id"] == "memento-query" and models[0]["cloud"] is False
    assert all(m["accent"] == "#0369a1" and m["accent_dark"] == "#00adee" for m in models if m["cloud"])


def test_a_running_job_is_not_handed_out_again_while_its_check_may_still_run():
    # warm-up answers are checked one call at a time (up to WARMUP_VERIFY_TIMEOUT); the queue must not re-hand a job it
    # thinks crashed (`stale_after_s`) while it is still being checked
    assert workers.WARMUP_STALE_S > checking.WARMUP_VERIFY_TIMEOUT + 480
