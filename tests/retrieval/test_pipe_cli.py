# The command-line answer path runs the Memento Query pipe itself (memento/retrieval/pipe_cli.py, behind run_query.sh and
# memento/retrieval/generate.py), so the pipe and the command line stay in sync.
import asyncio
import json
import sys
import threading
import types
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool
from memento.config.paths import PROJECT_ROOT
from memento.query import library_state, prompts, request, texts
from memento.query.scope import documents

ROOT = PROJECT_ROOT
sys.path.insert(0, str(ROOT))

from memento.retrieval import pipe_cli  # noqa: E402


def _engine(valves: dict | None):
    e = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    with e.begin() as c:
        c.execute(text("CREATE TABLE function (id TEXT PRIMARY KEY, valves TEXT)"))
        if valves is not None:
            c.execute(text("INSERT INTO function VALUES ('memento_query', :v)"), {"v": json.dumps(valves)})
    return e


def test_the_pipe_is_loaded_without_its_background_worker():
    mod = pipe_cli.load_pipe_module()
    assert not mod.__name__.startswith("function_") and not mod._should_start_worker(mod.__name__)
    assert not any(t.name.startswith("memento-warmup") for t in threading.enumerate())
    assert hasattr(mod, "Pipe") and request.THOROUGH_FILTER_ID == "memento_thorough"


def test_the_pipe_gets_the_saved_settings_and_the_database():
    mod = pipe_cli.load_pipe_module()
    e = _engine({"COLLECTION": "memento-served-rag", "RERANK_TOP": 7, "NOT_A_VALVE": 1, "DOMAIN_NAME": None})
    p = pipe_cli.make_pipe(mod, e)
    assert p.valves.COLLECTION == "memento-served-rag" and p.valves.RERANK_TOP == 7
    assert p.valves.DOMAIN_NAME == mod.Pipe.Valves().DOMAIN_NAME            # None (unset) keeps the default
    assert p.engine() is e and p.function_id == "memento_query"            # OpenWebUI's database, by DATABASE_URL


def test_no_saved_settings_is_an_error_not_a_silent_default():
    mod = pipe_cli.load_pipe_module()
    with pytest.raises(SystemExit, match="saved settings"):
        pipe_cli.make_pipe(mod, _engine(None))


class _FakePipe:
    def __init__(self, chunks=("Answer [1].",), events=()):
        self.calls, self._chunks, self._events = [], list(chunks), list(events)

    async def pipe(self, body, user, emit, files, metadata):
        self.calls.append({"body": body, "user": user, "files": files, "metadata": metadata})
        for e in self._events:
            await emit(e)
        for c in self._chunks:
            yield c


def _cite(name):
    return {"type": "citation", "data": {"source": {"name": name}, "metadata": [{"name": name}], "document": ["x"]}}


@pytest.mark.parametrize("model, thorough, want_model, want_filters", [
    (None, False, "memento_query.local", None),
    ("qwen36", True, "memento_query.qwen36", ["memento_thorough"]),
])
def test_the_question_reaches_the_pipe_as_openwebui_sends_it(model, thorough, want_model, want_filters):
    p = _FakePipe(events=[_cite("Doc A (p. 2)"), {"type": "status", "data": {"description": "x"}}])
    out = asyncio.run(pipe_cli.ask(p, "What is X?", model=model, thorough=thorough))
    call = p.calls[0]
    assert call["body"] == {"model": want_model, "stream": True,
                            "messages": [{"role": "user", "content": "What is X?"}]}
    assert (call["metadata"] or {}).get("filter_ids") == want_filters
    assert call["files"] is None
    assert out["answer"] == "Answer [1]." and out["sources"] == ["Doc A (p. 2)"]


def test_output_is_readable_in_a_terminal():
    raw = ("<think>plan</think>The value is 5 [1].\n\n![Doc p.3](data:image/png;base64,iVBORw0KGgo=)\n\n"
           "*Figure (1): Doc, p. 3*\n\n<details type=\"memento_sources\">\n<summary>Cited in this answer: all 1 "
           "sources Memento read</summary>\n\n- (1) Doc (p. 3)\n</details>\n")
    clean = pipe_cli.clean_output(raw, show_thinking=False)
    assert "base64" not in clean and "[figure: Doc p.3]" in clean
    assert "<think>" not in clean and "plan" not in clean
    assert "<details" not in clean and "<summary>" not in clean and "Cited in this answer" in clean
    assert "Thinking:\nplan" in pipe_cli.clean_output(raw, show_thinking=True)


def test_main_runs_the_pipe_never_the_old_retrieval(monkeypatch, capsys, tmp_path):
    import memento.retrieval.retrieve as old
    monkeypatch.setattr(old, "retrieve", lambda *a, **k: pytest.fail("old retrieve() called"))
    fake = _FakePipe(events=[_cite("Doc A (p. 2)")])
    monkeypatch.setattr(pipe_cli, "build", lambda function_id="memento_query": fake)
    out = tmp_path / "a.json"
    assert pipe_cli.main(["What is X?", "--model", "qwen35", "--thorough", "--json", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "Answer [1]." in printed and "[1] Doc A (p. 2)" in printed
    assert fake.calls[0]["body"]["model"] == "memento_query.qwen35"
    saved = json.loads(out.read_text())
    assert saved["question"] == "What is X?" and saved["sources"] == ["Doc A (p. 2)"] and saved["model"] == "qwen35"


@pytest.mark.parametrize("flag", ["--no-rerank", "--retrieve-k", "--paper", "--strategy", "--think"])
def test_old_flags_are_refused_with_a_reason(flag, capsys):
    with pytest.raises(SystemExit):
        pipe_cli.main(["q", flag, "5"] if flag in ("--retrieve-k", "--paper", "--strategy") else ["q", flag])
    assert "Memento Query" in capsys.readouterr().err


def test_generate_py_command_line_is_the_pipe(monkeypatch):
    import memento.retrieval.generate as gen
    seen = {}
    monkeypatch.setattr(pipe_cli, "main", lambda argv=None: seen.setdefault("argv", argv) and 0)
    monkeypatch.setattr(sys, "argv", ["generate.py", "What is X?"])
    gen.main()
    assert "argv" in seen
    assert callable(gen.generate)              # the importable function stays (frozen RQ2/RQ3 evaluation path)


def test_the_real_pipe_answers_through_the_command_line(monkeypatch):
    # end to end with the real Pipe: an empty library answers a general question directly and refuses a library one,
    # exactly as in OpenWebUI
    import openai
    import memento.query.router as router
    mod = pipe_cli.load_pipe_module()
    p = pipe_cli.make_pipe(mod, _engine({"COLLECTION": "memento-x-rag", "DOMAIN_NAME": "generic"}))

    class FakeLLM:
        def __init__(self, *a, **k):
            async def create(**kw):
                async def chunks():
                    yield types.SimpleNamespace(choices=[types.SimpleNamespace(delta=types.SimpleNamespace(
                        content="Today is Friday.", reasoning_content=None))])
                return chunks()
            self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=create))
    intent = {"v": "general"}

    async def fake_route(*a, **k):
        return router.RouterResponse(intent=intent["v"], query="q", reasoning="light")
    monkeypatch.setattr(openai, "AsyncOpenAI", FakeLLM)
    monkeypatch.setattr(router, "route", fake_route)
    monkeypatch.setattr(library_state, "collection_status", lambda *a: "empty")
    out = asyncio.run(pipe_cli.ask(p, "what is today's date?"))
    assert out["answer"].startswith("Today is Friday.") and texts.GENERAL_NOTE in out["answer"]
    intent["v"] = "ranked_retrieval"
    assert asyncio.run(pipe_cli.ask(p, "What is PTFE?"))["answer"] == texts.EMPTY_LIBRARY_MSG


def test_the_command_line_never_imports_openwebuis_database_module(monkeypatch):
    # every database access goes through _db_engine: importing open_webui.internal.db outside Open WebUI starts its own
    # engine (and migration step)
    mod = pipe_cli.load_pipe_module()
    e = _engine({"COLLECTION": "memento-x-rag"})
    with e.begin() as c:
        c.execute(text("CREATE TABLE memento_ingestion_status (source TEXT)"))
        c.execute(text("INSERT INTO memento_ingestion_status VALUES ('Paper being indexed')"))
    p = pipe_cli.make_pipe(mod, e)
    import builtins
    real = builtins.__import__

    def guard(name, *a, **k):
        if name.startswith("open_webui.internal"):
            raise AssertionError("open_webui.internal imported")
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", guard)
    assert library_state.in_progress_sources(p.engine) == {"Paper being indexed"}


class _RecordingLLM:
    """Router → a fixed route; the streamed answer call records its messages."""
    calls: list = []
    intent = "ranked_retrieval"

    def __init__(self, *a, **k):
        self.chat = self
        self.completions = self

    async def create(self, **kw):
        _RecordingLLM.calls.append(kw)
        if kw.get("stream"):
            async def gen():
                delta = types.SimpleNamespace(content="Answer [1].", reasoning_content=None, reasoning=None)
                yield types.SimpleNamespace(choices=[types.SimpleNamespace(delta=delta)])
            return gen()
        msg = types.SimpleNamespace(content='{"intent":"%s","query":"q","reasoning":"deep"}' % _RecordingLLM.intent)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


def _no_openwebui(monkeypatch):
    import builtins
    real = builtins.__import__

    def guard(name, *a, **k):
        if name.startswith("open_webui"):
            raise AssertionError(f"{name} imported on the command-line path")
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", guard)


def test_a_library_question_sends_the_model_exactly_what_the_chat_sends(monkeypatch):
    # the main path (router → filter → search → rerank → cap → prompt → citations) through the command line: the model
    # gets the messages the RQ3 fidelity test pins for the chat, and the sources are the chat's citation cards in order
    import openai
    import yaml
    import memento.retrieval.search as search
    import memento.common.model_servers as vc
    from memento.retrieval.answer_pipeline import build_messages, system_prompt_for
    from memento.retrieval.search import point_to_hit
    mod = pipe_cli.load_pipe_module()
    p = pipe_cli.make_pipe(mod, _engine({"COLLECTION": "memento-x-rag", "DOMAIN_NAME": "generic",
                                         "GRAPH_RETRIEVAL": False, "VALIDATION_ENABLED": False,
                                         "ENABLE_QUERY_REWRITE": True, "THINKING_AUTO": False, "RELATED_DOCS": False}))
    payloads = [{"text": "Alumina reached 99 % density.", "source": "P", "page": 0, "type": "text",
                 "section": "Results", "section_raw": "4.1 Density"},
                {"text": "SPS used 1400 °C.", "source": "Q", "page": 3, "type": "text"}]
    points = [types.SimpleNamespace(payload=d, score=0.9 - i / 10, id=i) for i, d in enumerate(payloads)]
    _RecordingLLM.calls, _RecordingLLM.intent = [], "ranked_retrieval"
    monkeypatch.setattr(openai, "AsyncOpenAI", _RecordingLLM)
    monkeypatch.setattr(library_state, "collection_status", lambda c, coll: "ok")
    monkeypatch.setattr(library_state, "load_documents", lambda *a: {"P": {}, "Q": {}})
    monkeypatch.setattr(documents, "_resolve_paper", lambda q, s, d=None: None)
    monkeypatch.setattr(library_state, "qdrant_query", lambda *a, **k: types.SimpleNamespace(points=points))
    monkeypatch.setattr(vc, "vl_embed", lambda *a, **k: [0.0] * 8)
    monkeypatch.setattr(search, "rerank", lambda q, hits, *a, **k: hits)
    _no_openwebui(monkeypatch)                     # in_progress_sources etc. run for real
    question = "What density did alumina reach?"
    out = asyncio.run(pipe_cli.ask(p, question))

    sent = [c for c in _RecordingLLM.calls if c.get("stream")]
    assert len(sent) == 1, "the pipe did not reach the answer step"
    hits = [point_to_hit(pt.payload, pt.score) for pt in points]
    dom_prompt = yaml.safe_load((ROOT / "config/domains/generic.yaml").read_text())["system_prompt"].rstrip() \
        + " " + prompts.PLAIN_FORMULAS
    assert sent[0]["messages"] == build_messages(system_prompt_for(dom_prompt, False), hits, question,
                                                 max_images=p.valves.MAX_IMAGES_TO_LLM)
    assert out["answer"].startswith("Answer [1].") and len(out["sources"]) == 2
    assert out["sources"][0].startswith("P") and out["sources"][1].startswith("Q")


def test_an_english_question_on_a_german_passage_gets_the_english_line_and_nothing_else(monkeypatch):
    # with a German passage among the sources the model gets the evaluated prompt plus EN_ANSWER_ADDENDUM; with English
    # passages only, the test above holds unchanged (the prompt the RQ3 evaluation pins)
    import openai
    import yaml
    import memento.retrieval.search as search
    import memento.common.model_servers as vc
    from memento.retrieval.answer_pipeline import build_messages, system_prompt_for
    from memento.retrieval.search import point_to_hit
    mod = pipe_cli.load_pipe_module()
    p = pipe_cli.make_pipe(mod, _engine({"COLLECTION": "memento-x-rag", "DOMAIN_NAME": "generic",
                                         "GRAPH_RETRIEVAL": False, "VALIDATION_ENABLED": False,
                                         "ENABLE_QUERY_REWRITE": True, "THINKING_AUTO": False, "RELATED_DOCS": False}))
    payloads = [{"text": "Alumina reached 99 % density.", "source": "P", "page": 0, "type": "text"},
                {"text": "Die Dichte der Proben wurde nach dem Sintern bei 1400 °C mit der Archimedes-Methode bestimmt "
                         "und war bei allen Proben höher als 98 %.", "source": "Q", "page": 3, "type": "text"}]
    points = [types.SimpleNamespace(payload=d, score=0.9 - i / 10, id=i) for i, d in enumerate(payloads)]
    _RecordingLLM.calls, _RecordingLLM.intent = [], "ranked_retrieval"
    monkeypatch.setattr(openai, "AsyncOpenAI", _RecordingLLM)
    monkeypatch.setattr(library_state, "collection_status", lambda c, coll: "ok")
    monkeypatch.setattr(library_state, "load_documents", lambda *a: {"P": {}, "Q": {}})
    monkeypatch.setattr(documents, "_resolve_paper", lambda q, s, d=None: None)
    monkeypatch.setattr(library_state, "qdrant_query", lambda *a, **k: types.SimpleNamespace(points=points))
    monkeypatch.setattr(vc, "vl_embed", lambda *a, **k: [0.0] * 8)
    monkeypatch.setattr(search, "rerank", lambda q, hits, *a, **k: hits)
    _no_openwebui(monkeypatch)
    question = "What density did alumina reach?"
    asyncio.run(pipe_cli.ask(p, question))

    sent = [c for c in _RecordingLLM.calls if c.get("stream")]
    assert len(sent) == 1, "the pipe did not reach the answer step"
    hits = [point_to_hit(pt.payload, pt.score) for pt in points]
    dom_prompt = yaml.safe_load((ROOT / "config/domains/generic.yaml").read_text())["system_prompt"].rstrip() \
        + " " + prompts.PLAIN_FORMULAS
    assert sent[0]["messages"] == build_messages(system_prompt_for(dom_prompt + texts.EN_ANSWER_ADDENDUM, False), hits,
                                                 question, max_images=p.valves.MAX_IMAGES_TO_LLM)
    assert sent[0]["messages"][0]["content"].count("The question is in English") == 1


def test_research_gaps_run_through_the_command_line_without_openwebui(monkeypatch):
    import openai
    mod = pipe_cli.load_pipe_module()
    p = pipe_cli.make_pipe(mod, _engine({"COLLECTION": "memento-x-rag", "DOMAIN_NAME": "generic",
                                         "ENABLE_QUERY_REWRITE": True}))
    _RecordingLLM.calls, _RecordingLLM.intent = [], "ranked_retrieval"
    monkeypatch.setattr(openai, "AsyncOpenAI", _RecordingLLM)
    monkeypatch.setattr(library_state, "collection_status", lambda c, coll: "ok")
    monkeypatch.setattr(library_state, "load_documents", lambda *a: {})
    _no_openwebui(monkeypatch)
    out = asyncio.run(pipe_cli.ask(p, "Research gaps across the library"))
    assert out["answer"] and "Error" not in out["answer"] and "`E-" not in out["answer"]
    assert not [c for c in _RecordingLLM.calls if c.get("stream")]       # empty ledger: no answer-model call


def test_the_model_choices_are_the_configured_models():
    from memento.query import models
    assert list(pipe_cli.model_ids()) == [v.id for v in models.load_variants()]    # config/query_models.yaml, in order
    with pytest.raises(SystemExit):
        pipe_cli.main(["question", "--model", "no-such-model"])
