# Answers within a time budget (about 2 minutes, 4–5 with Thorough check). Thinking that runs past the cut-off is closed
# and the request goes again without thinking; a cloud model too slow for that → the local model, with a note; an answer
# the reader cannot see yet (held) stops at the budget's end; optional steps need time left.
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from memento.config.paths import PROJECT_ROOT
from _owui_loader import load_function
from memento.query import generation, library_state, request, texts
from memento.query.scope import documents
from memento.query.flows import gap_answer, gap_report

ROOT = PROJECT_ROOT
sys.path.insert(0, str(ROOT))
qp = load_function(ROOT / "openwebui/functions/query_pipe.py", "memento_query")
from memento.query import models as qm  # noqa: E402

V = {v.id: v for v in qm.load_variants()}
LOCAL = next(v for v in V.values() if not v.is_cloud)
CLOUD = next(v for v in V.values() if v.is_cloud and v.thinking)


def _content(t):
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=t, reasoning_content=None))])


def _reasoning(t):
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=None, reasoning_content=t))])


class Stream:
    """A model stream: `items` with `delay` s before each; `forever` repeats the last item without end."""

    def __init__(self, items, delay=0.0, forever=False, start_delay=0.0):
        self.items, self.delay, self.forever, self.start_delay, self.i, self.closed = items, delay, forever, start_delay, 0, False
        self.response = SimpleNamespace(headers={})

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.closed:
            raise StopAsyncIteration
        await asyncio.sleep(self.start_delay if self.i == 0 else self.delay)
        if self.i >= len(self.items):
            if not self.forever:
                raise StopAsyncIteration
            return self.items[-1]
        self.i += 1
        return self.items[self.i - 1]

    async def close(self):
        self.closed = True


def _pipe(monkeypatch, tmp_path, make):
    """`make(base, thinking) -> Stream`: what the local ("http://local/v1") or cloud endpoint sends."""
    p = qp.Pipe()
    p.valves.LLM_BASE_URL, p.valves.LLM_MODEL = "http://local/v1", "qwen3.8-27b"
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path))
    monkeypatch.setenv(CLOUD.base_url_env, "http://cloud/v1")
    monkeypatch.setenv(CLOUD.key_env, "secret")
    calls, streams = [], []

    def client(base_url, api_key, cloud):
        async def create(**kw):
            thinking = ((kw.get("extra_body") or {}).get("chat_template_kwargs") or {}).get("enable_thinking")
            calls.append({"base": base_url, "thinking": thinking})
            s = make(base_url, thinking)
            streams.append(s)
            return s
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(generation, "client", client)
    return p, calls, streams


def _drain(p, variant, *, think=True, budget=None, held=False):
    async def go():
        chunks, answered, note = await generation.open_answer(p.valves, variant, messages=[{"role": "user", "content": "q"}],
                                                      think=think, budget=budget, held=held)
        text = await generation.drain_content(chunks, answered.is_cloud)
        return note + text, answered
    return asyncio.run(go())


def test_the_budget_has_a_thinking_cut_off_before_its_end():
    async def go():
        b = generation.Budget(120, 45)
        return round(b.left()), round(b.think_end - (b.end - 45), 3)
    assert asyncio.run(go()) == (120, 0.0)


def test_thinking_past_the_cut_off_is_asked_again_without_thinking(monkeypatch, tmp_path):
    def make(base, thinking):
        return Stream([_reasoning("hmm ")], delay=0.02, forever=True) if thinking else Stream([_content("plain answer")])
    p, calls, streams = _pipe(monkeypatch, tmp_path, make)
    text, answered = _drain(p, LOCAL, budget=_budget(0.6, 0.4))
    assert text == "plain answer" and answered.id == LOCAL.id
    assert [c["thinking"] for c in calls] == [True, False] and streams[0].closed


def test_an_answer_already_under_way_is_not_cut(monkeypatch, tmp_path):
    def make(base, thinking):
        return Stream([_reasoning("think "), _content("part one, "), _content("part two")], delay=0.15)
    p, calls, _ = _pipe(monkeypatch, tmp_path, make)
    text, _ = _drain(p, LOCAL, budget=_budget(0.8, 0.5))             # cut-off at 0.3 s, the text started at 0.3 s
    assert text == "part one, part two" and len(calls) == 1


def test_a_cloud_model_too_slow_without_thinking_hands_over_to_the_local_model(monkeypatch, tmp_path):
    def make(base, thinking):
        if base == "http://local/v1":
            return Stream([_content("local answer")])
        if thinking:
            return Stream([_reasoning("cloud thinks ")], delay=0.02, forever=True)
        return Stream([_content("too late")], start_delay=5)
    p, calls, _ = _pipe(monkeypatch, tmp_path, make)
    text, answered = _drain(p, CLOUD, budget=_budget(1.0, 0.6))
    assert text == texts.SLOW_NOTE + "local answer" and answered.id == LOCAL.id
    assert [(c["base"], c["thinking"]) for c in calls] == [("http://cloud/v1", True), ("http://cloud/v1", False),
                                                           ("http://local/v1", False)]


def test_a_held_answer_stops_at_the_budgets_end(monkeypatch, tmp_path):
    def make(base, thinking):
        return Stream([_content(f"w{k} ") for k in range(20)], delay=0.05)
    p, _, streams = _pipe(monkeypatch, tmp_path, make)
    held, _ = _drain(p, LOCAL, think=False, budget=_budget(0.3, 0.1), held=True)
    assert held.endswith(texts.TIMEOUT_NOTE) and "w19" not in held and streams[0].closed
    shown, _ = _drain(p, LOCAL, think=False, budget=_budget(0.3, 0.1))   # a visible answer finishes
    assert "w19" in shown


def test_without_a_budget_nothing_changes(monkeypatch, tmp_path):
    def make(base, thinking):
        return Stream([_reasoning("long "), _content("answer")], delay=0.05)
    p, calls, _ = _pipe(monkeypatch, tmp_path, make)
    text, _ = _drain(p, LOCAL)
    assert text == "answer" and len(calls) == 1


def _budget(seconds, reserve):
    """A budget made inside the event loop the answer runs in (asyncio.run makes a new one per test)."""
    return _LazyBudget(seconds, reserve)


class _LazyBudget:
    def __init__(self, seconds, reserve):
        self.seconds, self.reserve, self._b = seconds, reserve, None

    def _get(self):
        if self._b is None:
            self._b = generation.Budget(self.seconds, self.reserve)
        return self._b

    def __getattr__(self, name):
        return getattr(self._get(), name)


def test_the_extensions_are_skipped_without_enough_time(monkeypatch, tmp_path):
    p = qp.Pipe()

    async def never(*a, **k):
        pytest.fail("no model call without time left")
    monkeypatch.setattr(generation, "open_answer", lambda valves, *a, **k: never(*a, **k))

    async def go():
        return await gap_answer.gap_suggestions(p.valves, source="DocA", docs={}, variant=LOCAL, authors="Limits [1].", chunks=[],
                                        question="Suggest future work", think=True, budget=None, db_engine=qp._db_engine,
                                        time_budget=generation.Budget(10, 5))
    assert asyncio.run(go()) == texts.SUGGEST_FAILED_NOTE


@pytest.mark.parametrize("thorough, warm, want", [(False, False, 120), (True, False, 270), (False, True, None)])
def test_each_request_gets_its_budget(thorough, warm, want):
    p = qp.Pipe()

    async def go():
        b = generation.response_budget(p.valves, thorough=thorough, warm=warm)
        return None if b is None else round(b.left())
    assert asyncio.run(go()) == want


def test_thorough_check_gets_the_time_left_at_most():
    p = qp.Pipe()

    async def go():
        return (round(generation.check_seconds(p.valves, generation.Budget(200, 45))), round(generation.check_seconds(p.valves, None)),
                round(generation.check_seconds(p.valves, generation.Budget(3, 1))))
    assert asyncio.run(go()) == (200, p.valves.VERIFY_TIMEOUT, 10)


# ── Thorough check made visible while waiting ──
def _statuses(monkeypatch, filter_ids):
    import qdrant_client
    p = qp.Pipe()
    p.valves.COLLECTION = "memento-x-rag"

    async def fake_report(topic, *, variant, emit, status, document=None, page=1):
        yield "REPORT"
    monkeypatch.setattr(gap_report, "gap_report", lambda valves, *a_, db_engine, **k_: fake_report(*a_, **k_))
    monkeypatch.setattr(library_state, "collection_status", lambda *a, **k: "ok")

    class Q:
        def __init__(self, *a, **k):
            pass
    monkeypatch.setattr(qdrant_client, "QdrantClient", Q)
    said = []

    async def emit(e):
        if e.get("type") == "status":
            said.append((e.get("data") or {}).get("description"))
    body = {"model": "memento_query.local", "messages": [{"role": "user", "content": "Research gaps across the library"}]}

    async def go():
        return "".join([c async for c in p.pipe(body, {"id": "u1", "role": "user"}, emit, None,
                                                {"chat_id": "c1", "filter_ids": filter_ids}, None)])
    asyncio.run(go())
    return said


def test_the_first_status_says_thorough_check_is_on(monkeypatch):
    said = _statuses(monkeypatch, [request.THOROUGH_FILTER_ID])
    assert said and said[0] == texts.THOROUGH_ON_STATUS
    assert texts.THOROUGH_ON_STATUS not in _statuses(monkeypatch, [])
    assert "minute" not in texts.THOROUGH_ON_STATUS                    # no time promises


# ── the library list is typed out like other finished answers ──
def test_the_library_list_is_typed_out(monkeypatch):
    import qdrant_client
    from memento.query import router as qr
    p = qp.Pipe()
    p.valves.COLLECTION = "memento-x-rag"
    p.valves.REPLAY_CHARS_PER_S = 100000                 # typed, fast in the test
    names = {f"Doc{k}": {"title": f"Document number {k}"} for k in range(30)}
    monkeypatch.setattr(library_state, "collection_status", lambda *a, **k: "ok")
    monkeypatch.setattr(library_state, "load_documents", lambda *a, **k: names)
    monkeypatch.setattr(documents, "publish_library", lambda *a, **k: None)
    monkeypatch.setattr(library_state, "in_progress_sources", lambda *a, **k: set())
    monkeypatch.setattr(documents, "display_titles", lambda srcs, dom: {s: names[s]["title"] for s in srcs})

    async def route(*a, **k):
        return qr.RouterResponse(intent="corpus_listing", query="which papers are in the library", reasoning="light")
    monkeypatch.setattr(qr, "route", route)

    class Q:
        def __init__(self, *a, **k):
            pass
    monkeypatch.setattr(qdrant_client, "QdrantClient", Q)
    body = {"model": "memento_query.local", "messages": [{"role": "user", "content": "which papers are in the library"}]}

    async def go():
        return [c async for c in p.pipe(body, {"id": "u1", "role": "user"}, lambda e: asyncio.sleep(0), None,
                                         {"chat_id": "c1"}, None)]
    pieces = asyncio.run(go())
    text = "".join(pieces)
    assert "Document number 0" in text and "Show 10 more" in text
    assert len(pieces) > 50                              # typed in small pieces, not one paste
    assert any(p_.startswith("<details") for p_ in pieces)   # the collapsed rest arrives whole
