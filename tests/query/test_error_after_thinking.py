# A model that fails while it thinks: the error reaches the reader outside the collapsed "Thinking" block, and the block
# is closed once.
import asyncio
from types import SimpleNamespace

import pytest

from memento.query import generation, presentation, texts
from memento.query.flows import answer as answer_flow, gap_answer
from memento.research_gaps import passages


def _chunk(reasoning=None, content=None):
    delta = SimpleNamespace(reasoning_content=reasoning, content=content)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


async def _thinks_then_fails():
    yield _chunk(reasoning="weighing the passages")
    raise RuntimeError("stream broke")


async def _thinks_then_ends():
    yield _chunk(reasoning="weighing the passages")


def _opens(stream):
    async def open_answer(valves, variant, **kw):
        return stream(), variant, None
    return open_answer


async def _noop(*a, **k):
    pass


def _turn(**extra):
    valves = SimpleNamespace(MAX_FIGURES_SHOWN=0, FILES_URL="http://x", GAP_THINKING=True, GAP_THINKING_BUDGET=0,
                             VALIDATION_ENABLED=False, GAP_BUDGET_TOKENS=100)
    base = dict(valves=valves, variant=SimpleNamespace(id="local", is_cloud=False, label="Local"), think=True,
                whole_doc=False, budget=None, verify=False, system_prompt="s", answer_history=[], user_content="q",
                image_intent=False, query_intent="ranked_retrieval", status=_noop, emit=_noop, metadata={}, hits=[],
                original_query="q", query="q", paper=None, sources=set(), dom={}, suggest=False, history=[], doc_chat=object(),
                docs={}, rec=SimpleNamespace(key=None, skip_reason=None), gap_base_think=True, client=None, collection="c",
                engine=None, done=False)
    return SimpleNamespace(**{**base, **extra})


def _collect(gen):
    async def run():
        return [p async for p in gen]
    return "".join(asyncio.run(run()))


def _assert_error_outside_thinking(text, kind):
    message = texts.user_error(kind, RuntimeError("stream broke"))
    assert text.count("<think>") == 1 and text.count("</think>") == 1
    assert text.index("</think>") < text.index(message)


def test_answer_error_while_thinking_is_shown_outside_the_block(monkeypatch):
    monkeypatch.setattr(generation, "open_answer", _opens(_thinks_then_fails))
    _assert_error_outside_thinking(_collect(answer_flow.stream_answer(_turn())), "answer")


def test_answer_error_before_any_thinking_adds_no_closing_tag(monkeypatch):
    async def fails(*a, **k):
        raise RuntimeError("stream broke")
    monkeypatch.setattr(generation, "open_answer", fails)
    text = _collect(answer_flow.stream_answer(_turn()))
    assert "</think>" not in text and texts.user_error("answer", RuntimeError("stream broke")) in text


def test_a_later_error_does_not_close_the_block_twice(monkeypatch):
    monkeypatch.setattr(generation, "open_answer", _opens(_thinks_then_ends))

    def fails(*a, **k):
        raise RuntimeError("stream broke")
    monkeypatch.setattr(presentation, "sources_marker", fails)
    _assert_error_outside_thinking(_collect(answer_flow.stream_answer(_turn())), "answer")


def test_gap_answer_error_while_thinking_is_shown_outside_the_block(monkeypatch):
    monkeypatch.setattr(generation, "open_answer", _opens(_thinks_then_fails))

    async def target(*a, **k):
        return "Some document"
    monkeypatch.setattr(gap_answer, "gap_target", target)
    monkeypatch.setattr(gap_answer, "gap_passages", lambda *a, **k: ([{"text": "a gap"}], False))
    monkeypatch.setattr(presentation, "gap_citation", lambda *a, **k: {})
    monkeypatch.setattr(passages, "build_gap_prompt", lambda *a, **k: "prompt")
    text = _collect(gap_answer.reply(_turn(query_intent="gap_query")))
    _assert_error_outside_thinking(text, "gap")
