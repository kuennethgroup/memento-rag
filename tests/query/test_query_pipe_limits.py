# Time limits and the Stop button in the Query pipe. An answer that runs too long or goes silent is stopped with a note;
# Stop (Open WebUI cancels the task) closes the model connection, stops Thorough check before its next call, and sends
# nothing more to the chat.
import asyncio
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from memento.config.paths import PROJECT_ROOT
from _owui_loader import load_function
from memento.query import generation, texts
from memento.query.scope import conversation

ROOT = PROJECT_ROOT
sys.path.insert(0, str(ROOT))
qp = load_function(ROOT / "openwebui/functions/query_pipe.py", "memento_query")


def _chunk(text):
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text, reasoning_content=None))])


class SlowStream:
    """Chunks with a pause before each one from `slow_from` onwards; records close()."""

    def __init__(self, texts, pause=0.0, slow_from=0):
        self.texts, self.pause, self.slow_from, self.i, self.closed = texts, pause, slow_from, 0, False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.pause and self.i >= self.slow_from:
            await asyncio.sleep(self.pause)
        if self.i >= len(self.texts):
            raise StopAsyncIteration
        self.i += 1
        return _chunk(self.texts[self.i - 1])

    async def close(self):
        self.closed = True


async def _collect(chunks):
    return [c.choices[0].delta.content async for c in chunks]


def test_limited_passes_a_normal_stream_and_closes_it():
    s = SlowStream(["a", "b"])

    async def go():
        return await _collect(generation.limited(s, s, asyncio.get_running_loop().time() + 5))
    assert asyncio.run(go()) == ["a", "b"] and s.closed


def test_limited_stops_at_the_overall_limit(monkeypatch):
    s = SlowStream(["a", "b", "c", "d"], pause=0.05, slow_from=1)

    async def go():
        got = []
        with pytest.raises(generation.AnswerTimeout):
            async for c in generation.limited(s, s, asyncio.get_running_loop().time() + 0.12):
                got.append(c.choices[0].delta.content)
        return got
    got = asyncio.run(go())
    assert got[0] == "a" and len(got) < 4 and s.closed           # partial output kept by the caller, model closed


def test_limited_stops_a_silent_stream(monkeypatch):
    monkeypatch.setattr(generation, "STALL_TIMEOUT", 0.05)
    s = SlowStream(["a", "late"], pause=0.3, slow_from=1)

    async def go():
        with pytest.raises(generation.AnswerTimeout):
            await _collect(generation.limited(s, s, asyncio.get_running_loop().time() + 60))
    asyncio.run(go())
    assert s.closed


def test_stop_closes_the_model_stream():
    s = SlowStream(["a"] * 100, pause=0.05)

    async def go():
        task = asyncio.create_task(_collect(generation.limited(s, s, asyncio.get_running_loop().time() + 60)))
        await asyncio.sleep(0.2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    asyncio.run(go())
    assert s.closed


def test_local_answers_get_the_limit_too(monkeypatch):
    p = qp.Pipe()
    p.valves.LLM_BASE_URL, p.valves.LLM_MODEL, p.valves.ANSWER_TIMEOUT = "http://local/v1", "qwen3.8-27b", 0.1
    slow = SlowStream(["a", "b", "c", "d", "e"], pause=0.05, slow_from=1)

    def fake_client(base_url, api_key, cloud):
        async def create(**kw):
            return slow
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(generation, "client", fake_client)
    variant = next(v for v in generation.copy_variants() if not v.is_cloud)

    async def go():
        chunks, used, note = await generation.open_answer(p.valves, variant, messages=[{"role": "user", "content": "q"}], think=False)
        with pytest.raises(generation.AnswerTimeout):
            await _collect(chunks)
    asyncio.run(go())
    assert slow.closed


@pytest.mark.parametrize("cloud", [True, False])
def test_drain_content_ends_with_the_timeout_note(monkeypatch, cloud):
    monkeypatch.setattr(generation, "STALL_TIMEOUT", 0.05)
    s = SlowStream(["partial", "late"], pause=0.3, slow_from=1)

    async def go():
        return await generation.drain_content(generation.limited(s, s, asyncio.get_running_loop().time() + 60), cloud=cloud)
    assert asyncio.run(go()) == "partial\n\n" + texts.TIMEOUT_NOTE


def test_timeout_notes_leave_the_history():
    assert conversation.history_text("Partial\n\n" + texts.TIMEOUT_NOTE) == "Partial"
    assert conversation.history_text("Answer.\n\n" + texts.CHECK_TIMEOUT_NOTE) == "Answer."


def test_guarded_llm_stops_the_check_before_its_next_call():
    calls = []
    inner = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kw: calls.append(kw) or "reply")))
    stop = threading.Event()
    g = generation.GuardedLLM(inner, stop, time.monotonic() + 60)
    assert g.chat.completions.create(model="m") == "reply"
    assert 0 < calls[0]["timeout"] <= 60                        # one call can never outlast the check's limit
    stop.set()
    with pytest.raises(generation.CheckStopped):
        g.chat.completions.create(model="m")
    assert len(calls) == 1
    late = generation.GuardedLLM(inner, threading.Event(), time.monotonic() - 1)
    with pytest.raises(generation.CheckStopped):
        late.chat.completions.create(model="m")


def test_check_stopped_is_not_swallowed_by_the_verifiers():
    # verifiers turn a failed call (any Exception) into "unjudged" and carry on; a stopped check must end instead
    assert not issubclass(generation.CheckStopped, Exception) and issubclass(generation.CheckStopped, BaseException)


def test_after_stop_only_the_status_line_ends():
    sent = []

    async def pipe_like():
        try:
            await asyncio.sleep(10)
            sent.append("answer text")
        finally:
            sent.append("Stopped" if generation.stopped() else "Done")

    async def go():
        t = asyncio.create_task(pipe_like())
        await asyncio.sleep(0.05)
        t.cancel()
        with pytest.raises(asyncio.CancelledError):
            await t
        normal = asyncio.create_task(asyncio.sleep(0))
        await normal
        assert generation.stopped() is False                           # outside a stopped task: statuses go out as usual
    asyncio.run(go())
    assert sent == ["Stopped"]


# ── Thorough check made visible: a checked answer says so ──
def test_a_checked_answer_says_it_was_checked():
    ok = SimpleNamespace(arm="A", claims=[1, 2], trace=[], rounds=1)
    assert generation.checked_text("The answer [1].", ok, None, False) == "The answer [1].\n\n" + texts.CHECKED_NOTE
    assert generation.checked_text("The answer [1].", ok, RuntimeError("x"), False) == "The answer [1]."     # failed: no note
    assert generation.checked_text("The answer [1].", None, None, False) == "The answer [1]."
    late = "The answer [1].\n\n" + texts.CHECK_TIMEOUT_NOTE
    assert generation.checked_text(late, ok, None, True) == late                                       # the timeout note only
    assert "statements" not in texts.CHECKED_NOTE                  # the outcome stays in the log


def test_the_checked_note_leaves_the_history():
    assert conversation.history_text("Answer.\n\n" + texts.CHECKED_NOTE) == "Answer."


def test_the_slow_cloud_note_leaves_the_history():
    assert conversation.history_text(texts.SLOW_NOTE + "Answer.") == "Answer."


def test_a_partly_judged_check_says_so_and_is_not_stored_as_checked():
    # claims a judge call failed for stay in the text; never "Checked with Thorough check"
    part = SimpleNamespace(arm="A", claims=[1, 2, 3], unjudged=1, trace=[], rounds=1)
    out = generation.checked_text("The answer [1].", part, None, False)
    assert out == "The answer [1].\n\n" + texts.PARTLY_CHECKED_NOTE and texts.CHECKED_NOTE not in out
    none = SimpleNamespace(arm="A", claims=[1, 2], unjudged=2, trace=[], rounds=1)
    assert texts.CHECKED_NOTE not in generation.checked_text("The answer [1].", none, None, False)
    assert conversation.history_text("Answer.\n\n" + texts.PARTLY_CHECKED_NOTE) == "Answer."
    assert generation.check_ok(part, None, False) is False and generation.check_ok(
        SimpleNamespace(arm="A", claims=[1], unjudged=0), None, False) is True


# ── the log names a stop only when the reader stopped ──
# The thinking cut (generation.think_capped: asyncio.wait_for on the next chunk, at the budget) cancels the model stream
# too, and must not be logged as "answer stopped by the user".

def test_the_thinking_cut_is_not_logged_as_a_stop_by_the_user(caplog):
    s = SlowStream(["a"] * 100, pause=0.05)

    async def go():
        it = generation.limited(s, s, asyncio.get_running_loop().time() + 60).__aiter__()
        await it.__anext__()
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(it.__anext__(), 0.01)        # as generation.think_capped does at the budget
    with caplog.at_level("INFO"):
        asyncio.run(go())
    assert s.closed                                                # the model connection is still closed
    assert not [r for r in caplog.records if "stopped by the user" in r.getMessage()]


def test_a_stop_by_the_reader_is_logged_once(caplog):
    p = qp.Pipe()

    async def slow(body, user, emitter, files, metadata, rec):
        yield "a"
        await asyncio.sleep(60)
        yield "b"
    p._pipe = slow

    async def go():
        async def read():
            return [c async for c in p.pipe({"messages": [{"role": "user", "content": "What is PEI?"}]},
                                            {"id": "u"}, None, None, {})]
        task = asyncio.create_task(read())
        await asyncio.sleep(0.05)
        task.cancel()                                              # Open WebUI's Stop cancels the request's task
        with pytest.raises(asyncio.CancelledError):
            await task
    with caplog.at_level("INFO"):
        asyncio.run(go())
    assert len([r for r in caplog.records if "answer stopped by the user" in r.getMessage()]) == 1


def test_a_cancelled_warmup_is_not_a_stop_by_the_user(caplog):
    p = qp.Pipe()

    async def slow(body, user, emitter, files, metadata, rec):
        yield "a"
        await asyncio.sleep(60)
    p._pipe = slow

    async def go():
        async def read():
            return [c async for c in p.pipe({"messages": [{"role": "user", "content": 'Summarize the document "D"'}]},
                                            {"id": "w"}, None, None, {"warmup": True, "warmup_result": {}})]
        task = asyncio.create_task(read())
        await asyncio.sleep(0.05)
        task.cancel()                                              # the worker stops (newer code loaded)
        with pytest.raises(asyncio.CancelledError):
            await task
    with caplog.at_level("INFO"):
        asyncio.run(go())
    assert not [r for r in caplog.records if "stopped by the user" in r.getMessage()]
    assert [r for r in caplog.records if "warm-up answer cancelled" in r.getMessage()]
