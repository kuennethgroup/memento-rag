# The Query pipe's model variants.
import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import httpx
import openai
import pytest
from memento.config.paths import PROJECT_ROOT
from _owui_loader import load_function
from memento.query import generation, texts
from memento.query.scope import conversation

ROOT = PROJECT_ROOT
sys.path.insert(0, str(ROOT))
qp = load_function(ROOT / "openwebui/functions/query_pipe.py", "memento_query")
from memento.common import cloud_quota as cq  # noqa: E402
from memento.query import models as qm  # noqa: E402

V = {v.id: v for v in qm.load_variants()}
HEADERS = {"X-RateLimit-Remaining-Hour": "150", "X-RateLimit-Limit-Hour": "200", "X-RateLimit-Remaining-Day": "900",
           "X-RateLimit-Limit-Day": "1000", "X-RateLimit-Remaining-Month": "2900", "X-RateLimit-Limit-Month": "3000",
           "X-RateLimit-Remaining-Minute": "29", "X-RateLimit-Limit-Minute": "30", "RateLimit-Reset": "20"}


def _chunk(text):
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text, reasoning_content=None))])


class FakeStream:
    """An AsyncStream stand-in: response headers, optional delay before chunk `delay_from` onwards, optional break."""

    def __init__(self, chunks, headers=None, fail_after=None, delay=0.0, delay_from=0):
        self.chunks, self.fail_after, self.delay, self.delay_from, self.i = chunks, fail_after, delay, delay_from, 0
        self.response = SimpleNamespace(headers=headers or {})
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.delay and self.i >= self.delay_from:
            await asyncio.sleep(self.delay)
        if self.fail_after is not None and self.i == self.fail_after:
            raise openai.APIConnectionError(message="lost", request=httpx.Request("POST", "http://cloud/v1"))
        if self.i >= len(self.chunks):
            raise StopAsyncIteration
        self.i += 1
        return self.chunks[self.i - 1]

    async def close(self):
        self.closed = True


def _status_error(code):
    resp = httpx.Response(code, headers=HEADERS, request=httpx.Request("POST", "http://cloud/v1"))
    cls = openai.RateLimitError if code == 429 else openai.InternalServerError
    return cls("x", response=resp, body=None)


def _pipe(monkeypatch, tmp_path, cloud):
    """A Pipe whose clients are fakes: the local endpoint answers "local answer", the cloud one runs `cloud()`."""
    p = qp.Pipe()
    p.valves.LLM_BASE_URL, p.valves.LLM_MODEL = "http://local/v1", "qwen3.8-27b"
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path))
    monkeypatch.setenv("OPENAI_BASE_URL", "http://cloud/v1")
    monkeypatch.setenv("CLOUD_API_KEY", "secret")
    calls = []

    def fake_client(base_url, api_key, cloud_call):
        async def create(**kw):
            calls.append({"base": base_url, "key": api_key, **kw})
            if base_url == "http://local/v1":
                return FakeStream([_chunk("local answer")])
            return cloud()
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    monkeypatch.setattr(generation, "client", fake_client)
    return p, calls


def _open(p, variant, **kw):
    async def go():
        chunks, used, note = await generation.open_answer(p.valves, variant, messages=[{"role": "user", "content": "q"}], think=True, **kw)
        return "".join([c.choices[0].delta.content async for c in chunks]), used, note
    return asyncio.run(go())


def test_pipes_lists_the_variants():
    # the Query variants, then Memento Document Chat
    assert [p["id"] for p in qp.Pipe().pipes()] == ["local", "qwen35", "qwen36", "document-chat"]


def test_broken_config_keeps_the_local_model(monkeypatch, tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("- {id: a, provider: moon}")
    monkeypatch.setattr(qm, "CONFIG", bad)
    assert [p["id"] for p in qp.Pipe().pipes()] == ["local", "document-chat"]     # Document Chat is not a variant
    assert qm.resolve("memento_query.qwen35", generation.copy_variants()).id == "local"


def test_answered_by_line_and_notes_leave_the_history():
    assert generation.answered_by("Qwen 3.5 397B (Academic Cloud)", "Qwen 3.8 27B (local)") == \
        "\n\n_Answered by Qwen 3.5 397B (Academic Cloud), checked by Qwen 3.8 27B (local)._\n"
    text = texts.QUOTA_NOTE + "An answer." + generation.answered_by("Qwen 3.8 27B (local)")
    assert conversation.history_text(text) == "An answer."
    assert conversation.history_text("Partial" + "\n\n" + texts.STOPPED_NOTE) == "Partial"


def test_local_variant_uses_the_valves(monkeypatch, tmp_path):
    p, calls = _pipe(monkeypatch, tmp_path, lambda: FakeStream([_chunk("cloud")]))
    text, used, note = _open(p, V["local"])
    assert (text, used.id, note) == ("local answer", "local", "")
    assert calls[0]["model"] == "qwen3.8-27b" and calls[0]["extra_body"]["chat_template_kwargs"]["enable_thinking"]


def test_cloud_answer_records_quota_and_skips_the_thinking_switch_without_one(monkeypatch, tmp_path):
    # a cloud model without Qwen's thinking switch (an explicit fixture)
    import dataclasses
    nothink = dataclasses.replace(V["qwen36"], id="nothink", model="cloud-no-thinking", thinking=False)
    p, calls = _pipe(monkeypatch, tmp_path, lambda: FakeStream([_chunk("cloud answer")], headers=HEADERS))
    text, used, note = _open(p, nothink)
    assert (text, used.id, note) == ("cloud answer", "nothink", "")
    assert calls[0]["model"] == "cloud-no-thinking" and "extra_body" not in calls[0]
    assert calls[0]["base"] == "http://cloud/v1" and calls[0]["key"] == "secret"
    state = cq.read_state(cq.state_path(tmp_path))
    assert state["windows"]["hour"]["remaining"] == 150
    assert state["cloud_models"] == ["memento-query-qwen35", "memento-query-qwen36"]
    assert "secret" not in cq.state_path(tmp_path).read_text()


def test_images_only_for_vision_variants():
    msgs = [{"role": "system", "content": "s"},
            {"role": "user", "content": [{"type": "text", "text": "Sources ..."}, {"type": "text", "text": "Image for Source [2]:"},
                                         {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}}]}]
    out = generation.text_only(msgs)
    assert out[0] == msgs[0] and out[1] == {"role": "user", "content": "Sources ..."}


@pytest.mark.parametrize("code, note", [(429, "QUOTA_NOTE"), (500, "UNREACHABLE_NOTE")])
def test_fallback_before_output(monkeypatch, tmp_path, code, note):
    def boom():
        raise _status_error(code)
    p, calls = _pipe(monkeypatch, tmp_path, boom)
    text, used, got = _open(p, V["qwen35"])
    assert (text, used.id, got) == ("local answer", "local", getattr(texts, note))
    assert [c["model"] for c in calls] == ["qwen3.5-397b-a17b", "qwen3.8-27b"]
    if code == 429:                                   # a 429 carries the headers too: the picker learns at once
        assert cq.read_state(cq.state_path(tmp_path))["windows"]["hour"]["remaining"] == 150


def test_no_first_output_falls_back(monkeypatch, tmp_path):
    slow = FakeStream([_chunk("too late")], headers=HEADERS, delay=0.3)
    p, calls = _pipe(monkeypatch, tmp_path, lambda: slow)
    monkeypatch.setattr(generation, "FIRST_OUTPUT_TIMEOUT", 0.05)
    text, used, note = _open(p, V["qwen36"])
    assert (text, used.id, note) == ("local answer", "local", texts.UNREACHABLE_NOTE) and slow.closed


def test_an_empty_first_chunk_is_not_output(monkeypatch, tmp_path):
    stalled = FakeStream([_chunk(""), _chunk("too late")], headers=HEADERS, delay=0.3, delay_from=1)
    p, calls = _pipe(monkeypatch, tmp_path, lambda: stalled)
    monkeypatch.setattr(generation, "FIRST_OUTPUT_TIMEOUT", 0.05)
    text, used, note = _open(p, V["qwen35"])
    assert (text, used.id, note) == ("local answer", "local", texts.UNREACHABLE_NOTE)


def test_empty_chunks_before_output_are_kept(monkeypatch, tmp_path):
    p, calls = _pipe(monkeypatch, tmp_path, lambda: FakeStream([_chunk(""), _chunk("cloud"), _chunk(" answer")]))
    assert _open(p, V["qwen36"])[:2] == ("cloud answer", V["qwen36"])


def test_used_up_quota_goes_local_without_a_cloud_call(monkeypatch, tmp_path):
    p, calls = _pipe(monkeypatch, tmp_path, lambda: FakeStream([_chunk("cloud")]))
    state = cq.from_headers({**HEADERS, "X-RateLimit-Remaining-Hour": "2"}, datetime.now(timezone.utc))
    cq.write_state(cq.state_path(tmp_path), state, [])
    text, used, note = _open(p, V["qwen36"])
    assert (used.id, note) == ("local", texts.QUOTA_NOTE) and [c["model"] for c in calls] == ["qwen3.8-27b"]


def test_drain_content_cloud_break_adds_the_stopped_note():
    out = asyncio.run(generation.drain_content(FakeStream([_chunk("partial")], fail_after=1), cloud=True))
    assert out == "partial\n\n" + texts.STOPPED_NOTE


def test_drain_content_local_break_raises():
    with pytest.raises(openai.APIConnectionError):
        asyncio.run(generation.drain_content(FakeStream([_chunk("partial")], fail_after=1), cloud=False))
