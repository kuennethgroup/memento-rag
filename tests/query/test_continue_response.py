# Open WebUI's "Continue Response": the request ends with the model's own answer and whatever the pipe writes is
# appended to it. Memento's answers are complete when sent, so both pipes add nothing.
import asyncio
import sys
from pathlib import Path

import pytest
from memento.config.paths import PROJECT_ROOT
from memento.query import request, service
from memento.ingestion.upload import intake
from memento.ingestion.upload import workflow

ROOT = PROJECT_ROOT
sys.path.insert(0, str(ROOT))


def _load(path, function_id):
    from _owui_loader import load_function
    return load_function(ROOT / path, function_id)


qp = _load("openwebui/functions/query_pipe.py", "memento_query")
ip = _load("openwebui/functions/ingestion_pipe.py", "memento_ingestion")
CONTINUE = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "Hello! I am Memento.\n\n_Answered by X._"}]
ASKED = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "Hello!"},
         {"role": "user", "content": "And what is in the library?"}]


@pytest.mark.parametrize("messages, want", [(CONTINUE, True), (ASKED, False), ([], False),
                                            ([{"role": "assistant", "content": ""}], True)])
def test_a_request_ending_with_the_models_answer_is_a_continue(messages, want):
    # one check for both pipes (memento/query/request.py)
    assert request.continue_request({"messages": messages}) is want


def _drive_query(monkeypatch, messages, task=None):
    pipe, called = qp.Pipe(), []

    async def inner(turn):
        called.append(1)
        yield "ANSWER"
    monkeypatch.setattr(service, "run", inner)

    async def emit(e):
        return None

    async def go():
        return "".join([c async for c in pipe.pipe({"model": "memento-query", "messages": messages}, {"id": "u1"}, emit,
                                                   None, {"chat_id": "c1"}, task)])
    return asyncio.run(go()), called


def test_the_query_pipe_adds_nothing_to_a_continued_answer(monkeypatch):
    out, called = _drive_query(monkeypatch, CONTINUE)
    assert out == "" and called == []


def test_the_query_pipe_still_answers_questions_and_background_tasks(monkeypatch):
    assert _drive_query(monkeypatch, ASKED)[0] == "ANSWER"
    assert _drive_query(monkeypatch, CONTINUE, task="follow_up_generation")[0] == "ANSWER"


def test_the_upload_pipe_adds_nothing_to_a_continued_reply(monkeypatch):
    pipe, called = ip.Pipe(), []

    async def may(user):
        return True
    monkeypatch.setattr(intake, "may_ingest", lambda valves, *a_, **k_: may(*a_, **k_))
    monkeypatch.setattr(workflow, "pipe_sync", lambda ing, *a_, **k_: (lambda *a, **k: called.append(1) or iter(["REPLY"]))(*a_, **k_))

    async def go():
        return "".join([c async for c in pipe.pipe({"messages": CONTINUE}, {"id": "u1", "role": "user"}, [], None,
                                                   None, "c1", {})])
    assert asyncio.run(go()) == "" and called == []
