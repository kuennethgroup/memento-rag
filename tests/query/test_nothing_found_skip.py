# Nothing in the library matches → a plain reply, no model call. Measured: answerable questions score 0.31–0.86,
# questions not in the library 0.008–0.044; the cut-off is RELATED_LOW_SCORE (0.1), the mark that already says "nothing
# found".
import asyncio
import types

import pytest

from test_rq3_answer_pipeline import _RecordingAsyncOpenAI, _load_pipe
from test_query_pipe_document_chat import CHAT, DC_MODEL, PASSAGES, FakeLLM, _drive, _ready, services  # noqa: F401
from memento.query import library_state, texts
from memento.query.scope import documents
from memento.query.flows import answer as answer_flow


def _library_run(monkeypatch, scores, question="Explain about RAG-Anything", payload_extra=None, heads=None,
                 **valves):
    import openai
    import memento.retrieval.search as search
    import memento.common.model_servers as vc
    mod = _load_pipe()
    payloads = [{"text": f"passage {i}", "source": "P" if i % 2 else "Q", "page": i, "type": "text", **(payload_extra or {})}
                for i in range(len(scores))]
    points = [types.SimpleNamespace(payload=p, score=0.5, id=i) for i, p in enumerate(payloads)]

    def rerank(q, hits, *a, **k):
        return [dict(h, rerank_score=s) for h, s in zip(hits, scores)] if scores[0] is not None else hits
    _RecordingAsyncOpenAI.calls = []
    monkeypatch.setattr(openai, "AsyncOpenAI", _RecordingAsyncOpenAI)
    monkeypatch.setattr(library_state, "collection_status", lambda c, coll: "ok")
    monkeypatch.setattr(library_state, "load_documents", lambda *a: {"P": {}, "Q": {}})
    monkeypatch.setattr(library_state, "in_progress_sources", lambda *a: set())
    monkeypatch.setattr(documents, "_resolve_paper", lambda q, s, d=None: None)
    monkeypatch.setattr(library_state, "qdrant_query", lambda *a, **k: types.SimpleNamespace(points=points))
    monkeypatch.setattr(vc, "vl_embed", lambda *a, **k: [0.0] * 8)
    monkeypatch.setattr(search, "rerank", rerank)
    p = mod.Pipe()
    p.valves = p.Valves(DOMAIN_NAME="generic", COLLECTION="x", RERANK_ENABLED=True, GRAPH_RETRIEVAL=False,
                        VALIDATION_ENABLED=False, ENABLE_QUERY_REWRITE=True, THINKING_AUTO=False, RELATED_DOCS=False,
                        **valves)
    if heads is not None:                     # the closest documents for the "nothing relevant" reply
        monkeypatch.setattr(answer_flow, "related_heads", lambda *a, **k: heads)

    async def run():
        return [c async for c in p.pipe({"messages": [{"role": "user", "content": question}]}, {"id": "t"})]
    out = "".join(c for c in asyncio.run(run()) if isinstance(c, str))
    return mod, out, [c for c in _RecordingAsyncOpenAI.calls if c.get("stream")]


def test_nothing_relevant_in_the_library_gets_the_plain_reply_without_the_model(monkeypatch):
    # with the model reply switched off (NOTHING_FOUND_REPLY); on, see the tests below
    mod, out, streamed = _library_run(monkeypatch, [0.016, 0.012, 0.009], NOTHING_FOUND_REPLY=False)
    assert streamed == []                                     # the answer model is never called
    assert out.strip().startswith(texts.NOTHING_IN_LIBRARY_MSG.strip())
    assert "Library Upload" in texts.NOTHING_IN_LIBRARY_MSG and "rephras" in texts.NOTHING_IN_LIBRARY_MSG


def test_a_weak_but_real_match_still_gets_an_answer(monkeypatch):
    # the lowest answerable question measured (domain pinning, 0.31) stays well above the cut-off
    mod, out, streamed = _library_run(monkeypatch, [0.31, 0.2, 0.05])
    assert len(streamed) == 1 and "Answer [1]." in out and texts.NOTHING_IN_LIBRARY_MSG.strip() not in out


def test_no_scores_means_no_judgement(monkeypatch):
    # the reranker failed (vector order, no scores): answer as before rather than claim the library has nothing
    mod, out, streamed = _library_run(monkeypatch, [None, None, None])
    assert len(streamed) == 1


def test_document_chat_always_asks_the_model(services, monkeypatch):
    # in Document Chat the reader picked the files, so the model always answers from them, however low the scores (a
    # "main findings" question scores low against every passage)
    import memento.retrieval.search as search
    src = _ready()
    PASSAGES[0]["source"] = PASSAGES[1]["source"] = src
    monkeypatch.setattr(search, "rerank", lambda q, hits, *a, **k: [dict(h, rerank_score=0.02) for h in hits])
    import openwebui.functions.query_pipe as qp
    text, _ = _drive(services, {"model": DC_MODEL, "messages": [{"role": "user", "content":
                     "What are the main findings of the attached file, and what evidence supports them?"}]},
                     {"id": "u1"}, {"chat_id": CHAT}, files=[{"id": "f1", "name": "Report.pdf"}])
    assert [c for c in FakeLLM.calls if c.get("stream")], "Document Chat did not ask the model"
    assert texts.NOTHING_IN_LIBRARY_MSG.strip() not in text and not hasattr(qp, "NOTHING_IN_FILES_MSG")


# nothing relevant → a short model reply (no thinking) from the closest documents' passages; the fixed text when there
# are none or the model fails
CLOSE = [{"source": "P", "text": "polymer informatics with machine learning", "rerank_score": 0.05}]


def test_nothing_relevant_gets_a_short_model_reply_without_thinking(monkeypatch):
    mod, out, streamed = _library_run(monkeypatch, [0.016, 0.012, 0.009], heads=CLOSE)
    assert len(streamed) == 1
    call = streamed[0]
    assert call["messages"][0]["content"] == answer_flow.NOTHING_FOUND_PROMPT
    assert "polymer informatics with machine learning" in call["messages"][1]["content"]
    assert call["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    assert out.strip().startswith("Answer [1].") and "Answered by" in out
    assert texts.NOTHING_IN_LIBRARY_MSG.strip() not in out


def test_no_close_documents_or_a_failing_model_keep_the_fixed_text(monkeypatch):
    mod, out, streamed = _library_run(monkeypatch, [0.016, 0.012, 0.009], heads=[])
    assert streamed == [] and out.strip().startswith(texts.NOTHING_IN_LIBRARY_MSG.strip())

    real = _RecordingAsyncOpenAI.create

    async def answer_fails(self, **kw):           # the router still answers; the answer call fails
        if kw.get("stream"):
            raise RuntimeError("model down")
        return await real(self, **kw)
    monkeypatch.setattr(_RecordingAsyncOpenAI, "create", answer_fails)
    mod2, out2, _ = _library_run(monkeypatch, [0.016, 0.012, 0.009], heads=CLOSE)
    assert out2.strip().startswith(texts.NOTHING_IN_LIBRARY_MSG.strip())
