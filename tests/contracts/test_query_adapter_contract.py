# Contract of the Library Chat pipe: for each scenario, what the real pipe returns and sends (answer text, status and
# citation events, every model call, every Qdrant query) against fakes at the service boundaries only. The golden
# transcripts in tests/fixtures/contracts/query/ pin today's behaviour, so moving the pipe's code into modules can be
# checked against them; MEMENTO_UPDATE_CONTRACTS=1 rewrites them after an intended change.
import asyncio
import json
import os

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool

from _contract_fakes import FakeLLM, FakeQdrant, SyncFakeLLM, check_golden, fake_rerank

COLLECTION = "memento-x-rag"
DOCS = {"Alpha": "Paper Alpha: hardness measurements", "Beta": "Paper Beta: thermal stability",
        "Gamma": "Paper Gamma: processing routes"}
PASSAGES = ([{"text": f"Alpha passage {i}: the samples reach a hardness of {10 + i} GPa.", "source": "Alpha",
              "page": i, "type": "text", "section": "Results"} for i in range(4)]
            + [{"text": "Beta passage: the films are stable up to 400 °C.", "source": "Beta", "page": 1, "type": "text",
                "section": "Results"},
               {"text": "Figure 2: stability map of the films.", "source": "Beta", "page": 2, "type": "image",
                "section": "Results"},
               {"text": "Gamma passage: three processing routes are compared.", "source": "Gamma", "page": 1,
                "type": "text", "section": "Methods"}])
VALVES = {"COLLECTION": COLLECTION, "DOMAIN_NAME": "generic", "GRAPH_RETRIEVAL": False, "VALIDATION_ENABLED": False,
          "ENABLE_QUERY_REWRITE": True, "THINKING_AUTO": False, "RELATED_DOCS": False, "NOTHING_FOUND_REPLY": False,
          "FILES_URL": "http://files.test", "QDRANT_URL": "http://qdrant.test", "LLM_BASE_URL": "http://llm.test/v1",
          "LLM_MODEL": "local-model"}
ROUTE = {"intent": "ranked_retrieval", "reasoning": "deep", "suggest": False, "element": "none", "scope": "none",
         "author": ""}

SCENARIOS = {
    "ranked_answer": dict(question="What hardness do the samples reach?", route={"query": "sample hardness"}),
    "named_document": dict(question="What does Paper Alpha: hardness measurements report about hardness?",
                           route={"query": "hardness in Paper Alpha: hardness measurements", "scope": "document"}),
    "two_documents": dict(question="Compare Paper Alpha: hardness measurements with Paper Beta: thermal stability",
                          route={"query": "compare Paper Alpha: hardness measurements and Paper Beta: thermal "
                                          "stability", "scope": "documents"}),
    "missing_named_document": dict(question="What does Paper Omega: unknown topics report?",
                                   route={"query": "Paper Omega: unknown topics", "scope": "document"}),
    "my_uploads_without_any": dict(question="Summarize my latest uploads",
                                   route={"intent": "summary", "query": "my latest uploads", "scope": "documents"}),
    "library_listing": dict(question="What's in the library?", route={"intent": "corpus_listing", "query": "library"}),
    "general_question": dict(question="What is 12 times 12?", route={"intent": "general", "query": "12 times 12"}),
    "chitchat": dict(question="hello", route={"intent": "chitchat", "query": "hello"}),
    "figure_request": dict(question="Show the stability map figure",
                           route={"intent": "image_request", "query": "stability map figure", "element": "figure"}),
    "rework_without_stored_sources": dict(
        question="Summarize that in two bullets",
        route={"intent": "rework", "query": "summarize the previous answer in two bullets"},
        history=[{"role": "user", "content": "What hardness do the samples reach?"},
                 {"role": "assistant", "content": "The samples reach 10 GPa [1]."}]),
    "summary_of_a_document": dict(question='Summarize the document "Paper Alpha: hardness measurements"',
                                  route={"intent": "summary", "query": "summary of Paper Alpha: hardness measurements",
                                         "scope": "document"}),
    # the first run is stored, the second replayed
    "summary_replayed_from_the_cache": dict(question='Summarize the document "Paper Gamma: processing routes"',
                                            route={"intent": "summary", "query": "summary of Paper Gamma: processing "
                                                   "routes", "scope": "document"}, repeat=2),
    "thorough_check_with_an_unreadable_check_reply": dict(question="What hardness do the samples reach?",
                                                          route={"query": "sample hardness"}, thorough=True,
                                                          valves={"VALIDATION_ENABLED": True}),
    "document_chat_without_a_file": dict(question="What does the attached file say?", model="document-chat",
                                         route={"query": "attached file"}),
}


def _run(monkeypatch, question, route, history=None, valves=None, model=None, thorough=False, repeat=1):
    import openai
    import qdrant_client
    import memento.common.model_servers as vc
    from memento.library import documents as library_db
    from memento.retrieval import pipe_cli
    e = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    with e.begin() as c:
        c.execute(text("CREATE TABLE function (id TEXT PRIMARY KEY, valves TEXT)"))
        c.execute(text("INSERT INTO function VALUES ('memento_query', :v)"), {"v": json.dumps({**VALVES, **(valves or {})})})
    library_db.ensure_tables(e)
    for source, title in DOCS.items():
        library_db.add_document(e, collection=COLLECTION, entry={"source": source, "title": title},
                                content_hash=f"h-{source}", uploaded_by="u-other", output_root=None)
    # the picker's list as a library that has been used has it (the first request otherwise writes it in the
    # background while it reads it, a race)
    from memento.library.documents import write_library
    write_library(os.environ["OUTPUT_ROOT"], [{"source": s, "title": t} for s, t in DOCS.items()], COLLECTION)
    FakeQdrant.collections, FakeQdrant.log = {COLLECTION: PASSAGES}, []
    FakeLLM.calls, FakeLLM.route, FakeLLM.answer = [], {**ROUTE, **route}, "Answer [1]."
    monkeypatch.setattr(qdrant_client, "QdrantClient", FakeQdrant)
    monkeypatch.setattr(openai, "AsyncOpenAI", FakeLLM)
    monkeypatch.setattr(openai, "OpenAI", SyncFakeLLM)
    monkeypatch.setattr(vc, "vl_embed", lambda *a, **k: [0.0] * 8)
    monkeypatch.setattr(vc, "rerank", fake_rerank)
    mod = pipe_cli.load_pipe_module()
    p = pipe_cli.make_pipe(mod, e)
    runs = []
    for _ in range(repeat):
        FakeLLM.calls, FakeQdrant.log = [], []
        out = asyncio.run(_ask(p, question, model=model, thorough=thorough, history=history))
        runs.append(_transcript(out))
    return runs[0] if repeat == 1 else {"runs": runs}


async def _ask(pipe, question, *, model=None, thorough=False, history=None):
    """retrieval/pipe_cli.ask with a saved chat's metadata (Document Chat needs a chat id)."""
    events = []

    async def emit(e):
        events.append(e)
    body = {"model": f"memento_query.{model or 'local'}", "stream": True,
            "messages": list(history or []) + [{"role": "user", "content": question}]}
    meta = {"chat_id": "6f1c2a3e-0000-4000-8000-000000000001", "message_id": "msg-1", **({"filter_ids": ["memento_thorough"]} if thorough else {})}
    chunks = [c async for c in pipe.pipe(body, {"id": "u-asker", "role": "user", "name": "Asker"}, emit, None, meta)]
    return {"answer": "".join(c for c in chunks if isinstance(c, str)), "events": events}


def _transcript(out) -> dict:
    events = []
    for ev in out["events"]:
        if not isinstance(ev, dict):
            continue
        data = ev.get("data") or {}
        if ev.get("type") == "status":
            events.append({"status": data.get("description"), "done": data.get("done"), "hidden": data.get("hidden")})
        elif ev.get("type") == "citation":
            events.append({"citation": (data.get("source") or {}).get("name"),
                           "pages": [m.get("page") for m in data.get("metadata") or []]})
        else:
            events.append({"event": ev.get("type"), "keys": sorted(data) if isinstance(data, dict) else None})
    return {"answer": out["answer"], "events": events, "llm_calls": FakeLLM.calls, "qdrant": FakeQdrant.log}


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_library_chat_contract(monkeypatch, name):
    s = SCENARIOS[name]
    check_golden("query", name, _run(monkeypatch, s["question"], s["route"], s.get("history"), s.get("valves"),
                                     s.get("model"), s.get("thorough", False), s.get("repeat", 1)))
