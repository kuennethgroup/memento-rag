# Memento Query, a question naming two or more documents: like Document Chat's several files, each named document is
# searched on its own and gets an equal share of the passages, and "summarize A and B" summarizes each one whole.
import asyncio
import json
import sys
import types
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool
from memento.config.paths import PROJECT_ROOT
from memento.query import library_state
from memento.query.scope import documents
from memento.query.flows import summary

ROOT = PROJECT_ROOT
sys.path.insert(0, str(ROOT))

from memento.retrieval import pipe_cli  # noqa: E402

PAYLOADS = [{"text": f"Paper A passage {i}.", "source": "A", "page": i, "type": "text"} for i in range(30)] \
    + [{"text": "Paper B passage.", "source": "B", "page": 0, "type": "text"}]


class _LLM:
    calls: list = []
    intent = "ranked_retrieval"

    def __init__(self, *a, **k):
        self.chat = self
        self.completions = self

    async def create(self, **kw):
        _LLM.calls.append(kw)
        if kw.get("stream"):
            async def gen():
                delta = types.SimpleNamespace(content="Answer [1].", reasoning_content=None, reasoning=None)
                yield types.SimpleNamespace(choices=[types.SimpleNamespace(delta=delta)])
            return gen()
        msg = types.SimpleNamespace(content='{"intent":"%s","query":"Compare A and B","reasoning":"deep"}' % _LLM.intent)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


def _source_of(f):
    for c in (getattr(f, "must", None) or []):
        if getattr(c, "key", None) == "source" and hasattr(getattr(c, "match", None), "value"):
            return c.match.value
    return None


def _run(monkeypatch, intent):
    import openai
    import memento.retrieval.search as search
    import memento.common.model_servers as vc
    mod = pipe_cli.load_pipe_module()
    e = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    with e.begin() as c:
        c.execute(text("CREATE TABLE function (id TEXT PRIMARY KEY, valves TEXT)"))
        c.execute(text("INSERT INTO function VALUES ('memento_query', :v)"), {"v": json.dumps(
            {"COLLECTION": "memento-x-rag", "DOMAIN_NAME": "generic", "GRAPH_RETRIEVAL": False,
             "VALIDATION_ENABLED": False, "ENABLE_QUERY_REWRITE": True, "THINKING_AUTO": False, "RELATED_DOCS": False})})
    p = pipe_cli.make_pipe(mod, e)
    searched, floors = [], []

    def query(client, collection, qvec, q, flt, limit, hybrid, k):
        src = _source_of(flt)
        searched.append(src)
        pts = [types.SimpleNamespace(payload=d, score=0.9 - i / 100, id=i) for i, d in enumerate(PAYLOADS)
               if src is None or d["source"] == src]
        return types.SimpleNamespace(points=pts)

    def scroll_doc(client, collection, src):
        return [dict(d) for d in PAYLOADS if d["source"] == src]

    def rerank(q, hits, url, top, min_score, *a, **k):
        floors.append(min_score)
        return hits[:top]
    _LLM.calls, _LLM.intent = [], intent
    monkeypatch.setattr(openai, "AsyncOpenAI", _LLM)
    monkeypatch.setattr(library_state, "collection_status", lambda c, coll: "ok")
    monkeypatch.setattr(library_state, "load_documents", lambda *a: {"A": {}, "B": {}})
    monkeypatch.setattr(documents, "_resolve_paper", lambda q, s, d=None: None)
    monkeypatch.setattr(documents, "named_documents", lambda q, s, d=None: ["A", "B"])
    monkeypatch.setattr(library_state, "qdrant_query", query)
    monkeypatch.setattr(summary, "document_passages", scroll_doc)
    monkeypatch.setattr(vc, "vl_embed", lambda *a, **k: [0.0] * 8)
    monkeypatch.setattr(search, "rerank", rerank)
    out = asyncio.run(pipe_cli.ask(p, "Compare A and B"))
    answers = [c for c in _LLM.calls if c.get("stream")]
    users = [(c["messages"][-1]["content"] if isinstance(c["messages"][-1]["content"], str)
              else c["messages"][-1]["content"][0]["text"]) for c in answers]
    return out, searched, floors, answers, users


def test_two_named_documents_each_get_their_share(monkeypatch):
    out, searched, floors, answers, users = _run(monkeypatch, "ranked_retrieval")
    assert "A" in searched and "B" in searched                       # each named document searched on its own
    assert floors[-1] == 0.0                                          # each one's best passages count
    assert len(answers) == 1 and "Paper B passage." in users[0] and "Paper A passage 0." in users[0]
    assert "documents named in the question" in users[0]


def test_summarize_two_named_documents_reads_each_whole(monkeypatch):
    out, searched, floors, answers, users = _run(monkeypatch, "summary")
    assert len(answers) == 2                                          # one whole-document summary each
    assert "Paper A passage 29." in users[0] and "Paper B passage." in users[1]
    assert all(c["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False for c in answers)
    assert out["answer"].count("Answered by") == 1
