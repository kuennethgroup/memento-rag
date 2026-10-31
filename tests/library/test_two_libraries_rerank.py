# Two libraries in one Open WebUI: the reranker instruction comes from the copy's own domain, not from the process-wide
# config/.env domain. Both copies run in one process whose memento.config.settings.DOMAIN is the computational-materials profile;
# the second copy's questions must still be reranked with the polymer instruction.
import ast
import sys
import types
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from memento.config.paths import PROJECT_ROOT
from memento.query import library_state, request
from memento.query.scope import documents

ROOT = PROJECT_ROOT
sys.path.insert(0, str(ROOT))
from memento.research_gaps import ledger as gl, update as gu
from memento.retrieval import search  # noqa: E402

PROCESS, COPY = "PROCESS-DOMAIN instruction", "COPY-DOMAIN instruction"


@pytest.fixture
def process_domain(monkeypatch):
    """The process-wide domain (config/.env) carries PROCESS; a call without its own instruction sends that."""
    import memento.config.settings as cfg
    monkeypatch.setitem(cfg.DOMAIN, "rerank_instruction", PROCESS)
    return cfg


@pytest.fixture
def sent(monkeypatch):
    """The instructions that reach the vLLM reranker."""
    import memento.common.model_servers as vc
    seen = []

    def fake(url, model, query, docs, instruction, top_n=None, **k):
        seen.append(instruction)
        return [(i, 0.9) for i in range(min(len(docs), top_n or len(docs)))]
    monkeypatch.setattr(vc, "rerank", fake)
    return seen


HITS = [{"text": "a passage", "type": "text", "source": "D", "page": 1}]


# ── memento/retrieval/search.py ──────────────────────────────────────────────────────

def test_rerank_sends_the_callers_instruction(process_domain, sent):
    out = search.rerank("q", HITS, "http://reranker", 1, 0.0, 0.0, instruction=COPY)
    assert sent == [COPY] and out[0]["rerank_score"] == 0.9


def test_rerank_without_an_instruction_keeps_the_process_domain(process_domain, sent):
    search.rerank("q", HITS, "http://reranker", 1, 0.0, 0.0)
    assert sent == [PROCESS]


def test_search_passes_its_rerank_instruction_on(process_domain, sent, monkeypatch):
    point = types.SimpleNamespace(payload={"text": "a passage", "source": "D", "page": 1, "type": "text"}, score=0.5,
                                  id=1)
    monkeypatch.setattr(search, "qdrant_query", lambda *a, **k: types.SimpleNamespace(points=[point]))
    kw = dict(embedder_url="", query_instruction="i", retrieve_k=5, hybrid=False, hybrid_prefetch_k=5,
              rerank_enabled=True, reranker_url="http://reranker", rerank_top=1, rerank_min_score=0.0,
              image_rerank_ratio=0.0, qvec=[0.0] * 4)
    search.search(None, "c", "q", rerank_instruction=COPY, **kw)
    search.search(None, "c", "q", **kw)
    assert sent == [COPY, PROCESS]


# ── memento/research_gaps/update.py: the four reranking paths ─────────────────────────

@pytest.fixture
def engine():
    e = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    gu.ensure_tables(e)
    gl.ensure_tables(e)
    return e


class _LLM:
    """An OpenAI-compatible client whose every answer is the same short text."""
    def __init__(self, text="polymer foam methods"):
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=lambda **k: types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=text))])))


def _ctx(engine, instruction):
    return gu.Ctx(engine=engine, qdrant=None, llm_client=_LLM(), model="m", collection="col", embedder_url="",
                  embedder_model="", reranker_url="http://reranker", instruction="embed", log=lambda *a: None,
                  rerank_instruction=instruction)


@pytest.fixture
def library(monkeypatch):
    """One passage of document D in every search; embeddings are zeros."""
    import memento.common.model_servers as vc
    point = types.SimpleNamespace(payload={"text": "a passage", "source": "D", "page": 1, "type": "text"}, score=0.5,
                                  id=7)
    monkeypatch.setattr(search, "qdrant_query", lambda *a, **k: types.SimpleNamespace(points=[point]))
    monkeypatch.setattr(search, "related_heads", lambda *a, **k: [{"text": "a head", "type": "text", "source": "D",
                                                                     "page": 1, "chunk_id": "7"}])
    monkeypatch.setattr(vc, "vl_embed", lambda *a, **k: [0.0] * 4)
    monkeypatch.setattr(vc, "vl_embed_many", lambda url, model, items, **k: [[0.0] * 4 for _ in items])


def _gap(engine, statement="Is the foam stable?", **extra):
    gid = gl.add_gap(engine, "col", statement, "limitation")
    return {"id": gid, "statement": statement, "documents": ["OTHER"], "mentions": [], **extra}


@pytest.mark.parametrize("instruction, expected", [(COPY, COPY), (None, PROCESS)])
def test_check_against_reranks_with_the_copys_instruction(engine, process_domain, sent, library, instruction,
                                                          expected, monkeypatch):
    from memento.research_gaps import jobs as gj
    monkeypatch.setattr(gj, "judge_labels", lambda llm, build, cands: [{"relation": "not_concern", "reason": ""}
                                                                         for _ in cands])
    g = _gap(engine, library_checked_at="2026-10-14", library_status="open")
    gu.check_against(_ctx(engine, instruction), "D", [g])
    assert sent and set(sent) == {expected}


@pytest.mark.parametrize("instruction, expected", [(COPY, COPY), (None, PROCESS)])
def test_check_library_reranks_both_searches_with_the_copys_instruction(engine, process_domain, sent, library,
                                                                        instruction, expected, monkeypatch):
    from memento.research_gaps import jobs as gj
    monkeypatch.setattr(gj, "judge_labels", lambda llm, build, cands: [{"relation": "not_concern", "reason": ""}
                                                                         for _ in cands])
    gu.check_library(_ctx(engine, instruction), [_gap(engine)], 3)
    assert len(sent) == 2 and set(sent) == {expected}             # the passages, then the documents' best passages


@pytest.mark.parametrize("instruction, expected", [(COPY, COPY), (None, PROCESS)])
def test_check_literature_reranks_the_abstracts_with_the_copys_instruction(engine, process_domain, sent, instruction,
                                                                           expected, monkeypatch):
    from memento.research_gaps import literature as al, jobs as gj, literature as ls
    work = ls.Work(title="Foam stability", year=2020, doi="", abstract="About foams.", url="", source="openalex",
                   work_id="W1")
    monkeypatch.setattr(ls, "search_openalex", lambda q, **k: ([work], 100))
    monkeypatch.setattr(al, "available", lambda *a, **k: False)
    monkeypatch.setattr(gj, "judge_labels", lambda llm, build, cands: [{"relation": "not_concern", "reason": ""}
                                                                         for _ in cands])
    gu.check_literature(_ctx(engine, instruction), [_gap(engine)], [], [])
    assert sent == [expected]


def test_every_rerank_call_in_gap_update_passes_the_contexts_instruction():
    """All four calls (check_against, check_library twice, check_literature), and no new one without it."""
    tree = ast.parse((ROOT / "memento/research_gaps/update.py").read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "rerank"]
    assert len(calls) == 4
    for c in calls:
        kw = {k.arg: ast.unparse(k.value) for k in c.keywords}
        assert kw.get("instruction") == "ctx.rerank_instruction", ast.unparse(c)


# ── the context builders: the copy's domain decides ───────────────────────────

def _pipe_module():
    sys.path.insert(0, str(ROOT / "openwebui/functions"))
    import query_pipe
    return query_pipe


def test_the_copys_rerank_instruction_comes_from_its_own_domain(monkeypatch):
    qp = _pipe_module()
    monkeypatch.delenv("EMBED_INSTRUCTION", raising=False)
    polymer, comp = request.domain("polymer-engineering"), request.domain("computational-materials")
    assert polymer["rerank_instruction"] != comp["rerank_instruction"]          # the two profiles really differ
    assert request.rerank_instruction(polymer) == polymer["rerank_instruction"]
    assert request.rerank_instruction(comp) == comp["rerank_instruction"]
    assert request.rerank_instruction({}) is None                                   # unknown domain: process default


def test_an_embed_instruction_override_still_applies_to_every_copy(monkeypatch):
    # memento.config.settings applies EMBED_INSTRUCTION process-wide today; each copy keeps that rule
    qp = _pipe_module()
    monkeypatch.setenv("EMBED_INSTRUCTION", "OVERRIDE")
    assert request.rerank_instruction(request.domain("polymer-engineering")) == "OVERRIDE"


def test_the_updaters_context_carries_the_copys_rerank_instruction(monkeypatch):
    sys.path.insert(0, str(ROOT / "scripts"))
    import update_gap_ledger as ugl
    import qdrant_client
    qp = _pipe_module()
    monkeypatch.setattr(qdrant_client, "QdrantClient", lambda **k: None)      # no connection attempt
    monkeypatch.delenv("EMBED_INSTRUCTION", raising=False)
    V = qp.Pipe.Valves(DOMAIN_NAME="polymer-engineering", COLLECTION="memento-polymer-rag")
    ctx = ugl.make_ctx(None, V, 1)
    assert ctx.rerank_instruction == request.domain("polymer-engineering")["rerank_instruction"]


def _calls(src: str, name: str) -> list[ast.Call]:
    tree = ast.parse(src)
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call)
            and (getattr(n.func, "id", None) == name or getattr(n.func, "attr", None) == name)]


def _calls_in_query_code(name: str) -> list[ast.Call]:
    """`_calls` over the query pipe and the memento/query modules its code moved to."""
    files = [ROOT / "openwebui/functions/query_pipe.py"] + sorted((ROOT / "memento/query").rglob("*.py"))
    return [c for f in files for c in _calls(f.read_text(), name)]


def test_every_rerank_and_search_call_in_the_query_pipe_passes_the_copys_instruction():
    reranks = _calls_in_query_code("rerank") + _calls_in_query_code("_rerank")
    # direct calls and the asyncio.to_thread(_rerank, ...) form of the answer path
    threaded = [c for c in _calls_in_query_code("to_thread") if c.args and ast.unparse(c.args[0]) == "_rerank"]
    # gap report, gap proposal, paper cards, related heads, a list question's documents
    assert len(reranks) == 5 and len(threaded) == 1
    for c in reranks + threaded:
        kw = {k.arg: ast.unparse(k.value) for k in c.keywords}
        assert kw.get("instruction", "").startswith("request.rerank_instruction("), ast.unparse(c)[:160]
    searches = [c for c in _calls_in_query_code("_search")]
    assert searches
    for c in searches:
        kw = {k.arg: ast.unparse(k.value) for k in c.keywords}
        assert kw.get("rerank_instruction", "").startswith("request.rerank_instruction("), ast.unparse(c)[:160]
    ctxs = [c for c in _calls_in_query_code("Ctx")]
    assert ctxs
    for c in ctxs:
        kw = {k.arg: ast.unparse(k.value) for k in c.keywords}
        assert kw.get("rerank_instruction", "").startswith("request.rerank_instruction("), ast.unparse(c)[:160]


def test_the_answer_path_reranks_a_polymer_question_with_the_polymer_instruction(monkeypatch, process_domain):
    """End to end through pipe(): the second copy (polymer Valves) in a computational-materials process."""
    import asyncio
    import openai
    from test_rq3_answer_pipeline import _RecordingAsyncOpenAI, _load_pipe
    import memento.common.model_servers as vc
    mod = _load_pipe()
    seen = []
    points = [types.SimpleNamespace(payload={"text": f"passage {i}", "source": "P", "page": i, "type": "text"},
                                    score=0.5, id=i) for i in range(3)]

    def rerank(q, hits, *a, **k):
        seen.append(k.get("instruction"))
        return [dict(h, rerank_score=0.8) for h in hits]
    monkeypatch.setattr(openai, "AsyncOpenAI", _RecordingAsyncOpenAI)
    monkeypatch.setattr(library_state, "collection_status", lambda c, coll: "ok")
    monkeypatch.setattr(library_state, "load_documents", lambda *a: {"P": {}})
    monkeypatch.setattr(library_state, "in_progress_sources", lambda *a: set())
    monkeypatch.setattr(documents, "_resolve_paper", lambda q, s, d=None: None)
    monkeypatch.setattr(library_state, "qdrant_query", lambda *a, **k: types.SimpleNamespace(points=points))
    monkeypatch.setattr(vc, "vl_embed", lambda *a, **k: [0.0] * 8)
    monkeypatch.setattr(search, "rerank", rerank)
    monkeypatch.delenv("EMBED_INSTRUCTION", raising=False)
    p = mod.Pipe()
    p.valves = p.Valves(DOMAIN_NAME="polymer-engineering", COLLECTION="memento-polymer-rag", RERANK_ENABLED=True,
                        QDRANT_URL="http://127.0.0.1:9",            # a closed port: never the live Qdrant
                        GRAPH_RETRIEVAL=False, VALIDATION_ENABLED=False, ENABLE_QUERY_REWRITE=False,
                        THINKING_AUTO=False, RELATED_DOCS=False)

    async def run():
        return [c async for c in p.pipe({"messages": [{"role": "user", "content": "How stable are the foams?"}]},
                                        {"id": "t"})]
    asyncio.run(run())
    polymer = request.domain("polymer-engineering")["rerank_instruction"]
    assert seen and set(seen) == {polymer} and polymer != PROCESS
