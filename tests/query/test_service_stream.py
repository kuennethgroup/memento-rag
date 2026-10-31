# The query pipe hands each message to memento/query/service.py: the adapter's stream closes the service's own at once,
# each request reads its own copy of the Valves, the command line goes through the same stream, two copies in one
# process keep their own state, and every reply a step hands back goes through contextlib.aclosing.
import ast
import asyncio
import inspect
import json

import pytest

from _contract_fakes import FakeLLM, FakeQdrant, SyncFakeLLM, fake_rerank
from memento.config.paths import PROJECT_ROOT
from memento.query import library_state, request
from test_two_libraries_together import both  # noqa: F401  (the two copies' fixture)

VALVES = {"DOMAIN_NAME": "generic", "GRAPH_RETRIEVAL": False, "VALIDATION_ENABLED": False,
          "ENABLE_QUERY_REWRITE": True, "THINKING_AUTO": False, "RELATED_DOCS": False, "NOTHING_FOUND_REPLY": False,
          "FILES_URL": "http://files.test", "QDRANT_URL": "http://qdrant.test", "LLM_BASE_URL": "http://llm.test/v1",
          "LLM_MODEL": "local-model"}
ROUTE = {"intent": "ranked_retrieval", "reasoning": "deep", "suggest": False, "element": "none", "scope": "none",
         "author": "", "query": "sample hardness"}


def _passages(source):
    return [{"text": f"{source} passage {i}: the samples reach a hardness of {10 + i} GPa.", "source": source,
             "page": i, "type": "text", "section": "Results"} for i in range(3)]


def _fakes(monkeypatch, collections, route=None):
    import openai
    import qdrant_client
    import memento.common.model_servers as vc
    FakeQdrant.collections, FakeQdrant.log = collections, []
    FakeLLM.calls, FakeLLM.route, FakeLLM.answer = [], {**ROUTE, **(route or {})}, "Answer [1]."
    monkeypatch.setattr(qdrant_client, "QdrantClient", FakeQdrant)
    monkeypatch.setattr(openai, "AsyncOpenAI", FakeLLM)
    monkeypatch.setattr(openai, "OpenAI", SyncFakeLLM)
    monkeypatch.setattr(vc, "vl_embed", lambda *a, **k: [0.0] * 8)
    monkeypatch.setattr(vc, "rerank", fake_rerank)


def _cli_pipe(monkeypatch, route=None, **valves):
    """The command line's pipe on an in-memory database with the contract fakes (tests/contracts)."""
    from sqlalchemy import create_engine, text
    from sqlalchemy.pool import StaticPool
    from memento.retrieval import pipe_cli
    e = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    with e.begin() as c:
        c.execute(text("CREATE TABLE function (id TEXT PRIMARY KEY, valves TEXT)"))
        c.execute(text("INSERT INTO function VALUES ('memento_query', :v)"),
                  {"v": json.dumps({**VALVES, "COLLECTION": "lib-a", **valves})})
    _fakes(monkeypatch, {"lib-a": _passages("Alpha")}, route)
    return pipe_cli.make_pipe(pipe_cli.load_pipe_module(), e), pipe_cli


def _body(question):
    return {"model": "memento_query.local", "messages": [{"role": "user", "content": question}]}


def test_pipe_cli_iterates_the_pipe(monkeypatch):
    p, pipe_cli = _cli_pipe(monkeypatch)
    out = asyncio.run(pipe_cli.ask(p, "What hardness do the samples reach?"))
    answer = out if isinstance(out, str) else "".join(c for c in out if isinstance(c, str)) if isinstance(out, list) \
        else out.get("answer", "")
    assert "Answer [1]." in answer


def test_closing_the_stream_resets_the_request(monkeypatch):
    p, _ = _cli_pipe(monkeypatch, route={"intent": "general", "query": "12 mal 12"})
    before = request.LANG.get(None)

    async def go():
        stream = p.pipe(_body("Was ist das Ergebnis von zwölf mal zwölf?"), {"id": "u"}, None, None, {"chat_id": "c"})
        first = await stream.__anext__()
        during = request.LANG.get(None)
        await stream.aclose()
        return first, during, request.LANG.get(None)       # right after the close: no other await in between
    first, during, after = asyncio.run(go())
    assert isinstance(first, str) and during == "de"
    assert after == before


def test_valves_are_copied_per_request(monkeypatch):
    p, _ = _cli_pipe(monkeypatch)

    async def emit(event):                               # the first status line: the request has started
        p.valves.LLM_MODEL = "changed-while-streaming"

    async def go():
        return [c async for c in p.pipe(_body("What hardness do the samples reach?"), {"id": "u"}, emit, None,
                                        {"chat_id": "c"})]
    asyncio.run(go())
    assert FakeLLM.calls and FakeLLM.calls[0]["model"] == "local-model"   # the router, asked after the change


def _closed_rule_breaks(path):
    """`async for x in <step>(turn): yield x` (a step the split made, called plainly) and a plain `async for` over
    service.handle(...). A step handed in as a parameter keeps its form: Document Chat's hand-over to run is today's
    delegation (the plan's Global Constraints)."""
    out = []
    tree = ast.parse(path.read_text())
    handed = set()
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            params = {a.arg for a in fn.args.args + fn.args.kwonlyargs}
            handed |= {id(x) for x in ast.walk(fn) if isinstance(x, ast.AsyncFor) and isinstance(x.iter, ast.Call)
                       and isinstance(x.iter.func, ast.Name) and x.iter.func.id in params}
    for n in ast.walk(tree):
        if not isinstance(n, ast.AsyncFor) or not isinstance(n.iter, ast.Call) or id(n) in handed:
            continue
        call = n.iter
        name = ast.unparse(call.func)
        step = (len(call.args) == 1 and not call.keywords and isinstance(call.args[0], ast.Name)
                and call.args[0].id == "turn")
        only_yield = (len(n.body) == 1 and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Yield)
                      and isinstance(n.target, ast.Name) and isinstance(n.body[0].value.value, ast.Name)
                      and n.body[0].value.value.id == n.target.id)
        if (step and only_yield) or name.endswith("service.handle"):
            out.append(f"{path.relative_to(PROJECT_ROOT)}:{n.lineno} {name}")
    return out


def test_pure_delegations_are_closed():
    files = [PROJECT_ROOT / "openwebui/functions/query_pipe.py"] + sorted((PROJECT_ROOT / "memento/query").rglob("*.py")) \
        + sorted((PROJECT_ROOT / "memento/document_chat").rglob("*.py"))
    assert [b for f in files for b in _closed_rule_breaks(f)] == []
    src = inspect.getsource(__import__("openwebui.functions.query_pipe", fromlist=["Pipe"]).Pipe.pipe)
    assert "contextlib.aclosing(service.handle(" in src


def test_two_copies_do_not_share_state(both, monkeypatch):
    # the default copy and the second one (tests/library/test_two_libraries_together.py), one question through each
    from test_two_libraries_together import COMP, POLY
    _fakes(monkeypatch, {COMP: _passages("Alpha"), POLY: _passages("Beta")})
    library_state._STATE_CACHE.clear()
    pa, pb = both.qc.Pipe(), both.qs.Pipe()
    pa.valves, pb.valves = both.vc, both.vs
    seen = {}
    for name, p, model in (("a", pa, "memento_query.local"), ("b", pb, "memento_query_polymer.local")):
        FakeQdrant.log = []

        async def go():
            return [c async for c in p.pipe({"model": model, "messages": [{"role": "user",
                                             "content": "What hardness do the samples reach?"}]},
                                            {"id": "u"}, None, None, {"chat_id": "c"})]
        out = "".join(c for c in asyncio.run(go()) if isinstance(c, str))
        assert "Answer [1]." in out, (name, out[:200])
        seen[name] = {e["collection"] for e in FakeQdrant.log}
    assert seen == {"a": {COMP}, "b": {POLY}}
    assert (pa.function_id, pb.function_id) == ("memento_query", "memento_query_polymer")
    assert {fid for fid, _ in library_state._STATE_CACHE} == {"memento_query_polymer"}   # the default copy reads none
    assert library_state.people_lock(pa.function_id) is not library_state.people_lock(pb.function_id)
