"""Adaptive thinking: the query router's "reasoning" field and the pipe's THINKING_AUTO Valve."""
import inspect
import re
import ast
import asyncio
import types
from pathlib import Path

import pytest

from memento.query import generation, library_state, presentation, prompts, request, router as qr, search_step, service, texts
from memento.query.flows import answer as answer_flow, figures, gap_answer
from memento.config.paths import PROJECT_ROOT
from memento.query.scope import documents, resolve

ROOT = PROJECT_ROOT


class _FakeLLM:
    def __init__(self, reply=None, exc=None):
        self.reply, self.exc = reply, exc
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))
        self.kwargs = None

    async def _create(self, **kw):
        self.kwargs = kw
        if self.exc:
            raise self.exc
        msg = types.SimpleNamespace(content=self.reply)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


def _route(reply=None, exc=None):
    llm = _FakeLLM(reply, exc)
    return asyncio.run(qr.route(llm, "m", "orig question", [])), llm


@pytest.mark.parametrize("reply,intent,reasoning,think", [
    ('{"intent":"ranked_retrieval","query":"q","reasoning":"light"}', "ranked_retrieval", "light", False),
    ('{"intent":"ranked_retrieval","query":"q","reasoning":"deep"}', "ranked_retrieval", "deep", True),
    ('{"intent":"ranked_retrieval","query":"q"}', "ranked_retrieval", "deep", True),          # field missing
    ('{"intent":"ranked_retrieval","query":"q","reasoning":"maybe"}', "ranked_retrieval", "deep", True),
    ('{"intent":"set_query","query":"q","reasoning":"light"}', "set_query", "light", True),   # synthesis intents
    ('{"intent":"gap_query","query":"q","reasoning":"light"}', "gap_query", "light", True),
    ('{"intent":"proposal","query":"q","reasoning":"light"}', "proposal", "light", True),
    ('{"intent":"multi_hop","query":"q","reasoning":"light"}', "multi_hop", "light", True),
    ('```json\n{"intent":"ranked_retrieval","query":"q","reasoning":"light"}\n```', "ranked_retrieval", "light", False),
])
def test_route_parsing_and_overrides(reply, intent, reasoning, think):
    r, llm = _route(reply)
    assert (r.intent, r.reasoning, r.needs_thinking) == (intent, reasoning, think)
    assert llm.kwargs["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}  # router never thinks


@pytest.mark.parametrize("reply,exc", [("not json", None), (None, RuntimeError("down"))])
def test_route_failure_keeps_thinking(reply, exc):
    r, _ = _route(reply, exc)
    assert (r.intent, r.query, r.needs_thinking) == ("ranked_retrieval", "orig question", True)


def test_empty_query_falls_back_to_original():
    r, _ = _route('{"intent":"ranked_retrieval","query":"","reasoning":"light"}')
    assert r.query == "orig question"


def test_classify_and_rewrite_unchanged_contract():
    llm = _FakeLLM('{"intent":"multi_hop","query":"rewritten","reasoning":"light"}')
    assert asyncio.run(qr.classify_and_rewrite(llm, "m", "q", [])) == ("multi_hop", "rewritten")


def test_pipe_uses_one_thinking_switch():
    """Both answer calls (normal + research gap) use the per-answer `think`, default Valve keeps old behaviour."""
    replace_imports = pytest.importorskip("open_webui.utils.plugin").replace_imports
    src = replace_imports((ROOT / "openwebui/functions/query_pipe.py").read_text())
    ast.parse(src)
    # both answer calls go through generation.open_answer (the chosen model variant), which applies the switch once
    gen = (ROOT / "memento/query/generation.py").read_text()
    assert gen.count('kwargs["extra_body"] = gen_extra(valves, think, ') == 1
    # main answer; a whole-document summary without thinking
    assert inspect.getsource(answer_flow.stream_answer).count(
        "await generation.open_answer(\n            turn.valves, turn.variant, think=turn.think and not turn.whole_doc,") == 1
    # gap: the same switch only when GAP_THINKING is on, or Thorough check; the local model's thinking capped; the
    # decision is made before the live rule (generation.live_thinks), so the stored gap answers are written with it
    gap = inspect.getsource(gap_answer.reply)
    assert gap.count("gap_think = generation.gap_thinks(turn.gap_base_think, thorough, turn.valves.GAP_THINKING)") == 1
    assert gap.count("await generation.open_answer(\n                            turn.valves, turn.variant, think=gap_think, "
                     "local_thinking_budget=gap_budget,") == 1
    assert "self.valves.ENABLE_THINKING}" not in src
    mod = types.ModuleType("qp")
    exec(compile(src, "query_pipe.py", "exec"), mod.__dict__)
    pipe = mod.Pipe()
    v = pipe.valves
    assert v.ENABLE_THINKING is True and v.THINKING_AUTO is False
    # thinking-token cap: runaway reasoning must not use all of MAX_TOKENS (an empty answer)
    assert v.THINKING_BUDGET == 12288 < v.MAX_TOKENS
    assert generation.gen_extra(pipe.valves, True) == {"chat_template_kwargs": {"enable_thinking": True}, "thinking_token_budget": 12288}
    assert generation.gen_extra(pipe.valves, False) == {"chat_template_kwargs": {"enable_thinking": False}}
    v.THINKING_BUDGET = 0
    assert generation.gen_extra(pipe.valves, True) == {"chat_template_kwargs": {"enable_thinking": True}}


# ── Neo4j isolation: a stopped graph store must not freeze a chat ──
def _pipe_module():
    replace_imports = pytest.importorskip("open_webui.utils.plugin").replace_imports
    src = replace_imports((ROOT / "openwebui/functions/query_pipe.py").read_text())
    mod = types.ModuleType("qp_graph")
    exec(compile(src, "query_pipe.py", "exec"), mod.__dict__)
    return mod, src


def _closed_port() -> int:
    import socket
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    return port


def test_listing_shortcut_off_without_graph_valve(monkeypatch):
    mod, _ = _pipe_module()
    from memento.graph import store as graph_store
    monkeypatch.setattr(graph_store, "Neo4jStore", lambda *a, **k: pytest.fail("Neo4j must not be touched"))
    q = "list all papers about polymer foams"
    assert documents.entity_scoped_listing(q, None, {"neo4j_uri": "bolt://localhost:7689"}, enabled=False) is None


def test_listing_and_title_lookup_fast_when_graph_down():
    import time
    mod, _ = _pipe_module()
    dom = {"neo4j_uri": f"bolt://127.0.0.1:{_closed_port()}"}
    t = time.perf_counter()
    assert documents._domain_graph(dom) is None and documents._domain_graph({}) is None
    assert documents.entity_scoped_listing("list all papers about polymer foams", None, dom, enabled=True) is None
    assert documents._resolve_paper("what does Diss_Garcia say", ["Diss_Garcia"], dom) == "Diss_Garcia"   # id fallback
    assert time.perf_counter() - t < 3


def test_graph_helpers_run_off_the_event_loop():
    _, src = _pipe_module()
    run = inspect.getsource(service.run)
    assert "asyncio.to_thread(documents.entity_scoped_listing, turn.query, turn.paper, turn.dom,\n" in run
    assert "turn.valves.GRAPH_RETRIEVAL and turn.doc_chat is None)" in run     # Document Chat never reads the graph
    # document resolution goes through documents.resolve_reference, off the event loop
    assert "asyncio.to_thread(documents.resolve_reference," not in src
    assert inspect.getsource(resolve.detect).count("asyncio.to_thread(documents.resolve_reference,") == 2  # + own uploads
    assert inspect.getsource(gap_answer).count("asyncio.to_thread(documents.resolve_reference,") == 1  # the gap target
    assert "get_store()" not in src and "get_store()" not in run   # the domain's own uri, never the process-wide config


# ── User-facing errors + empty library ──
def test_user_error_messages():
    mod, _ = _pipe_module()
    busy = texts.user_error("answer", RuntimeError("Error code: 429 - rate limit"))
    down = texts.user_error("answer", RuntimeError("Connection error."))
    search = texts.user_error("search", RuntimeError("404 Collection x doesn't exist"))
    assert "busy" in busy and "not reachable" in down and "searching the library failed" in search
    for m in (busy, down, search):
        assert "`E-" in m and "Traceback" not in m and "404" not in m


def test_empty_library_answers_without_model_call(monkeypatch):
    # a collection that does not exist reads as empty (library_state.collection_status); the model is unreachable
    mod, _ = _pipe_module()
    monkeypatch.setattr(library_state, "collection_status", lambda *a: "empty")
    p = mod.Pipe()
    p.valves = p.Valves(QDRANT_URL=f"http://127.0.0.1:{_closed_port()}", COLLECTION="memento-does-not-exist-test",
                        LLM_BASE_URL=f"http://127.0.0.1:{_closed_port()}/v1", DOMAIN_NAME="generic")

    async def run():
        return [c async for c in p.pipe({"messages": [{"role": "user", "content": "What is PTFE?"}]}, {"id": "t"})]
    out = "".join(asyncio.run(run()))
    assert out == texts.EMPTY_LIBRARY_MSG


def test_trial_query_model_has_starter_prompts():
    import importlib.util
    spec = importlib.util.spec_from_file_location("so", ROOT / "scripts/setup_openwebui.py")
    so = importlib.util.module_from_spec(spec); spec.loader.exec_module(so)
    q = next(m for m in so.MODELS if m["id"] == "memento-query")
    sp = q["meta"]["suggestion_prompts"]
    assert len(sp) >= 3 and all(isinstance(x["title"], list) and x["content"] for x in sp)
    # suggestions go through retrieval: no library-listing shortcut; one German question
    assert not any("documents are in the library" in x["content"] for x in sp)
    assert any("Dokumenten" in x["content"] for x in sp)
    # the description describes the model (local variant); cloud variants carry no suggestions
    assert "English or German" in q["meta"]["description"] and "\n" not in q["meta"]["description"]      # one line
    assert all("suggestion_prompts" not in m["meta"] for m in so.MODELS if m["id"].startswith("memento-query-"))


# ── Intent fixes: chitchat label; "hello" must not list the library ──
def test_chitchat_is_a_valid_intent_and_needs_no_special_thinking():
    r, _ = _route('{"intent":"chitchat","query":"hello","reasoning":"light"}')
    assert r.intent == "chitchat" and qr.pipe_intent("chitchat") == "chitchat"


def test_router_prompt_separates_listing_verbatim_and_chitchat():
    p = qr.ROUTER_SYSTEM_PROMPT
    assert '"is there a paper about X?"' in p and "NOT corpus_listing" in p
    assert "NOT verbatim" in p and '"chitchat"' in p


def test_pipe_answers_chitchat_before_any_retrieval():
    src = inspect.getsource(service.run)
    i_chat, i_detect = src.index('if turn.query_intent in ("general", "chitchat"):'), src.index("# ── 1. Detect named paper")
    assert i_chat < i_detect


def test_a_greeting_is_answered_by_the_model_without_the_no_sources_note(monkeypatch):
    # chitchat goes to the answer model directly, like general questions
    import openai
    import memento.query.router as router
    mod, _ = _pipe_module()
    calls = []

    class FakeLLM:
        def __init__(self, *a, **k):
            async def create(**kw):
                calls.append(kw)

                async def chunks():
                    yield types.SimpleNamespace(choices=[types.SimpleNamespace(delta=types.SimpleNamespace(
                        content="Hello! Ask me about the library.", reasoning_content=None))])
                return chunks()
            self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=create))

    async def fake_route(*a, **k):
        return router.RouterResponse(intent="chitchat", query="hello", reasoning="light")
    monkeypatch.setattr(openai, "AsyncOpenAI", FakeLLM)
    monkeypatch.setattr(router, "route", fake_route)
    monkeypatch.setattr(library_state, "collection_status", lambda *a: "ready")
    p = mod.Pipe()

    async def run():
        return [c async for c in p.pipe({"messages": [{"role": "user", "content": "hello"}]}, {"id": "t"})]
    out = "".join(asyncio.run(run()))
    assert out.startswith("Hello! Ask me about the library.") and texts.GENERAL_NOTE not in out
    assert out.endswith(generation.answered_by("Qwen 3.8 Flash-Next (Local)"))
    assert len(calls) == 1 and "Library Chat" in calls[0]["messages"][0]["content"]


def test_router_failure_is_flagged_and_keywords_only_fall_back():
    r, _ = _route(None, RuntimeError("down"))
    assert r.ok is False and _route('{"intent":"ranked_retrieval","query":"q"}')[0].ok is True
    _, src = _pipe_module()
    assert "(not turn.router_ok and figures.detect_image_intent(turn.query))" in inspect.getsource(search_step.filters)
    assert "_wants_verbatim_request" not in src      # verbatim only on the router's explicit intent


def test_figures_go_through_the_safe_converter(tmp_path):
    mod, src = _pipe_module()
    assert "data:image/jpeg" not in src
    svg = tmp_path / "f.svg"
    svg.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="40" height="30"><rect width="40" height="30" fill="red"/></svg>')
    assert prompts.verified_image_url(str(svg)).startswith("data:image/png;base64,")
    bad = tmp_path / "broken.jpg"; bad.write_bytes(b"not an image")
    assert prompts.verified_image_url(str(bad)) is None and prompts.verified_image_url(str(tmp_path / "missing.png")) is None


# ── Not about the library → answered directly ──
def test_general_is_a_valid_intent_and_the_prompt_keeps_science_on_retrieval():
    r, _ = _route('{"intent":"general","query":"what is today\'s date?","reasoning":"light"}')
    assert r.intent == "general" and qr.pipe_intent("general") == "general"
    p = qr.ROUTER_SYSTEM_PROMPT
    assert '"general"' in p and '"what is DFT?"' in p and 'are NOT general' in p


def test_pipe_answers_general_before_any_retrieval():
    src = inspect.getsource(service.run)
    assert src.index('if turn.query_intent in ("general", "chitchat"):') < src.index("# ── 1. Detect named paper")


def test_general_question_gets_one_direct_call_with_the_date_and_no_sources(monkeypatch):
    import asyncio
    import types
    import openai
    import memento.query.router as router
    mod, _ = _pipe_module()
    calls = []

    class FakeLLM:
        def __init__(self, *a, **k):
            async def create(**kw):
                calls.append(kw)

                async def chunks():          # answers stream through generation.open_answer (model variants)
                    yield types.SimpleNamespace(choices=[types.SimpleNamespace(delta=types.SimpleNamespace(
                        content="<think></think>Today is Monday.", reasoning_content=None))])
                return chunks()
            self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=create))

    async def fake_route(*a, **k):
        return router.RouterResponse(intent="general", query="what is today's date?", reasoning="light")
    monkeypatch.setattr(openai, "AsyncOpenAI", FakeLLM)
    monkeypatch.setattr(router, "route", fake_route)
    monkeypatch.setattr(library_state, "collection_status", lambda *a: "ready")
    events = []
    p = mod.Pipe()

    async def emit(e):
        events.append(e)

    async def run():
        return [c async for c in p.pipe({"messages": [{"role": "user", "content": "what is today's date?"}]},
                                        {"id": "t"}, __event_emitter__=emit)]
    out = "".join(asyncio.run(run()))
    assert out.startswith("Today is Monday.") and "<think>" not in out
    assert out.endswith(texts.GENERAL_NOTE + generation.answered_by("Qwen 3.8 Flash-Next (Local)"))    # no model chosen → local
    assert len(calls) == 1 and calls[0]["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    import datetime
    assert datetime.datetime.now().astimezone().strftime("%Y") in calls[0]["messages"][0]["content"]
    assert not any(e.get("type") == "citation" for e in events)


@pytest.mark.parametrize("intent", ["general", "ranked_retrieval"])
def test_empty_library_still_answers_general_questions(monkeypatch, intent):
    # general questions are answered directly on an empty library too; anything that needs documents still gets the
    # empty-library message
    import openai
    import memento.query.router as router
    mod, _ = _pipe_module()

    class FakeLLM:
        def __init__(self, *a, **k):
            async def create(**kw):
                async def chunks():
                    yield types.SimpleNamespace(choices=[types.SimpleNamespace(delta=types.SimpleNamespace(
                        content="Today is Friday.", reasoning_content=None))])
                return chunks()
            self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=create))

    async def fake_route(*a, **k):
        return router.RouterResponse(intent=intent, query="q", reasoning="light")
    monkeypatch.setattr(openai, "AsyncOpenAI", FakeLLM)
    monkeypatch.setattr(router, "route", fake_route)
    monkeypatch.setattr(library_state, "collection_status", lambda *a: "empty")
    p = mod.Pipe()

    async def run():
        return [c async for c in p.pipe({"messages": [{"role": "user", "content": "what is today's date?"}]},
                                        {"id": "t"})]
    out = "".join(asyncio.run(run()))
    if intent == "general":
        assert out.startswith("Today is Friday.") and texts.GENERAL_NOTE in out
    else:
        assert out == texts.EMPTY_LIBRARY_MSG


def test_query_ignores_openwebuis_attached_files_block():
    mod, _ = _pipe_module()
    block = '<attached_files>\n<file type="file" id="f1" name="paper.pdf"/>\n</attached_files>\n\n'
    assert mod.request.message_text(block + "What is DFT?") == "What is DFT?"
    assert mod.request.message_text([{"type": "text", "text": block}, {"type": "text", "text": "What is DFT?"}]).strip() \
        == "What is DFT?"


def test_attachment_note_only_for_files_on_this_message():
    # __files__ holds every file of the chat: an upload to Ingestion earlier in the same chat must not trigger the Query
    # note; Open WebUI's <attached_files> block marks the message that actually carries files
    src = (ROOT / "memento/query/service.py").read_text()                 # the message's set-up step
    assert 'attached_now = "<attached_files>" in' in src
    # images get IMAGE_NOTE; Memento Document Chat reads attachments itself
    assert "    if attached_now and (files_now or not images_now) and turn.doc_chat is None:\n" in src
    assert "        if attached:\n            yield (\"_Note: files attached here" not in src


def test_citation_names_carry_the_section():
    # several passages from one page are one source in the source list
    mod, _ = _pipe_module()
    h = {"source": "Becker 2026", "page": 5, "section_raw": "2.4.2 | The Full Molar Mass Distribution Histogram"}
    assert presentation.source_name(h) == "Becker 2026 (p. 6, §2.4.2)"
    assert presentation.source_name({**h, "section_raw": "Introduction"}) == "Becker 2026 (p. 6, Introduction)"
    long = presentation.source_name({**h, "section_raw": "Results on the thermal stability of the foams"})
    assert long.endswith("…)") and len(long) < 60
    assert presentation.source_name({"source": "X", "page": 0, "page_end": 1}) == "X (pp. 1–2)"


def test_attached_images_are_told_apart_from_files():
    mod, _ = _pipe_module()
    block = ('<attached_files>\n<file id="1" url="1" content_type="image/png" name="sem.png"/>\n</attached_files>\n\n'
             "What does this figure show?")
    assert request.attachment_kinds([{"type": "text", "text": block}, {"type": "image_url", "image_url": {"url": "data:"}}],
                                 block) == (True, False)
    both = block.replace("</attached_files>", '<file id="2" url="2" content_type="application/pdf" name="a.pdf"/>\n</attached_files>')
    assert request.attachment_kinds(both, both) == (True, True)
    pdf = '<attached_files>\n<file id="2" url="2" name="a.pdf"/>\n</attached_files>\n\nq'
    assert request.attachment_kinds(pdf, pdf) == (False, True)
    assert request.attachment_kinds("plain question", "plain question") == (False, False)


def test_any_attached_image_gets_only_the_image_reply():
    # no routing rules: an image means the fixed reply, with or without text
    src = (ROOT / "memento/query/service.py").read_text()                 # the message's set-up step
    assert re.search(r"    if images_now and turn.doc_chat is None: .*\n        yield texts.IMAGE_MSG\n        turn.done = True\n"
                     r"        return\n", src)
    assert "can't read attachments directly" in texts.IMAGE_MSG and "Library Upload" in texts.IMAGE_MSG
    assert "about_attachment" not in src


# ── future-work questions and objections ──
def test_router_parses_suggest():
    r, _ = _route('{"intent":"gap_query","query":"q","reasoning":"deep","suggest":true}')
    assert r.intent == "gap_query" and r.suggest is True
    assert _route('{"intent":"gap_query","query":"q","reasoning":"deep"}')[0].suggest is False      # missing
    assert _route('{"intent":"gap_query","query":"q","suggest":"yes"}')[0].suggest is False         # only JSON true
    assert _route(exc=RuntimeError("down"))[0].suggest is False                                     # router failure


def test_router_prompt_routes_future_work_and_objections():
    p = qr.ROUTER_SYSTEM_PROMPT
    gap = p[p.index('"gap_query"'):p.index('"set_query"')]
    assert all(w in gap for w in ("suggest", "extend", "next steps", "future work"))
    prop = p[p.index('   - "proposal"'):p.index('   - "summary"')]
    assert "research proposal" in prop and "future work" not in prop
    rework = p[p.index('   - "rework"'):p.index('   - "ranked_retrieval"')]
    assert "objection" in rework and "NOT rework" in rework
    assert '"suggest"' in p and "asks to DO something" not in p and "ONLY for asking what the gaps ARE" not in p
