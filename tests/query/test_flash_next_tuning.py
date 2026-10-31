# Tuning for the local model: a finished answer typed out at its speed, live answers without thinking unless Thorough
# check is on, Thorough check described as what it does, the default library's switch to the same model.
import importlib.util
import inspect
import re
import sys
from pathlib import Path
from memento.config.paths import PROJECT_ROOT
from _owui_loader import load_function
from memento.query import cache, generation, library_state, prompts, request, service
from memento.query.scope import conversation, documents
from memento.query.flows import gap_answer, summary

ROOT = PROJECT_ROOT
sys.path.insert(0, str(ROOT))
qp = load_function(ROOT / "openwebui/functions/query_pipe.py", "memento_query")


# ── a finished answer is typed out at about the local model's speed ──
# Measured about 270–370 characters a second (one to four requests at once): 300 for every library, never faster than a
# real answer.

def test_a_finished_answer_is_typed_at_300_characters_a_second():
    src = (ROOT / "openwebui/functions/query_pipe.py").read_text()
    assert 'REPLAY_CHARS_PER_S: float = float(os.getenv("REPLAY_CHARS_PER_S", "300"))' in src


# ── Thorough check says what it does: thinking, then the statement check ──
# Live answers think only with Thorough check on; its texts (the switch's hover tip, the button's description in two
# places, the help line the model uses when asked about it) say both steps, no durations.

THOROUGH_TEXT = "Takes time to think the question through, then checks each statement against the sources."


def test_the_thorough_switch_tip_names_both_steps():
    js = (ROOT / "openwebui/loader.js").read_text()
    tip = js[js.index("const TIP = '<b>Thorough check</b>'"):js.index("const tipEl")]
    tip = re.sub(r"'\s*\+\s*'", "", tip)                              # the literal's pieces, joined
    assert "think the question through" in tip and "checks each statement against its sources" in tip
    assert "a quick answer" in tip                                  # what Off means
    assert "—" not in tip                                           # no em dashes in user-facing text


def test_the_thorough_button_description_is_the_same_in_both_places():
    sys.path.insert(0, str(ROOT / "scripts"))
    filt = (ROOT / "openwebui/functions/thorough_filter.py").read_text()
    assert f"description: {THOROUGH_TEXT} Answers take longer." in filt
    src = (ROOT / "scripts/setup_openwebui.py").read_text()
    meta = src[src.index('"id":   "memento_thorough"'):src.index('"id":   "memento_thorough"') + 600]
    assert THOROUGH_TEXT[:40] in meta and "Answers take longer." in meta


def test_the_help_line_tells_the_model_what_thorough_check_does():
    src = re.sub(r'"\s*\n\s*"', "", (ROOT / "memento/query/flows/general.py").read_text())    # joined literals
    assert ('The \\"Thorough check\\" switch next to + makes Memento take time to think the question through and '
            'check each statement against the sources; answers take longer.') in src


# ── live answers think only with Thorough check on ──
# An ordinary answer's thinking mostly hit the 75 s cut and was dropped (the answer then came much later, written
# without it). The background (warm-up, pre-warm) keeps ENABLE_THINKING; research gaps keep their own rule
# (GAP_THINKING), so their stored answers keep their keys.

def test_a_live_answer_does_not_think_without_thorough_check():
    assert generation.live_thinks(True, thorough=False, warm=False, enabled=False) is False


def test_thorough_check_keeps_the_thinking_the_router_chose():
    assert generation.live_thinks(True, thorough=True, warm=False, enabled=False) is True
    assert generation.live_thinks(False, thorough=True, warm=False, enabled=False) is False    # a plain lookup: the router


def test_a_warmup_and_the_valve_think_as_before():
    assert generation.live_thinks(True, thorough=False, warm=True, enabled=False) is True
    assert generation.live_thinks(True, thorough=False, warm=False, enabled=True) is True
    src = (ROOT / "openwebui/functions/query_pipe.py").read_text()
    assert 'LIVE_THINKING:  bool  = os.getenv("LIVE_THINKING", "false")' in src


def test_research_gaps_keep_their_own_thinking_and_their_stored_keys():
    route = inspect.getsource(service._route)
    assert route.index("turn.think = generation.list_thinks(") < route.index("turn.gap_base_think = turn.think") < \
        route.index("turn.think = generation.live_thinks(")
    gap = inspect.getsource(gap_answer.reply)
    assert re.search(r'prompt_version=cache\.prompt_version\("gaps", "capped", turn\.dom, turn\.valves, think=turn\.gap_base_think,', gap)
    assert "gap_think = generation.gap_thinks(turn.gap_base_think, thorough, turn.valves.GAP_THINKING)" in gap


def _ask(monkeypatch, metadata):
    """One ordinary question through the real pipe (fake search, a recording model): the answer call's thinking."""
    import asyncio
    import types
    import openai
    import memento.retrieval.search as search
    import memento.common.model_servers as vc
    from test_rq3_answer_pipeline import _RecordingAsyncOpenAI, _load_pipe
    mod = _load_pipe()
    payloads = [{"text": "Alumina reached 99 % density.", "source": "P", "page": 0, "type": "text"}]
    points = [types.SimpleNamespace(payload=p, score=0.9, id=i) for i, p in enumerate(payloads)]
    _RecordingAsyncOpenAI.calls = []
    monkeypatch.setattr(openai, "AsyncOpenAI", _RecordingAsyncOpenAI)
    monkeypatch.setattr(library_state, "collection_status", lambda c, coll: "ok")
    monkeypatch.setattr(library_state, "load_documents", lambda *a: {"P": {}})
    monkeypatch.setattr(library_state, "in_progress_sources", lambda *a: set())
    monkeypatch.setattr(documents, "_resolve_paper", lambda q, s, d=None: None)
    monkeypatch.setattr(library_state, "qdrant_query", lambda *a, **k: types.SimpleNamespace(points=points))
    monkeypatch.setattr(vc, "vl_embed", lambda *a, **k: [0.0] * 8)
    monkeypatch.setattr(search, "rerank", lambda q, hits, *a, **k: hits)
    p = mod.Pipe()
    p.valves = p.Valves(DOMAIN_NAME="generic", COLLECTION="x", GRAPH_RETRIEVAL=False, VALIDATION_ENABLED=False,
                        THINKING_AUTO=False, ENABLE_THINKING=True)

    async def run():
        return [c async for c in p.pipe({"messages": [{"role": "user", "content": "What density did alumina reach?"}]},
                                        {"id": "t"}, __metadata__=metadata)]
    asyncio.run(run())
    sent = [c for c in _RecordingAsyncOpenAI.calls if c.get("stream")]
    assert len(sent) == 1, "the pipe did not reach the answer step"
    return sent[0]["extra_body"]["chat_template_kwargs"]["enable_thinking"]


def test_the_pipe_answers_a_live_question_without_thinking(monkeypatch):
    assert _ask(monkeypatch, {}) is False


def test_the_pipe_thinks_with_thorough_check_on(monkeypatch):
    assert _ask(monkeypatch, {"filter_ids": ["memento_thorough"]}) is True


# ── "Which of them …?" after an answer about several documents ──
# A plural reference after an answer that cited two or more library documents searches within those documents (not only
# one cited document, conversation.this_document).

def test_a_plural_reference_means_the_documents_the_previous_answer_cited():
    assert conversation.these_documents({"B", "A"}, False) == ["A", "B"]
    assert conversation.these_documents({"A"}, False) is None                 # one document: this_document decides
    assert conversation.these_documents({"A", "B"}, True) is None             # a citation that cannot be mapped: ask
    assert conversation.these_documents(set(), False) is None


def test_the_follow_up_searches_within_the_cited_documents(monkeypatch):
    import asyncio
    import json
    import types
    import openai
    import memento.retrieval.search as search
    import memento.common.model_servers as vc
    from test_rq3_answer_pipeline import _load_pipe
    mod = _load_pipe()
    follow = "Which of them reported the lowest peak heat release rate?"
    rewritten = "Which of the dissertations on phosphorus-based flame retardants (Steinmetz, Garcia) reported the lowest pHRR?"

    class Model:
        calls = []

        def __init__(self, *a, **k):
            self.chat = self; self.completions = self

        async def create(self, **kw):
            Model.calls.append(kw)
            if kw.get("stream"):
                async def gen():
                    delta = types.SimpleNamespace(content="Garcia reported the lowest [1].", reasoning_content=None,
                                                  reasoning=None)
                    yield types.SimpleNamespace(choices=[types.SimpleNamespace(delta=delta)])
                return gen()
            route = {"intent": "ranked_retrieval", "query": rewritten, "reasoning": "deep", "scope": "documents"}
            msg = types.SimpleNamespace(content=json.dumps(route))
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

    filters = []
    points = [types.SimpleNamespace(payload={"text": "pHRR 180 kW/m2", "source": "Vortrag_Garcia", "page": 3,
                                             "type": "text"}, score=0.9, id=1)]

    def query(client, collection, qvec, text, query_filter, *a, **k):
        filters.append(query_filter)
        return types.SimpleNamespace(points=points)

    async def previous(metadata):
        stored = [{"metadata": [{"name": "[1] Vortrag_Steinmetz", "source": "Vortrag_Steinmetz#3"}]},
                  {"metadata": [{"name": "[2] Vortrag_Garcia", "source": "Vortrag_Garcia#7"}]}]
        return ("Which dissertations worked on phosphorus-based flame retardants?",
                "Steinmetz used DOPO [1]; Garcia used ammonium polyphosphate [2].", stored)
    docs = {"Vortrag_Steinmetz": {}, "Vortrag_Garcia": {}, "Other": {}}
    monkeypatch.setattr(openai, "AsyncOpenAI", Model)
    monkeypatch.setattr(conversation, "stored_previous_turn", previous)
    monkeypatch.setattr(library_state, "collection_status", lambda c, coll: "ok")
    monkeypatch.setattr(library_state, "load_documents", lambda *a: docs)
    monkeypatch.setattr(library_state, "in_progress_sources", lambda *a: set())
    monkeypatch.setattr(documents, "_resolve_paper", lambda q, s, d=None: None)
    monkeypatch.setattr(library_state, "qdrant_query", query)
    monkeypatch.setattr(vc, "vl_embed", lambda *a, **k: [0.0] * 8)
    monkeypatch.setattr(search, "rerank", lambda q, hits, *a, **k: hits)
    p = mod.Pipe()
    p.valves = p.Valves(DOMAIN_NAME="generic", COLLECTION="x", GRAPH_RETRIEVAL=False, VALIDATION_ENABLED=False)
    body = {"messages": [{"role": "user", "content": "Which dissertations worked on phosphorus-based flame retardants?"},
                         {"role": "assistant", "content": "Steinmetz used DOPO [1]; Garcia used ammonium polyphosphate [2]."},
                         {"role": "user", "content": follow}]}

    async def run():
        return "".join([c async for c in p.pipe(body, {"id": "t"}, __metadata__={"chat_id": "c1"}) if isinstance(c, str)])
    out = asyncio.run(run())
    assert "could not find that document" not in out
    assert any(c.get("stream") for c in Model.calls), "the follow-up was never answered"
    allowed = [set(cond.match.any) for f in filters if f is not None for cond in (f.must or [])
               if getattr(cond, "key", None) == "source" and hasattr(cond.match, "any")]
    assert allowed and all(a == {"Vortrag_Steinmetz", "Vortrag_Garcia"} for a in allowed)


# ── the default library on the same local model ──

def _setup_module():
    s_spec = importlib.util.spec_from_file_location("so_flash", ROOT / "scripts/setup_openwebui.py")
    so = importlib.util.module_from_spec(s_spec)
    s_spec.loader.exec_module(so)
    return so


class _R:
    def __init__(self, body, status_code=200):
        self.body, self.status_code, self.text = body, status_code, ""

    def json(self):
        return self.body

    def raise_for_status(self):
        pass


def _fake_api(monkeypatch, so, saved):
    posted = []

    def get(url, **kw):
        fid = url.split("/functions/id/")[1].split("/")[0] if "/functions/id/" in url else None
        return _R(dict(saved[fid])) if fid in saved else _R(None, 404)

    def post(url, json=None, **kw):
        posted.append((url.split("/functions/id/")[1].split("/")[0], json))
        saved[posted[-1][0]] = dict(json)
        return _R({"ok": True})
    monkeypatch.setattr(so.requests, "get", get)
    monkeypatch.setattr(so.requests, "post", post)
    return posted


def test_the_switch_sets_flash_next_and_keeps_every_other_setting(monkeypatch):
    so = _setup_module()
    saved = {"memento_query": {"LLM_BASE_URL": "http://localhost:8090/v1", "LLM_MODEL": "qwen3.8-27b",
                               "COLLECTION": "memento-comp-mat-rag", "THINKING_AUTO": True},
             "memento_ingestion": {"LLM_BASE_URL": "http://localhost:8090/v1", "LLM_MODEL": "qwen3.8-27b",
                                   "COLLECTION": "memento-comp-mat-rag", "BACKEND": "vlm-engine"}}
    posted = _fake_api(monkeypatch, so, saved)
    so.cmd_flash_next("http://x", {}, dry=False)
    assert [fid for fid, _ in posted] == ["memento_query", "memento_ingestion"]
    q, i = saved["memento_query"], saved["memento_ingestion"]
    assert (q["LLM_BASE_URL"], q["LLM_MODEL"], q["LLM_API_KEY_ENV"]) == (
        "http://localhost:8023/v1", "qwen-flash-next", "REASONING_API_KEY")
    assert q["METRICS_URL"] == "http://localhost:8023/metrics"
    assert q["METRICS_API_KEY_ENV"] == "REASONING_API_KEY"   # the warm-up reads the same server's load
    assert (i["LLM_BASE_URL"], i["LLM_MODEL"], i["LLM_API_KEY_ENV"]) == (q["LLM_BASE_URL"], q["LLM_MODEL"],
                                                                         q["LLM_API_KEY_ENV"])
    assert "METRICS_URL" not in i                                      # an upload has no warm-up of its own
    assert (q["COLLECTION"], q["THINKING_AUTO"], i["BACKEND"]) == ("memento-comp-mat-rag", True, "vlm-engine")


def test_the_switch_dry_run_and_a_second_run_post_nothing(monkeypatch):
    so = _setup_module()
    saved = {"memento_query": {"LLM_MODEL": "qwen3.8-27b"}, "memento_ingestion": {"LLM_MODEL": "qwen3.8-27b"}}
    posted = _fake_api(monkeypatch, so, saved)
    so.cmd_flash_next("http://x", {}, dry=True)
    assert posted == []
    so.cmd_flash_next("http://x", {}, dry=False)
    n = len(posted)
    so.cmd_flash_next("http://x", {}, dry=False)                      # already on the model: nothing to write
    assert len(posted) == n == 2


def test_the_switch_needs_the_model_server_url(monkeypatch):
    import pytest
    so = _setup_module()
    posted = _fake_api(monkeypatch, so, {"memento_query": {}, "memento_ingestion": {}})
    monkeypatch.setattr(so, "FLASH_NEXT_URL", "")
    with pytest.raises(SystemExit, match="REASONING_URL"):
        so.cmd_flash_next("http://x", {}, dry=False)
    assert posted == []


def test_the_switch_stops_on_a_missing_copy(monkeypatch):
    import pytest
    so = _setup_module()
    _fake_api(monkeypatch, so, {"memento_query": {"LLM_MODEL": "qwen3.8-27b"}})     # no ingestion function
    with pytest.raises(SystemExit):
        so.cmd_flash_next("http://x", {}, dry=False)


def test_without_a_model_server_url_the_trial_leaves_the_upload_model_alone(monkeypatch):
    so = _setup_module()
    monkeypatch.setattr(so, "FLASH_NEXT_URL", "")
    i = so.trial_valves("memento_ingestion", "generic", {})
    assert not {"LLM_BASE_URL", "LLM_MODEL", "LLM_API_KEY_ENV"} & set(i) and i["SKIP_PARSE"] is False


def test_the_trial_settings_and_the_model_list_name_flash_next():
    import yaml
    so = _setup_module()
    i = so.trial_valves("memento_ingestion", "generic", {"REASONING_MODEL_NAME": "qwen3.8-27b"})
    assert (i["LLM_BASE_URL"], i["LLM_MODEL"]) == ("http://localhost:8023/v1", "qwen-flash-next")
    local = next(v for v in yaml.safe_load((ROOT / "config/query_models.yaml").read_text()) if v["id"] == "local")
    study = next(v for v in yaml.safe_load((ROOT / "config/query_models_study.yaml").read_text()) if v["id"] == "local")
    assert (local["name"], local["label"], local["model"]) == (study["name"], study["label"], study["model"]) == (
        "Library Chat (Qwen 3.8 Flash-Next)", "Qwen 3.8 Flash-Next (Local)", "qwen-flash-next")


# ── whole-document summaries use the local model's window ──
# The model has a 131,072-token window: a 100,000-token document budget summarises most documents whole in the same
# time. The research-gap passage budget stays at 20,000: a larger one puts more gaps under a wrong heading.

def test_whole_document_summaries_use_flash_next_s_window():
    src = (ROOT / "openwebui/functions/query_pipe.py").read_text()
    assert 'SUMMARY_MAX_TOKENS: int = int(os.getenv("SUMMARY_MAX_TOKENS", "100000"))' in src
    assert summary.SUMMARY_PROMPT_TOKENS == 106000
    assert summary.SUMMARY_PROMPT_TOKENS + 4000 + 16384 <= 131072   # document + history, instructions, the answer


def test_a_thesis_of_about_70000_tokens_is_now_summarised_whole():
    ps = [{"text": "x" * 300, "source": "D", "page": i // 10, "section": "S"} for i in range(600)]
    est = sum(len(p["text"]) + prompts.SUMMARY_LABEL_CHARS for p in ps) / request.CHARS_PER_TOKEN
    assert 60000 < est < 100000
    assert summary.summary_hits(ps, 0, 100000, summary.SUMMARY_PROMPT_TOKENS)[1] == "whole"
    assert summary.summary_hits(ps, 0, 40000, 46000)[1] == "partial"                         # before


def test_research_gap_passages_keep_their_budget():
    src = (ROOT / "openwebui/functions/query_pipe.py").read_text()
    assert 'GAP_BUDGET_TOKENS: int = int(os.getenv("GAP_BUDGET_TOKENS", "20000"))' in src


# ── the pipe's own lists show the library's titles ──
# The library listing, documents by a person, the document lookup, related documents and the source labels the model
# reads use the cleaned library title (kind prefix, unique; the picker uses them), not the raw first-page title.

def test_the_lists_show_the_library_title_with_the_file_name(monkeypatch):
    from sqlalchemy import create_engine
    from memento.library import documents as ldb
    e = create_engine("sqlite://")
    ldb.ensure_tables(e)
    rows = [("k2_Kap9_APA_ok2", "Chapter 9 (Lindemann): Zusammenfassung zu Kapitel 9", "**Abb. 9. 9.: REM-Aufnahme"),
            ("Vortrag_X", "Vortrag_X", "Micro- and nanostructured polymer blends"),          # no better library title
            ("Plain_Name", "Plain_Name", None), ("Spaced_Name", "Spaced Name", None)]
    for src, title, parsed in rows:
        ldb.add_document(e, collection="c", entry={"source": src, "title": title, "parsed_title": parsed},
                         content_hash="h", uploaded_by="u", output_root=None)
    monkeypatch.setattr(qp, "_db_engine", lambda: e)
    assert library_state.load_documents("c", None, qp._db_engine)
    shown = documents.display_titles(["k2_Kap9_APA_ok2", "Vortrag_X", "Plain_Name", "Spaced_Name"], None)
    assert shown["k2_Kap9_APA_ok2"] == "Chapter 9 (Lindemann): Zusammenfassung zu Kapitel 9 (k2_Kap9_APA_ok2)"
    assert shown["Vortrag_X"] == "Micro- and nanostructured polymer blends (Vortrag_X)"     # the first-page title
    assert shown["Plain_Name"] == "Plain_Name"                                               # the file name is the title
    assert shown["Spaced_Name"] == "Spaced Name (Spaced_Name)"                               # as any other title
