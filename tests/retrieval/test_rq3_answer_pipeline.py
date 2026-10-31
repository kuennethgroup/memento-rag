"""RQ3 task 1: the production answer step as a library (memento/retrieval/answer_pipeline.py)."""
import asyncio
import types
from pathlib import Path

import pytest
from PIL import Image

from memento.retrieval.answer_pipeline import (apply_context_hygiene, build_context, build_messages, chunk_label,
                                       generate_answer, system_prompt_for)
from memento.config.paths import PROJECT_ROOT
from memento.query import library_state, request
from memento.query.scope import documents

ROOT = PROJECT_ROOT
H = lambda src, page, text, typ="text", sec=None, raw=None, img=None: {
    "source": src, "page": page, "text": text, "type": typ, "section": sec, "section_raw": raw, "img_path": img}


def _png(path, color="red"):
    Image.new("RGB", (40, 30), color).save(path)
    return str(path)


# ── unit tests ────────────────────────────────────────────────────────────────
def test_chunk_label_matches_query_pipe_format():
    assert chunk_label(H("Diss_X", 4, "t")) == "Diss_X, page 5"
    assert chunk_label(H("Diss_X", 4, "t", sec="Results")) == "Diss_X, page 5, Section: Results"
    assert chunk_label(H("Diss_X", 4, "t", sec="Results", raw="4.2 Fatigue")) == "Diss_X, page 5, Section: Results › 4.2 Fatigue"
    assert chunk_label(H("Diss_X", 4, "t", sec="Results", raw="results")) == "Diss_X, page 5, Section: Results"


def test_context_and_history_note():
    assert build_context([H("A", 0, "one"), H("B", 2, "two")]) == "[1] A, page 1\none\n\n---\n\n[2] B, page 3\ntwo"
    assert system_prompt_for("SYS", False) == "SYS" and system_prompt_for("SYS", True).startswith("SYS\nEarlier turns")


def test_build_messages_text_only_is_a_plain_string():
    msgs = build_messages("SYS", [H("P", 0, "text one")], "Q?")
    assert msgs == [{"role": "system", "content": "SYS"},
                    {"role": "user", "content": "Sources:\n[1] P, page 1\ntext one\n\nQuestion: Q?"}]


def test_build_messages_images_follow_the_pipe_cap(tmp_path):
    hits = [H("P", i, f"fig {i}", "image", img=_png(tmp_path / f"f{i}.png")) for i in range(8)]
    parts = build_messages("SYS", hits, "Q?", max_images=6)[-1]["content"]
    # the cap counts figures, not label + image parts
    assert [p["text"] for p in parts if p["type"] == "text"][1:] == [f"\n\nImage for Source [{i}]:" for i in range(1, 7)]
    assert sum(p["type"] == "image_url" for p in parts) == 6


def test_image_labels_stay_apart_from_the_question(tmp_path):
    # the model's chat template joins text parts with nothing in between: the question and the first image label must
    # not run together
    hits = [H("P", 0, "text"), H("P", 1, "fig", "image", img=_png(tmp_path / "f.png"))]
    parts = build_messages("SYS", hits, "Explain about RAG-Anything", max_images=6)[-1]["content"]
    joined = "".join(p["text"] for p in parts if p["type"] == "text")
    assert "Question: Explain about RAG-Anything\n\nImage for Source [2]:" in joined


def test_unreadable_image_is_skipped(tmp_path):
    bad = tmp_path / "broken.jpg"; bad.write_bytes(b"not an image")
    msgs = build_messages("SYS", [H("P", 0, "cap", "image", img=str(bad))], "Q?")
    assert isinstance(msgs[-1]["content"], str)


def test_hygiene_drops_toc_and_near_duplicates_keeping_best_rank():
    para = ("Daher wurden zusätzlich TSC-Messungen an den medium und hochverstreckten Filmen mit Glashohlkugeln "
            "vorgenommen, um diesen Sachverhalt zu verifizieren.")
    hits = [H("Diss_Lindemann", 86, para), H("i_Kap7_IEEE_Diss_ok", 0, "**" + para + "**"),
            H("b_Inhaverz_Diss_ok", 0, "7.3.1.2. Medium verstreckte i-PP-Filme mit Glashohlkugeln.............................70"),
            H("Diss_Lindemann", 25, "A different paragraph about charge decay in cellular films."),
            H("Diss_Lindemann", 30, para, "image")]            # a caption is never dropped as a duplicate
    kept, stats = apply_context_hygiene(hits)
    assert [(h["source"], h["page"]) for h in kept] == [("Diss_Lindemann", 86), ("Diss_Lindemann", 25), ("Diss_Lindemann", 30)]
    assert stats == {"dropped_toc": 1, "dropped_dup": 1}


def test_generate_answer_separates_thinking():
    class LLM:
        def __init__(self): self.chat = self; self.completions = self; self.kw = None
        def create(self, **kw):
            self.kw = kw
            msg = types.SimpleNamespace(content="<think>plan</think>\nAnswer [1].", reasoning_content=None)
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)], usage=None)
    llm = LLM()
    out = generate_answer(llm, "m", [{"role": "user", "content": "q"}], enable_thinking=False)
    assert out["answer"] == "Answer [1]." and out["thinking"] == "plan"
    assert llm.kw["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert llm.kw["temperature"] == 0.15 and llm.kw["max_tokens"] == 16384
    generate_answer(llm, "m", [{"role": "user", "content": "q"}], enable_thinking=True)
    assert llm.kw["extra_body"] == {"chat_template_kwargs": {"enable_thinking": True}, "thinking_token_budget": 12288}


# ── fidelity: the real pipe with fake search + a recording model ──────────────
def _load_pipe():
    replace_imports = pytest.importorskip("open_webui.utils.plugin").replace_imports
    src = replace_imports((ROOT / "openwebui/functions/query_pipe.py").read_text())
    mod = types.ModuleType("qp_fidelity")
    exec(compile(src, "query_pipe.py", "exec"), mod.__dict__)
    return mod


class _RecordingAsyncOpenAI:
    """Router call → a fixed JSON route; streamed answer call → records its messages and streams 'Answer [1].'"""
    calls: list = []

    def __init__(self, *a, **k):
        self.chat = self; self.completions = self

    async def create(self, **kw):
        _RecordingAsyncOpenAI.calls.append(kw)
        if kw.get("stream"):
            async def gen():
                delta = types.SimpleNamespace(content="Answer [1].", reasoning_content=None, reasoning=None)
                yield types.SimpleNamespace(choices=[types.SimpleNamespace(delta=delta)])
            return gen()
        msg = types.SimpleNamespace(content='{"intent":"ranked_retrieval","query":"q","reasoning":"deep"}')
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


@pytest.mark.parametrize("with_images", [False, True])
def test_messages_identical_to_the_live_pipe(tmp_path, monkeypatch, with_images):
    import openai
    import memento.retrieval.search as search
    import memento.common.model_servers as vc
    from memento.retrieval.search import point_to_hit
    mod = _load_pipe()

    payloads = [{"text": "Alumina reached 99 % density.", "source": "P", "page": 0, "type": "text",
                 "section": "Results", "section_raw": "4.1 Density"},
                {"text": "SPS used 1400 °C.", "source": "Q", "page": 3, "type": "text"}]
    if with_images:
        payloads += [{"text": f"Fig. {k} caption", "source": "P", "page": k, "type": "chart" if k % 2 else "image",
                      "img_path": _png(tmp_path / f"c{k}.png", ("red", "blue", "green")[k % 3])} for k in range(8)]
    points = [types.SimpleNamespace(payload=p, score=0.9 - i / 10, id=i) for i, p in enumerate(payloads)]

    _RecordingAsyncOpenAI.calls = []
    monkeypatch.setattr(openai, "AsyncOpenAI", _RecordingAsyncOpenAI)
    monkeypatch.setattr(library_state, "collection_status", lambda c, coll: "ok")
    monkeypatch.setattr(library_state, "load_documents", lambda *a: {"P": {}, "Q": {}})
    monkeypatch.setattr(library_state, "in_progress_sources", lambda *a: set())
    monkeypatch.setattr(documents, "_resolve_paper", lambda q, s, d=None: None)
    monkeypatch.setattr(library_state, "qdrant_query", lambda *a, **k: types.SimpleNamespace(points=points))
    monkeypatch.setattr(vc, "vl_embed", lambda *a, **k: [0.0] * 8)
    monkeypatch.setattr(search, "rerank", lambda q, hits, *a, **k: hits)

    p = mod.Pipe()
    p.valves = p.Valves(DOMAIN_NAME="generic", COLLECTION="x", RERANK_ENABLED=True, GRAPH_RETRIEVAL=False,
                        VALIDATION_ENABLED=False, ENABLE_QUERY_REWRITE=True, THINKING_AUTO=False)
    question = "What density did alumina reach?"

    async def run():
        return [c async for c in p.pipe({"messages": [{"role": "user", "content": question}]}, {"id": "t"})]
    asyncio.run(run())

    sent = [c for c in _RecordingAsyncOpenAI.calls if c.get("stream")]
    assert len(sent) == 1, "the pipe did not reach the answer step"
    hits = [point_to_hit(pt.payload, pt.score) for pt in points]
    import yaml
    from memento.query.prompts import PLAIN_FORMULAS                  # the pipe adds it to every domain
    dom_prompt = yaml.safe_load((ROOT / "config/domains/generic.yaml").read_text())["system_prompt"].rstrip() \
        + " " + PLAIN_FORMULAS
    expected = build_messages(system_prompt_for(dom_prompt, False), hits, question, max_images=p.valves.MAX_IMAGES_TO_LLM)
    assert sent[0]["messages"] == expected
    if with_images:   # 8 figures among the sources → the default cap of 6 figures is reached exactly
        assert sum(p["type"] == "image_url" for p in sent[0]["messages"][-1]["content"]) == p.valves.MAX_IMAGES_TO_LLM
        joined = "".join(x["text"] for x in sent[0]["messages"][-1]["content"] if x["type"] == "text")
        assert f"Question: {question}\n\nImage for Source [" in joined     # not glued to the question
    assert sent[0]["temperature"] == p.valves.TEMPERATURE and sent[0]["max_tokens"] == p.valves.MAX_TOKENS


def test_follow_up_answer_reads_earlier_turns_without_their_citations(monkeypatch):
    """The answer model reads earlier answers without their citations, so a follow-up does not copy "(document, p. 4,
    §2.6)" as its own; the router still reads them with the document names."""
    import openai
    import memento.retrieval.search as search
    import memento.common.model_servers as vc
    from memento.query import presentation, texts
    mod = _load_pipe()
    payloads = [{"text": "SPS used 1400 °C.", "source": "Q", "page": 3, "type": "text"}]
    points = [types.SimpleNamespace(payload=p, score=0.9, id=i) for i, p in enumerate(payloads)]
    _RecordingAsyncOpenAI.calls = []
    monkeypatch.setattr(openai, "AsyncOpenAI", _RecordingAsyncOpenAI)
    monkeypatch.setattr(library_state, "collection_status", lambda c, coll: "ok")
    monkeypatch.setattr(library_state, "load_documents", lambda *a: {"P": {}, "Q": {}})
    monkeypatch.setattr(library_state, "in_progress_sources", lambda *a: set())
    monkeypatch.setattr(documents, "_resolve_paper", lambda q, s, d=None: None)
    monkeypatch.setattr(library_state, "qdrant_query", lambda *a, **k: types.SimpleNamespace(points=points))
    monkeypatch.setattr(vc, "vl_embed", lambda *a, **k: [0.0] * 8)
    monkeypatch.setattr(search, "rerank", lambda q, hits, *a, **k: hits)
    p = mod.Pipe()
    p.valves = p.Valves(DOMAIN_NAME="generic", COLLECTION="x", RERANK_ENABLED=True, GRAPH_RETRIEVAL=False,
                        VALIDATION_ENABLED=False, ENABLE_QUERY_REWRITE=True, THINKING_AUTO=False)
    earlier = "Alumina reached 99 % density [1]." + presentation.sources_marker(
        [{"source": "P", "page": 0, "section_raw": "4.1 Density"}], {0})
    messages = [{"role": "user", "content": "What density did alumina reach?"},
                {"role": "assistant", "content": earlier},
                {"role": "user", "content": "Which sintering temperature was used?"}]

    async def run():
        return [c async for c in p.pipe({"messages": messages}, {"id": "t"})]
    asyncio.run(run())

    answer = [c for c in _RecordingAsyncOpenAI.calls if c.get("stream")]
    assert len(answer) == 1, "the pipe did not reach the answer step"
    sent = answer[0]["messages"]
    assert sent[1] == {"role": "user", "content": "What density did alumina reach?"}
    assert sent[2] == {"role": "assistant", "content": "Alumina reached 99 % density."}
    assert sent[0]["content"].endswith(texts.HISTORY_RULE)
    router = [c for c in _RecordingAsyncOpenAI.calls if not c.get("stream")]
    assert any("(P, p. 1, §4.1)" in str(c.get("messages")) for c in router)


# ── verify-before-showing in the live pipe ──
THOROUGH_ON = {"filter_ids": ["memento_thorough"]}


def _run_pipe_with_validation(tmp_path, monkeypatch, run_strategy_fake, metadata=THOROUGH_ON, **valves):
    import openai
    import memento.retrieval.search as search
    import memento.common.model_servers as vc
    import memento.agentic_validation.strategies as strategies
    mod = _load_pipe()
    payloads = [{"text": "Alumina reached 99 % density.", "source": "P", "page": 0, "type": "text"},
                {"text": "SPS used 1400 °C.", "source": "Q", "page": 3, "type": "text"}]
    points = [types.SimpleNamespace(payload=p, score=0.9, id=i) for i, p in enumerate(payloads)]
    _RecordingAsyncOpenAI.calls = []
    monkeypatch.setattr(openai, "AsyncOpenAI", _RecordingAsyncOpenAI)
    monkeypatch.setattr(library_state, "collection_status", lambda c, coll: "ok")
    monkeypatch.setattr(library_state, "load_documents", lambda *a: {"P": {}, "Q": {}})
    monkeypatch.setattr(library_state, "in_progress_sources", lambda *a: set())
    monkeypatch.setattr(documents, "_resolve_paper", lambda q, s, d=None: None)
    monkeypatch.setattr(library_state, "qdrant_query", lambda *a, **k: types.SimpleNamespace(points=points))
    monkeypatch.setattr(vc, "vl_embed", lambda *a, **k: [0.0] * 8)
    monkeypatch.setattr(search, "rerank", lambda q, hits, *a, **k: hits)
    monkeypatch.setattr(strategies, "run_strategy", run_strategy_fake)
    p = mod.Pipe()
    p.valves = p.Valves(DOMAIN_NAME="generic", COLLECTION="x", GRAPH_RETRIEVAL=False, THINKING_AUTO=False,
                        VALIDATION_ENABLED=True, **valves)
    events = []

    async def emitter(e):
        events.append(e)

    async def run():
        return [c async for c in p.pipe({"messages": [{"role": "user", "content": "What density?"}]}, {"id": "t"},
                                        __event_emitter__=emitter, __metadata__=metadata)]
    return "".join(asyncio.run(run())), events, mod


def test_validation_holds_the_answer_and_shows_the_verified_one(tmp_path, monkeypatch):
    from memento.agentic_validation.strategies import P1_ADDENDUM, ArmResult
    seen = {}

    def fake(name, ctx, *, p1_answer=None, max_rounds=2):
        seen.update(name=name, p1=p1_answer, rounds=max_rounds, n_hits=len(ctx.hits))
        extra = {"type": "text", "text": "New source.", "source": "R", "page": 1, "section": None, "section_raw": None,
                 "bbox": None, "page_end": None, "img_path": None}
        return ArmResult("A", "Verified answer [3].", hits=ctx.hits + [extra],
                         trace=[{"round": 0, "failing": {"c1": "x"}}, {"round": 1, "failing": {}}], rounds=1)
    out, events, _ = _run_pipe_with_validation(tmp_path, monkeypatch, fake, VALIDATION_STRATEGY="a")
    assert "Answer [1]." not in out and "Verified answer [3]." in out           # raw answer never shown
    assert seen == {"name": "A", "p1": "Answer [1].", "rounds": 2, "n_hits": 2}
    sent = [c for c in _RecordingAsyncOpenAI.calls if c.get("stream")][0]
    assert sent["messages"][0]["content"].endswith(P1_ADDENDUM)                # strict-citation prompt when verifying
    cards = [e for e in events if e["type"] == "citation"]
    assert len(cards) == 3 and cards[2]["data"]["metadata"][0]["name"].startswith("[3] R")
    assert not any("revised" in (e["data"].get("description") or "") or "Checked" in (e["data"].get("description") or "")
                   for e in events if e["type"] == "status")                  # outcome is logged, never shown


def test_validation_outcome_is_logged_once_per_statement_not_shown(tmp_path, monkeypatch, caplog):
    """The check's outcome (statements checked and corrected) goes to the log only, never to a status line."""
    import logging
    caplog.set_level(logging.INFO, logger="memento.verify")
    from memento.agentic_validation.strategies import ArmResult
    from memento.agentic_validation.claims import Claim

    def fake(name, ctx, *, p1_answer=None, max_rounds=2):
        return ArmResult("A", "Verified [1].", claims=[Claim(f"c{i}", "x", (1,), (0, 1)) for i in range(8)], hits=ctx.hits,
                         trace=[{"round": 0, "failing": {f"c{i}": "x" for i in range(7)}},
                                {"round": 1, "failing": {"c1": "x", "c2": "x"}}, {"round": 2, "failing": {"c2": "x"}}],
                         rounds=2)
    _, events, _ = _run_pipe_with_validation(tmp_path, monkeypatch, fake, VALIDATION_STRATEGY="A")
    msgs = [e["data"].get("description") or "" for e in events if e["type"] == "status"]
    assert not any("revised" in m or "Checked" in m for m in msgs)
    log = [r.getMessage() for r in caplog.records if r.name == "memento.verify"]
    assert any("claims=8 revised=7 unconfirmed=1 rounds=2" in m and "thorough=True" in m for m in log)   # check = button on


def test_validation_failure_shows_the_unverified_answer_with_a_note(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("verifier down")
    out, _, _ = _run_pipe_with_validation(tmp_path, monkeypatch, boom)
    assert "Answer [1]." in out and "Verification unavailable" in out


def test_validation_off_by_default_and_old_block_removed():
    mod = _load_pipe()
    v = mod.Pipe().valves
    assert v.VALIDATION_ENABLED is False and v.VALIDATION_STRATEGY == "A" and v.VERIFIER_URL == ""
    src = (ROOT / "openwebui/functions/query_pipe.py").read_text()
    assert "validate_answer" not in src and "Validation correction" not in src


@pytest.mark.parametrize("filter_ids,checked", [
    (None, False),                                               # no metadata (e.g. API call) → no check
    ([], False),                                                 # button off (the default)
    (["some_other_filter"], False),
    (["memento_thorough"], True),                                # "Thorough check" on → the full check
])
def test_thorough_button_switches_the_check_on_and_off(tmp_path, monkeypatch, filter_ids, checked):
    """The check takes time on long answers, so it runs only with the button; on = the RQ3 loop with the thinking round
    (it fixes more hard answers than without thinking)."""
    from memento.agentic_validation.strategies import ArmResult
    seen = {}

    def fake(name, ctx, *, p1_answer=None, max_rounds=2):
        seen["revise_thinking"] = ctx.revise_thinking
        return ArmResult("A", "Verified answer [1].", hits=ctx.hits)
    meta = None if filter_ids is None else {"filter_ids": filter_ids}
    out, events, mod = _run_pipe_with_validation(tmp_path, monkeypatch, fake, metadata=meta, VALIDATION_STRATEGY="A")
    assert request.THOROUGH_FILTER_ID == "memento_thorough"
    shown = any("Thorough check: refining" in (e["data"].get("description") or "") for e in events if e["type"] == "status")
    if checked:
        assert seen["revise_thinking"] == "escalate" and shown and "Verified answer [1]." in out
    else:
        assert not seen and not shown and "Answer [1]." in out and "Verified" not in out   # the first answer, unchecked


def test_thorough_filter_is_a_toggle_with_icon_and_no_side_effects():
    replace_imports = pytest.importorskip("open_webui.utils.plugin").replace_imports
    src = replace_imports((ROOT / "openwebui/functions/thorough_filter.py").read_text())
    mod = types.ModuleType("thorough_filter")
    exec(compile(src, "thorough_filter.py", "exec"), mod.__dict__)
    f = mod.Filter()
    assert f.toggle is True and f.icon.startswith("data:image/svg+xml")
    body = {"messages": [{"role": "user", "content": "q"}]}
    assert asyncio.run(f.inlet(dict(body))) == body
    import importlib.util, sys as _sys
    spec = importlib.util.spec_from_file_location("setup_openwebui", ROOT / "scripts/setup_openwebui.py")
    setup = importlib.util.module_from_spec(spec); spec.loader.exec_module(setup)
    assert any(fn["id"] == "memento_thorough" and fn["file"] == "thorough_filter.py" for fn in setup.FUNCTIONS)
    # every Memento Query model variant has the button (the check itself always runs locally)
    assert all(m["meta"]["filterIds"] == ["memento_thorough"] for m in setup.MODELS if m["id"] in setup.QUERY_MODEL_IDS)
    assert len(setup.QUERY_MODEL_IDS) == 3
    # and Memento Document Chat (its check runs locally too)
    assert next(m for m in setup.MODELS if m["id"] == setup.DOC_CHAT_ID)["meta"]["filterIds"] == ["memento_thorough"]
    assert all("filterIds" not in m["meta"] for m in setup.MODELS
               if m["id"] not in setup.QUERY_MODEL_IDS + [setup.DOC_CHAT_ID])
