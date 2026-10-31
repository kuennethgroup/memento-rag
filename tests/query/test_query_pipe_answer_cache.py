# The Query pipe serves and stores exact picker answers.
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from memento.config.paths import PROJECT_ROOT
from _owui_loader import load_function
from memento.query import cache, presentation, service, texts
from memento.query.flows import figures

ROOT = PROJECT_ROOT
sys.path.insert(0, str(ROOT))
qp = load_function(ROOT / "openwebui/functions/query_pipe.py", "memento_query")
from memento.library import answers as ac  # noqa: E402

FIELDS = dict(collection="col", source="doc", content_hash="h", action="summary", language="en",
              model="local:qwen3.8-27b", mode="whole", prompt_version="p")


def test_strip_think():
    assert cache.strip_think("<think>plan</think>The answer. [1]") == "The answer. [1]"
    assert cache.strip_think("A<think>x</think>B<think>y</think>C") == "ABC"
    assert cache.strip_think("<think>never closed") == ""


def test_regeneration_detected():
    msgs = {"u1": {"id": "u1", "role": "user", "childrenIds": ["a1"]},
            "a1": {"id": "a1", "role": "assistant", "parentId": "u1", "childrenIds": []}}
    assert cache._regenerated(msgs, "a1") is False                                   # the first answer
    msgs["u1"]["childrenIds"] = ["a1", "a2"]
    msgs["a2"] = {"id": "a2", "role": "assistant", "parentId": "u1", "childrenIds": []}
    assert cache._regenerated(msgs, "a2") is True                                    # a second answer to the question
    assert cache._regenerated(msgs, "missing") is False and cache._regenerated({}, None) is False


def test_eligibility_rules():
    ok = dict(picked=("doc", "summary"), history=[], content_hash="h", in_progress=set())
    assert cache.cache_eligible(**ok) == (True, "")
    assert cache.cache_eligible(**{**ok, "picked": None}) == (False, "not a picker question")
    assert cache.cache_eligible(**{**ok, "history": [{"role": "user", "content": "x"}]}) == (False, "history")
    assert cache.cache_eligible(**{**ok, "content_hash": None}) == (False, "no fingerprint")
    assert cache.cache_eligible(**{**ok, "in_progress": {"other"}}) == (False, "library changing")


def test_model_in_key():
    valves = SimpleNamespace(LLM_MODEL="qwen3.8-27b")
    local = SimpleNamespace(id="local", is_cloud=False, model="ignored")
    cloud = SimpleNamespace(id="mistral", is_cloud=True, model="mistral-medium-3.5")
    assert cache.model_key(local, valves) == "local:qwen3.8-27b"
    assert cache.model_key(cloud, valves) == "mistral:mistral-medium-3.5"
    k1 = ac.cache_key(**{**FIELDS, "model": cache.model_key(local, valves)})
    valves.LLM_MODEL = "another-model"                               # the model behind "local" swapped
    assert ac.cache_key(**{**FIELDS, "model": cache.model_key(local, valves)}) != k1


def test_prompt_versions_follow_mode_and_settings():
    v = SimpleNamespace(SUMMARY_MAX_TOKENS=40000, TEMPERATURE=0.15, MAX_TOKENS=16384, GAP_BUDGET_TOKENS=20000)
    assert cache.summary_prompt_version("sys", "whole", v) != cache.summary_prompt_version("sys", "partial", v)
    assert cache.summary_prompt_version("sys", "whole", v) != cache.summary_prompt_version("other", "whole", v)
    assert cache.gap_prompt_version(2000, v) != cache.gap_prompt_version(None, v)


def _run_wrapper(monkeypatch, rec_setup, chunks, citations=()):
    """Drive Pipe.pipe (the recording wrapper) with a stub service.run; returns (yielded text, stored row or None)."""
    pipe = qp.Pipe()
    engine = qp._db_engine()
    ac.ensure_table(engine)

    async def fake(turn):
        rec_setup(turn.rec)
        for ev in citations:
            await turn.emit(ev)
        for c in chunks:
            yield c
    monkeypatch.setattr(service, "run", fake)
    seen = []

    async def emitter(ev):
        seen.append(ev)

    async def go():
        return [c async for c in pipe.pipe({"messages": []}, {}, emitter, None, {})]
    out = asyncio.run(go())
    return "".join(out), ac.get(engine, ac.cache_key(**FIELDS)), seen


def _clean(rec):
    rec.key, rec.fields, rec.clean, rec.generated_by = ac.cache_key(**FIELDS), FIELDS, True, "Qwen 3.8 27B (local)"


def test_recorder_stores_only_clean_answers(monkeypatch):
    cite = {"type": "citation", "data": {"document": ["p"], "metadata": [{"name": "[1] doc (p. 1)"}], "source": {}}}
    text, row, seen = _run_wrapper(monkeypatch, _clean, ["<think>plan</think>", "The summary. [1]", "\n\n_Answered by X._"],
                                   [cite, {"type": "status", "data": {}}])
    assert text.startswith("<think>") and seen[0] == cite                      # the reader's stream is untouched
    assert row["answer"] == "The summary. [1]\n\n_Answered by X._" and row["citations"] == [cite]  # citations only
    assert row["generated_by"] == "Qwen 3.8 27B (local)"


def test_unclean_answers_are_not_stored(monkeypatch):
    def not_clean(rec):
        _clean(rec)
        rec.clean = False
    assert _run_wrapper(monkeypatch, not_clean, ["partial"])[1] is None
    assert _run_wrapper(monkeypatch, _clean, ["partial", "\n\n" + texts.TIMEOUT_NOTE])[1] is None
    assert _run_wrapper(monkeypatch, _clean, ["partial", "\n\n" + texts.STOPPED_NOTE])[1] is None

    def hit(rec):
        _clean(rec)
        rec.hit = True
    assert _run_wrapper(monkeypatch, hit, ["replayed"])[1] is None


def test_hit_replays_citations_then_text():
    row = {"answer": "Stored answer. [1]", "citations": [{"type": "citation", "data": {"n": 1}},
                                                           {"type": "citation", "data": {"n": 2}}]}
    events = []

    async def emit(ev):
        events.append(ev)

    async def go():
        return [c async for c in cache.replay(row, emit)]
    assert asyncio.run(go()) == ["Stored answer. [1]"] and events == row["citations"]


def test_lookup_skips_on_regeneration():
    engine = qp._db_engine()
    ac.ensure_table(engine)
    ac.put(engine, ac.cache_key(**FIELDS), fields=FIELDS, answer="a", citations=[], generated_by="m")
    assert cache.cache_lookup(engine, FIELDS, regenerated=False)["answer"] == "a"
    assert cache.cache_lookup(engine, FIELDS, regenerated=True) is None


def test_answer_in_another_language_is_not_stored(monkeypatch):
    # a German research-gap answer is not stored under an English picker question
    german = ("### Einschränkungen\n1. Die Sauerstoffpositionen, die für die Unterscheidung polarer und nichtpolarer "
              "Gitter entscheidend sind, lassen sich nicht bestimmen [1].\n2. Die Proben wurden nur bei Raumtemperatur "
              "gemessen, und der Einfluss der Dotierung ist noch offen [2].")
    assert _run_wrapper(monkeypatch, _clean, [german])[1] is None
    english = "### Limitations\n1. The samples were only measured at room temperature, so the effect is still open [2]."
    assert _run_wrapper(monkeypatch, _clean, [english])[1]["answer"] == english


def _gate(pieces):
    g = presentation.HeadingGate()
    return "".join(g.feed(p) for p in pieces) + g.finish(), g


def test_heading_gate_drops_reasoning_before_the_first_heading():
    # capped thinking that runs out and keeps planning in the visible answer is not stored
    out, g = _gate(["Need maybe [1] is not a gap. Let's final.\n\n", "### Limi", "tations\n1. Only two [2].\n",
                    "### Future work\n2. Test more [3]."])
    assert out == "### Limitations\n1. Only two [2].\n### Future work\n2. Test more [3]."
    assert g.opened and g.dropped > 0 and g.storable


def test_heading_gate_passes_a_normal_answer_unchanged():
    out, g = _gate(["### Limitations\n1. Only", " two [2].\n"])
    assert out == "### Limitations\n1. Only two [2].\n" and g.opened and g.dropped == 0 and g.storable
    out, g = _gate(["\n### Einschränkungen\n1. Nur zwei [2]."])            # a leading blank line is not reasoning
    assert out == "### Einschränkungen\n1. Nur zwei [2]." and g.dropped == 0


def test_heading_gate_without_heading():
    # "If no genuine gaps are found, say so plainly in one sentence": shown and stored
    out, g = _gate(["The document names no ", "research gaps."])
    assert out == "The document names no research gaps." and not g.opened and g.storable
    # a long answer without any heading is shown (nothing is lost) but never stored
    long = "Maybe [1] is a limitation? " * 40
    out, g = _gate([long])
    assert out == long and not g.opened and not g.storable
    # a "#" inside a line is not a heading
    out, g = _gate(["Sample #3 failed early [1]."])
    assert out == "Sample #3 failed early [1]." and not g.opened


def test_qr_code_figures_are_not_appended(tmp_path):
    # a cited "figure" that is a journal QR code is not shown under the answer (page-1 images are shown only when a
    # figure is asked for)
    from test_qr_code import chart_png, qr_png
    qr, chart = qr_png(tmp_path / "qr.png"), chart_png(tmp_path / "chart.png")
    assert figures.shown_figure({"type": "image", "img_path": str(chart), "page": 3})
    assert not figures.shown_figure({"type": "image", "img_path": str(qr), "page": 3})
    assert not figures.shown_figure({"type": "text", "img_path": str(chart)})
    assert not figures.shown_figure({"type": "chart", "img_path": ""})
