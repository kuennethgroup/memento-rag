# One parser for research-gap answers: parse_gap_answer decides once: known sections (with a flag when anything was
# dropped), a no-gap statement (every sentence of it), or rejected; display and storing use the final parse; a rejected
# answer gets one retry without thinking, then a plain message; a formatted stored answer is checked before replay.
import asyncio

import pytest
import json
import types

from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool

from memento.research_gaps import passages as gr
from memento.retrieval import pipe_cli
from memento.query import cache, checking, library_state
from memento.query.scope import documents
from memento.query.flows import gap_answer

P1 = "1. Only one material was tested [1]."
P4 = "4. Higher temperatures are left for future work [2]."
GOOD = f"### Limitations\n{P1}\n\n### Future work\n{P4}\n"


# ── the parser ───────────────────────────────────────────────────────────────────────────────────────────────────────

def test_known_sections_are_accepted():
    p = gr.parse_gap_answer(GOOD)
    assert p.kind == "sections" and not p.rejected and P1 in p.text and P4 in p.text


def test_unknown_heading_only_is_rejected():
    p = gr.parse_gap_answer("### Let's think about classification\n1. Could this source be a limitation [1]?\n")
    assert p.kind == "rejected" and p.text == ""


def test_mixed_known_and_unknown_keeps_the_known_and_flags_it():
    p = gr.parse_gap_answer(f"{GOOD}\n### Draft notes\n5. maybe [3]\n")
    assert p.kind == "sections" and p.rejected and "Draft notes" not in p.text and P4 in p.text


def test_deliberation_after_a_point_is_dropped_and_flagged():
    p = gr.parse_gap_answer(f"### Limitations\n{P1}\nLet's think about whether [2] is suitable.\n")
    assert p.kind == "sections" and p.rejected and P1 in p.text and "Let's think" not in p.text
    p = gr.parse_gap_answer(f"### Limitations\n{P1}\n2. Could [9] be a limitation?\n")
    assert p.rejected and "Could [9]" not in p.text


def test_a_wrapped_point_is_kept():
    p = gr.parse_gap_answer("### Limitations\n1. The model was trained only on\nhomopolymers and short chains [1].\n")
    assert p.kind == "sections" and not p.rejected and "homopolymers and short chains" in p.text


def test_scientific_sentences_that_start_alike_are_kept():
    for point in ("1. Waiting times above 10 h were not studied [3].", "1. Could not be measured above 600 °C [2].",
                  "1. Hmm-shaped peaks were not explained [4]."):
        p = gr.parse_gap_answer(f"### Limitations\n{point}\n")
        assert not p.rejected and point in p.text, point


def test_no_gap_statements():
    for t in ("No explicit limitations or future work are stated in the passages.",
              "The paper does not name any research gaps.",
              "In den Passagen werden keine Einschränkungen oder zukünftigen Arbeiten genannt.",
              "Die Arbeit nennt keine Forschungslücken."):
        p = gr.parse_gap_answer(t)
        assert p.kind == "no_gaps" and not p.rejected and p.text == t, t


def test_short_heading_free_text_that_is_not_a_no_gap_statement_is_rejected():
    for t in ("Let's think about classification.", "Alumina reaches 18 GPa [1].",
              "No explicit limitations are stated. Let's think about classification.",
              "Keine Einschränkungen genannt. Wait, maybe [2] is one."):
        assert gr.parse_gap_answer(t).kind == "rejected", t


def test_calm_layout_never_returns_rejected_text():
    assert gr.calm_gap_answer("### Let's think about classification\n1. Could this source be a limitation [1]?\n") == ""


def test_gap_answer_clean_follows_the_parser():
    assert gr.gap_answer_clean(GOOD)
    assert not gr.gap_answer_clean(f"### Limitations\n{P1}\nLet's think about whether [2] is suitable.\n")
    assert not gr.gap_answer_clean("### Let's think\n1. x [1]\n")


# ── stored answers in the calm layout ────────────────────────────────────────────────────────────────────────────────

def test_a_formatted_stored_answer_is_checked_before_replay():
    shown = gr.calm_gap_answer(GOOD)
    assert gr.stored_gap_ok(shown)
    leaked = shown.replace(P1, f"{P1}\nLet's think about whether [2] is suitable.")
    assert not gr.stored_gap_ok(leaked)
    unknown = shown.replace("<summary>Limitations (1)</summary>", "<summary>Draft notes (1)</summary>")
    assert not gr.stored_gap_ok(unknown)
    assert gr.stored_gap_ok("The paper does not name any research gaps.")
    focused = gr.calm_gap_answer(GOOD, "Future work")
    assert gr.stored_gap_ok(focused)


# ── the pipe ─────────────────────────────────────────────────────────────────────────────────────────────────────────

class _LLM:
    calls: list = []
    route: dict = {}
    replies: list = []

    def __init__(self, *a, **k):
        self.chat = self
        self.completions = self

    async def create(self, **kw):
        _LLM.calls.append(kw)
        if kw.get("stream"):
            content = _LLM.replies.pop(0) if _LLM.replies else "### Limitations\n1. fallback [1].\n"

            async def gen():
                delta = types.SimpleNamespace(content=content, reasoning_content=None, reasoning=None)
                yield types.SimpleNamespace(choices=[types.SimpleNamespace(delta=delta)])
            return gen()
        msg = types.SimpleNamespace(content=json.dumps(_LLM.route))
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


def _gap(monkeypatch, replies, *, thorough_final=None):
    import openai
    import memento.common.model_servers as vc
    m = pipe_cli.load_pipe_module()
    e = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    with e.begin() as c:
        c.execute(text("CREATE TABLE function (id TEXT PRIMARY KEY, valves TEXT)"))
        c.execute(text("INSERT INTO function VALUES ('memento_query', :v)"), {"v": json.dumps(
            {"COLLECTION": "memento-x-rag", "DOMAIN_NAME": "generic", "GRAPH_RETRIEVAL": False,
             "VALIDATION_ENABLED": thorough_final is not None, "ENABLE_QUERY_REWRITE": True, "THINKING_AUTO": False,
             "RELATED_DOCS": False})})
    p = pipe_cli.make_pipe(m, e)
    recs = []

    class Rec(cache.Recorder):
        def __init__(self):
            super().__init__()
            recs.append(self)

    async def thorough_check(self, **kw):
        from memento.agentic_validation.strategies import ArmResult
        return thorough_final, ArmResult("A", thorough_final), None, False

    passages = [{"text": "Only one material was tested.", "source": "A", "page": 1, "type": "text", "section": "Results",
                 "gap_relevant": True, "gap_type": "limitation", "gap_signal": "only one material"},
                {"text": "Higher temperatures are left for future work.", "source": "A", "page": 2, "type": "text",
                 "section": "Conclusion", "gap_relevant": True, "gap_type": "future_work",
                 "gap_signal": "higher temperatures"}]

    def no_search(*a, **k):
        raise AssertionError("a research-gap answer searches nothing")
    _LLM.calls, _LLM.replies = [], list(replies)
    _LLM.route = {"intent": "gap_query", "query": "research gaps of paper A", "reasoning": "deep", "suggest": False,
                  "element": "none", "scope": "document", "author": ""}
    monkeypatch.setattr(openai, "AsyncOpenAI", _LLM)
    monkeypatch.setattr(cache, "Recorder", Rec)
    monkeypatch.setattr(library_state, "collection_status", lambda c, coll: "ok")
    # a start-screen question (picked, with a fingerprint): its answer may be stored, so rec.clean is the decision
    monkeypatch.setattr(library_state, "load_documents", lambda *a: {"A": {"content_hash": "h1"}, "B": {"content_hash": "h2"}})
    monkeypatch.setattr(documents, "picked_document", lambda q, s, collection=None: ("A", "gap_query"))
    monkeypatch.setattr(library_state, "in_progress_sources", lambda *a: set())
    monkeypatch.setattr(documents, "resolve_reference", lambda q, s, d=None: "A")
    monkeypatch.setattr(documents, "named_documents", lambda q, s, d=None: [])
    monkeypatch.setattr(gap_answer, "gap_passages", lambda client, coll, src, budget: (passages, False))
    monkeypatch.setattr(library_state, "qdrant_query", no_search)
    monkeypatch.setattr(vc, "vl_embed", lambda *a, **k: [0.0] * 8)
    if thorough_final is not None:
        monkeypatch.setattr(checking, "thorough_check", thorough_check)
    out = asyncio.run(pipe_cli.ask(p, "What are the research gaps of paper A?", thorough=thorough_final is not None))
    streams = [c for c in _LLM.calls if c.get("stream")]
    assert recs[-1].key, "the question must be one whose answer may be stored"
    return out["answer"], recs[-1], streams


def test_pipe_text_before_the_heading_is_shown_without_it_and_not_stored(monkeypatch):
    answer, rec, _ = _gap(monkeypatch, ["Let me plan the headings first.\n" + GOOD])
    assert "Only one material was tested" in answer and "Let me plan" not in answer
    assert not rec.clean


def test_pipe_rejected_answer_gets_one_retry_without_thinking(monkeypatch):
    answer, rec, streams = _gap(monkeypatch, ["### Let's think about classification\n1. Could [1] be one?\n", GOOD])
    assert len(streams) == 2
    assert streams[1]["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    assert "Only one material was tested" in answer and "Let's think" not in answer
    assert rec.clean                                               # the retry's clean answer may be stored


def test_pipe_rejected_twice_says_so_and_stores_nothing(monkeypatch):
    answer, rec, streams = _gap(monkeypatch, ["### Let's think\n1. x [1]\n", "Let's think about classification."])
    assert len(streams) == 2
    assert "Sorry, I could not write the research gaps this time. Please try again." in answer
    assert "Let's think" not in answer and not rec.clean


def test_pipe_malformed_revision_falls_back_to_the_first_draft(monkeypatch):
    malformed = "### Let's think again\n1. Could [1] be?\n\n_Checked with Thorough check._"
    answer, rec, _ = _gap(monkeypatch, [GOOD], thorough_final=malformed)
    assert "Only one material was tested" in answer and "Let's think" not in answer
    assert "Verification unavailable for this answer" in answer and "Checked with Thorough check." not in answer
    assert not rec.clean


def test_a_stored_answer_that_fails_the_check_is_written_anew(monkeypatch):
    m = pipe_cli.load_pipe_module()
    leaked = gr.calm_gap_answer(GOOD).replace(P1, f"{P1}\nLet's think about whether [2] is suitable.")
    rows = {"leaked": {"answer": leaked, "citations": [], "checked": False, "effective_checked": False},
            "clean": {"answer": gr.calm_gap_answer(GOOD), "citations": [], "checked": False, "effective_checked": False}}
    pick = ["leaked"]
    monkeypatch.setattr(cache, "cache_lookup", lambda engine, fields, regen, count_hit=True: dict(rows[pick[0]]))
    monkeypatch.setattr(cache, "model_key", lambda variant, valves: "local:m")
    monkeypatch.setattr(m, "_db_engine", lambda: None)

    async def not_regen(meta):
        return False
    monkeypatch.setattr(cache, "is_regeneration", not_regen)
    p = m.Pipe()
    run = lambda: asyncio.run(cache.prepare(p.valves,
        cache.Recorder(), picked=("A", "gap_query"), history=[], docs={"A": {"content_hash": "h1"}}, in_progress=set(),
        source="A", action="gaps", mode="capped", prompt_version="p", variant=None, metadata={}, thorough=False,
        valid=gr.stored_gap_ok, db_engine=m._db_engine))
    assert run() is None                                          # leaked: a miss, so the answer is written anew
    pick[0] = "clean"
    assert run() is not None


@pytest.mark.parametrize("t", ["The provided passages do not name any genuine research gaps.",
                               "No genuine research gaps are named in the provided passages."])
def test_real_no_gap_answers(t):
    # calibration: the production prompt on documents without tagged passages gave exactly these two sentences;
    # documents with one tag gave sections that parse cleanly
    p = gr.parse_gap_answer("\n\n" + t)
    assert p.kind == "no_gaps" and not p.rejected


# ── a no-gap answer is a complete absence statement, and only the pipe's own notes pass as italic lines ──

@pytest.mark.parametrize("t", ["I am not sure which limitations to report.",
                               "I need to decide which limitations to list.",
                               "Maybe there are no gaps; let me check the passages again.",
                               "No limitations? I should look at [3] again.",
                               "Ich bin nicht sicher, welche Einschränkungen ich nennen soll."])
def test_uncertainty_and_planning_are_not_no_gap_statements(t):
    assert gr.parse_gap_answer(t).kind == "rejected", t


@pytest.mark.parametrize("t", ["There are no genuine research gaps in the provided passages.",
                               "The authors do not state any limitations or future work.",
                               "Es werden keine Forschungslücken genannt.",
                               "Die bereitgestellten Passagen nennen keine echten Forschungslücken."])
def test_more_absence_statements_are_recognised(t):
    p = gr.parse_gap_answer(t)
    assert p.kind == "no_gaps" and not p.rejected, t


def test_an_unknown_italic_line_is_flagged_not_trusted():
    p = gr.parse_gap_answer("### Limitations\n1. Only one material was tested [1].\n\n_Let's think about classification._\n")
    assert p.rejected and "Let's think" not in p.text and "Only one material" in p.text


@pytest.mark.parametrize("note", ["_Checked with Thorough check._", "_Geprüft mit Thorough check._",
                                  "_⚠ Verification unavailable for this answer (revision unreadable)._",
                                  "_Answered by Qwen 3.8 27B (Local)._"])
def test_the_pipes_own_notes_still_pass(note):
    raw = f"### Limitations\n1. Only one material was tested [1].\n\n{note}\n"
    p = gr.parse_gap_answer(raw)
    assert not p.rejected and note not in p.text
    assert gr.stored_gap_ok(gr.calm_gap_answer("### Limitations\n1. Only one material was tested [1].\n") + "\n" + note + "\n")


def test_italic_scientific_content_inside_a_point_is_kept():
    raw = "### Limitations\n1. Only measured at room temperature [1].\n_In vivo_ behaviour was not studied.\n"
    p = gr.parse_gap_answer(raw)
    assert not p.rejected and "_In vivo_ behaviour" in p.text


def test_a_stored_answer_with_an_unknown_italic_line_is_written_anew():
    good = gr.calm_gap_answer("### Limitations\n1. Only one material was tested [1].\n")
    assert gr.stored_gap_ok(good)
    assert not gr.stored_gap_ok(good + "\n_Let's think about classification._\n")


# emphasis is removed before judging a numbered point too; a real point that starts in italics stays
@pytest.mark.parametrize("point", ["1. _In situ_ measurements were not performed [1].",
                                   "1. **Only one alloy** was tested [2]."])
def test_an_emphasised_real_point_is_kept(point):
    raw = f"### Limitations\n{point}\n"
    p = gr.parse_gap_answer(raw)
    assert not p.rejected and point in p.text
    assert gr.stored_gap_ok(gr.calm_gap_answer(raw))
