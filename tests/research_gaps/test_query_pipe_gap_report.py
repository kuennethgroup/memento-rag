# The open-research-questions report in the Query pipe: reached by an exact phrase (the start-screen button, or typed),
# built from the ledger only, the prose checked before it is shown (no new searches); unchecked prose is never shown.
import asyncio
import json
import re
import sys
import types
from pathlib import Path

import pytest
from memento.config.paths import PROJECT_ROOT
from _owui_loader import load_function
from memento.query import generation, library_state, texts
from memento.query.scope import documents
from memento.query.flows import gap_report
from memento.document_chat import cleanup

ROOT = PROJECT_ROOT
sys.path.insert(0, str(ROOT))
qp = load_function(ROOT / "openwebui/functions/query_pipe.py", "memento_query")
from memento.research_gaps import ledger as gl  # noqa: E402


@pytest.mark.parametrize("text, want", [
    ("Open research gaps across the library", (None,)),
    ("open research gaps", (None,)),
    ('Open research gaps on "ferroelectric hafnia"', ("ferroelectric hafnia",)),
    ("Open research gaps on dopants in HfO2", ("dopants in HfO2",)),
    ("Open research gaps on  ", (None,)),
    ("What research gaps does the document \"X\" name?", None),
    ("Are there open research gaps on hafnia in the documents?", None),
])
def test_the_report_phrase(text, want):
    assert gap_report.gap_report_request(text) == want


# ── the route, with fake services ─────────────────────────────────────────────

class FakeQdrant:
    def __init__(self, *a, **k):
        pass

    def retrieve(self, collection_name, ids, with_payload=True):
        return [types.SimpleNamespace(id=i, payload={"text": f"passage {i}", "source": "DocA", "page": 1,
                                                     "type": "text"}) for i in ids]

    def query_points(self, collection_name, query=None, limit=10, **kw):
        assert collection_name.endswith("-gaps")
        return types.SimpleNamespace(points=[types.SimpleNamespace(id=g, score=0.9) for g in GAP_IDS])

    def collection_exists(self, name):
        return True


GAP_IDS: list = []
PROSE = '{"overview": "Hafnia work [1].", "why": {"1": "It limits devices [1]."}, ' \
        '"suggestions": [{"text": "Try X.", "builds_on": [1]}]}'


@pytest.fixture
def ledger(monkeypatch):
    engine = qp._db_engine()
    gl.ensure_tables(engine)
    gid = gl.add_gap(engine, "memento-x-rag", "How does X affect Y?", "limitation")
    gl.add_mention(engine, gid, "memento-x-rag", source="DocA", point_id="11", page=1, signal="s", claim="X open",
                   passage="DocA says X is open.", judge="supported")
    gl.set_library_status(engine, gid, status="open", docs_checked=3, evidence=[])
    GAP_IDS[:] = [gid]
    import qdrant_client
    import memento.retrieval.search as search
    import memento.common.model_servers as vc
    monkeypatch.setattr(qdrant_client, "QdrantClient", FakeQdrant)
    monkeypatch.setattr(vc, "vl_embed", lambda *a, **k: [0.0] * 8)
    monkeypatch.setattr(search, "rerank", lambda q, hits, *a, **k: [{**h, "rerank_score": 0.5} for h in hits])
    # a library of 4: "checked against the other 3 documents" (the count is capped at the library)
    monkeypatch.setattr(library_state, "load_documents", lambda *a, **k: {"DocA": {"title": "Document A"}, "DocB": {"title": "B"},
                                                                "DocC": {"title": "C"}, "DocD": {"title": "D"}})
    return engine


def _run(pipe, text, monkeypatch, *, prose=PROSE, check=lambda ctx, t: types.SimpleNamespace(answer=t, claims=[1],
                                                                                              removed=[]),
         document=None, sent=None):
    from memento.agentic_validation import strategies

    async def fake_open(variant, *, messages, think, max_tokens=None, local_thinking_budget=None, **kw):
        if sent is not None:
            sent.append(messages)

        async def gen():
            yield types.SimpleNamespace(choices=[types.SimpleNamespace(
                delta=types.SimpleNamespace(content=prose, reasoning_content=None, reasoning=None))])
        return gen(), variant, ""
    monkeypatch.setattr(generation, "open_answer", lambda valves, *a, **k: fake_open(*a, **k))
    monkeypatch.setattr(strategies, "run_arm_c", check)
    events = []

    async def emit(e):
        events.append(e)

    async def go():
        out = []
        kw = {"document": document} if document is not None else {}
        async for c in gap_report.gap_report(pipe.valves, None if text is None else text, db_engine=qp._db_engine, variant=generation.copy_variants()[0],
                                        emit=emit, status=lambda *a, **k: asyncio.sleep(0), **kw):
            out.append(c)
        return "".join(out)
    return asyncio.run(go()), events


def _pipe():
    p = qp.Pipe()
    p.valves.COLLECTION = "memento-x-rag"
    return p


def test_a_whole_library_report_is_built_from_the_ledger(ledger, monkeypatch):
    md, events = _run(_pipe(), None, monkeypatch)
    assert md.startswith("## Open research questions in the library")
    assert "### 1. How does X affect Y?" in md and "Document A [1]" in md and "It limits devices [1]." in md
    assert "**In the library:** nothing answers it (checked against the other 3 documents" in md
    assert "| Try X. | question 1 |" in md and "were checked against the sources" in md
    cites = [e for e in events if e.get("type") == "citation"]
    assert len(cites) == 1 and cites[0]["data"]["metadata"][0]["name"].startswith("[1] ")


def test_the_check_runs_without_new_searches_and_unchecked_prose_is_never_shown(ledger, monkeypatch):
    seen = {}

    def check(ctx, text):
        seen["retrieve_fn"] = ctx.retrieve_fn
        raise RuntimeError("judge down")
    md, _ = _run(_pipe(), None, monkeypatch, check=check)
    assert seen["retrieve_fn"] is None
    assert "It limits devices" not in md and "Hafnia work" not in md                 # prose left out
    assert "### 1. How does X affect Y?" in md and "could not be checked" in md      # the facts stay


def test_unreadable_prose_still_gives_the_facts(ledger, monkeypatch):
    md, _ = _run(_pipe(), None, monkeypatch, prose="sorry, no JSON")
    assert "### 1. How does X affect Y?" in md and "Why it matters" not in md


def test_an_empty_ledger_says_so(monkeypatch):
    engine = qp._db_engine()
    gl.ensure_tables(engine)
    md, _ = _run(_pipe(), None, monkeypatch)
    assert md == texts.GAP_REPORT_EMPTY_MSG


def test_a_topic_without_enough_questions_says_so(ledger, monkeypatch):
    import memento.retrieval.search as search
    monkeypatch.setattr(search, "rerank", lambda q, hits, *a, **k: [{**h, "rerank_score": 0.01} for h in hits])
    md, _ = _run(_pipe(), "quantum gravity", monkeypatch)
    assert "quantum gravity" in md and "no open questions" in md.lower()


# ── a draft proposal from the report's questions ──

PROPOSAL = ('{"title": "X and Y", "motivation": "X matters [1].", "background": "Known so far [1].", '
            '"objectives": ["Measure X"], "approach": "Simulate (Q1).", "outcomes": "A model.", "risks": "None."}')


def _run_proposal(pipe, monkeypatch, request, report, *, prose=PROPOSAL, tmp_path=None):
    from memento.agentic_validation import strategies

    async def fake_open(variant, *, messages, think, max_tokens=None, local_thinking_budget=None, **kw):
        async def gen():
            yield types.SimpleNamespace(choices=[types.SimpleNamespace(
                delta=types.SimpleNamespace(content=prose, reasoning_content=None, reasoning=None))])
        return gen(), variant, ""
    monkeypatch.setattr(generation, "open_answer", lambda valves, *a, **k: fake_open(*a, **k))
    monkeypatch.setattr(strategies, "run_arm_c", lambda ctx, t: types.SimpleNamespace(answer=t, claims=[1],
                                                                                         removed=[]))
    events = []

    async def emit(e):
        events.append(e)

    async def go():
        return "".join([c async for c in gap_report.gap_proposal(pipe.valves, request, report, db_engine=qp._db_engine, variant=generation.copy_variants()[0],
                                                             emit=emit, status=lambda *a, **k: asyncio.sleep(0))])
    return asyncio.run(go()), events


REPORT_MD = "## Open research questions in the library\n\n### 1. How does X affect Y?\n\n- **Raised by:** A [1]\n"


def test_a_proposal_is_drafted_from_the_reports_question_with_a_pdf(ledger, monkeypatch, tmp_path):
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path))
    pipe = _pipe()
    pipe.valves.FILES_URL = "https://memento.example/files"
    md, events = _run_proposal(pipe, monkeypatch, [1], REPORT_MD)
    assert md.startswith("## Draft research proposal: X and Y")
    assert "**Q1. How does X affect Y?**" in md and "X matters [1]." in md
    assert "<summary>References (" in md and '<details type="memento_proposed">' in md      # collapsed reply
    assert "nothing answers it" in md                                      # the facts, rendered by code
    m = re.search(r"\[Download as PDF\]\((https://memento\.example/files/report/([0-9a-f]{24})\.pdf)\)", md)
    assert m and (tmp_path / "_runtime" / "reports" / f"{m.group(2)}.pdf").read_bytes().startswith(b"%PDF")
    assert [e for e in events if e.get("type") == "citation"]


def test_a_number_the_report_does_not_have_is_answered_plainly(ledger, monkeypatch):
    md, _ = _run_proposal(_pipe(), monkeypatch, [4], REPORT_MD)
    assert "question 4" in md and "1" in md and "Draft research proposal" not in md


def test_the_proposal_route_needs_a_report_before_it():
    assert gap_report.is_gap_report("## Open research questions on hafnia\n\n### 1. Q?") is True
    assert gap_report.is_gap_report("Some other answer") is False


def test_old_proposal_pdfs_are_removed(tmp_path):
    import os
    import time as _t
    d = tmp_path / "_runtime" / "reports"
    d.mkdir(parents=True)
    old, new, other = d / ("a" * 24 + ".pdf"), d / ("b" * 24 + ".pdf"), d / "notes.txt"
    for f in (old, new, other):
        f.write_bytes(b"x")
    os.utime(old, (_t.time() - 31 * 86400,) * 2)
    os.utime(other, (_t.time() - 31 * 86400,) * 2)
    assert cleanup.remove_old_reports(str(tmp_path), days=30) == 1
    assert not old.exists() and new.exists() and other.exists()
    assert cleanup.remove_old_reports(str(tmp_path / "missing")) == 0


# ── a document uploaded but not yet in the ledger: processed at once ──

def _pending_setup(monkeypatch, sources, busy=()):
    from memento.research_gaps import update as gu
    engine = qp._db_engine()
    gl.ensure_tables(engine)
    monkeypatch.setattr(library_state, "load_documents", lambda *a, **k: {s: {"title": s, "content_hash": "h"} for s in sources})
    done, said = [], []
    state = {s: 2 for s in busy}                                   # "in progress" for two polls, then finished

    def in_progress(e, c, s):
        if state.get(s, 0) > 0:
            state[s] -= 1
            return True
        return False

    def process(ctx, src, *, owner, status=None, literature=True):
        assert owner == "report" and ctx.workers == 3
        import time as _t
        for k in ("extract", "against", "library", "literature"):
            status(k)
            _t.sleep(0.05)                                         # a real step takes seconds
        gl.set_document(engine, "memento-x-rag", src, content_hash="h", status="done")
        done.append(src)
        return {"questions": 0}
    monkeypatch.setattr(gu, "in_progress", in_progress)
    monkeypatch.setattr(gu, "process_document", process)
    real_sleep = asyncio.sleep
    monkeypatch.setattr(asyncio, "sleep", lambda *_a: real_sleep(0))

    async def status(msg, done=False):
        said.append(msg)
    return engine, done, said, status


def _run_pending(pipe, engine, status):
    import logging
    asyncio.run(gap_report.process_pending(pipe.valves, engine, "memento-x-rag", {}, status, logging.getLogger("t"), db_engine=qp._db_engine))


def test_a_pending_document_is_processed_before_the_report(monkeypatch):
    engine, done, said, status = _pending_setup(monkeypatch, ["New"])
    _run_pending(_pipe(), engine, status)
    assert done == ["New"]
    assert said[0] == "Adding the new paper first..."
    assert "minute" not in said[0]                                  # no time promises
    assert [m for m in said if m in texts.GAP_REPORT_STEPS.values()] == list(texts.GAP_REPORT_STEPS.values())


def test_at_most_two_pending_documents_are_processed_in_a_request(monkeypatch):
    engine, done, said, status = _pending_setup(monkeypatch, ["A", "B", "C"])
    _run_pending(_pipe(), engine, status)
    assert done == ["A", "B"] and said[0] == "Adding 2 new papers first..."


def test_a_document_the_updater_has_is_waited_for_not_redone(monkeypatch):
    engine, done, said, status = _pending_setup(monkeypatch, ["New"], busy=["New"])
    _run_pending(_pipe(), engine, status)
    assert done == [] and "New paper almost ready..." in said


def test_no_pending_documents_no_delay(monkeypatch):
    engine, done, said, status = _pending_setup(monkeypatch, [])
    _run_pending(_pipe(), engine, status)
    assert done == [] and said == []


def test_a_failure_while_processing_never_fails_the_report(monkeypatch):
    from memento.research_gaps import update as gu
    engine, done, said, status = _pending_setup(monkeypatch, ["New"])
    monkeypatch.setattr(gu, "process_document", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("model down")))
    _run_pending(_pipe(), engine, status)                                  # no exception


@pytest.mark.parametrize("text, want", [
    ("Research gaps across the library", (None,)),
    ('Research gaps on "dopants in hafnia"', ("dopants in hafnia",)),
    ("research gaps", (None,)),
])
def test_the_renamed_button_phrase_is_recognised(text, want):
    # the buttons are "Limitations and future work" (one document) and "Research gaps" (the library)
    assert gap_report.gap_report_request(text) == want


# ── "Draft a proposal" from a topic (start-screen button) ──

@pytest.mark.parametrize("text, want", [
    ('Draft a proposal on "dopants in hafnia"', "dopants in hafnia"),
    ('draft a research proposal about "2D materials"', "2D materials"),
    ("Draft a proposal on question 2", None),                         # the report follow-up, not a topic
    ("Draft a proposal on 2D materials", None),                       # unquoted: never mistaken for numbers below
])
def test_the_topic_proposal_phrase(text, want):
    assert gap_report.proposal_topic_request(text) == want


def test_a_topic_proposal_builds_on_the_topics_best_question(ledger, monkeypatch, tmp_path):
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path))
    pipe = _pipe()
    seen = {}

    async def fake_build(gaps, *, variant, emit, status, intro=""):
        seen["gaps"], seen["intro"] = [g["statement"] for g in gaps], intro
        yield "PROPOSAL"
    monkeypatch.setattr(gap_report, "proposal_for", lambda valves, *a_, db_engine, **k_: fake_build(*a_, **k_))
    monkeypatch.setattr(gap_report, "process_pending", lambda *a, **k: asyncio.sleep(0))

    async def go():
        return "".join([c async for c in gap_report.gap_proposal_topic(pipe.valves, "hafnia", db_engine=qp._db_engine, variant=generation.copy_variants()[0],
                                                                    emit=lambda e: asyncio.sleep(0),
                                                                    status=lambda *a, **k: asyncio.sleep(0))])
    out = asyncio.run(go())
    assert out == "PROPOSAL" and seen["gaps"] == ["How does X affect Y?"]
    assert "How does X affect Y?" in seen["intro"] and "Draft a proposal" in seen["intro"]


def test_a_topic_without_questions_says_so(ledger, monkeypatch):
    import memento.retrieval.search as search
    monkeypatch.setattr(search, "rerank", lambda q, hits, *a, **k: [{**h, "rerank_score": 0.01} for h in hits])
    pipe = _pipe()
    monkeypatch.setattr(gap_report, "process_pending", lambda *a, **k: asyncio.sleep(0))

    async def go():
        return "".join([c async for c in gap_report.gap_proposal_topic(pipe.valves, "quantum gravity", db_engine=qp._db_engine, variant=generation.copy_variants()[0],
                                                                    emit=lambda e: asyncio.sleep(0),
                                                                    status=lambda *a, **k: asyncio.sleep(0))])
    assert "quantum gravity" in asyncio.run(go())


# ── one document's open questions (start screen "Draft a proposal" → a picked document) ──
# The picker sends 'Research gaps in the document "<name>"' (the name as the picker lists it: the title, or the file
# name when two documents share a title); the report is the usual one, limited to the questions that document raises, so
# "Draft a proposal on question N" and the buttons under each question work on it unchanged.

@pytest.mark.parametrize("text, want", [
    ('Research gaps in the document "Document A"', "Document A"),
    ('  research gaps in the document "DocA"  ', "DocA"),
    ('Research gaps in the document "Doc \'A\' (2)"', "Doc 'A' (2)"),
    ('Research gaps in the document ""', None),
    ('Research gaps in the document "   "', None),
    ('What research gaps does the document "Document A" name?', None),
    ("Research gaps in the library", None),
    ('Research gaps on "hafnia"', None),
    ('Which research gaps in the document "Document A" matter most?', None),
])
def test_the_document_report_phrase(text, want):
    assert gap_report.doc_report_request(text) == want


@pytest.mark.parametrize("text", ['Research gaps in the document "Document A"', 'research gaps in the document "X"'])
def test_the_document_phrase_is_never_taken_for_a_library_or_topic_report(text):
    assert gap_report.gap_report_request(text) is None and gap_report.proposal_topic_request(text) is None


@pytest.fixture
def doc_ledger(ledger):
    """Gap 1 raised by DocA only (the fixture's), gap 2 by DocB only, gap 3 by DocA and DocB; every document of the
    library done in the ledger, so nothing is pending."""
    engine, coll = ledger, "memento-x-rag"
    g2 = gl.add_gap(engine, coll, "Why is Z unstable?", "limitation")
    gl.add_mention(engine, g2, coll, source="DocB", point_id="21", page=2, signal="s", claim="Z open",
                   passage="DocB says Z is open.", judge="supported")
    gl.set_library_status(engine, g2, status="open", docs_checked=3, evidence=[])
    g3 = gl.add_gap(engine, coll, "Can W be scaled?", "future_work")
    for src, pid in (("DocA", "31"), ("DocB", "32")):
        gl.add_mention(engine, g3, coll, source=src, point_id=pid, page=3, signal="s", claim="W open",
                       passage=f"{src} says W is open.", judge="supported")
    gl.set_library_status(engine, g3, status="partly", docs_checked=3, evidence=[])
    for src in ("DocA", "DocB", "DocC", "DocD"):
        gl.set_document(engine, coll, src, content_hash=None, status="done", n_questions=1)
    return engine


def test_a_document_report_has_only_the_questions_that_document_raises(doc_ledger, monkeypatch):
    md, events = _run(_pipe(), None, monkeypatch, document="Document A")
    assert md.startswith('## Open research questions in "Document A"\n')
    assert "How does X affect Y?" in md and "Can W be scaled?" in md and "Why is Z unstable?" not in md
    assert '**2 open research questions** raised by "Document A", 1 of them also by other documents.' in md
    assert "Draft a proposal on this question" in md
    from memento.research_gaps import proposal as gp
    asked = gp.report_questions(md)                    # "Draft a proposal on question N" reads these numbers
    assert gap_report.is_gap_report(md) and sorted(asked.values()) == ["Can W be scaled?", "How does X affect Y?"]


def test_the_document_is_found_by_its_file_name_too(doc_ledger, monkeypatch):
    md, _ = _run(_pipe(), None, monkeypatch, document="DocB")       # two documents with one title: the file name
    assert md.startswith('## Open research questions in "B"\n')
    assert "Why is Z unstable?" in md and "Can W be scaled?" in md and "How does X affect Y?" not in md


def test_the_writing_model_and_the_check_are_told_the_document(doc_ledger, monkeypatch):
    sent, seen = [], {}

    def check(ctx, text):
        seen["query"] = ctx.question
        return types.SimpleNamespace(answer=text, claims=[1], removed=[])
    _run(_pipe(), None, monkeypatch, document="Document A", sent=sent, check=check)
    assert 'Topic: the open questions of the document "Document A"' in sent[0][1]["content"]
    assert "Why is Z unstable?" not in sent[0][1]["content"]
    assert seen["query"] == 'the open questions of the document "Document A"'


def test_a_document_not_in_the_library_says_so(doc_ledger, monkeypatch):
    md, _ = _run(_pipe(), None, monkeypatch, document="No such paper")
    assert md == gap_report.DOC_REPORT_UNKNOWN_MSG.format(name="No such paper")
    assert "Draft a proposal" in md and "start screen" in md


def test_a_document_without_open_questions_points_to_limitations(doc_ledger, monkeypatch):
    md, _ = _run(_pipe(), None, monkeypatch, document="C")                     # DocC: done, raises no question
    assert md == gap_report.DOC_REPORT_NONE_MSG.format(name="C")
    assert "Limitations & future work" in md


def test_a_document_whose_questions_are_not_checked_yet_says_so(doc_ledger, monkeypatch):
    g = gl.add_gap(doc_ledger, "memento-x-rag", "Is V reversible?", "limitation")
    gl.add_mention(doc_ledger, g, "memento-x-rag", source="DocD", point_id="41", page=1, signal="s", claim="V open",
                   passage="DocD says V is open.", judge="supported")                # no library check stored yet
    md, _ = _run(_pipe(), None, monkeypatch, document="D")
    assert md == gap_report.DOC_REPORT_CHECKING_MSG.format(name="D")
    assert "try again later" in md.lower() and "minute" not in md


def test_a_document_still_processing_says_so(doc_ledger, monkeypatch):
    gl.set_document(doc_ledger, "memento-x-rag", "DocD", content_hash="new", status="done")   # changed since read
    monkeypatch.setattr(library_state, "load_documents", lambda *a, **k: {"DocA": {"title": "Document A"}, "DocB": {"title": "B"},
                                                                "DocC": {"title": "C"},
                                                                "DocD": {"title": "D", "content_hash": "newer"}})
    pipe = _pipe()
    monkeypatch.setattr(gap_report, "process_pending", lambda *a, **k: asyncio.sleep(0))   # nothing finished in time
    md, _ = _run(pipe, None, monkeypatch, document="D")
    assert md == gap_report.DOC_REPORT_PENDING_MSG.format(name="D")
    assert "Paper still processing, try again later" in md


def test_a_document_the_ledger_could_not_read_says_so(doc_ledger, monkeypatch):
    gl.set_document(doc_ledger, "memento-x-rag", "DocC", content_hash=None, status="failed", error="model down")
    md, _ = _run(_pipe(), None, monkeypatch, document="C")
    assert md == gap_report.DOC_REPORT_FAILED_MSG.format(name="C")
    assert "Limitations & future work" in md and "model down" not in md


def test_a_document_whose_questions_the_library_answers_lists_them(doc_ledger, monkeypatch):
    for g in gl.list_gaps(doc_ledger, "memento-x-rag"):
        if "DocA" in g["documents"]:
            gl.set_library_status(doc_ledger, g["id"], status="addressed", docs_checked=3, evidence=[])
    md, _ = _run(_pipe(), None, monkeypatch, document="Document A")
    assert md.startswith('## Open research questions in "Document A"')
    assert "may already be answered by other documents" in md and "<summary>Possibly already answered" in md
    assert "Why is Z unstable?" not in md


def test_more_questions_than_the_report_shows_are_listed_after(doc_ledger, monkeypatch):
    for k in range(12):
        g = gl.add_gap(doc_ledger, "memento-x-rag", f"Extra question {k}?", "limitation")
        gl.add_mention(doc_ledger, g, "memento-x-rag", source="DocA", point_id=f"5{k}", page=1, signal="s", claim="c",
                       passage="p", judge="supported")
        gl.set_library_status(doc_ledger, g, status="open", docs_checked=3, evidence=[])
    pipe = _pipe()
    md, _ = _run(pipe, None, monkeypatch, document="Document A")
    assert md.count("<summary>Evidence for question") == pipe.valves.GAP_REPORT_FULL      # one document: no 2 cap
    assert "<summary>Further open questions (" in md


def test_a_picked_pending_document_is_processed_alone(monkeypatch):
    # one document's report needs only that document: never a wait for other people's uploads
    engine, done, said, status = _pending_setup(monkeypatch, ["A", "B", "C"])
    import logging
    asyncio.run(gap_report.process_pending(_pipe().valves, engine, "memento-x-rag", {}, status, logging.getLogger("t"), only="C", db_engine=qp._db_engine))
    assert done == ["C"] and said[0] == "Adding the new paper first..."


def test_a_picked_document_already_in_the_ledger_starts_at_once(monkeypatch):
    engine, done, said, status = _pending_setup(monkeypatch, ["A", "B", "C"])
    gl.set_document(engine, "memento-x-rag", "C", content_hash="h", status="done")
    import logging
    asyncio.run(gap_report.process_pending(_pipe().valves, engine, "memento-x-rag", {}, status, logging.getLogger("t"), only="C", db_engine=qp._db_engine))
    assert done == [] and said == []


def _route(monkeypatch, text, previous=None):
    """Pipe.pipe with the report routes recorded: (topic, document) per report asked."""
    import qdrant_client
    pipe = _pipe()
    seen = []

    async def fake_report(topic, *, variant, emit, status, document=None):
        seen.append((topic, document))
        yield "REPORT"
    monkeypatch.setattr(gap_report, "gap_report", lambda valves, *a_, db_engine, **k_: fake_report(*a_, **k_))
    monkeypatch.setattr(library_state, "collection_status", lambda *a, **k: "ok")
    monkeypatch.setattr(qdrant_client, "QdrantClient", FakeQdrant)
    msgs = ([{"role": "user", "content": "earlier"}, {"role": "assistant", "content": previous}] if previous else [])
    body = {"model": "memento_query.local", "messages": msgs + [{"role": "user", "content": text}]}

    async def go():
        return "".join([c async for c in pipe.pipe(body, {"id": "u1", "role": "user"}, lambda e: asyncio.sleep(0),
                                                   None, {"chat_id": "c1"}, None)])
    return asyncio.run(go()), seen


def test_the_document_phrase_reaches_the_document_report(monkeypatch):
    out, seen = _route(monkeypatch, 'Research gaps in the document "Document A"')
    assert "REPORT" in out and seen == [(None, "Document A")]


@pytest.mark.parametrize("text, want", [("Research gaps across the library", (None, None)),
                                        ('Research gaps on "hafnia"', ("hafnia", None))])
def test_the_library_and_topic_reports_are_unchanged(monkeypatch, text, want):
    out, seen = _route(monkeypatch, text)
    assert "REPORT" in out and seen == [want]


# ── the side panel names where an outside paper was found ──
def test_an_outside_papers_citation_names_the_services_that_found_it():
    paper = {"kind": "paper", "title": "High- Tc superconductor candidates", "source": "x", "year": "2024",
             "services": ["semantic_scholar", "alexandria"], "url": "https://doi.org/10.1/x", "text": "Abstract."}
    ev = gap_report.report_citation(1, paper, "http://files")
    assert ev["data"]["source"]["name"] == "[2] High- Tc superconductor candidates (2024; Semantic Scholar, Offline index)"
    assert ev["data"]["source"]["url"] == "https://doi.org/10.1/x" and ev["data"]["document"] == ["Abstract."]
    plain = gap_report.report_citation(0, {**paper, "services": [], "year": None}, "http://files")
    assert plain["data"]["source"]["name"] == "[1] High- Tc superconductor candidates (n.d.)"


def test_possibly_answered_questions_cite_the_answering_page_outside_the_writers_sources(doc_ledger, monkeypatch):
    g = gl.add_gap(doc_ledger, "memento-x-rag", "Is U already solved?", "limitation")
    gl.add_mention(doc_ledger, g, "memento-x-rag", source="DocC", point_id="61", page=1, signal="s", claim="U open",
                   passage="DocC says U is open.", judge="supported")
    gl.set_library_status(doc_ledger, g, status="addressed", docs_checked=3,
                          evidence=[{"source": "DocA", "point_id": "88", "page": 1, "relation": "addresses",
                                     "reason": "DocA solves U."}])
    sent = []
    md, events = _run(_pipe(), None, monkeypatch, sent=sent)
    cites = [e for e in events if e.get("type") == "citation"]
    last = len(cites)
    i = md.index("<summary>Possibly already answered in the library (1)")
    assert f"| Is U already solved? | Document A [{last}] |" in md[i:md.index("</details>", i)]
    assert cites[-1]["data"]["metadata"][0]["name"].startswith(f"[{last}] ")          # the page opens in the panel
    assert f"[{last}] " not in sent[0][1]["content"] and f"[{last - 1}] " in sent[0][1]["content"]


def test_a_proposals_chat_lines_leave_the_page_to_the_pill_and_its_pdf_keeps_it(ledger, monkeypatch, tmp_path):
    # the page once (the pill shows it too) and reasons not cut off
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path))
    gid = GAP_IDS[0]
    long = ("The passage describes using X as a tool within a workflow, giving a partial example of how it is used, "
            "but it does not detail the retraining or the structural changes that the question asks about at all.")
    gl.set_library_status(ledger, gid, status="partly", docs_checked=3,
                          evidence=[{"source": "DocA", "point_id": "12", "page": 1, "relation": "partly",
                                     "reason": long}])
    from memento.research_gaps import proposal as gp
    pdf = {}
    real = gp.write_pdf
    monkeypatch.setattr(gp, "write_pdf", lambda md, path: pdf.setdefault("md", md) and real(md, path))
    md, _ = _run_proposal(_pipe(), monkeypatch, [1], REPORT_MD)
    line = next(x for x in md.splitlines() if x.strip().startswith("- Partly: Document A"))
    assert ", p. " not in line and line.rstrip().endswith(long)
    assert re.search(r"- Partly: Document A, p\. 2 \[\d+\]: " + re.escape(long), pdf["md"])


# ── the next page of a report: "more research gaps" ──
@pytest.mark.parametrize("text, want", [("more research gaps", True), ("More", True), ("next", True),
                                        ("show more", True), ("Next page", True), ("more questions", True),
                                        ("show me the next 10", True), ("more research gaps please.", True),
                                        ("tell me more about question 2", False), ("more about hafnia", False),
                                        ("What is next for X?", False)])
def test_the_next_page_phrase(text, want):
    assert gap_report.more_request(text) is want


def _route_page(monkeypatch, previous, text="more research gaps"):
    import qdrant_client
    pipe = _pipe()
    seen = []

    async def fake_report(topic, *, variant, emit, status, document=None, page=1):
        seen.append((topic, document, page))
        yield "REPORT"
    monkeypatch.setattr(gap_report, "gap_report", lambda valves, *a_, db_engine, **k_: fake_report(*a_, **k_))
    monkeypatch.setattr(library_state, "collection_status", lambda *a, **k: "ok")
    monkeypatch.setattr(qdrant_client, "QdrantClient", FakeQdrant)
    body = {"model": "memento_query.local", "messages": [{"role": "user", "content": "q"},
                                                         {"role": "assistant", "content": previous},
                                                         {"role": "user", "content": text}]}

    async def go():
        return "".join([c async for c in pipe.pipe(body, {"id": "u1", "role": "user"}, lambda e: asyncio.sleep(0),
                                                   None, {"chat_id": "c1"}, None)])
    return asyncio.run(go()), seen


def _report_page(head, last):
    return head + "\n\n" + "\n\n".join(f"### {k}. Q{k}?" for k in range(last - 9, last + 1)) + "\n"


@pytest.mark.parametrize("head, last, want", [
    ("## Open research questions in the library", 10, (None, None, 2)),
    ("## Open research questions in the library (questions 11–20)", 20, (None, None, 3)),
    ("## Open research questions on hafnia films", 10, ("hafnia films", None, 2)),
    ('## Open research questions in "Document A"', 10, (None, "Document A", 2)),
])
def test_more_after_a_report_gives_its_next_page(monkeypatch, head, last, want):
    out, seen = _route_page(monkeypatch, _report_page(head, last))
    assert "REPORT" in out and seen == [want]


def test_more_without_a_report_before_it_is_an_ordinary_question(monkeypatch):
    out, seen = _route_page(monkeypatch, "Hafnia is a ferroelectric oxide [1].")
    assert seen == []


@pytest.fixture
def big_ledger(ledger):
    """25 open questions over 5 documents (DocA…DocE), all documents done in the ledger."""
    coll = "memento-x-rag"
    for d in "ABCDE":
        for q in range(5):
            g = gl.add_gap(ledger, coll, f"Question {d}{q}?", "limitation")
            gl.add_mention(ledger, g, coll, source=f"Doc{d}", point_id=f"{ord(d)}{q}", page=1, signal="s", claim="c",
                           passage="p", judge="supported")
            gl.set_library_status(ledger, g, status="open", docs_checked=3, evidence=[])
    for d in "ABCDE":
        gl.set_document(ledger, coll, f"Doc{d}", content_hash=None, status="done", n_questions=5)
    return ledger


def test_the_library_report_lists_documents_and_offers_the_next_page(big_ledger, monkeypatch):
    monkeypatch.setattr(library_state, "load_documents", lambda *a, **k: {f"Doc{d}": {"title": f"Document {d}"}
                                                                for d in "ABCDE"})
    md, _ = _run(_pipe(), None, monkeypatch)
    assert "<summary>Open questions by document (5)</summary>" in md and "| Document A | 6 |" in md
    assert "<summary>Further open questions" not in md and "reply **more research gaps**" in md


def test_page_two_of_the_library_is_numbered_on(big_ledger, monkeypatch):
    monkeypatch.setattr(library_state, "load_documents", lambda *a, **k: {f"Doc{d}": {"title": f"Document {d}"}
                                                                for d in "ABCDE"})
    pipe = _pipe()

    async def go(page):
        return "".join([c async for c in gap_report.gap_report(pipe.valves, None, db_engine=qp._db_engine, variant=generation.copy_variants()[0], page=page,
                                                          emit=lambda e: asyncio.sleep(0),
                                                          status=lambda *a, **k: asyncio.sleep(0))])
    from memento.agentic_validation import strategies

    async def fake_open(variant, *, messages, think, max_tokens=None, local_thinking_budget=None, **kw):
        async def gen():
            yield types.SimpleNamespace(choices=[types.SimpleNamespace(
                delta=types.SimpleNamespace(content=PROSE, reasoning_content=None, reasoning=None))])
        return gen(), variant, ""
    monkeypatch.setattr(generation, "open_answer", lambda valves, *a, **k: fake_open(*a, **k))
    monkeypatch.setattr(strategies, "run_arm_c", lambda ctx, t: types.SimpleNamespace(answer=t, claims=[1], removed=[]))
    md = asyncio.run(go(2))
    assert md.startswith("## Open research questions in the library (questions 11–20)")
    assert "### 11. " in md and "Open questions by document" not in md
    assert asyncio.run(go(4)) == texts.GAP_REPORT_NO_MORE_MSG


# ── paper cards: an outside paper's citation opens our card in the side panel ──
ABSTRACT = "We study P in thin films. P raises Y at room temperature. Other results follow."


def _with_paper(engine):
    gl.set_web_status(engine, GAP_IDS[0], status="partly", query="x y", sources=["openalex", "alexandria"],
                      evidence=[{"origin": "openalex", "source": "https://openalex.org/W77", "title": "Paper P",
                                 "year": "2023", "doi": "10.5/p", "url": "https://doi.org/10.5/p", "text": ABSTRACT,
                                 "relation": "partly", "reason": "Shows P partly."}])


def _paper_citation(events):
    return next(e for e in events if e.get("type") == "citation"
                and e["data"]["metadata"][0]["source"].startswith("paper#"))


def test_a_report_cites_an_outside_paper_with_its_card(ledger, monkeypatch, tmp_path):
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path))
    _with_paper(ledger)
    import memento.retrieval.search as search
    monkeypatch.setattr(search, "rerank", lambda q, hits, *a, **k: [
        {**h, "rerank_score": 0.9 if "raises Y" in h.get("text", "") else 0.01} for h in hits])
    pipe = _pipe()
    pipe.valves.FILES_URL = "https://memento.example/files"
    md, events = _run(pipe, None, monkeypatch)
    src = _paper_citation(events)["data"]["source"]
    m = re.fullmatch(r"https://memento\.example/files/paper/([0-9a-f]{24})", src.get("embed_url", ""))
    assert m and src["url"] == "https://doi.org/10.5/p"
    card = json.loads((tmp_path / "_runtime" / "papers" / f"{m.group(1)}.json").read_text())
    assert card["title"] == "Paper P" and card["year"] == "2023" and card["doi"] == "10.5/p"
    assert card["sentences"] == ["We study P in thin films.", "P raises Y at room temperature.", "Other results follow."]
    assert card["services"] == [{"name": "OpenAlex", "url": "https://openalex.org/W77"}]
    assert card["questions"] == [{"label": "Question 1", "statement": "How does X affect Y?", "verdict": "Partly",
                                  "reason": "Shows P partly.", "marks": [1]}]


def test_a_card_without_the_reranker_has_no_marks(ledger, monkeypatch, tmp_path):
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path))
    _with_paper(ledger)
    import memento.retrieval.search as search

    def down(q, hits, *a, **k):
        if any("raises Y" in h.get("text", "") for h in hits):
            raise RuntimeError("reranker down")
        return [{**h, "rerank_score": 0.5} for h in hits]
    monkeypatch.setattr(search, "rerank", down)
    pipe = _pipe()
    pipe.valves.FILES_URL = "https://memento.example/files"
    md, events = _run(pipe, None, monkeypatch)
    tok = _paper_citation(events)["data"]["source"]["embed_url"].rsplit("/", 1)[1]
    card = json.loads((tmp_path / "_runtime" / "papers" / f"{tok}.json").read_text())
    assert card["questions"][0]["marks"] == [] and "### 1. How does X affect Y?" in md


def test_a_card_that_cannot_be_written_leaves_the_citation_as_before(ledger, monkeypatch, tmp_path):
    blocked = tmp_path / "not_a_dir"
    blocked.write_text("x")                                   # OUTPUT_ROOT is a file: no folder can be made
    monkeypatch.setenv("OUTPUT_ROOT", str(blocked))
    _with_paper(ledger)
    pipe = _pipe()
    pipe.valves.FILES_URL = "https://memento.example/files"
    md, events = _run(pipe, None, monkeypatch)
    src = _paper_citation(events)["data"]["source"]
    assert "embed_url" not in src and src["url"] == "https://doi.org/10.5/p" and "### 1." in md


def test_a_proposals_papers_get_cards_labelled_by_their_q_number(ledger, monkeypatch, tmp_path):
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path))
    _with_paper(ledger)
    pipe = _pipe()
    pipe.valves.FILES_URL = "https://memento.example/files"
    md, events = _run_proposal(pipe, monkeypatch, [1], REPORT_MD)
    tok = _paper_citation(events)["data"]["source"]["embed_url"].rsplit("/", 1)[1]
    card = json.loads((tmp_path / "_runtime" / "papers" / f"{tok}.json").read_text())
    assert card["questions"][0]["label"] == "Q1"


def test_old_paper_cards_are_removed(tmp_path):
    import os
    import time as _t
    d = tmp_path / "_runtime" / "papers"
    d.mkdir(parents=True)
    old, new = d / ("a" * 24 + ".json"), d / ("b" * 24 + ".json")
    for f in (old, new):
        f.write_text("{}")
    os.utime(old, (_t.time() - 31 * 86400,) * 2)
    assert cleanup.remove_old_reports(str(tmp_path), days=30) == 1 and not old.exists() and new.exists()


# ── typed phrases: Open WebUI's chat box turns straight quotes into curly ones ──
@pytest.mark.parametrize("text, want", [('Research gaps in the document “Document A”', "Document A"),
                                        ('Research gaps in the document "Document A”', "Document A")])
def test_the_document_phrase_accepts_curly_quotes(text, want):
    assert gap_report.doc_report_request(text) == want


def test_the_picker_phrases_accept_curly_quotes():
    assert documents.PICKED.match('Summarize the document “Doc A”').group(2) == "Doc A"
    assert documents.PICKED.match('What research gaps does the document “Doc A” name?').group(2) == "Doc A"


def test_a_report_whose_check_judged_only_some_statements_says_so(ledger, monkeypatch):
    part = lambda ctx, t: types.SimpleNamespace(answer=t, claims=[1, 2, 3], removed=[], unjudged=1)
    md, _ = _run(_pipe(), None, monkeypatch, check=part)
    assert "It limits devices [1]." in md and "some statements could not be checked this time" in md
    none = lambda ctx, t: types.SimpleNamespace(answer=t, claims=[1, 2], removed=[], unjudged=2)
    md2, _ = _run(_pipe(), None, monkeypatch, check=none)
    assert "It limits devices" not in md2 and "could not be checked" in md2          # nothing judged: left out


def test_prose_with_no_statement_to_check_stays_and_says_it_was_not_checked(ledger, monkeypatch):
    empty = lambda ctx, t: types.SimpleNamespace(answer=t, claims=[], removed=[], unjudged=0)
    md, _ = _run(_pipe(), None, monkeypatch, check=empty)
    assert "It limits devices [1]." in md and "were not checked against the sources" in md
    assert "were checked against the sources" not in md


# ── full texts stay on this server ────────────────────────────

PASSAGE = "SYNTHETIC-PASSAGE: the blend failed above 413 kelvin."
REASON = "SYNTHETIC-REASON: it failed above 413 kelvin."
CLOUD_URL = "http://cloud.test/v1"


def _judged(engine, text):
    gl.set_web_status(engine, GAP_IDS[0], status="partly", query="x y", sources=["openalex"], evidence=[
        {"origin": "openalex", "source": "W1", "title": "A paper", "year": 2020, "doi": "10.9999/x",
         "url": "https://doi.org/10.9999/x", "relation": "partly", "reason": REASON, "text": text}])


@pytest.fixture
def read_in_full(ledger):
    from memento.research_gaps import fulltext as ft
    _judged(ledger, f"A short abstract.\n\n{ft.MARK}\n{PASSAGE}")
    return ledger


def _leaving(monkeypatch, prose):
    """Every request to a model as the network would see it: (its server, what it is sent)."""
    sent = []

    class Stream:
        def __init__(self):
            self.left = [prose]

        def __aiter__(self):
            return self

        async def __anext__(self):
            if not self.left:
                raise StopAsyncIteration
            delta = types.SimpleNamespace(content=self.left.pop(), reasoning_content=None, reasoning=None)
            return types.SimpleNamespace(choices=[types.SimpleNamespace(delta=delta)])

        async def close(self):
            pass

    def client(base, key, cloud):
        async def create(**kw):
            sent.append((base, json.dumps(kw["messages"], ensure_ascii=False)))
            return Stream()
        return types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))
    monkeypatch.setattr(generation, "client", client)
    return sent


def _cloud(pipe, monkeypatch):
    import dataclasses
    from memento.agentic_validation import strategies
    monkeypatch.setattr(strategies, "run_arm_c", lambda ctx, t: types.SimpleNamespace(answer=t, claims=[1], removed=[]))
    monkeypatch.setenv("CLOUD_URL", CLOUD_URL)
    monkeypatch.setenv("CLOUD_KEY", "k")
    local = next(v for v in generation.copy_variants(pipe.valves) if not v.is_cloud)
    return dataclasses.replace(local, id="cloud", provider="cloud", model="m", base_url_env="CLOUD_URL",
                               key_env="CLOUD_KEY")


def _collect(flow):
    async def go():
        return "".join([c async for c in flow])
    return asyncio.run(go())


def _flow_kw(variant, events):
    async def emit(e):
        events.append(e)
    return dict(db_engine=qp._db_engine, variant=variant, emit=emit, status=lambda *a, **k: asyncio.sleep(0))


def test_a_report_from_full_texts_never_goes_to_a_cloud_model(read_in_full, monkeypatch):
    pipe = _pipe()
    sent = _leaving(monkeypatch, PROSE)
    md = _collect(gap_report.gap_report(pipe.valves, None, **_flow_kw(_cloud(pipe, monkeypatch), [])))
    assert sent and {server for server, _ in sent} == {pipe.valves.LLM_BASE_URL}
    assert md.startswith(texts.FULLTEXT_NOTE)        # first, so a report cut short still keeps its chat local


def test_a_proposal_from_full_texts_never_goes_to_a_cloud_model(read_in_full, monkeypatch, tmp_path):
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path))
    pipe = _pipe()
    sent = _leaving(monkeypatch, PROPOSAL)
    md = _collect(gap_report.gap_proposal(pipe.valves, [1], REPORT_MD, **_flow_kw(_cloud(pipe, monkeypatch), [])))
    assert sent and {server for server, _ in sent} == {pipe.valves.LLM_BASE_URL}
    assert md.startswith(texts.FULLTEXT_NOTE)


def test_a_report_from_abstracts_goes_to_the_model_picked(ledger, monkeypatch):
    _judged(ledger, "A short abstract.")
    pipe = _pipe()
    sent = _leaving(monkeypatch, PROSE)
    md = _collect(gap_report.gap_report(pipe.valves, None, **_flow_kw(_cloud(pipe, monkeypatch), [])))
    assert sent and {server for server, _ in sent} == {CLOUD_URL} and texts.FULLTEXT_NOTE.strip() not in md


def test_a_report_from_full_texts_is_marked_on_the_local_model_too(read_in_full, monkeypatch):
    # the reader can still switch to a cloud model later in this chat
    pipe = _pipe()
    _leaving(monkeypatch, PROSE)
    _cloud(pipe, monkeypatch)
    local = next(v for v in generation.copy_variants(pipe.valves) if not v.is_cloud)
    assert _collect(gap_report.gap_report(pipe.valves, None, **_flow_kw(local, []))).startswith(texts.FULLTEXT_NOTE)


def test_passages_stay_out_of_citations_and_paper_cards(read_in_full, monkeypatch, tmp_path):
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path))
    pipe = _pipe()
    pipe.valves.FILES_URL = "https://memento.example/files"
    _leaving(monkeypatch, PROSE)
    events = []
    _collect(gap_report.gap_report(pipe.valves, None, **_flow_kw(_cloud(pipe, monkeypatch), events)))
    assert [e for e in events if e.get("type") == "citation"] and PASSAGE not in json.dumps(events, ensure_ascii=False)
    cards = list((tmp_path / "_runtime" / "papers").glob("*.json"))
    assert cards and PASSAGE not in cards[0].read_text() and json.loads(cards[0].read_text())["full_text"] is True


@pytest.mark.parametrize("lang", ["en", "de"])
def test_a_chat_holding_such_a_report_stays_on_the_local_model(monkeypatch, lang):
    from memento.query import request, service
    pipe = _pipe()
    cloud = _cloud(pipe, monkeypatch)
    token = request.LANG.set(lang)
    try:
        note = texts.tr(texts.FULLTEXT_NOTE)
    finally:
        request.LANG.reset(token)
    turn = types.SimpleNamespace(valves=pipe.valves, doc_chat=None, body={"model": "any", "messages": [
        {"role": "user", "content": "Open research gaps"}, {"role": "assistant", "content": f"{note}The report."},
        {"role": "user", "content": "Translate it"}]})
    import memento.query.models as models
    monkeypatch.setattr(models, "resolve", lambda *a: cloud)
    service._choose_model(turn)
    assert not turn.variant.is_cloud
    turn.body["messages"] = [{"role": "user", "content": "Translate it"}]
    service._choose_model(turn)
    assert turn.variant.is_cloud


def test_the_note_is_not_sent_to_the_model():
    assert texts.MODEL_NOTE_RE.sub("", f"{texts.FULLTEXT_NOTE}The report.").strip() == "The report."


def test_a_report_under_the_note_is_still_recognised_for_its_next_step():
    from memento.query.flows import rework
    chat = [{"role": "assistant", "content": f"{texts.FULLTEXT_NOTE}{REPORT_MD}"}, {"role": "user", "content": "a draft"}]
    assert gap_report.is_gap_report(rework.previous_answer(chat))
