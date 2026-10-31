# The research-gap report (gap ledger step 4): code renders every fact from the ledger; the model writes only the
# overview, "why it matters" and suggestions, whose citations and question numbers are checked mechanically; the prose
# is checked against the sent sources.
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
from memento.config.paths import PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT))
from memento.research_gaps import report as gr  # noqa: E402

AT = datetime(2026, 10, 2, 1, 0, tzinfo=timezone.utc)


def _gap(gid, status, docs=("A",), web="nothing_found", sources="openalex,semantic_scholar", n_docs=24):
    return {"id": gid, "statement": f"Question {gid}?", "library_status": status, "library_docs_checked": n_docs,
            "library_checked_at": AT, "web_status": web, "web_sources": sources, "web_checked_at": AT,
            "documents": list(docs),
            "mentions": [{"source": d, "point_id": f"{gid}{d}", "page": 2, "passage": f"{d} says {gid} is open.",
                          "claim": f"{gid} is open"} for d in docs]}


# ── selection ─────────────────────────────────────────────────────────────────

def test_whole_library_selection_orders_by_status_then_documents():
    gaps = [_gap(1, "partly", docs=("P",)), _gap(2, "open", docs=("Q",)), _gap(3, "contested", docs=("R",)),
            _gap(4, "open", docs=("S", "T")), _gap(5, "addressed"), _gap(6, None)]
    sel = gr.select(gaps, full=3)
    assert [g["id"] for g in sel["full"]] == [3, 4, 2] and [g["id"] for g in sel["more"]] == [1]
    assert [g["id"] for g in sel["addressed"]] == [5]                    # unchecked questions are left out


def test_topic_selection_keeps_only_relevant_questions_by_score():
    gaps = [_gap(1, "open"), _gap(2, "open"), _gap(3, "partly"), _gap(4, "addressed")]
    sel = gr.select(gaps, relevance={3: 0.9, 1: 0.4, 4: 0.8}, full=10)
    assert [g["id"] for g in sel["full"]] == [3, 1] and [g["id"] for g in sel["addressed"]] == [4]


# ── sources ───────────────────────────────────────────────────────────────────

def _passage(pid, src, page=4):
    return {"chunk_id": pid, "text": f"text of {pid}", "source": src, "page": page, "type": "text", "section": None,
            "section_raw": "3 Results", "bbox": [1, 2, 3, 4], "page_end": None, "score": 0, "img_path": None}


def test_sources_are_numbered_once_and_unresolvable_passages_are_dropped():
    g1, g2 = _gap(1, "partly", docs=("A",)), _gap(2, "open", docs=("A", "B"))
    ev = {1: [dict(origin="library", source="C", point_id="p1", page=4, relation="partly", reason="r"),
              dict(origin="library", source="D", point_id="gone", page=1, relation="addresses", reason="r"),
              dict(origin="openalex", source="W1", title="Paper", year=2024, url="https://doi.org/x", text="Abs.",
                   relation="partly", reason="r"),
              dict(origin="semantic_scholar", source="p2", title="No text", year=2020, url="u", text=None,
                   relation="partly", reason="r"),
              dict(origin="openalex", source="W9", title="Near", year=2021, url="u", text="t", relation="near_miss",
                   reason="r")],
          2: [dict(origin="library", source="C", point_id="p1", page=4, relation="partly", reason="r")]}
    passages = {"p1": _passage("p1", "C"), "1A": _passage("1A", "A", 2)}
    sources, refs = gr.gather_sources([g1, g2], ev, passages)
    kinds = [(s["kind"], s["source"]) for s in sources]
    assert kinds == [("stating", "A"), ("library", "C"), ("paper", "Paper"), ("stating", "A"), ("stating", "B")]
    assert refs[1] == {"stating": [1], "library": [(2, "partly")], "papers": [(3, "partly")],
                       "unnumbered": [("No text", 2020, "u", "partly")], "reasons": {2: "r", 3: "r"},
                       "unnumbered_reasons": ["r"]}
    assert refs[2]["library"] == [(2, "partly")]                          # the same passage keeps its number
    assert sources[0]["bbox"] == [1, 2, 3, 4] and sources[3]["text"] == "A says 2 is open."   # from the mention


# ── facts rendered by code ────────────────────────────────────────────────────

def test_status_lines_are_rendered_by_code_in_plain_words():
    g = _gap(7, "open", web="no_results", sources="semantic_scholar")
    refs = {"stating": [1], "library": [], "papers": [], "unnumbered": [], "reasons": {}, "unnumbered_reasons": []}
    sources = [{"kind": "stating", "source": "A", "page": 1}]
    md = gr.render_question(1, g, refs, why="It matters because X [1].", names={"A": "Doc A"}, sources=sources)
    assert md.startswith("### 1. Question 7?")
    assert "**Raised by:** Doc A [1]" in md
    assert "**In the library:** nothing answers it (checked against the other 24 documents" in md
    assert "**In the literature:** no papers found for the search (Semantic Scholar, 2 Oct 2026)" in md
    assert "**Why it matters:** It matters because X [1]." in md


def test_every_piece_of_evidence_says_which_document_or_paper_and_how():
    # each question says which document answers it, fully or partly
    g = _gap(8, "partly", web="possibly_addressed")
    sources = [{"kind": "stating", "source": "A", "page": 1},
               {"kind": "library", "source": "B", "page": 3}, {"kind": "library", "source": "C", "page": 6},
               {"kind": "paper", "source": "Paper one", "title": "Paper one", "year": 2023},
               {"kind": "paper", "source": "Paper two", "title": "Paper two", "year": 2024}]
    refs = {"stating": [1], "library": [(2, "addresses"), (3, "partly")], "papers": [(4, "addresses"), (5, "partly")],
            "unnumbered": [("No abstract paper", 2020, "https://doi.org/x", "partly")],
            "reasons": {2: "reports the fraction", 3: "only for ZrO2", 4: "IR separates the phases", 5: "DFT only"},
            "unnumbered_reasons": ["related method"]}
    md = gr.render_question(2, g, refs, why="", names={"B": "Doc B", "C": "Doc C"}, sources=sources)
    assert "**In the library:** partly answered (checked against the other 24 documents, 2 Oct 2026):" in md
    # in chat the citation pill shows the page ("[2] p. 4"): the text names the document only
    assert "  - Answers it: Doc B [2]: reports the fraction" in md and "p. 4" not in md
    assert "  - Partly: Doc C [3]: only for ZrO2" in md
    assert "**In the literature:** possibly answered (Semantic Scholar and OpenAlex, 2 Oct 2026; judged from " \
           "abstracts):" in md
    assert "  - Possibly answers it: Paper one (2023) [4]: IR separates the phases" in md
    assert "  - Partly: Paper two (2024) [5]: DFT only" in md
    assert "  - Partly: No abstract paper (2020), https://doi.org/x: related method" in md
    assert md.index("Answers it: Doc B") < md.index("Partly: Doc C")


# ── the model's prose ─────────────────────────────────────────────────────────

def test_prose_is_parsed_and_its_citations_and_numbers_checked():
    raw = ('```json\n{"overview": "Hafnia films are studied [1][9].", "why": {"1": "Because [2].", "2": "Also [4]."},'
           ' "suggestions": [{"text": "Combine A and B [3].", "builds_on": [1, 2]}, '
           '{"text": "Invented", "builds_on": [7]}, {"text": "", "builds_on": [1]}]}\n```')
    p = gr.parse_prose(raw, n_sources=4, n_questions=2)
    assert p["overview"] == "Hafnia films are studied [1]."             # [9] was never sent: removed
    assert p["why"] == {1: "Because [2].", 2: "Also [4]."}
    assert p["suggestions"] == [{"text": "Combine A and B.", "builds_on": [1, 2]}]   # no citations, valid numbers
    with pytest.raises(ValueError):
        gr.parse_prose("no json here", n_sources=4, n_questions=2)


def test_checked_text_round_trips_through_its_markers():
    prose = {"overview": "O [1].", "why": {1: "W1 [2].", 3: "W3 [1]."}, "suggestions": []}
    text = gr.checkable_text(prose)
    back = gr.split_checked(text)
    assert back == {"overview": "O [1].", "why": {1: "W1 [2].", 3: "W3 [1]."}}
    # a section the check emptied loses its heading (strategies._remove_spans): it is simply missing
    assert gr.split_checked(text.replace("W3 [1].", "").replace(f"{gr.MARK}W3\n\n", "")) == \
        {"overview": "O [1].", "why": {1: "W1 [2]."}}


def test_messages_carry_the_numbered_sources_and_the_questions():
    sources = [{"kind": "stating", "source": "A", "page": 1, "text": "A text"},
               {"kind": "paper", "source": "Paper", "year": 2024, "page": 0, "text": "Abstract"}]
    msgs = gr.prose_messages([{"n": 1, "statement": "Q?", "facts": "Raised by [1]", "sources": [1, 2]}], sources,
                             topic="hafnia")
    user = msgs[1]["content"]
    assert "[1] A, page 2" in user and "[2] Paper (2024), abstract" in user and "1. Q?" in user and "hafnia" in user
    assert "JSON" in msgs[0]["content"]


def _block(md, title):
    """The Markdown of the collapsed block whose summary starts with `title`."""
    i = md.index(f"<summary>{title}")
    return md[i:md.index("</details>", i)]


def test_the_report_is_assembled_with_collapsed_tables_and_how_it_was_made():
    # the further, possibly-answered and suggested parts as tables in collapsed blocks (they can be long); the footer in
    # its own block
    g1, g2, g3 = _gap(1, "open"), _gap(2, "partly"), _gap(3, "addressed")
    g4 = _gap(4, "open", web="partly")
    sel = {"full": [g1, g2], "more": [g4], "addressed": [g3]}
    ev = {3: [dict(origin="library", source="Doc C", point_id="77", relation="addresses", reason="C measures it.")]}
    passages = {"77": {"source": "Doc C", "page": 4, "text": "Doc C answers question 3.", "type": "text"}}
    sources, refs = gr.gather_sources([g1, g2], ev, passages, answered=[g3])
    prose = {"overview": "Both concern hafnia [1].", "why": {1: "It blocks devices [1]."},
             "suggestions": [{"text": "Study both together.", "builds_on": [1, 2]}]}
    md = gr.assemble(sel, sources, refs, prose, names={"A": "Doc A", "Doc C": "Title C"}, topic="hafnia",
                     n_documents=25, evidence=ev, checked=True)
    assert md.startswith("## Open research questions on hafnia")
    assert md.index("Both concern hafnia [1].") < md.index("### 1. Question 1?") < md.index("### 2. Question 2?")
    assert "It blocks devices [1]." in md
    assert ("click **Draft a proposal on this question** under a question, or reply **draft a proposal on "
            "question 1** or **questions 1 and 2**._") in md
    # the hint right after the questions, then the three blocks, then how it was made, in this order
    order = [md.index(t) for t in ("Draft a proposal on this question", "<summary>Further open questions (1)",
                                   "<summary>Possibly already answered in the library (1)",
                                   "<summary>Directions suggested by the AI (1)", "<summary>How this report was made")]
    assert order == sorted(order)
    more = _block(md, "Further open questions")
    assert "| Question | Library | Literature |" in more and "| Question 4? | nothing answers it | related findings |" in more
    answered = _block(md, "Possibly already answered in the library")
    n = len(sources)                                         # the answering passage: numbered after the report's own
    assert sources[n - 1]["source"] == "Doc C" and "| Question | Answered by |" in answered
    assert f"| Question 3? | Title C [{n}] |" in answered
    sug = _block(md, "Directions suggested by the AI")
    assert "Suggested by the AI, not from the documents. Please verify." in sug          # the label of AI suggestions
    assert "| Direction | Builds on |" in sug and "| Study both together. | questions 1 and 2 |" in sug
    how = _block(md, "How this report was made")
    for t in ("**Questions:** what the documents themselves name as limitations or future work",
              "**In the library:** each question was searched across all passages of all other documents in the "
              "library; the closest passages were read to judge whether they answer it.",
              "**Published papers:** searched in Semantic Scholar, OpenAlex and an offline index, and judged from titles "
              "and abstracts only.",
              "**Written by the AI:** the overview, \"why it matters\" and the suggested directions.",
              "were checked against the sources", "Documents added after the checks are not included yet.",
              "A domain expert should judge the questions before relying on them."):
        assert t in how, t
    assert not re.search(r"\b\d+ (?:other )?documents\b", how)          # no count
    assert "### Further open questions" not in md and "### Suggested directions" not in md and "\n---\n" not in md
    unchecked = gr.assemble(sel, sources, refs, prose, names={}, topic=None, n_documents=25, evidence={},
                            checked=False)
    assert unchecked.startswith("## Open research questions in the library")
    assert "were left out this time because they could not be checked" in _block(unchecked, "How this report was made")


def test_empty_parts_have_no_block():
    g1 = _gap(1, "open")
    sources, refs = gr.gather_sources([g1], {}, {})
    md = gr.assemble({"full": [g1], "more": [], "addressed": []}, sources, refs,
                     {"overview": "", "why": {}, "suggestions": []}, names={}, topic=None, n_documents=5, evidence={},
                     checked=True)
    for t in ("Further open questions", "Possibly already answered", "Directions suggested by the AI"):
        assert f"<summary>{t}" not in md
    assert "<summary>How this report was made</summary>" in md


def test_an_answered_question_whose_passage_is_gone_still_names_the_document():
    g3 = _gap(3, "addressed")
    ev = {3: [dict(origin="library", source="Doc C", point_id="99", relation="addresses")]}   # passage deleted
    sources, refs = gr.gather_sources([], ev, {}, answered=[g3])
    md = gr.assemble({"full": [], "more": [], "addressed": [g3]}, sources, refs,
                     {"overview": "", "why": {}, "suggestions": []}, names={"Doc C": "Title C"}, topic=None,
                     n_documents=5, evidence=ev, checked=True)
    assert "| Question 3? | Title C |" in _block(md, "Possibly already answered in the library")


def test_the_writer_gets_only_the_reports_own_sources():
    g1, g3 = _gap(1, "open"), _gap(3, "addressed")
    ev = {3: [dict(origin="library", source="Doc C", point_id="77", relation="addresses")]}
    passages = {"77": {"source": "Doc C", "page": 4, "text": "C.", "type": "text"}}
    sources, refs = gr.gather_sources([g1], ev, passages, answered=[g3])
    assert gr.own_source_count(refs, [g1]) == len(sources) - 1


# ── citations, dates and report parts ──

def test_citation_lists_and_ranges_become_separate_citations():
    p = gr.parse_prose('{"overview": "Two fields [1, 4, 5] and more [2-3]; none [7, 1].", "why": {}}',
                       n_sources=5, n_questions=1)
    assert p["overview"] == "Two fields [1][4][5] and more [2][3]; none [1]."


def test_the_instruction_forbids_comma_citations_and_talk_about_the_sources():
    assert "[1][4]" in gr.PROSE_SYSTEM and "never [1, 4]" in gr.PROSE_SYSTEM
    assert "provided text" in gr.PROSE_SYSTEM


def test_dates_are_shown_in_local_time():
    late = datetime(2026, 10, 1, 22, 30, tzinfo=timezone.utc)               # 00:30 on 2 Oct in Berlin time
    assert gr._date(late, tz="Europe/Berlin") == "2 Oct 2026"


def test_at_most_two_questions_per_document_in_full_and_connected_questions_first():
    gaps = [_gap(i, "open", docs=("A",)) for i in range(1, 5)] + [_gap(9, "open", docs=("B",))]
    sel = gr.select(gaps, full=3, related_counts={4: 5, 9: 1})
    assert [g["id"] for g in sel["full"]] == [4, 9, 1]                    # A at most twice; links break ties
    assert [g["id"] for g in sel["more"]] == [2, 3]


def test_topic_floor_is_relative_to_the_best_question():
    assert gr.relevant_only({1: 0.25, 2: 0.11, 3: 0.09}) == {1: 0.25, 2: 0.11}
    assert gr.relevant_only({1: 0.04, 2: 0.03}) == {}


def test_paper_titles_are_one_line():
    from memento.research_gaps import literature as ls
    w = ls.from_s2({"paperId": "x", "title": "Effects of strain on \nHfO2\n thin films"})
    assert w.title == "Effects of strain on HfO2 thin films"
    refs = {"stating": [1], "library": [], "papers": [], "unnumbered": [("A\n title", 2020, "u", "partly")]}
    g = _gap(1, "open", web="partly")
    assert "A title (2020)" in gr.literature_line(g, refs, {})


def test_the_cap_per_document_never_leaves_slots_empty():
    # a topic whose questions mostly come from one document
    gaps = [_gap(i, "open", docs=("A",)) for i in range(1, 6)] + [_gap(9, "partly", docs=("B",))]
    sel = gr.select(gaps, full=4)
    assert [g["id"] for g in sel["full"]] == [1, 2, 3, 9] and [g["id"] for g in sel["more"]] == [4, 5]   # ranking order


def test_documents_still_being_checked_are_named_at_the_top():
    # asked right after an upload: the report says the new document is not included yet
    g = _gap(1, "open")
    sel = {"full": [g], "more": [], "addressed": []}
    sources, refs = gr.gather_sources([g], {}, {})
    md = gr.assemble(sel, sources, refs, {"overview": "", "why": {}, "suggestions": []}, names={"N": "New paper"},
                     topic=None, n_documents=26, evidence={}, checked=True, pending=["N"])
    head = md.split("### 1.")[0]
    assert 'Paper still processing, try again later: "New paper"' in head
    two = gr.assemble(sel, sources, refs, {"overview": "", "why": {}, "suggestions": []}, names={}, topic=None,
                      n_documents=27, evidence={}, checked=True, pending=["N", "M"])
    assert "Papers still processing, try again later" in two
    none = gr.assemble(sel, sources, refs, {"overview": "", "why": {}, "suggestions": []}, names={}, topic=None,
                       n_documents=25, evidence={}, checked=True)
    assert "still processing" not in none


def test_the_offline_source_is_named_in_the_report():
    # the offline literature index as a third literature source
    assert gr._services("semantic_scholar,openalex,alexandria") == "Semantic Scholar, OpenAlex and Offline index"
    assert gr._services("alexandria") == "Offline index"


def test_the_overview_starts_with_a_sentence_that_stands_on_its_own():
    # sentences that only make sense after an earlier one are dropped from the start (the check may remove the
    # overview's first sentence)
    assert gr.standalone_start("They also investigate X [1]. Dopants matter for Y [2].") == "Dopants matter for Y [2]."
    assert gr.standalone_start("However, it is unclear [1]. Moreover it fails [2]. The questions cover Z [3].") \
        == "The questions cover Z [3]."
    assert gr.standalone_start("The documents study dopants [1]. They also model phases [2].") \
        == "The documents study dopants [1]. They also model phases [2]."
    assert gr.standalone_start("It is open [1].") == ""
    assert gr.standalone_start("") == ""


def test_assemble_shows_the_overview_from_its_first_standalone_sentence():
    sel = {"full": [], "more": [], "addressed": []}
    md = gr.assemble(sel, [], {}, {"overview": "They also investigate X [1]. Hafnia dopants are studied [1].",
                                   "why": {}, "suggestions": []},
                     names={}, topic="dopants", n_documents=3, evidence={}, checked=True)
    assert "They also" not in md and "Hafnia dopants are studied" in md


def test_every_report_opens_with_the_same_fact_line():
    # the start does not depend on what survives the overview check: a line written by the code from the ledger's facts
    # comes first, the checked overview (if any) after it
    full = [{"id": 1, "statement": "Q1?", "library_status": "open", "web_status": "partly", "documents": ["A", "B"],
             "mentions": []},
            {"id": 2, "statement": "Q2?", "library_status": "partly", "web_status": None, "documents": ["B"],
             "mentions": []}]
    line = gr.opening_line(full, more=0, topic="dopants in hafnia")
    assert line == ("**2 open research questions** on dopants in hafnia, raised by 2 documents. Each was checked "
                    "against the other documents in the library and against published papers.")
    assert gr.opening_line(full[:1], more=3, topic=None).startswith(
        "**1 open research question** in the library, raised by 2 documents.")
    assert "Further ones are listed after them." in gr.opening_line(full[:1], more=3, topic=None)
    assert "published papers" not in gr.opening_line(full[1:], more=0, topic=None)
    assert gr.opening_line([], more=0, topic=None) == ""


def test_documents_checked_is_never_more_than_the_library_has():
    # a stored counter that drifted from the library's document count
    g = _gap(1, "open", docs=("A",), n_docs=34)
    capped = gr.cap_docs_checked([g], library_size=27)[0]
    assert capped["library_docs_checked"] == 26 and g["library_docs_checked"] == 34     # a copy; the input unchanged
    assert gr.cap_docs_checked([_gap(2, "open", n_docs=20)], library_size=27)[0]["library_docs_checked"] == 20
    assert "other 26 documents" in gr.library_line(capped, {"library": []})
    two = _gap(3, "partly", docs=("A", "B"), n_docs=40)
    assert gr.cap_docs_checked([two], library_size=27)[0]["library_docs_checked"] == 25
    assert gr.cap_docs_checked([g], library_size=None)[0]["library_docs_checked"] == 34   # size unknown: unchanged



# ── layout: a table first, then one evidence block per question ──
def _report(short=None):
    g1, g2 = _gap(1, "open"), _gap(2, "partly", web="partly")
    sel = {"full": [g1, g2], "more": [], "addressed": []}
    sources, refs = gr.gather_sources([g1, g2], {}, {})
    prose = {"overview": "Both concern hafnia [1].", "why": {1: "It blocks devices [1]."}, "suggestions": [],
             "short": short or {}}
    return gr.assemble(sel, sources, refs, prose, names={"A": "Doc A"}, topic="hafnia", n_documents=25, evidence={},
                       checked=True)


def test_report_table_then_evidence_blocks():
    md = _report(short={1: "Phase identity of hafnia films", 2: "Dopants and wake-up"})
    table = md.index("| # | Question | Library | Literature |")
    assert md.index("Both concern hafnia [1].") < table < md.index('<details type="memento_evidence">')
    assert "| 1 | Phase identity of hafnia films | nothing answers it | nothing found |" in md
    assert "| 2 | Dopants and wake-up | partly answered | related findings |" in md
    assert md.count('<details type="memento_evidence">') == 2
    block1 = md[md.index("<summary>Evidence for question 1</summary>"):md.index("</details>")]
    assert "### 1. Question 1?" in block1 and "**Raised by:**" in block1 and "**In the library:**" in block1
    assert "**Why it matters:** It blocks devices [1]." in block1


def test_short_titles_fall_back():
    assert gr.short_title("How does the oxygen vacancy concentration change the wake-up of doped hafnia films?",
                          None) == "How does the oxygen vacancy concentration change the…"
    assert gr.short_title("Short question?", None) == "Short question?"
    long = " ".join(["word"] * 15)
    assert gr.short_title("Q?", long) == " ".join(["word"] * 12) + "…"
    md = _report()                                          # no short titles from the model: the question's words
    assert "| 1 | Question 1? |" in md


def test_short_titles_are_parsed_from_the_prose():
    raw = '{"overview": "O [1].", "why": {}, "suggestions": [], "short": {"1": "Phase identity [1]", "7": "x"}}'
    out = gr.parse_prose(raw, n_sources=3, n_questions=2)
    assert out["short"] == {1: "Phase identity"}


# ── one document's open questions (start screen "Draft a proposal" → a document) ──
def test_a_document_report_names_the_document_in_its_heading_and_first_line():
    g1, g2 = _gap(1, "open", docs=("A",)), _gap(2, "partly", docs=("A", "B"), web="partly")
    sel = {"full": [g1, g2], "more": [], "addressed": []}
    sources, refs = gr.gather_sources([g1, g2], {}, {})
    md = gr.assemble(sel, sources, refs, {"overview": "", "why": {}, "suggestions": []}, names={"A": "Doc A"},
                     topic=None, document="Doc A", n_documents=25, evidence={}, checked=True)
    assert md.startswith('## Open research questions in "Doc A"\n')
    assert re.match(r"\s*## Open research questions\b", md)           # the proposal route and buttons still find it
    assert ('**2 open research questions** raised by "Doc A", 1 of them also by other documents. Each was checked '
            'against the other documents in the library and against published papers.') in md
    assert "### 1. Question 1?" in md and "### 2. Question 2?" in md and "Draft a proposal on this question" in md


def test_a_document_opening_line_counts_the_questions_other_documents_share():
    one = [{"id": 1, "statement": "Q1?", "library_status": "open", "web_status": None, "documents": ["A"]}]
    shared = [{"id": 2, "statement": "Q2?", "library_status": "open", "web_status": None, "documents": ["A", "B"]}]
    assert gr.opening_line(one, more=0, topic=None, document="Doc A") == (
        '**1 open research question** raised by "Doc A". Each was checked against the other documents in the library.')
    assert gr.opening_line(shared, more=0, topic=None, document="Doc A").startswith(
        '**1 open research question** raised by "Doc A", also by other documents.')
    assert gr.opening_line(shared + shared, more=0, topic=None, document="Doc A").startswith(
        '**2 open research questions** raised by "Doc A", both also by other documents.')
    assert "Further ones are listed after them." in gr.opening_line(one, more=2, topic=None, document="Doc A")


def test_a_document_whose_questions_the_library_answers_says_so_first():
    g = _gap(3, "addressed", docs=("A",))
    sel = {"full": [], "more": [], "addressed": [g]}
    md = gr.assemble(sel, [], {}, {"overview": "", "why": {}, "suggestions": []}, names={"A": "Doc A"}, topic=None,
                     document="Doc A", n_documents=25, evidence={}, checked=True)
    head, first = md.split("\n\n")[:2]
    assert head == '## Open research questions in "Doc A"'
    assert first == ('The open questions "Doc A" names may already be answered by other documents in the library: '
                     'see below.')
    assert "<summary>Possibly already answered in the library (1)" in md and "| Question 3? |" in md
    assert "Draft a proposal on this question" not in md and "| # | Question |" not in md


# ── one paper, one number: identity by DOI, never by title ──
# The same paper reached through two services (each with its own id, its own title spelling and year) is one source.
def _paper(gid_rel, origin, source, doi, title, year=None, text="An abstract.", url=""):
    return {"origin": origin, "source": source, "doi": doi, "title": title, "year": year, "text": text,
            "url": url, "relation": gid_rel, "reason": f"{origin} says so"}


def test_one_paper_found_by_two_services_is_one_source_with_the_best_details():
    g1, g2 = _gap(1, "open"), _gap(2, "open")
    ev = {1: [_paper("partly", "alexandria", "10.1088/2632-2153/ae04c1", "10.1088/2632-2153/ae04c1",
                     "High- T c superconductor candidates proposed by machine learning", text="Short.")],
          2: [_paper("addresses", "semantic_scholar", "460645287f16c94ce7a2cc38bc21c4dd85bd9e60",
                     "10.1088/2632-2153/AE04C1", "High- Tc superconductor candidates proposed by machine learning",
                     year="2024", text="A longer abstract of the same paper.")]}
    sources, refs = gr.gather_sources([g1, g2], ev, {})
    papers = [s for s in sources if s["kind"] == "paper"]
    assert len(papers) == 1
    p = papers[0]
    n = sources.index(p) + 1
    assert refs[1]["papers"] == [(n, "partly")] and refs[2]["papers"] == [(n, "addresses")]   # each keeps its verdict
    assert p["year"] == "2024" and p["title"] == "High- Tc superconductor candidates proposed by machine learning"
    assert p["text"] == "A longer abstract of the same paper." and p["services"] == ["semantic_scholar", "alexandria"]
    assert refs[1]["reasons"][n] == "alexandria says so" and refs[2]["reasons"][n] == "semantic_scholar says so"


@pytest.mark.parametrize("a, b", [("10.1000/ABC.1", "https://doi.org/10.1000/abc.1"), ("doi:10.1000/abc.1", "10.1000/abc.1"),
                                  ("http://dx.doi.org/10.1000/abc.1", " 10.1000/ABC.1 "), ("10.1000/abc.1.", "10.1000/abc.1")])
def test_the_same_doi_written_differently_is_one_paper(a, b):
    assert gr.doi_key(a) == gr.doi_key(b) == "10.1000/abc.1"


def test_papers_without_a_doi_are_never_merged_by_title():
    g = _gap(1, "open")
    ev = {1: [_paper("partly", "openalex", "W1", None, "Same title"), _paper("partly", "semantic_scholar", "S1", "",
                                                                            "Same title")]}
    sources, refs = gr.gather_sources([g], ev, {})
    assert len([s for s in sources if s["kind"] == "paper"]) == 2 and len(refs[1]["papers"]) == 2


def test_one_paper_twice_in_one_questions_evidence_is_listed_once():
    g = _gap(1, "open")
    ev = {1: [_paper("addresses", "openalex", "W9", "10.5/x", "T", year="2020"),
              _paper("partly", "alexandria", "10.5/x", "10.5/x", "T")]}
    sources, refs = gr.gather_sources([g], ev, {})
    assert [rel for _, rel in refs[1]["papers"]] == ["addresses"]           # the stronger verdict, once
    assert sources[refs[1]["papers"][0][0] - 1]["services"] == ["openalex", "alexandria"]


def test_a_paper_without_abstract_takes_the_number_its_doi_has_elsewhere():
    g1, g2 = _gap(1, "open"), _gap(2, "open")
    ev = {1: [_paper("partly", "openalex", "W5", "10.7/y", "T", year="2021")],
          2: [_paper("partly", "semantic_scholar", "S5", "10.7/y", "T", text="")]}
    sources, refs = gr.gather_sources([g1, g2], ev, {})
    assert refs[2]["unnumbered"] == [] and refs[2]["papers"] == refs[1]["papers"]


def test_the_literature_line_numbers_one_paper_once_in_the_report():
    g1, g2 = _gap(1, "open", web="partly"), _gap(2, "open", web="possibly_addressed")
    ev = {1: [_paper("partly", "alexandria", "10.9/z", "10.9/z", "Paper Z")],
          2: [_paper("addresses", "openalex", "W7", "10.9/Z", "Paper Z", year="2019")]}
    sources, refs = gr.gather_sources([g1, g2], ev, {})
    md = gr.assemble({"full": [g1, g2], "more": [], "addressed": []}, sources, refs,
                     {"overview": "", "why": {}, "suggestions": []}, names={"A": "Doc A"}, topic=None, n_documents=25,
                     evidence=ev, checked=True)
    assert md.count("Paper Z (2019) [") == 2 and "(n.d.)" not in md and len(set(re.findall(r"Paper Z \(2019\) \[(\d+)\]", md))) == 1



# ── proposal evidence lines: the page once, reasons not cut off ──
def _lib_case(reason):
    g = _gap(8, "partly")
    sources = [{"kind": "stating", "source": "A", "page": 1}, {"kind": "library", "source": "B", "page": 1}]
    refs = {"stating": [1], "library": [(2, "partly")], "papers": [], "unnumbered": [], "reasons": {2: reason},
            "unnumbered_reasons": []}
    return g, refs, sources


def test_the_pdf_and_the_writer_keep_the_page_the_chat_leaves_it_to_the_pill():
    g, refs, sources = _lib_case("short")
    assert "  - Partly: Doc B, p. 2 [2]: short" in gr.library_block(g, refs, sources, {"B": "Doc B"})
    assert "  - Partly: Doc B [2]: short" in gr.library_block(g, refs, sources, {"B": "Doc B"}, pages=False)


def test_a_long_reason_is_shown_whole():
    long = ("The passage describes using polyBERT as a fingerprinting tool within an inverse design workflow, providing "
            "a partial example of how fingerprints are used for design, but it does not detail the retraining or "
            "structural changes that the question asks about, nor how the model would be updated for new targets.")
    g, refs, sources = _lib_case(long)
    assert any(line.endswith(long) for line in gr.library_block(g, refs, sources, {}, pages=False))
    assert "…" not in "".join(gr.library_block(g, refs, sources, {}))


# ── a library report that stays short as the library grows ──
# Page 1: the 10 most open questions; "Open questions by document" (each row a button to that document's report); at
# most 20 rows per table; "more research gaps" for the next 10 (numbered on, so proposals by number work).
def _many(n_docs=6, per_doc=5, start_id=100):
    return [_gap(start_id + d * per_doc + q, ("open", "partly", "contested")[q % 3], docs=(f"D{d}",))
            for d in range(n_docs) for q in range(per_doc)]


def test_pages_continue_each_other_without_overlap():
    gaps = _many()                                           # 30 open questions, 5 per document
    p1, p2, p3, p4 = (gr.page_select(gaps, page=k, per_page=10) for k in (1, 2, 3, 4))
    assert [g["id"] for g in p1["full"]] == [g["id"] for g in gr.select(gaps, full=10)["full"]]   # page 1 as before
    ids = [g["id"] for p in (p1, p2, p3) for g in p["full"]]
    assert len(ids) == 30 and len(set(ids)) == 30 and p4["full"] == []
    assert (p1["start"], p2["start"], p3["start"]) == (1, 11, 21) and p1["total"] == p2["total"] == 30


def test_open_questions_by_document_counts_and_names_the_most_open_one():
    gaps = [_gap(1, "partly", docs=("A",)), _gap(2, "contested", docs=("A",)), _gap(3, "open", docs=("B",)),
            _gap(4, "addressed", docs=("B",)), _gap(5, "open", docs=("A", "C"))]
    rows = gr.by_document(gaps, {"A": "Doc A", "B": "Doc B", "C": "Doc C"})
    assert [(r["name"], r["count"]) for r in rows] == [("Doc A", 3), ("Doc B", 1), ("Doc C", 1)]
    assert rows[0]["top"] == "Question 2?"                    # contested first, as the report ranks them


def _page_md(sel, prose=None, **kw):
    sources, refs = gr.gather_sources(sel["full"], {}, {})
    return gr.assemble(sel, sources, refs, prose or {"overview": "", "why": {}, "suggestions": []},
                       names={f"D{d}": f"Doc {d}" for d in range(9)}, topic=kw.pop("topic", None), n_documents=30,
                       evidence={}, checked=True, **kw)


def test_page_two_is_numbered_on_and_can_be_drafted_from():
    gaps = _many()
    sel = gr.page_select(gaps, page=2, per_page=10)
    prose = {"overview": "", "why": {1: "First of this page [1]."}, "short": {2: "Second short"},
             "suggestions": [{"text": "Join them.", "builds_on": [1, 2]}]}
    md = _page_md(sel, prose)
    assert md.startswith("## Open research questions in the library (questions 11–20)\n")
    assert "**Questions 11–20 of 30** open research questions in the library" in md
    assert "### 11. " in md and "### 20. " in md and "### 1. " not in md and "<summary>Evidence for question 11<" in md
    assert "| 11 | " in md and "| 12 | Second short |" in md and "First of this page [1]." in md
    assert "| Join them. | questions 11 and 12 |" in md
    assert "reply **draft a proposal on question 11** or **questions 11 and 12**" in md
    from memento.research_gaps import proposal as gp
    assert sorted(gp.report_questions(md)) == list(range(11, 21))
    assert gr.report_scope(md) == {"kind": "library", "value": None, "last": 20}


def test_the_library_page_one_lists_documents_instead_of_every_further_question():
    gaps = _many(n_docs=25, per_doc=3)                      # 75 open questions in 25 documents
    sel = gr.page_select(gaps, page=1, per_page=10)
    rows = gr.by_document(gaps, {f"D{d}": f"Doc {d}" for d in range(25)})
    md = _page_md(sel, by_document=rows)
    assert "<summary>Further open questions" not in md
    i = md.index("<summary>Open questions by document (20 of 25)</summary>")
    block = md[i:md.index("</details>", i)]
    assert "| Document | Open questions | Most open one |" in block and block.count("\n| Doc ") == 20
    assert "**Show its questions**" in block
    assert "reply **more research gaps**" in md and "**10 of 75 open research questions** in the library" in md


def test_topic_and_document_reports_keep_a_short_further_table():
    gaps = _many(n_docs=1, per_doc=40)                      # one document, 40 open questions
    sel = gr.page_select(gaps, page=1, per_page=10, per_document=10)
    md = _page_md(sel, document="Doc 0")
    i = md.index("<summary>Further open questions (20 of 30)</summary>")
    assert len(re.findall(r"\n\| Question \d+\? ", md[i:md.index("</details>", i)])) == 20
    assert gr.report_scope(md) == {"kind": "document", "value": "Doc 0", "last": 10}
    assert gr.report_scope(_page_md(gr.page_select(gaps, page=1, per_page=10), topic="hafnia films")) == {
        "kind": "topic", "value": "hafnia films", "last": 10}


def test_possibly_answered_is_capped_with_its_total():
    gaps = [_gap(200 + k, "addressed", docs=("D1",)) for k in range(30)] + [_gap(1, "open")]
    sel = gr.page_select(gaps, page=1, per_page=10)
    md = _page_md(sel)
    i = md.index("<summary>Possibly already answered in the library (20 of 30)</summary>")
    assert len(re.findall(r"\n\| Question \d+\? ", md[i:md.index("</details>", i)])) == 20


def test_the_last_page_says_there_is_no_more():
    gaps = _many(n_docs=1, per_doc=12)
    md = _page_md(gr.page_select(gaps, page=2, per_page=10, per_document=10))
    assert "more research gaps" not in md and "### 11. " in md and "### 12. " in md


# ── paper cards: an outside paper in the side panel like a library page ──
def test_an_abstract_is_split_into_sentences_and_nothing_is_lost():
    text = ("We train polyBERT on 100 million polymers. It predicts 29 properties (R2 = 0.8). The model is 2x faster "
            "than PG, e.g. for Tg! Fingerprints are learned; e.g. polymer chemistry. Open question remains?")
    s = gr.paper_sentences(text)
    assert s[0] == "We train polyBERT on 100 million polymers." and len(s) == 5
    assert " ".join(s) == " ".join(text.split())
    assert gr.paper_sentences("") == [] and gr.paper_sentences("One sentence without a stop") == [
        "One sentence without a stop"]


@pytest.mark.parametrize("scores, want", [([0.9, 0.5, 0.1], [0, 1]), ([0.9, 0.3, 0.1], [0]), ([0.02, 0.01], []),
                                          ([0.2, 0.6, 0.31], [1, 2]), ([], [])])
def test_the_marked_sentences_are_the_best_one_and_a_close_second(scores, want):
    assert gr.pick_marks(scores) == want


def test_a_paper_keeps_its_id_at_every_service_that_found_it():
    g1, g2 = _gap(1, "open"), _gap(2, "open")
    ev = {1: [_paper("partly", "openalex", "https://openalex.org/W123", "10.4/q", "T", year="2022")],
          2: [_paper("addresses", "semantic_scholar", "abc123", "10.4/Q", "T")]}
    sources, _ = gr.gather_sources([g1, g2], ev, {})
    [paper] = [x for x in sources if x["kind"] == "paper"]
    assert paper["ids"] == {"openalex": "https://openalex.org/W123", "semantic_scholar": "abc123"}


def test_service_links_are_built_only_from_well_formed_ids():
    links = gr.service_links({"semantic_scholar": "460645287f16c94ce7a2cc38bc21c4dd85bd9e60",
                              "openalex": "https://openalex.org/W4389", "alexandria": "10.1/x"})
    assert links == [
        {"name": "Semantic Scholar", "url": "https://www.semanticscholar.org/paper/460645287f16c94ce7a2cc38bc21c4dd85bd9e60"},
        {"name": "OpenAlex", "url": "https://openalex.org/W4389"},
        {"name": "Offline index", "url": ""}]
    assert gr.service_links({"semantic_scholar": "x\"><script>", "openalex": "javascript:alert(1)"}) == [
        {"name": "Semantic Scholar", "url": ""}, {"name": "OpenAlex", "url": ""}]


# ── a partly judged check: the prose stays, the footer says so ──
def test_a_partly_checked_report_keeps_its_prose_and_says_some_could_not_be_checked():
    g1 = _gap(1, "open")
    sources, refs = gr.gather_sources([g1], {}, {})
    prose = {"overview": "Hafnia work [1].", "why": {1: "It limits devices [1]."}, "suggestions": []}
    md = gr.assemble({"full": [g1], "more": [], "addressed": []}, sources, refs, prose, names={}, topic=None,
                     n_documents=5, evidence={}, checked="partial")
    assert "Hafnia work [1]." in md and "It limits devices [1]." in md
    assert "and some statements could not be checked this time." in md
    full = gr.assemble({"full": [g1], "more": [], "addressed": []}, sources, refs, prose, names={}, topic=None,
                       n_documents=5, evidence={}, checked=True)
    assert "could not be checked this time" not in full


@pytest.mark.skipif(__import__("shutil").which("node") is None, reason="node not installed")
def test_the_proposal_buttons_sit_in_the_question_table():
    # the report's questions are in closed "Evidence for question N" blocks: the buttons go in the table's rows
    # (loader.js, tests/openwebui/js/loader_propose.test.mjs)
    import subprocess
    root = PROJECT_ROOT
    r = subprocess.run(["node", str(root / "tests/openwebui/js/loader_propose.test.mjs")], capture_output=True, text=True,
                       timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
