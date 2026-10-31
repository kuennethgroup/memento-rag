# A draft research proposal from questions of the open-research-questions report (text reply + PDF). Facts and
# references by code; motivation and background by the model, checked against the sources; the proposed work labelled as
# the model's suggestion.
import sys
from pathlib import Path

import pytest
from memento.config.paths import PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT))
from memento.research_gaps import proposal as gp  # noqa: E402

REPORT = """## Open research questions on hafnia

Overview [1].

### 1. Is the phase Pca21 or Pbca?

- **Raised by:** Doc A [1]

### 2. How do dopants change wake-up?

- **Raised by:** Doc B [2]
"""


@pytest.mark.parametrize("text, want", [
    ("Draft a proposal on question 2", [2]),
    ("draft a research proposal for questions 1 and 2.", [1, 2]),
    ("Write a proposal on 2, 1", [1, 2]),
    ("proposal on question 3 please", [3]),
    ("Summarize question 2", None),
    ("Draft a proposal", None),
])
def test_the_request_names_question_numbers(text, want):
    assert gp.proposal_request(text) == want


def test_the_reports_questions_are_read_back_by_number():
    assert gp.report_questions(REPORT) == {1: "Is the phase Pca21 or Pbca?", 2: "How do dopants change wake-up?"}
    assert gp.report_questions("no report here") == {}


def test_the_proposal_prose_is_parsed_with_its_checks():
    raw = ('{"title": "Phase [3] identity", "motivation": "Devices need it [1, 2].", "background": "Known [9].", '
           '"objectives": ["Find the phase [1]", ""], "approach": "Use IR (Q1) [2].", "outcomes": "A map.", '
           '"risks": "Films vary."}')
    p = gp.parse_proposal(raw, n_sources=3)
    assert p["title"] == "Phase identity"                                      # no citations in the title
    assert p["motivation"] == "Devices need it [1][2]." and p["background"] == "Known."   # [9] never sent
    assert p["objectives"] == ["Find the phase"]                                # proposed work: no source citations
    assert p["approach"] == "Use IR (Q1)." and p["outcomes"] == "A map." and p["risks"] == "Films vary."
    with pytest.raises(ValueError):
        gp.parse_proposal("nothing", n_sources=3)


def test_motivation_and_background_round_trip_through_the_check_markers():
    p = {"motivation": "M [1].", "background": "B [2]."}
    assert gp.split_checked(gp.checkable_text(p)) == {"motivation": "M [1].", "background": "B [2]."}
    assert gp.split_checked(gp.checkable_text(p).replace("M [1].", "")) == {"motivation": "", "background": "B [2]."}


def test_the_proposal_is_assembled_with_labels_facts_and_references():
    sources = [{"kind": "stating", "source": "A", "page": 1, "text": "t"},
               {"kind": "paper", "source": "P", "title": "Paper P", "year": 2024, "url": "https://doi.org/10.1/p",
                "text": "abs"}]
    prose = {"title": "Phase identity in hafnia films", "motivation": "M [1].", "background": "B [2].",
             "objectives": ["O1", "O2"], "approach": "A", "outcomes": "E", "risks": "R"}
    md = gp.assemble(prose, questions=["Is the phase Pca21 or Pbca?"], facts=["- **In the library:** nothing."],
                     sources=sources, names={"A": "Doc A"}, checked=True)
    assert md.startswith("## Draft research proposal: Phase identity in hafnia films")
    assert "### Research question" in md and "Is the phase Pca21 or Pbca?" in md and "nothing." in md
    assert md.index("### Motivation") < md.index("### Background") < md.index("### Proposed work")
    assert "Suggested by the AI, not from the documents. Please verify." in md            # the label of AI suggestions
    assert "1. O1" in md and "2. O2" in md
    refs = md[md.index("### References"):]
    assert "[1] Doc A, p. 2" in refs and "[2] Paper P (2024). https://doi.org/10.1/p" in refs
    assert "domain expert" in md
    unchecked = gp.assemble(prose, questions=["Q?"], facts=[], sources=sources, names={}, checked=False)
    assert "M [1]." not in unchecked and "could not be checked" in unchecked


def test_the_pdf_is_written(tmp_path):
    path = gp.write_pdf("## Title\n\nSome **bold** text [1].\n\n- a point\n", tmp_path / "x.pdf")
    data = path.read_bytes()
    assert data.startswith(b"%PDF") and len(data) > 500
    import fitz
    assert "Some bold text [1]." in fitz.open(path)[0].get_text().replace("\n", " ")


def test_evidence_lines_stay_nested_in_the_pdf():
    html = gp.to_html("- **In the library:** partly:\n  - Partly: Doc, p. 2 [2]: reason\n")
    assert "<li><strong>In the library:</strong> partly:\n<ul>" in html or "<ul>\n<li>Partly" in html
    assert html.count("<ul>") == 2



def test_report_questions_parse_the_new_layout():
    # each question's heading sits inside its evidence block; "Draft a proposal on question N" still works
    md = ("## Open research questions on hafnia\n\n| # | Question | Library | Literature |\n|---|---|---|---|\n"
          "| 1 | Phase | nothing answers it | nothing found |\n\n<details type=\"memento_evidence\">\n"
          "<summary>Evidence for question 1</summary>\n\n### 1. Is the phase Pca21 or Pbca?\n\n- **Raised by:** A [1]"
          "\n\n</details>\n")
    assert gp.report_questions(md) == {1: "Is the phase Pca21 or Pbca?"}



# ── the proposal, collapsed: title, summary and the PDF link visible; sections folded ──
def test_proposal_summary_is_parsed():
    raw = ('{"title": "T", "summary": "Is it Pca21 or Pbca [2]? We combine IR and TEM. Then we link it to wake-up. '
           'A fourth sentence.", "motivation": "", "background": "", "objectives": [], "approach": "", '
           '"outcomes": "", "risks": ""}')
    p = gp.parse_proposal(raw, n_sources=3)
    assert p["summary"] == "Is it Pca21 or Pbca? We combine IR and TEM. Then we link it to wake-up."
    assert gp.parse_proposal(raw.replace('"summary"', '"x"'), n_sources=3)["summary"] == ""


def _prose():
    return {"title": "Phase identity", "summary": "Is the phase Pca21 or Pbca? IR, TEM and XRD together.",
            "motivation": "M [1].", "background": "B [2].", "objectives": ["O1"], "approach": "A", "outcomes": "E",
            "risks": "R"}


def test_collapsed_proposal_keeps_title_summary_and_link_visible():
    sources = [{"kind": "stating", "source": "A", "page": 1, "text": "t"}]
    md = gp.assemble(_prose(), questions=["Is the phase Pca21 or Pbca?"], facts=["- **In the library:** nothing."],
                     sources=sources, names={"A": "Doc A"}, checked=True, collapsed=True,
                     link="\n[Download as PDF](/files/report/x.pdf)\n")
    head = md[:md.index("<details")]
    assert head.startswith("## Draft research proposal: Phase identity")
    assert "Is the phase Pca21 or Pbca? IR, TEM and XRD together." in head and "[Download as PDF]" in head
    for title in ("Research question and evidence", "Motivation", "Background", "References (1)"):
        assert f"<summary>{title}</summary>" in md, title
    proposed = md[md.index('<details type="memento_proposed">'):]
    assert "<summary>Proposed work (suggested by the AI)</summary>" in proposed
    assert "_Suggested by the AI, not from the documents. Please verify._" in proposed and "1. O1" in proposed
    assert md.count("<details") == md.count("</details>") == 5


def test_pdf_markdown_stays_plain():
    md = gp.assemble(_prose(), questions=["Q?"], facts=[], sources=[], names={}, checked=True)
    assert "<details" not in md and "### Motivation" in md and "Is the phase Pca21 or Pbca?" in md


def test_a_partly_checked_proposal_says_some_statements_could_not_be_checked():
    # a partly judged check
    assert "and some statements could not be checked this time." in gp._how_made("partial")
    assert "could not be checked this time" not in gp._how_made(True)
    assert "left out" in gp._how_made(False)
