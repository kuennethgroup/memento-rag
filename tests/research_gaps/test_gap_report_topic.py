# The report heading shows the topic, not the whole question. The search still uses the full question.
import pytest

from memento.research_gaps import report as gr


@pytest.mark.parametrize("q, want", [
    ("What are the open research questions in polymer informatics?", "polymer informatics"),
    ("find potential research gaps across the library", None),
    ("find potential research gaps?", None),
    ("What are the research gaps in ferroelectric hafnia?", "ferroelectric hafnia"),
    ("Which open questions are there on polymer fingerprinting", "polymer fingerprinting"),
    ("Welche offenen Fragen gibt es zu Hafniumoxid?", "Hafniumoxid"),
    ("polymer informatics", "polymer informatics"),
    (None, None),
    ("", None),
])
def test_display_topic(q, want):
    assert gr.display_topic(q) == want


def test_a_question_that_cannot_be_reduced_is_quoted():
    assert gr.display_topic("How do dopants and strain interact when films are annealed?") == \
        '"How do dopants and strain interact when films are annealed?"'


def test_heading_and_first_line_use_it():
    sel = {"full": [], "more": [], "start": 1, "total": 0}
    prose = {"overview": "", "why": {}, "suggestions": []}
    out = gr.assemble(sel, [], {}, prose, names={}, topic="find potential research gaps across the library",
                      n_documents=1, evidence={}, checked=True)
    assert out.startswith("## Open research questions in the library") and "find potential" not in out
    out = gr.assemble(sel, [], {}, prose, names={}, topic="What are the open research questions in polymer informatics?",
                      n_documents=1, evidence={}, checked=True)
    assert out.startswith("## Open research questions on polymer informatics")
