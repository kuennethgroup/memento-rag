# When a research-gap answer's table is shown again: hidden only for the same document, when the request asks for
# extensions alone or bans repeating the gaps, and never when it asks to see them. Also: the suggestions leave out "Gaps
# in earlier work it addresses" in a stored answer's calm layout too.
import pytest

import openwebui.functions.query_pipe as qp
from memento.research_gaps import passages as gr
from memento.query import request
from memento.query.flows import gap_answer

GAPS = [{"role": "user", "content": "What research gaps does paper A name?"},
        {"role": "assistant", "content": '| What the authors state | Points |\n|---|---|\n| Limitations | 1 |\n\n'
                                         '<details type="memento_gaps">\n<summary>Limitations (1)</summary>\n\n1. x [1]\n'
                                         '\n</details>\n'}]


def show(q, *, same=True, history=GAPS, suggest=True):
    return request.show_gap_table(suggest=suggest, history=history, question=q, same_document=same)


@pytest.mark.parametrize("q", ["How could this work be extended?", "Only suggest extensions",
                               "Do not repeat the gaps; just suggest extensions",
                               "Suggest extensions without repeating the gaps",
                               "Schlag Erweiterungen vor, ohne die Lücken zu wiederholen",
                               "Nur Erweiterungen vorschlagen, bitte"])
def test_extensions_alone_for_the_same_document_hide_the_table(q):
    assert not show(q)


@pytest.mark.parametrize("q", ["Show the gaps again and suggest new extensions without repeating your previous suggestions",
                               "Show those again and add your ideas",
                               "What research gaps does it name, and suggest extensions",
                               "Show its limitations again and suggest extensions",
                               "Zeig die Lücken nochmal und schlag neue Erweiterungen vor, ohne deine Vorschläge zu "
                               "wiederholen"])
def test_a_request_to_see_the_gaps_shows_them(q):
    assert show(q)


def test_another_document_or_an_unknown_one_shows_the_table():
    assert show("Only suggest extensions", same=False)


def test_no_gap_answer_before_or_no_suggestions_shows_the_table():
    assert show("Only suggest extensions", history=[])
    assert show("What research gaps does it name?", suggest=False)


def test_the_pipe_decides_after_finding_the_document():
    import inspect
    src = inspect.getsource(gap_answer.reply)
    assert "same_document=" in src and "conversation.previous_document(" in src


# ── the suggestions' basis ───────────────────────────────────────────────────────────────────────────────────────────

def test_study_points_leave_out_earlier_work_in_the_calm_layout():
    raw = ("### Limitations\n1. Only one material [1].\n\n### Gaps in earlier work it addresses\n2. Earlier work "
           "ignored X [2].\n")
    calm = gr.calm_gap_answer(raw)
    assert "Earlier work ignored" in calm
    points = gr.study_points(calm)
    assert "Only one material [1]." in points and "Earlier work ignored" not in points and "[2]" not in points
    assert gr.study_points(raw) == "### Limitations\n1. Only one material [1].\n"
