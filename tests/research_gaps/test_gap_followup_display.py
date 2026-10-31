# "Suggest how this study could be extended" right after a gap answer shows the suggestions, not the same table again.
import openwebui.functions.query_pipe as qp
from memento.query import request

GAPS = ("| What the authors state | Points |\n|---|---|\n| Future work | 3 |\n\n"
        '<details type="memento_gaps">\n<summary>Future work (3)</summary>\n\n1. More [2].\n\n</details>\n')


def test_a_previous_gap_answer_is_recognised():
    assert request.answer_was_gaps([{"role": "user", "content": "future work?"}, {"role": "assistant", "content": GAPS}])
    assert not request.answer_was_gaps([{"role": "assistant", "content": "polyBERT is a language model [1]."}])
    assert not request.answer_was_gaps([])


def test_the_table_is_shown_unless_only_suggestions_follow_a_gap_answer():
    hist = [{"role": "assistant", "content": GAPS}]
    assert request.show_gap_table(suggest=True, history=hist) is False
    assert request.show_gap_table(suggest=False, history=hist) is True       # a new gap question: the table as usual
    assert request.show_gap_table(suggest=True, history=[]) is True          # suggestions without an earlier table: both
