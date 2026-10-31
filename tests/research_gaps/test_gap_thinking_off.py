# Research-gap thinking and never storing an answer with leaked reasoning: a capped thinking budget can run out and the
# model go on deliberating in the visible answer, which must never be stored and replayed.
import openwebui.functions.query_pipe as qp
from memento.research_gaps import passages as gr
from memento.query import cache, generation

CLEAN = ("### Limitations\n1. Only one alloy system [3].\n2. The surrogate does not generalise in sparse regions,\n"
         "   which misleads the error metrics [16][17].\n\n### Future work\n3. Test other systems [8].\n\n"
         "_Answered by Qwen 3.8 27B (Local)._\n")
LEAK = ("### Limitations\n1. The GPR does not generalize well [16].\n\nCould [9] be limitation? If put under Untested "
        "conditions, okay. Let's think.\n\n### Untested conditions\n2. Data-sparse regions are untested [9].\n")
SKELETON = ("### Limitations\n1. ... [4]\n2. ... [7]\n...\n### Uncertainties\n6. ... [1][2]\n...\n\n"
            "Need ensure each sentence ends with passage(s). If sentence has multiple clauses, final token\n")


def test_gap_answers_think_while_the_valve_is_on():
    # measured: without thinking, books and reviews were treated as studies; with the 2,000-token cap correctly, so
    # thinking stays on (the Valve can switch it off)
    assert generation.gap_thinks(think=True, thorough=False, enabled=True) is True
    assert generation.gap_thinks(think=True, thorough=True, enabled=True) is True
    assert generation.gap_thinks(think=False, thorough=True, enabled=True) is True
    assert generation.gap_thinks(think=True, thorough=False, enabled=False) is False
    assert generation.gap_thinks(think=True, thorough=True, enabled=False) is False


def test_the_valve_is_on_by_default():
    assert qp.Pipe().valves.GAP_THINKING is True


def test_thorough_keeps_the_cap():
    # uncapped thinking with Thorough check leaves the check too little time
    assert generation.gap_thinking_budget(True, 2000) == 2000 and generation.gap_thinking_budget(False, 2000) == 2000
    assert generation.gap_thinking_budget(True, 0) is None


def test_leaked_lines_are_not_shown():
    out = gr.calm_gap_answer(LEAK)
    assert "Let's think" not in out and "Could [9]" not in out
    assert "The GPR does not generalize well [16]." in out and "Data-sparse regions are untested [9]." in out
    assert gr.strip_leaked(CLEAN) == CLEAN


def test_stored_answers_are_keyed_by_thinking():
    v = qp.Pipe().valves
    assert cache.gap_prompt_version(2000, v) == cache.gap_prompt_version(2000, v, thinking=True)   # old keys unchanged
    assert cache.gap_prompt_version(2000, v, thinking=False) != cache.gap_prompt_version(2000, v, thinking=True)


def test_a_clean_gap_answer_passes():
    assert gr.gap_answer_clean(CLEAN) is True
    assert gr.gap_answer_clean("The document names no research gaps.") is True   # the one-sentence "none" answer
    assert gr.gap_answer_clean("") is True


def test_leaked_reasoning_and_a_planning_skeleton_fail():
    assert gr.gap_answer_clean(LEAK) is False
    assert gr.gap_answer_clean(SKELETON) is False


def test_an_answer_with_leaked_reasoning_is_never_stored():
    assert cache.gap_storable(True, CLEAN) is True
    assert cache.gap_storable(True, LEAK) is False
    assert cache.gap_storable(False, CLEAN) is False                               # the heading gate still decides too
