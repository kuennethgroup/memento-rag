# Thorough check thinks with a cap of its own: measured, a 2,000-token cap halves the time of the 12,288 one with no
# loss in the judged outcome. The check's rewrite step takes the same cap.
import inspect

import openwebui.functions.query_pipe as qp
from memento.agentic_validation import strategies
from memento.agentic_validation.claims import Claim
from memento.agentic_validation.verifiers import Verdict
from memento.query import checking, generation
from memento.query.flows import answer as answer_flow


def test_the_valve_and_its_fallback():
    v = qp.Pipe().valves
    assert v.THOROUGH_THINKING_BUDGET == 2000
    assert generation.thorough_thinking_budget(v) == 2000
    v.THOROUGH_THINKING_BUDGET = 0                         # 0: the normal cap, as before
    assert generation.thorough_thinking_budget(v) == v.THINKING_BUDGET


def test_the_rewrite_thinks_with_the_contexts_cap(monkeypatch):
    claims = [Claim("c1", "Alumina reaches 18 GPa", (1,), (0, 10))]
    rounds = iter([({"c1": Verdict("c1", "unsupported")}), ({"c1": Verdict("c1", "supported")})])

    def check(ctx, llm, answer, hits, check_gap=False, previous=None):
        return claims, next(rounds), {}
    calls = []

    def gen(llm, model, messages, **kw):
        calls.append(kw)
        return {"answer": "Alumina reaches 18 GPa [1]."}
    monkeypatch.setattr(strategies, "check_answer", check)
    monkeypatch.setattr(strategies, "generate_answer", gen)
    ctx = strategies.ArmContext(object(), "m", "system", "q", [{"text": "t", "source": "s", "page": 1}], None,
                                revise_thinking=True, thinking_budget=2000)
    strategies.run_arm_a(ctx, "Alumina reaches 18 GPa [1].", max_rounds=2)
    assert calls and calls[0]["enable_thinking"] is True and calls[0]["thinking_budget"] == 2000


def test_the_context_keeps_the_evaluated_cap_by_default():
    assert strategies.ArmContext(object(), "m", "s", "q", [], None).thinking_budget == 12288   # RQ3 runs unchanged


def test_the_pipe_caps_both_the_answer_and_the_rewrite():
    src = inspect.getsource(answer_flow.stream_answer)
    assert "local_thinking_budget=generation.thorough_thinking_budget(turn.valves) if turn.verify else None" in src
    assert "thinking_budget=generation.thorough_thinking_budget(valves)" in inspect.getsource(checking.thorough_check)
