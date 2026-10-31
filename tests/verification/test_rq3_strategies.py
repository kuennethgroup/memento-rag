"""RQ3 task 5: arms P0/P1/A/C (memento/agentic_validation/strategies.py)."""
import json
import re
import types

import pytest

from memento.agentic_validation.claims import SPLIT_SYSTEM_PROMPT, markers, strip_markers
from memento.agentic_validation.strategies import (NO_ANSWER, P1_ADDENDUM, REWRITE_INSTRUCTION, SUBQUERY_PROMPT, ArmContext,
                                           finalize_for_display, run_arm_a, run_arm_c, run_strategy)
from memento.agentic_validation.verifiers import Verdict


class FakeLLM:
    """Split requests → one claim per numbered sentence (or a scripted split); rewrite → queued revisions;
    sub-query → fixed query; anything else → a generated answer. Records every request."""
    def __init__(self, rewrites=(), split=None, answer="Answer [1]."):
        self.rewrites, self.split, self.answer, self.requests = list(rewrites), split, answer, []
        self.chat = self; self.completions = self

    def create(self, **kw):
        self.requests.append(kw)
        msgs = kw["messages"]
        if msgs[0]["content"] == SPLIT_SYSTEM_PROMPT:
            if self.split:
                out = self.split(msgs[1]["content"])
            else:
                out = []
                only = re.search(r"Split ONLY these sentences: ([S\d, ]+)\.", msgs[1]["content"])   # re-split after
                only = {int(x) for x in re.findall(r"S(\d+)", only.group(1))} if only else None       # a revision
                for line in msgs[1]["content"].splitlines():
                    m = re.match(r"S(\d+): (.*)", line)
                    if not m:
                        continue
                    k, sent = m.groups()
                    if only is not None and int(k) not in only:
                        continue
                    if re.search(r"\w", strip_markers(sent)) and not sent.startswith("#"):
                        out.append({"text": strip_markers(sent), "cited": list(markers(sent)), "s": [int(k)]})
            content = json.dumps(out)
        elif msgs[0]["content"] == SUBQUERY_PROMPT:
            content = "zirconia toughening"
        elif isinstance(msgs[-1]["content"], str) and msgs[-1]["content"].startswith(REWRITE_INSTRUCTION):
            content = self.rewrites.pop(0)
        else:
            content = self.answer
        return types.SimpleNamespace(choices=[types.SimpleNamespace(
            message=types.SimpleNamespace(content=content, reasoning_content=None))], usage=None)


class FakeVerifier:
    """Labels by the first matching substring rule; claims checked against all hits (gap check) use `gap_rules`."""
    name = "fake"

    def __init__(self, rules=None, gap_rules=None):
        self.rules, self.gap_rules, self.requests, self.calls = rules or {}, gap_rules, [], 0

    def verify(self, pairs):
        self.requests.append(pairs); self.calls += 1
        out = {}
        for cid, text, src in pairs:
            rules = self.gap_rules if (self.gap_rules is not None and src.count("[") > 2) else self.rules
            label = next((lab for sub, lab in rules.items() if sub in text), "supported")
            out[cid] = Verdict(cid, label, None, "text", "reason for " + text[:20])
        return out


HITS = [{"type": "text", "text": "Alumina reached 99 % density.", "source": "P", "page": 0},
        {"type": "text", "text": "SPS used 1400 °C.", "source": "P", "page": 1}]
P1 = "Alumina reached 99 % density [1]. SPS used 1600 °C [2]."


def ctx(llm, verifier, **kw):
    return ArmContext(llm, "m", "SYS", "What happened to the alumina?", list(HITS), verifier, **kw)


# ── C ─────────────────────────────────────────────────────────────────────────
def test_arm_c_removes_the_failing_sentence_by_number_check():
    r = run_arm_c(ctx(FakeLLM(), FakeVerifier()), P1)            # verifier approves; 1600 ≠ 1400 °C in the source
    assert r.answer == "Alumina reached 99 % density [1]." and r.removed == ["c2"]
    assert r.issues["c2"][0].kind == "number_not_in_source" and r.collateral == []


def test_arm_c_counts_collateral_and_keeps_unjudged():
    split = lambda numbered: [{"text": "Alumina reached 99 % density", "cited": [1], "s": [1]},
                              {"text": "SPS was slow", "cited": [2], "s": [1]},
                              {"text": "SPS used 1400 °C", "cited": [2], "s": [2]}]
    ans = "Alumina reached 99 % density and SPS was slow [1][2]. SPS used 1400 °C [2]."
    r = run_arm_c(ctx(FakeLLM(split=split), FakeVerifier({"slow": "unsupported", "1400": "unjudged"})), ans)
    assert r.removed == ["c2"] and r.collateral == ["c1"] and r.unjudged == 1
    assert r.answer == "SPS used 1400 °C [2]."


def test_arm_c_everything_removed_gives_a_no_answer_in_the_question_language():
    c = ctx(FakeLLM(), FakeVerifier({"": "contradicted"}))
    assert run_arm_c(c, P1).answer == NO_ANSWER["en"]
    c.question = "Welche Dichte wurde bei der Sinterung erreicht und wie hoch war die Temperatur?"
    assert run_arm_c(c, P1).answer == NO_ANSWER["de"]


def test_arm_c_tidies_empty_list_items_and_headings():
    ans = "## Results\n- **Density:** Alumina reached 99 % density [1].\n- SPS used 1600 °C [2].\n\n## Extra\n- SPS used 1600 °C [2]."
    r = run_arm_c(ctx(FakeLLM(), FakeVerifier()), ans)
    assert r.answer == "## Results\n- **Density:** Alumina reached 99 % density [1]."


# ── A ─────────────────────────────────────────────────────────────────────────
def test_arm_a_rewrites_only_failing_claims_with_reasons():
    fixed = "Alumina reached 99 % density [1]. SPS used 1400 °C [2]."
    llm = FakeLLM(rewrites=[fixed])
    r = run_arm_a(ctx(llm, FakeVerifier()), P1, max_rounds=2)
    assert r.answer == fixed and r.rounds == 1 and r.trace[-1]["failing"] == {}
    rw = next(q for q in llm.requests if isinstance(q["messages"][-1]["content"], str)
              and q["messages"][-1]["content"].startswith(REWRITE_INSTRUCTION))
    assert rw["messages"][-2] == {"role": "assistant", "content": P1}          # previous answer as context
    assert '"SPS used 1600 °C." → number_not_in_source' in rw["messages"][-1]["content"]
    assert "Alumina reached" not in rw["messages"][-1]["content"].split("Statements to fix:")[1]
    assert r.llm_calls == 5 and r.verifier_calls == 2                         # split, verify, rewrite, split, verify


def test_arm_a_reretrieves_on_an_evidence_gap_and_appends_sources():
    calls = []

    def retrieve(q):
        calls.append(q)
        return [HITS[0], {"type": "text", "text": "Zirconia was toughened by yttria.", "source": "Z", "page": 3}]
    # unsupported by its cited source AND by all sources together → evidence gap → re-retrieval
    v = FakeVerifier({"Zirconia was toughened.": "unsupported"}, gap_rules={"Zirconia was toughened.": "unsupported"})
    llm = FakeLLM(rewrites=["Zirconia was toughened by yttria [3]."])
    r = run_arm_a(ctx(llm, v, retrieve_fn=retrieve), "Zirconia was toughened [1].", max_rounds=1)
    assert calls == ["zirconia toughening"] and r.subqueries == calls
    assert len(r.hits) == 3 and r.hits[2]["source"] == "Z"                   # duplicate of hit 1 not re-added
    assert r.answer == "Zirconia was toughened by yttria [3]."


def test_arm_a_stops_at_max_rounds():
    llm = FakeLLM(rewrites=[P1, P1])
    r = run_arm_a(ctx(llm, FakeVerifier()), P1, max_rounds=2)
    assert r.rounds == 2 and len(r.trace) == 3 and r.trace[-1]["failing"]


def _rewrite_thinking(llm):
    return [kw["extra_body"]["chat_template_kwargs"]["enable_thinking"] for kw in llm.requests
            if isinstance(kw["messages"][-1]["content"], str) and kw["messages"][-1]["content"].startswith(REWRITE_INSTRUCTION)]


@pytest.mark.parametrize("mode,answer_thinks,expected", [
    ("escalate", True, [False, True]),        # default: fast first retry, thinking only for leftovers
    ("escalate", False, [False, False]),      # never think when the answer itself did not
    (None, True, [True, True]),               # the verifier's default
    (False, True, [False, False]),
])
def test_arm_a_revision_thinking(mode, answer_thinks, expected):
    llm = FakeLLM(rewrites=[P1, P1])
    run_arm_a(ctx(llm, FakeVerifier(), enable_thinking=answer_thinks, revise_thinking=mode), P1, max_rounds=2)
    assert _rewrite_thinking(llm) == expected


def test_arm_a_default_revision_thinking_is_escalate():
    assert ArmContext(None, "m", "S", "q", [], None).revise_thinking == "escalate"


def test_arm_a_nothing_to_fix_makes_no_rewrite():
    llm = FakeLLM()
    r = run_arm_a(ctx(llm, FakeVerifier()), "Alumina reached 99 % density [1].")
    assert r.rounds == 0 and r.llm_calls == 2 and r.verifier_calls == 1


# ── P0 / P1, dispatch, display ────────────────────────────────────────────────
def test_prompt_arms_and_thinking_flag():
    llm = FakeLLM(answer="Answer [1].")
    r0 = run_strategy("P0", ctx(llm, FakeVerifier(), enable_thinking=False))
    r1 = run_strategy("P1", ctx(llm, FakeVerifier()))
    s0, s1 = llm.requests[0], llm.requests[1]
    assert s0["messages"][0]["content"] == "SYS" and s1["messages"][0]["content"] == "SYS" + P1_ADDENDUM
    assert s0["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    assert s1["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True
    assert r0.answer == r1.answer == "Answer [1]." and r0.llm_calls == 1
    with pytest.raises(ValueError):
        run_strategy("C", ctx(llm, FakeVerifier()))


def test_finalize_for_display():
    from memento.agentic_validation.strategies import ArmResult
    assert finalize_for_display("A [1].", ArmResult("C", "B [1]."), None) == "B [1]."
    out = finalize_for_display("A [1].", None, RuntimeError("verifier down"))
    assert out.startswith("A [1].") and "Verification unavailable" in out and "RuntimeError" in out


# ── how complete a check was: unjudged claims are not "checked" ──
from memento.agentic_validation.claims import Claim  # noqa: E402
from memento.agentic_validation.strategies import ArmResult, check_state  # noqa: E402


def _res(n, unjudged):
    return ArmResult("C", "x", claims=[Claim(f"c{i}", "s", (1,), (0, 1)) for i in range(n)], unjudged=unjudged)


@pytest.mark.parametrize("result, error, late, want", [
    (_res(5, 0), None, False, "complete"),
    (_res(0, 0), None, False, "no_claims"),             # no claim extracted: says nothing about the answer's statements
    (_res(5, 2), None, False, "partial"),
    (_res(5, 5), None, False, "unavailable"),           # claims, none judged
    (_res(5, 0), RuntimeError("judge down"), False, "unavailable"),
    (_res(5, 0), None, True, "unavailable"),            # out of time
    (None, None, False, "unavailable"),
])
def test_check_state(result, error, late, want):
    assert check_state(result, error, late) == want
