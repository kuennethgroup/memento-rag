# tests/evaluation/test_rq2_p3_necessity.py
import json
import sys

sys.argv = [sys.argv[0], "--domain", "ceramics"]
from memento.config.cli_domain import apply_domain_arg  # noqa: E402
apply_domain_arg()
from evaluation.RQ2.questions.multihop import necessity  # noqa: E402


class ScriptLLM:
    """Answers: 'ANS:<n>' for answer calls; for judge calls returns the scripted label."""
    def __init__(self, labels):
        self.labels = list(labels)
        self.chat = self
        self.completions = self

    def create(self, **kw):
        sys_msg = kw["messages"][0]["content"]
        content = json.dumps({"verdict": self.labels.pop(0)}) if "grade" in sys_msg.lower() else "some answer"
        msg = type("M", (), {"content": content})
        return type("R", (), {"choices": [type("C", (), {"message": msg})]})


Q = {"question": "Q?", "answer": "A"}


def test_necessity_requires_both_and_neither_alone():
    assert necessity(ScriptLLM(["correct", "incorrect", "incorrect"]), "m", Q, "a", "b")["passes"]
    assert not necessity(ScriptLLM(["correct", "correct", "incorrect"]), "m", Q, "a", "b")["passes"]   # A alone suffices
    assert not necessity(ScriptLLM(["correct", "incorrect", "correct"]), "m", Q, "a", "b")["passes"]   # B alone suffices
    assert not necessity(ScriptLLM(["incorrect", "incorrect", "incorrect"]), "m", Q, "a", "b")["passes"]


def test_unparsed_verdict_fails_closed():
    assert not necessity(ScriptLLM(["correct", "maybe", "incorrect"]), "m", Q, "a", "b")["passes"]
