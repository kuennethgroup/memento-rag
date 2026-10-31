# tests/evaluation/test_rq2_p4_answer.py
import sys

sys.argv = [sys.argv[0], "--domain", "ceramics"]
from memento.config.cli_domain import apply_domain_arg  # noqa: E402
apply_domain_arg()
from evaluation.RQ2.harness.answer import answers_for, build_messages  # noqa: E402
from evaluation.RQ2.harness.arms import ArmOut  # noqa: E402


def _out(arm, design="multihop", block=""):
    return ArmOut(arm, design, "q1", [{"source": "P1", "page": 2, "section": "Results", "type": "text",
                                       "text": "HfO2 was sintered."}], block, ["P1"])


def test_messages_follow_the_pipe_template():
    from memento.config.settings import DOMAIN
    m = build_messages("Q?", _out("g2", block="Linking entities: HfO2"))
    assert m[0] == {"role": "system", "content": DOMAIN["system_prompt"]}
    user = m[1]["content"] if isinstance(m[1]["content"], str) else m[1]["content"][0]["text"]
    assert user.startswith("Sources:\n[1] P1, page 3, Section: Results\nHfO2 was sintered.")
    assert "Linking entities: HfO2" in user and user.endswith("Question: Q?")


def test_routed_to_g0_reuses_g0_answer():
    outs = {o.key: o for o in (_out("g0", "rag"), _out("g2"))}
    table = [{"qid": "q1", "policy": "real", "arm": "g2", "key": "q1|g0|rag"},
             {"qid": "q1", "policy": "true", "arm": "g2", "key": "q1|g2|multihop"},
             {"qid": "q1", "policy": "true", "arm": "g0", "key": "q1|g0|rag"}]
    calls = []

    def gen(o):
        calls.append(o.key); return {"answer": f"ans {o.key}"}
    ans = answers_for(table, outs, gen)
    assert sorted(calls) == ["q1|g0|rag", "q1|g2|multihop"]
    assert ans["q1|g0|rag"]["answer"] == "ans q1|g0|rag"


def test_generator_settings_match_the_pipe_valves():
    import re
    from evaluation.RQ2.harness import answer
    src = open("openwebui/functions/query_pipe.py").read()
    assert float(re.search(r"TEMPERATURE:\s+float = ([\d.]+)", src).group(1)) == answer.TEMPERATURE
    assert int(re.search(r"MAX_TOKENS:\s+int\s+= (\d+)", src).group(1)) == answer.MAX_TOKENS
    assert (re.search(r"ENABLE_THINKING: bool = (\w+)", src).group(1) == "True") == answer.ENABLE_THINKING
    assert int(re.search(r'MAX_IMAGES_TO_LLM", "(\d+)"', src).group(1)) == answer.MAX_IMAGES_TO_LLM
