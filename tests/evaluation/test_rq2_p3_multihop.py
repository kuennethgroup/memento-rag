# tests/evaluation/test_rq2_p3_multihop.py
import json
import sys

sys.argv = [sys.argv[0], "--domain", "ceramics"]
from memento.config.cli_domain import apply_domain_arg  # noqa: E402
apply_domain_arg()
from evaluation.RQ2.questions.multihop import anchor_pairs, bridge_leaks, compose  # noqa: E402


def _c(pid, src, text, section="Results"):
    return {"point_id": pid, "source": src, "page": 1, "type": "text", "section": section, "text": text}


REG = {"D1": {"type": "dissertation", "student": "S, A"}, "T1": {"type": "defense", "parent_diss": "D1"},
       "P2": {"type": "paper"}, "P3": {"type": "paper"}}


def test_pairs_exclude_same_work_and_same_document():
    chunks = [_c("1", "D1", "TGMDA was cured at 180 C in D1."), _c("2", "D1", "TGMDA again in D1."),
              _c("3", "T1", "TGMDA in the talk."), _c("4", "P2", "TGMDA networks were toughened.")]
    df = {"TGMDA": {"D1", "T1", "P2"}}
    pairs = anchor_pairs(chunks, df, REG, n_research_docs=400, seed=0, max_pairs=10)
    assert pairs and all({p["a"]["source"], p["b"]["source"]} not in ({"D1", "T1"}, {"D1"}, {"T1"}) for p in pairs)
    assert all(p["a"]["source"] != p["b"]["source"] for p in pairs)


def test_hub_terms_and_singletons_are_not_bridges():
    chunks = [_c(str(i), f"P{i % 3 + 2}" if i % 3 else "P2", "Temperature and ZrB2") for i in range(6)]
    df = {"Temperature": {f"X{i}" for i in range(50)}, "ZrB2": {"P2"}}
    assert anchor_pairs(chunks, df, {f"P{i}": {"type": "paper"} for i in range(2, 5)},
                        n_research_docs=400, seed=0, max_pairs=10) == []


def test_bridge_leak_detects_variants():
    assert bridge_leaks("How was the carbon fibre sized?", "Carbon-Fibre", [])
    assert bridge_leaks("Welche Kohlenstofffasern wurden verwendet?", "carbon fibre", ["Kohlenstofffasern"])
    assert not bridge_leaks("Which reinforcement was sized with epoxy?", "carbon fibre", ["Kohlenstofffaser"])


class FakeLLM:
    def __init__(self, payload):
        self.payload = payload
        self.chat = self
        self.completions = self

    def create(self, **kw):
        msg = type("M", (), {"content": json.dumps(self.payload)})
        return type("R", (), {"choices": [type("C", (), {"message": msg})]})


def test_compose_returns_none_when_fields_missing():
    pair = {"bridge": "TGMDA", "a": _c("1", "P2", "a"), "b": _c("2", "P3", "b")}
    assert compose(FakeLLM({"question": "Q?"}), "m", pair) is None
    ok = compose(FakeLLM({"question": "Q?", "answer": "A", "fact_a": "fa", "fact_b": "fb", "bridge_forms": ["x"]}), "m", pair)
    assert ok["question"] == "Q?" and ok["bridge_forms"] == ["x"]


class RecLLM(FakeLLM):
    def create(self, **kw):
        self.messages = kw["messages"]
        return super().create(**kw)


def test_compose_states_the_target_language_from_the_passages():
    # questions are written in the passages' language
    ok = {"question": "Q?", "answer": "A", "fact_a": "fa", "fact_b": "fb"}
    en = {"bridge": "TGMDA", "a": _c("1", "P2", "The resin was cured with the hardener."),
          "b": _c("2", "P3", "The network was toughened by the particles.")}
    de = {"bridge": "TGMDA", "a": _c("1", "P2", "Das Harz wurde mit dem Härter bei der Temperatur gehärtet."),
          "b": _c("2", "P3", "Die Netzwerke werden mit den Partikeln zäh modifiziert.")}
    llm = RecLLM(ok)
    assert compose(llm, "m", en)["lang"] == "en" and "in English" in llm.messages[1]["content"]
    assert compose(llm, "m", de)["lang"] == "de" and "in German" in llm.messages[1]["content"]


def test_question_problem_flags_wrong_language_and_passage_references():
    from evaluation.RQ2.questions.common import question_problem
    assert question_problem("Welches Pulver wird in der Synthese von Diboriden verwendet?", "en") == "language"
    assert question_problem("Which metal named in passage A raises the hardness?", "en") == "meta_reference"
    assert question_problem("Welches in Passage A genannte Metall erhöht die Härte?", "de") == "meta_reference"
    assert question_problem("What grain size did the sintered (Hf-Ta-Zr-Nb)C reach?", "en") is None
    assert question_problem("Welche Korngröße erreichte das gesinterte Karbid?", "de") is None
