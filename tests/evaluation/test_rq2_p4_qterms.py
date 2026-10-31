# tests/evaluation/test_rq2_p4_qterms.py
import json
import sys

sys.argv = [sys.argv[0], "--domain", "ceramics"]
from memento.config.cli_domain import apply_domain_arg  # noqa: E402
apply_domain_arg()
from evaluation.RQ2.harness.qterms import query_terms, seed_ids_from_terms  # noqa: E402
from evaluation.RQ2.harness.view import QuestionView  # noqa: E402


class LLM:
    def __init__(self, payload):
        self.payload, self.calls = payload, 0
        self.chat = self.completions = self

    def create(self, **kw):
        self.calls += 1
        self.user = kw["messages"][1]["content"]
        msg = type("M", (), {"content": json.dumps(self.payload)})
        return type("R", (), {"choices": [type("C", (), {"message": msg})]})


def test_query_terms_use_only_the_question_and_are_cached():
    llm = LLM({"terms": [{"term": "hafnia", "variants": ["HfO2", "Hafniumoxid"]}], "combine": "or", "relation": None})
    v = QuestionView("q1", "How many works report on HfO2?", "en")
    cache = {}
    t1, t2 = query_terms(llm, "m", v, cache), query_terms(llm, "m", v, cache)
    assert t1 == t2 and llm.calls == 1 and llm.user.strip().endswith("How many works report on HfO2?")
    assert t1["terms"][0]["variants"] == ["HfO2", "Hafniumoxid", "hafnia"] or "hafnia" in t1["terms"][0]["variants"]


def test_bad_answer_gives_empty_terms():
    t = query_terms(LLM("nonsense"), "m", QuestionView("q2", "Q?", "en"), {})
    assert t == {"terms": [], "combine": "or", "relation": None}


class Store:
    def existing_entity_ids(self, ids):
        return {i for i in ids if i in {"hfo2", "exafs"}}


def test_seed_ids_use_the_build_normalisation(monkeypatch):
    import memento.graph as graph
    monkeypatch.setattr(graph, "canonicalize", lambda name: None)       # not a registry term: slug path
    terms = {"terms": [{"term": "HfO₂", "variants": ["HfO2", "hafnia"]}, {"term": "EXAFS", "variants": []}],
             "combine": "or", "relation": None}
    assert seed_ids_from_terms(Store(), terms) == {"hfo2": 1.0, "exafs": 1.0}


class SeqLLM:
    def __init__(self, *payloads):
        self.payloads, self.calls = list(payloads), 0
        self.chat = self.completions = self

    def create(self, **kw):
        self.calls += 1
        p = self.payloads.pop(0)
        msg = type("M", (), {"content": p if isinstance(p, str) else json.dumps(p)})
        return type("R", (), {"choices": [type("C", (), {"message": msg})]})


def test_query_terms_retry_once_on_a_bad_answer():
    # Smoke run: ceramics:corpus_list:1 got empty terms from one unparsable answer.
    good = {"terms": [{"term": "EXAFS", "variants": ["EXAFS"]}], "combine": "or", "relation": None}
    llm = SeqLLM('{"terms": [{"term": "EXAFS", "variants": ["EX', good)
    t = query_terms(llm, "m", QuestionView("q3", "Which documents use EXAFS?", "en"), {})
    assert llm.calls == 2 and t["terms"][0]["term"] == "EXAFS"


def test_at_most_three_specific_terms_are_kept():
    # Smoke v2: ~6 terms incl. broad context words; the soft AND then kept 1 of 3 gold documents.
    from evaluation.RQ2.harness.qterms import TERMS_SYSTEM
    many = {"terms": [{"term": t, "variants": [t]} for t in ("EXAFS", "homogeneity", "cations", "HEM", "local")],
            "combine": "and", "relation": None}
    t = query_terms(LLM(many), "m", QuestionView("q4", "Which documents used EXAFS?", "en"), {})
    assert [x["term"] for x in t["terms"]] == ["EXAFS", "homogeneity", "cations"]
    assert "1 to 3" in TERMS_SYSTEM
