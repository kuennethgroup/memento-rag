# tests/evaluation/test_rq2_p3_corpus_level.py
import sys

sys.argv = [sys.argv[0], "--domain", "polymer-engineering"]
from memento.config.cli_domain import apply_domain_arg  # noqa: E402
apply_domain_arg()
from evaluation.RQ2.questions.corpus_level import count_answer, seed_terms, variant_scan  # noqa: E402


def test_count_unit_uses_works_for_how_many():
    reg = {"Diss_A": {"type": "dissertation"}, "Vortrag_A": {"type": "defense", "parent_diss": "Diss_A"},
           "Diss_B": {"type": "dissertation"}}
    assert count_answer(["Diss_A", "Vortrag_A", "Diss_B"], reg, "works") == (2, ["Diss_A", "Diss_B"])
    assert count_answer(["Diss_A", "Vortrag_A", "Diss_B"], reg, "documents") == (3, [])


def test_seed_terms_respect_document_frequency_band():
    df = {"rare": {"a"}, "mid": {"a", "b", "c"}, "hub": {str(i) for i in range(50)}}
    assert seed_terms(df, lo=3, hi=30, seed=0, n=10) == ["mid"]


def test_variant_scan_is_case_insensitive_and_boundary_aware():
    chunks = [{"source": "d1", "text": "Die Kohlenstofffasern wurden geschlichtet."},
              {"source": "d2", "text": "carbon fibre reinforced"},
              {"source": "d3", "text": "carbonfibreless"}]
    hits = variant_scan(chunks, ["Carbon fibre", "Kohlenstofffaser"])
    assert set(hits) == {"d1", "d2"}


def test_corpus_question_states_the_target_language():
    import json as _j
    from evaluation.RQ2.questions.corpus_level import compose_corpus_question

    class L:
        def __init__(self):
            self.chat = self.completions = self

        def create(self, **kw):
            self.messages = kw["messages"]
            msg = type("M", (), {"content": _j.dumps({"question": "Q?", "criterion": "c"})})
            return type("R", (), {"choices": [type("C", (), {"message": msg})]})
    llm = L()
    passages = [{"source": "P2", "text": "Die Proben wurden mit dem Harz und der Faser hergestellt."}]
    out = compose_corpus_question(llm, "m", "Harz", passages, "list")
    assert out["lang"] == "de" and "in German" in llm.messages[1]["content"]
