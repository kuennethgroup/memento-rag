# tests/evaluation/test_rq2_p4_pool.py
import sys

sys.argv = [sys.argv[0], "--domain", "ceramics"]
from memento.config.cli_domain import apply_domain_arg  # noqa: E402
apply_domain_arg()
from evaluation.RQ2.harness import pool  # noqa: E402
from evaluation.RQ2.harness.arms import ArmOut  # noqa: E402
from evaluation.RQ2.questions.common import Question  # noqa: E402


def _q():
    return Question("q", "ceramics", "multihop", "Q?", "A", "en", "pilot",
                    gold_pages=[{"source": "A", "page": 1, "role": "A", "fact": "a"},
                                {"source": "B", "page": 2, "role": "B", "fact": "b"}],
                    meta={"fact_a": "fa", "fact_b": "fb"})


def test_multihop_pool_passes_known_and_adds_new(monkeypatch):
    seen = {}

    def fake(client, model, facts, lists, known=None):
        seen["known"], seen["lists"] = known, lists
        return {"gold": known["gold"] + [{"source": "C", "page": 0, "fact": "b", "found_by": ["q|g2|multihop"]}],
                "unparsed": [], "n_candidates": 3, "judged": []}
    monkeypatch.setattr(pool, "pool_and_judge", fake)
    out = ArmOut("g2", "multihop", "q", [{"source": "C", "page": 0, "text": "t"}])
    q = pool.extend_multihop_gold(None, "m", _q(), [out])
    assert {(g["source"], g["page"]) for g in seen["known"]["gold"]} == {("A", 1), ("B", 2)}
    assert [g["role"] for g in q.gold_pages] == ["A", "B", "pooled"] and q.gold_pages[2]["fact"] == "b"


def test_corpus_pool_screens_only_new_docs(monkeypatch):
    screened = {}

    def fake_screen(client, model, criterion, docs):
        screened.update(docs); return {"gold_docs": ["N1"], "unparsed_docs": [], "n_screened": len(docs)}
    monkeypatch.setattr(pool, "screen", fake_screen)
    q = Question("c", "ceramics", "corpus_count", "How many?", "", "en", "pilot", gold_docs=["G1"],
                 meta={"criterion": "c", "unit": "documents"})
    outs = [ArmOut("lx", "corpus", "c", [{"source": "N1", "page": 0, "text": "t"}], docs=["G1", "N1", "N2"])]
    q2 = pool.extend_corpus_gold(None, "m", q, outs,
                                 fetch=lambda d: [{"source": d, "page": 0, "text": "fetched"}])
    assert set(screened) == {"N1", "N2"} and q2.gold_docs == ["G1", "N1"]
