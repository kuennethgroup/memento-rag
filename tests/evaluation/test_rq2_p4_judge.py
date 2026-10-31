# tests/evaluation/test_rq2_p4_judge.py
import json

from evaluation.RQ2.harness.judge import judge_absolute, judge_pair


class Seq:
    """Returns the queued JSON payloads in order."""
    def __init__(self, *payloads):
        self.payloads, self.n, self.chat, self.completions = list(payloads), 0, self, self

    def create(self, **kw):
        self.n += 1
        p = self.payloads.pop(0)
        msg = type("M", (), {"content": p if isinstance(p, str) else json.dumps(p)})
        return type("R", (), {"choices": [type("C", (), {"message": msg})]})


def test_absolute_scores_and_none():
    assert judge_absolute(Seq({"score": 1}), "m", "Q", "R", "A") == 1.0
    assert judge_absolute(Seq({"score": 0.5}), "m", "Q", "R", "A") == 0.5
    assert judge_absolute(Seq("garbage"), "m", "Q", "R", "A") is None
    assert judge_absolute(Seq(), "m", "Q", "", "A") is None


def test_pairwise_needs_both_orders_to_agree():
    assert judge_pair(Seq({"winner": "A"}, {"winner": "B"}), "m", "Q", "R", "arm", "g0") == "win"
    assert judge_pair(Seq({"winner": "B"}, {"winner": "A"}), "m", "Q", "R", "arm", "g0") == "loss"
    assert judge_pair(Seq({"winner": "A"}, {"winner": "A"}), "m", "Q", "R", "arm", "g0") == "tie"
    assert judge_pair(Seq({"winner": "A"}, "garbage"), "m", "Q", "R", "arm", "g0") is None
    s = Seq()
    assert judge_pair(s, "m", "Q", "R", "same", "same") == "tie" and s.n == 0


def test_citations_are_mapped_to_documents_for_corpus_judging():
    # Pilot: corpus references are document ids, answers cite [N] (production prompt forbids naming
    # documents), so every list answer scored 0. The judge gets the [N] → document map.
    from evaluation.RQ2.harness.judge import cited_docs, with_citations
    ctx = [{"source": "D1"}, {"source": "D2"}, {"source": "D1"}, {"source": "D3"}]
    ans = "AFM was used [1], [2][3]; see also [9]."
    assert cited_docs(ans, ctx) == ["D1", "D2"]
    assert with_citations(ans, ctx).endswith("Documents cited: [1] D1; [2] D2; [3] D1")
    assert with_citations("no citations", ctx) == "no citations"


def test_cross_family_skips_failed_cloud_calls():
    # Pilot re-judge: one Academic Cloud 500 aborted the whole judge stage and lost every local judgement.
    from evaluation.RQ2.harness.judge import cross_family
    calls = []

    def fn(cloud, model, q, ref, ans):
        calls.append(q)
        if q == "boom":
            raise RuntimeError("Error code: 500")
        return 1.0
    items = [{"question": q, "reference": "r", "answer": "a", "local": 1.0} for q in ("q1", "boom", "q3")]
    out = cross_family(None, "m", items, max_calls=10, min_gap_s=0.0, judge_fn=fn)
    assert out["n"] == 2 and out["errors"] == 1 and out["agreement"] == 1.0 and len(calls) == 3
