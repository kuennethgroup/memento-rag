# tests/evaluation/test_rq2_p4_metrics.py
from evaluation.RQ2.harness.metrics import control_hits, doc_prf, fact_coverage, reaches_b

GOLD = [{"source": "A", "page": 1, "fact": "a"}, {"source": "B", "page": 2, "fact": "b"},
        {"source": "C", "page": 0, "fact": "a+b"}]


def test_fact_coverage_counts_facts_not_pages():
    assert fact_coverage([{"source": "C", "page": 0}], GOLD) == {"a": True, "b": True, "complete": True,
                                                                  "fact_recall": 1.0, "page_recall": 1 / 3}
    got = fact_coverage([{"source": "A", "page": 1}, {"source": "X", "page": 0}], GOLD)
    assert got["complete"] is False and got["fact_recall"] == 0.5


def test_reaches_b_needs_the_arms_own_chunk():
    ctx = [{"source": "B", "page": 2, "origin": "g0"}, {"source": "C", "page": 0, "origin": "g2"}]
    assert reaches_b(ctx, GOLD, "g2") and not reaches_b(ctx[:1], GOLD, "g2")


def test_doc_prf_by_unit():
    reg = {"T": {"parent_diss": "D"}, "D": {}}
    assert doc_prf(["T", "X"], ["D"], "works", reg)["r"] == 1.0
    assert doc_prf(["T", "X"], ["D"], "documents", reg) == {"p": 0.0, "r": 0.0, "f1": 0.0}


def test_control_hits_use_whole_context():
    assert control_hits([{"source": "X", "page": 0}, {"source": "A", "page": 1}], GOLD[:1]) == {"hit": True,
                                                                                                   "recall": 1.0}


def test_stream_reaches_b_counts_the_graph_stream_before_the_merge():
    # "reaches B" is measured on the ranked stream, before merging (whether the traversal reaches document B);
    # merging would drop B chunks as duplicates when G0 already has B
    from evaluation.RQ2.harness.metrics import stream_reaches_b
    stream = [["X", 0], ["B", 2], ["Y", 1]]
    assert stream_reaches_b(stream, GOLD, k=2) and not stream_reaches_b(stream, GOLD, k=1)
    assert stream_reaches_b([["C", 0]], GOLD, k=5)          # C states fact b as well
