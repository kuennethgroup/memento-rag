# tests/evaluation/test_rq2_p4_core.py
import sys

sys.argv = [sys.argv[0], "--domain", "ceramics"]
from memento.config.cli_domain import apply_domain_arg  # noqa: E402
apply_domain_arg()
import pytest  # noqa: E402

from evaluation.RQ2.harness import budget, scope  # noqa: E402
from evaluation.RQ2.harness.view import QuestionView, view_of  # noqa: E402
from evaluation.RQ2.questions.common import Question  # noqa: E402


def _h(i, text="word " * 10, src="P"):
    return {"chunk_id": str(i), "source": f"{src}{i}", "page": 0, "type": "text", "text": text}


def test_arms_only_get_a_question_view():
    q = Question("c:multihop:1", "ceramics", "multihop", "Q?", "A", "en", "pilot",
                 meta={"bridge": "ZrB2", "fact_a": "x", "variants": ["ZrB2"]})
    v = view_of(q)
    assert v == QuestionView("c:multihop:1", "Q?", "en")
    for leaked in ("meta", "reference_answer", "gold_pages", "qtype"):
        with pytest.raises(AttributeError):
            getattr(v, leaked)


def test_source_filter_scopes_to_the_universe():
    assert scope.source_filter(None) is None
    f = scope.source_filter({"B", "A"})
    assert f.must[0].key == "source" and f.must[0].match.any == ["A", "B"]


def test_g0_passes_the_filter_into_search(monkeypatch):
    seen = {}

    def fake_search(client, coll, q, **kw):
        seen.update(kw); return [_h(1)], []
    monkeypatch.setattr(scope, "search", fake_search)
    ctx = scope.ArmContext(domain="ceramics", qdrant=None, collection="c", allowed={"P1"}, llm=None, model="m",
                           registry={}, store=None)
    hits, qvec = scope.g0_retrieve(ctx, "q", qvec=[0.0])
    assert seen["query_filter"].must[0].match.any == ["P1"] and seen["rerank_top"] == 20 and hits


def test_merge_keeps_top_and_respects_budget():
    g0 = [_h(i) for i in range(20)]
    stream = [_h(3), _h(21), _h(21), _h(22), _h(23), _h(24), _h(25), _h(26)]   # 3 duplicates G0, 21 twice
    plain, _ = budget.merge_streams(g0, stream, k_slots=5, origin="g2")
    assert [h["chunk_id"] for h in plain] == [str(i) for i in range(15)] + ["21", "22", "23", "24", "25"]
    merged, block = budget.merge_streams(g0, stream, k_slots=5, origin="g2", block="x " * 5000)
    g0_part = [h["chunk_id"] for h in merged if h["origin"] == "g0"]
    assert g0_part == [str(i) for i in range(len(g0_part))]              # trimming drops G0's lowest first
    assert [h["chunk_id"] for h in merged if h["origin"] == "g2"] == ["21", "22", "23", "24", "25"]
    assert budget.hit_tokens(merged) + budget.n_tokens(block) <= budget.hit_tokens(g0)
    assert budget.n_tokens(block) <= int(budget.BLOCK_SHARE * budget.hit_tokens(g0))


def test_merge_with_empty_stream_is_g0():
    g0 = [_h(i) for i in range(20)]
    merged, block = budget.merge_streams(g0, [], k_slots=5, origin="d2")
    assert [h["chunk_id"] for h in merged] == [h["chunk_id"] for h in g0] and block == ""


def test_merge_with_a_short_g0_never_exceeds_its_budget():
    # the stream gets K/20 of G0's list (at least 1) and trimming may also drop stream chunks, so a stream cannot
    # replace G0's context and grow it
    g0 = [_h(i) for i in range(4)]
    stream = [_h(30 + i, text="long " * 200) for i in range(5)]
    merged, block = budget.merge_streams(g0, stream, k_slots=5, origin="g2", block="Linking entities: x")
    assert budget.hit_tokens(merged) + budget.n_tokens(block) <= budget.hit_tokens(g0)
    assert [h["chunk_id"] for h in merged if h["origin"] == "g0"][:2] == ["0", "1"]    # G0's top half survives
    assert sum(h["origin"] == "g2" for h in merged) <= 1


def test_stream_share_is_proportional_to_g0():
    g0 = [_h(i) for i in range(8)]
    merged, _ = budget.merge_streams(g0, [_h(30 + i) for i in range(5)], k_slots=5, origin="d2")
    assert [h["chunk_id"] for h in merged] == [str(i) for i in range(6)] + ["30", "31"]
