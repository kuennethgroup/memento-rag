# tests/evaluation/test_rq2_p4_arms.py
import json
import sys

sys.argv = [sys.argv[0], "--domain", "ceramics"]
from memento.config.cli_domain import apply_domain_arg  # noqa: E402
apply_domain_arg()
from evaluation.RQ2.harness import arms  # noqa: E402
from evaluation.RQ2.harness.scope import ArmContext  # noqa: E402
from evaluation.RQ2.harness.view import QuestionView  # noqa: E402


def _h(i, src=None):
    return {"chunk_id": str(i), "source": src or f"P{i}", "page": 0, "type": "text", "text": "w " * 20}


class LLM:
    def __init__(self, text):
        self.text, self.chat, self.completions = text, self, self

    def create(self, **kw):
        msg = type("M", (), {"content": self.text})
        return type("R", (), {"choices": [type("C", (), {"message": msg})]})


class Store:
    def papers_for_entities(self, eids, need=1):
        return [{"src": "P1", "hits": 3}, {"src": "P2", "hits": 1}, {"src": "OUT", "hits": 9}]

    def existing_entity_ids(self, ids):
        return set(ids)                      # every candidate id exists in this fake graph

    def relations_among(self, ids, limit=10):
        return []

    def chunk_entities(self, ids, max_df=None):
        return {}

    def entities_by_ids(self, ids):
        return {}


def _ctx(**kw):
    base = dict(domain="ceramics", qdrant=None, collection="c", allowed={"P1", "P2", "P3"}, llm=LLM("follow-up"),
                model="m", registry={}, store=Store())
    return ArmContext(**(base | kw))


def _patch(monkeypatch, g0_hits, second=None, graph_ids=()):
    calls = []

    def fake_g0(ctx, text, qvec=None):
        calls.append(text); return (g0_hits if len(calls) == 1 else (second or [])), [0.0]
    monkeypatch.setattr(arms, "g0_retrieve", fake_g0)
    monkeypatch.setattr(arms, "rank_stream", lambda ctx, q, hits: list(hits))
    monkeypatch.setattr(arms, "graph_candidate_ids", lambda *a, **k: list(graph_ids))
    monkeypatch.setattr(arms, "payload_hits", lambda ctx, ids: [_h(i) for i in ids])
    monkeypatch.setattr(arms, "best_chunk_per_doc", lambda ctx, docs, qv: [_h(d[1:], d) for d in docs])
    return calls


def test_g0_and_d2_share_the_merge_rule(monkeypatch):
    calls = _patch(monkeypatch, [_h(i) for i in range(20)], second=[_h(30), _h(31)])
    ctx, v = _ctx(), QuestionView("q", "Q?", "en")
    g0 = arms.arm_g0(ctx, v)
    d2 = arms.arm_d2(ctx, v, g0)
    assert calls == ["Q?", "follow-up"]
    assert [h["chunk_id"] for h in d2.context[-2:]] == ["30", "31"] and len(d2.context) == 20


def test_graph_arm_fills_k_slots(monkeypatch):
    _patch(monkeypatch, [_h(i) for i in range(20)], graph_ids=[40, 41, 42, 43, 44, 45, 46])
    ctx, v = _ctx(k_slots=5), QuestionView("q", "Q?", "en")
    g0 = arms.arm_g0(ctx, v)
    g1 = arms.arm_graph(ctx, v, g0, "g1", {"terms": [], "combine": "or", "relation": None})
    assert [h["origin"] for h in g1.context].count("g1") == 5 and g1.block == ""


def test_corpus_lookup_scopes_and_orders_documents(monkeypatch):
    _patch(monkeypatch, [_h(i) for i in range(20)])
    ctx, v = _ctx(), QuestionView("q", "How many works report on HfO2?", "en")
    g0 = arms.arm_g0(ctx, v)
    out = arms.corpus_lookup(ctx, v, g0, "g2", {"terms": [{"term": "HfO2", "variants": ["HfO2"]}],
                                                "combine": "or", "relation": None})
    assert out.docs == ["P1", "P2"] and out.block.startswith("Documents found (2)")


def test_corpus_lookup_abstains_to_g0(monkeypatch):
    _patch(monkeypatch, [_h(i) for i in range(20)])
    ctx, v = _ctx(), QuestionView("q", "Q?", "en")
    g0 = arms.arm_g0(ctx, v)
    out = arms.corpus_lookup(ctx, v, g0, "g2", {"terms": [], "combine": "or", "relation": None})
    assert out.info["abstained"] and out.context == g0.context


def test_lx_combines_terms_with_and(monkeypatch):
    _patch(monkeypatch, [_h(i) for i in range(20)])
    chunks = [{"source": "P1", "text": "HfO2 and EXAFS"}, {"source": "P2", "text": "HfO2 only"},
              {"source": "P3", "text": "EXAFS only"}]
    ctx, v = _ctx(), QuestionView("q", "Q?", "en")
    g0 = arms.arm_g0(ctx, v)
    out = arms.corpus_lx(ctx, v, g0, {"terms": [{"term": "HfO2", "variants": ["HfO2"]},
                                                {"term": "EXAFS", "variants": ["EXAFS"]}],
                                      "combine": "and", "relation": None}, chunks)
    assert out.docs == ["P1"]


def test_and_relaxes_to_documents_matching_most_terms(monkeypatch):
    # an AND over terms that matches no document falls back to the documents matching the most terms, in both the graph
    # lookup and LX (same rule)
    _patch(monkeypatch, [_h(i) for i in range(20)])

    class S(Store):
        def papers_for_entities(self, eids, need=1):        # one document per term: no doc has both
            return [{"src": "P1", "hits": 2}] if "hfo2" in eids else [{"src": "P2", "hits": 1}]
    ctx, v = _ctx(store=S()), QuestionView("q", "Q?", "en")
    g0 = arms.arm_g0(ctx, v)
    terms = {"terms": [{"term": "HfO2", "variants": ["HfO2"]}, {"term": "EXAFS", "variants": ["EXAFS"]}],
             "combine": "and", "relation": None}
    out = arms.corpus_lookup(ctx, v, g0, "g2", terms)
    assert out.docs == ["P1", "P2"] and out.info["and_relaxed_to"] == 1
    chunks = [{"source": "P2", "text": "HfO2 only"}, {"source": "P3", "text": "EXAFS only"}]
    lx = arms.corpus_lx(ctx, v, g0, terms, chunks)
    assert sorted(lx.docs) == ["P2", "P3"] and lx.info["and_relaxed_to"] == 1


def test_and_counts_terms_not_variants(monkeypatch):
    # Smoke3: variants of one term became separate seeds, so AND over 3 terms required 5 seeds.
    import memento.graph as graph
    monkeypatch.setattr(graph, "canonicalize", lambda name: None)
    _patch(monkeypatch, [_h(i) for i in range(20)])
    where = {"exafs": ["P1"], "x_ray_absorption_fine_structure": ["P2"], "hfo2": ["P1", "P2", "P3"]}

    class S(Store):
        def papers_for_entities(self, eids, need=1):
            hits = {}
            for e in eids:
                for d in where.get(e, []):
                    hits[d] = hits.get(d, 0) + 1
            return [{"src": d, "hits": h} for d, h in sorted(hits.items())]
    ctx, v = _ctx(store=S()), QuestionView("q", "Q?", "en")
    g0 = arms.arm_g0(ctx, v)
    terms = {"terms": [{"term": "EXAFS", "variants": ["EXAFS", "X-ray absorption fine structure"]},
                       {"term": "HfO2", "variants": ["HfO2"]}], "combine": "and", "relation": None}
    out = arms.corpus_lookup(ctx, v, g0, "g2", terms)
    assert sorted(out.docs) == ["P1", "P2"] and "and_relaxed_to" not in out.info


def test_lx_ranks_by_terms_matched_first(monkeypatch):
    _patch(monkeypatch, [_h(i) for i in range(20)])
    chunks = [{"source": "P1", "text": "HfO2"}, {"source": "P1", "text": "HfO2 again"}, {"source": "P1", "text": "HfO2 3"},
              {"source": "P2", "text": "HfO2 and EXAFS"}]
    ctx, v = _ctx(), QuestionView("q", "Q?", "en")
    out = arms.corpus_lx(ctx, v, arms.arm_g0(ctx, v), {"terms": [{"term": "HfO2", "variants": ["HfO2"]},
                                                               {"term": "EXAFS", "variants": ["EXAFS"]}],
                                                     "combine": "or", "relation": None}, chunks)
    assert out.docs == ["P2", "P1"]


def test_graph_arm_records_its_ranked_stream(monkeypatch):
    _patch(monkeypatch, [_h(i) for i in range(20)], graph_ids=[3, 40, 41])
    ctx, v = _ctx(k_slots=5), QuestionView("q", "Q?", "en")
    g1 = arms.arm_graph(ctx, v, arms.arm_g0(ctx, v), "g1", {"terms": [], "combine": "or", "relation": None})
    assert g1.info["stream_pages"] == [["P3", 0], ["P40", 0], ["P41", 0]]     # incl. the chunk G0 already had


def test_graph_stream_is_not_pre_filtered_by_g0(monkeypatch):
    _patch(monkeypatch, [_h(i) for i in range(20)])
    seen = {}

    def spy(*a, **k):
        seen.update(k); return []
    monkeypatch.setattr(arms, "graph_candidate_ids", spy)
    ctx, v = _ctx(), QuestionView("q", "Q?", "en")
    arms.arm_graph(ctx, v, arms.arm_g0(ctx, v), "g2", {"terms": [], "combine": "or", "relation": None})
    assert not seen.get("exclude_ids")          # the merge drops duplicates; reach is measured on the full stream


def test_without_docs_removes_bs_document_from_the_first_hop():
    # the bridge check gives the graph arm G0's context minus B's document(s), so B is not re-found trivially from G0's
    # first hop
    g0 = arms.ArmOut("g0", "rag", "q", [_h(1, "A"), _h(2, "B"), _h(3, "C")], "", ["A", "B", "C"], {"qvec": [0.1]})
    cut = arms.without_docs(g0, {"B"})
    assert [h["source"] for h in cut.context] == ["A", "C"] and cut.docs == ["A", "C"] and cut.info["qvec"] == [0.1]
    assert [h["source"] for h in g0.context] == ["A", "B", "C"]


def test_followup_falls_back_to_the_question_on_blank_output(monkeypatch):
    _patch(monkeypatch, [_h(i) for i in range(20)], second=[_h(30)])
    ctx, v = _ctx(llm=LLM("  \n ")), QuestionView("q", "Q?", "en")
    d2 = arms.arm_d2(ctx, v, arms.arm_g0(ctx, v))
    assert d2.info["followup"] == "Q?"
