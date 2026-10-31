# tests/graph/test_rq2_p4_rungs.py
import sys

sys.argv = [sys.argv[0], "--domain", "ceramics"]
from memento.config.cli_domain import apply_domain_arg  # noqa: E402
apply_domain_arg()
import pytest  # noqa: E402

from memento.graph.query import graph_retrieval as gr  # noqa: E402


class Forbidden(Exception):
    pass


class RungStore:
    """Answers the structural reads; entity/relation reads are allowed only when listed."""
    def __init__(self, allow=()):
        self.allow, self.calls = set(allow), []

    def _guard(self, name):
        self.calls.append(name)
        if name not in self.allow:
            raise Forbidden(name)

    def expand_via_references(self, ids, limit=50):
        return [{"chunk_id": "9", "source": "P9", "via_chunk": ids[0]}]

    def citing_neighbors(self, pids):
        return []

    def same_work_neighbors(self, pids):
        return []

    def chunk_entities(self, ids, max_df=None):
        self._guard("chunk_entities"); return {"e1": 1.0}

    def subgraph_chunks(self, seeds, max_hops=2, limit=50, max_df=None):
        self._guard("subgraph_chunks")
        if max_hops != 0:
            raise Forbidden("relation hop in subgraph_chunks")
        return [{"chunk_id": "7", "source": "P7", "score": 1.0, "via": ["e1"]}]

    def cross_paper_chunks(self, pids, eids, limit=50):
        self._guard("cross_paper_chunks"); return []

    def entity_edges(self, *a, **k):
        self._guard("entity_edges"); return []

    def get_meta(self, k):
        self._guard("get_meta"); return None

    def chunk_mentions_for(self, ids):
        self._guard("chunk_mentions_for"); return []

    def close(self):
        pass


def _run(store, **kw):
    return gr.graph_candidates("q", dense_chunk_ids=["1", "2"], dense_sources=["P1"], store=store, enabled=True,
                               seed_mode="dict", use_llm=False, **kw)


def test_g1_reads_no_entity_edges():
    rows = _run(RungStore(), streams="structural")
    assert [r["chunk_id"] for r in rows] == ["9"]


def test_g2_reads_no_relation_edges():
    s = RungStore(allow={"chunk_entities", "subgraph_chunks", "cross_paper_chunks"})
    rows = _run(s, streams="all", traversal="bfs", max_hops=0, first_hop=True)
    assert "7" in [r["chunk_id"] for r in rows] and "entity_edges" not in s.calls


def test_g3_uses_ppr_over_relations():
    s = RungStore(allow={"chunk_entities", "cross_paper_chunks", "entity_edges", "get_meta", "chunk_mentions_for"})
    _run(s, streams="all", traversal="ppr", first_hop=True)
    assert "entity_edges" in s.calls


def test_extra_seeds_reach_the_entity_stream():
    s = RungStore(allow={"chunk_entities", "subgraph_chunks", "cross_paper_chunks"})
    seen = {}
    orig = s.subgraph_chunks

    def spy(seeds, **k):
        seen["seeds"] = list(seeds); return orig(seeds, **k)
    s.subgraph_chunks = spy
    _run(s, streams="all", traversal="bfs", max_hops=0, first_hop=False, extra_seeds={"hfo2": 1.0})
    assert "hfo2" in seen["seeds"]


def test_same_work_stream_is_structural(monkeypatch):
    s = RungStore()
    s.same_work_neighbors = lambda pids: ["P2"]
    monkeypatch.setattr(gr, "_same_work_chunks_by_vector",
                        lambda store, src, qv, qc, limit: [{"chunk_id": "5", "source": "P2", "score": 0.9}])
    rows = _run(s, streams="structural", query_vector=[0.1])
    assert "5" in [r["chunk_id"] for r in rows]


def test_copy_chunks_map_to_parent_chunks(monkeypatch):
    reg = {"C1": {"copy_of": "D1"}, "P1": {}}
    monkeypatch.setattr(gr, "_nearest_in_doc", lambda qc, cid, doc: ("77", 0.97) if doc == "D1" else (None, 0.0))
    assert gr.first_hop_chunk_ids(["10", "11"], {"10": "C1", "11": "P1"}, reg, qdrant_client=object()) == ["77", "11"]
