# tests/graph/test_rq2_p2_ppr_cache.py
import random

import pytest

from memento.graph.ppr import personalized_pagerank
from memento.graph.ppr_cache import build_entity_graph, load_or_build, ppr_sparse


def test_sparse_ppr_matches_reference():
    rng = random.Random(0)
    nodes = [f"e{i}" for i in range(40)]
    edges = [(rng.choice(nodes), rng.choice(nodes)) for _ in range(90)]
    seeds = {"e1": 1.0, "e7": 0.5, "lonely": 0.3}              # a seed with no edges
    ref = personalized_pagerank(edges, seeds, damping=0.5)
    got = ppr_sparse(build_entity_graph(edges), seeds, damping=0.5)
    for k, v in ref.items():
        assert got.get(k, 0.0) == pytest.approx(v, abs=1e-6)


class FakeStore:
    def __init__(self):
        self.build_id, self.calls, self.edges = "b1", 0, [("a", "b")]

    def get_meta(self, k):
        return self.build_id

    def entity_edges(self, types="RELATION|SIMILAR", min_weight=0):
        self.calls += 1
        return list(self.edges)


def test_ppr_cache_rebuilds_on_new_build_id(tmp_path):
    s = FakeStore()
    load_or_build(s, tmp_path)
    load_or_build(s, tmp_path)
    assert s.calls == 1                                           # second call served from cache
    s.build_id, s.edges = "b2", [("a", "b"), ("b", "c")]
    g = load_or_build(s, tmp_path)
    assert s.calls == 2 and "c" in g.index


# ── stale-cache and cross-corpus fixes ──
class _Rec:
    def __init__(self, n):
        self.records = [{"n": n, "v": None}]


class _Driver:
    """Records every query; MATCH-based edge writes report all rows written."""
    def __init__(self):
        self.queries = []

    def execute_query(self, q, parameters_=None, database_=None):
        self.queries.append((q, parameters_ or {}))
        return _Rec(len((parameters_ or {}).get("rows", [])))

    def close(self):
        pass


def _meta_writes(d):
    return [p["v"] for q, p in d.queries if q.startswith("MERGE (m:Meta") and p.get("k") == "build_id"]


def test_every_graph_write_changes_build_id():
    from memento.graph.store import GraphEdge, Neo4jStore
    d = _Driver()
    s = Neo4jStore(driver=d)
    s.upsert_edges([GraphEdge(start_id="a", start_label="Entity", end_id="b", end_label="Entity",
                              type="SIMILAR", props={})])
    s.clear_paper("p1", orphan_cleanup=False)
    s.delete_orphans()
    ids = _meta_writes(d)
    assert len(ids) == 3 and len(set(ids)) == 3                   # a fresh, unique id per write


def test_apply_merges_changes_build_id():
    from memento.graph.build.entity_merge import apply_merges

    class S:
        def __init__(self):
            self.marked = 0

        def _run(self, q, **p):
            class R:
                records = [{"n_merged": 1, "mentions_moved": 2, "relations_moved": 0}]
            return R()

        def mark_changed(self):
            self.marked += 1
    s = S()
    apply_merges(s, {"x": "y"})
    assert s.marked == 1


def test_cache_key_includes_store_identity(tmp_path):
    a, b = FakeStore(), FakeStore()
    a.identity, b.identity = "bolt://localhost:7690/neo4j", "bolt://localhost:7691/neo4j"
    b.edges = [("x", "y")]
    load_or_build(a, tmp_path)
    g = load_or_build(b, tmp_path)                                # same build_id, other corpus
    assert b.calls == 1 and "x" in g.index


def test_cache_write_leaves_no_temp_files(tmp_path):
    load_or_build(FakeStore(), tmp_path)
    assert sorted(p.suffix for p in tmp_path.iterdir()) == [".json", ".npz"]
