# tests/graph/test_rq2_p2_merge.py
import math
import os

import pytest

from memento.graph.build.entity_merge import plan_merges


def _v(deg):
    r = math.radians(deg)
    return [math.cos(r), math.sin(r)]


ENTS = [
    {"id": "alumina", "name": "Alumina", "type": "Material", "canonical": True, "n_mentions": 90},
    {"id": "alpha_alumina", "name": "alpha-Al2O3", "type": "Material", "canonical": True, "n_mentions": 20},
    {"id": "aluminium_oxide_powder", "name": "aluminium oxide powder", "type": "Material", "canonical": False, "n_mentions": 3},
    {"id": "al2o3_ceramic", "name": "Al2O3 ceramic", "type": "Material", "canonical": False, "n_mentions": 5},
    {"id": "alumina_sintering", "name": "alumina sintering", "type": "Process", "canonical": False, "n_mentions": 4},
]
VEC = [_v(0), _v(1), _v(2), _v(3), _v(1.5)]          # all within cos > 0.999


def test_plan_merges_never_joins_two_canonicals_or_types():
    m = plan_merges(ENTS, VEC, threshold=0.99)["mapping"]
    assert "alumina" not in m and "alpha_alumina" not in m               # canonicals stay separate
    assert m.get("aluminium_oxide_powder") in {"alumina", "alpha_alumina"}
    assert m.get("al2o3_ceramic") in {"alumina", "alpha_alumina"}
    assert "alumina_sintering" not in m                                   # different type


def test_representative_without_canonical_is_most_mentioned():
    ents = [{"id": "a", "name": "PP foam", "type": "Material", "canonical": False, "n_mentions": 2},
            {"id": "b", "name": "polypropylene foam", "type": "Material", "canonical": False, "n_mentions": 9}]
    assert plan_merges(ents, [_v(0), _v(1)], 0.99)["mapping"] == {"a": "b"}


@pytest.mark.skipif(os.getenv("NEO4J_TEST_URI") is None, reason="set NEO4J_TEST_URI=bolt://localhost:7690")
def test_apply_merges_rewires_edges_and_keeps_alias():
    from memento.graph.build.entity_merge import apply_merges
    from memento.graph.store import GraphEdge, GraphNode, Neo4jStore
    s = Neo4jStore(uri=os.environ["NEO4J_TEST_URI"])
    s._run("MATCH (n) DETACH DELETE n")
    s.upsert_nodes([GraphNode("c1", "Chunk", {"source": "p"}), GraphNode("c2", "Chunk", {"source": "p"}),
                    GraphNode("x", "Entity", {"name": "Al2O3 ceramic", "type": "Material"}),
                    GraphNode("alumina", "Entity", {"name": "Alumina", "type": "Material", "aliases": ["Al2O3"]}),
                    GraphNode("sps", "Entity", {"name": "SPS", "type": "Process"})])
    s.upsert_edges([GraphEdge("c1", "Chunk", "x", "Entity", "MENTIONS", {"paper_id": "p"}),
                    GraphEdge("c2", "Chunk", "alumina", "Entity", "MENTIONS", {"paper_id": "p"}),
                    GraphEdge("x", "Entity", "sps", "Entity", "RELATION",
                              {"rel_type": "processed_by", "chunk_id": "c1"})])
    out = apply_merges(s, {"x": "alumina"})
    rows = s._run("MATCH (c:Chunk)-[:MENTIONS]->(e:Entity {id:'alumina'}) RETURN c.id AS c ORDER BY c").records
    assert [r["c"] for r in rows] == ["c1", "c2"]
    assert s._run("MATCH (:Entity {id:'alumina'})-[r:RELATION]->(:Entity {id:'sps'}) RETURN r.rel_type AS t").records[0]["t"] == "processed_by"
    assert s._run("MATCH (e:Entity {id:'x'}) RETURN count(e) AS n").records[0]["n"] == 0
    assert "Al2O3 ceramic" in s._run("MATCH (e:Entity {id:'alumina'}) RETURN e.aliases AS a").records[0]["a"]
    assert out["n_merged"] == 1
    s._run("MATCH (n) DETACH DELETE n")
    s.close()
