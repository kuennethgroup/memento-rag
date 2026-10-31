# tests/graph/test_rq2_p4_store.py
import os

import pytest

from memento.graph.store import GraphEdge, Neo4jStore


class _Rec:
    def __init__(self, rows):
        self.records = rows


class _Driver:
    def __init__(self):
        self.queries = []

    def execute_query(self, q, parameters_=None, database_=None):
        self.queries.append((q, parameters_ or {}))
        return _Rec([{"n": len((parameters_ or {}).get("rows", []))}])

    def close(self):
        pass


def test_relation_merge_is_keyed_per_statement():
    d = _Driver()
    s = Neo4jStore(driver=d)
    s.upsert_edges([GraphEdge(start_id="a", start_label="Entity", end_id="b", end_label="Entity", type="RELATION",
                              props={"rel_type": "raises hardness of", "paper_id": "p", "chunk_id": "1"})])
    q = next(q for q, _ in d.queries if "RELATION" in q and "MERGE" in q)
    assert "rel_type: row.props.rel_type" in q and "chunk_id: row.props.chunk_id" in q


def test_other_edges_still_merge_per_pair():
    d = _Driver()
    Neo4jStore(driver=d).upsert_edges([GraphEdge(start_id="c", start_label="Chunk", end_id="e", end_label="Entity",
                                                 type="MENTIONS", props={})])
    q = next(q for q, _ in d.queries if "MENTIONS" in q and "MERGE" in q)
    assert "MERGE (a)-[r:MENTIONS]->(b)" in q


@pytest.mark.skipif(os.getenv("NEO4J_TEST_URI") is None, reason="set NEO4J_TEST_URI=bolt://localhost:7690")
def test_two_statements_between_one_pair_are_two_edges_and_one_ppr_pair():
    from memento.graph.store import GraphNode
    s = Neo4jStore(uri=os.environ["NEO4J_TEST_URI"])
    s._run("MATCH (n) DETACH DELETE n")
    s.upsert_nodes([GraphNode(id=i, label="Entity", props={"name": i}) for i in ("x", "y")])
    s.upsert_nodes([GraphNode(id="p1", label="Paper", props={}), GraphNode(id="p2", label="Paper", props={})])
    for cid, desc in (("1", "raises"), ("2", "lowers")):
        s.upsert_edges([GraphEdge(start_id="x", start_label="Entity", end_id="y", end_label="Entity", type="RELATION",
                                  props={"rel_type": desc, "paper_id": "p1", "chunk_id": cid})])
    s.upsert_edges([GraphEdge(start_id="p1", start_label="Paper", end_id="p2", end_label="Paper", type="SAME_WORK",
                              props={})])
    assert s._run("MATCH ()-[r:RELATION]->() RETURN count(r) AS n").records[0]["n"] == 2
    assert s.entity_edges() == [("x", "y")] or s.entity_edges() == [("y", "x")]
    assert {r["description"] for r in s.relations_among(["x", "y"])} == {"raises", "lowers"}
    assert s.same_work_neighbors(["p1"]) == ["p2"]
    assert s.entities_by_ids(["x"]) == {"x": "x"} and s.existing_entity_ids(["x", "zz"]) == {"x"}
    s._run("MATCH (n) DETACH DELETE n")
    s.close()
