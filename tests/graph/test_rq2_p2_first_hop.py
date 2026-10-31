# tests/graph/test_rq2_p2_first_hop.py
import os

import pytest


def test_first_hop_seeds_reach_entity_traversal(monkeypatch):
    import memento.graph.query.graph_retrieval as gr
    got = {}

    class Store:
        def chunk_entities(self, ids, max_df=None):
            got["ids"] = ids
            return {"dopo": 3.0, "epoxy": 1.0}

        def subgraph_chunks(self, seed_ids, max_hops=2, limit=50, max_df=None):
            got["seeds"] = sorted(seed_ids)
            return [{"chunk_id": "9", "source": "B", "score": 1.0, "via": ["dopo"]}]

        def expand_via_references(self, ids, limit=50):
            return []

        def close(self):
            pass

    monkeypatch.setattr(gr, "_build_seeds", lambda *a, **k: {})          # question names no entity
    rows = gr.graph_candidates("q", dense_chunk_ids=[str(i) for i in range(20)], store=Store(),
                               enabled=True, streams="entity", traversal="bfs",
                               first_hop=True, first_hop_k=5)
    assert got["ids"] == ["0", "1", "2", "3", "4"] and got["seeds"] == ["dopo", "epoxy"]
    assert rows[0]["chunk_id"] == "9"


def test_first_hop_off_keeps_old_behaviour(monkeypatch):
    import memento.graph.query.graph_retrieval as gr
    monkeypatch.setattr(gr, "_build_seeds", lambda *a, **k: {})

    class Store:
        def close(self):
            pass

    assert gr.graph_candidates("q", dense_chunk_ids=["1"], store=Store(), enabled=True,
                               streams="entity", first_hop=False) == []


@pytest.mark.skipif(os.getenv("NEO4J_TEST_URI") is None, reason="set NEO4J_TEST_URI=bolt://localhost:7690")
def test_chunk_entities_idf_and_cap():
    from memento.graph.store import GraphEdge, GraphNode, Neo4jStore
    s = Neo4jStore(uri=os.environ["NEO4J_TEST_URI"])
    s._run("MATCH (n) DETACH DELETE n")
    s.upsert_nodes([GraphNode(f"c{i}", "Chunk", {"source": "p"}) for i in range(10)]
                   + [GraphNode("hub", "Entity", {"name": "T", "type": "Condition"}),
                      GraphNode("rare", "Entity", {"name": "DOPO", "type": "Material"})])
    s.upsert_edges([GraphEdge(f"c{i}", "Chunk", "hub", "Entity", "MENTIONS") for i in range(10)]
                   + [GraphEdge("c0", "Chunk", "rare", "Entity", "MENTIONS")])
    w = s.chunk_entities(["c0"])
    assert w["rare"] > w["hub"] == 0.0
    assert set(s.chunk_entities(["c0"], max_df=5)) == {"rare"}
    s._run("MATCH (n) DETACH DELETE n")
    s.close()
