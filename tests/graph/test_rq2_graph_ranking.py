# tests/graph/test_rq2_graph_ranking.py
import os
import pytest

needs_neo4j = pytest.mark.skipif(os.getenv("NEO4J_TEST_URI") is None,
                                 reason="set NEO4J_TEST_URI=bolt://localhost:7690")


@pytest.fixture()
def store():
    from memento.graph.store import Neo4jStore
    s = Neo4jStore(uri=os.environ["NEO4J_TEST_URI"])
    s._run("MATCH (n) DETACH DELETE n")
    s.ensure_schema()
    yield s
    s._run("MATCH (n) DETACH DELETE n")
    s.close()


def _graph(store, n_hub_chunks=30):
    from memento.graph.store import GraphEdge, GraphNode
    nodes = [GraphNode("hub", "Entity", {"name": "Temperature", "type": "Condition"}),
             GraphNode("rare", "Entity", {"name": "DOPO", "type": "Material"})]
    edges = []
    for i in range(n_hub_chunks):
        nodes.append(GraphNode(f"c{i}", "Chunk", {"source": f"p{i % 3}", "type": "text"}))
        edges.append(GraphEdge(f"c{i}", "Chunk", "hub", "Entity", "MENTIONS"))
    edges.append(GraphEdge("c7", "Chunk", "rare", "Entity", "MENTIONS"))
    store.upsert_nodes(nodes)
    store.upsert_edges(edges)


@needs_neo4j
def test_subgraph_chunks_ranks_rare_entity_first(store):
    _graph(store)
    rows = store.subgraph_chunks(["hub", "rare"], max_hops=0, limit=5)
    assert rows[0]["chunk_id"] == "c7"
    assert rows[0]["score"] > rows[1]["score"]
    assert [r["score"] for r in rows] == sorted((r["score"] for r in rows), reverse=True)


@needs_neo4j
def test_hub_cap_drops_hub_entity(store):
    _graph(store)
    rows = store.subgraph_chunks(["hub", "rare"], max_hops=0, limit=50, max_df=10)
    assert [r["chunk_id"] for r in rows] == ["c7"]


@needs_neo4j
def test_upsert_edges_counts_dropped(store):
    from memento.graph.store import GraphEdge, GraphNode
    store.upsert_nodes([GraphNode("a", "Entity", {"name": "A", "type": "Material"})])
    n = store.upsert_edges([GraphEdge("a", "Entity", "missing", "Entity", "RELATION", {"rel_type": "x"})])
    assert n == 0 and store.last_dropped == 1


@needs_neo4j
def test_citing_neighbors(store):
    from memento.graph.store import GraphEdge, GraphNode
    store.upsert_nodes([GraphNode(p, "Paper", {"source": p}) for p in ("p1", "p2", "p3")])
    store.upsert_edges([GraphEdge("p1", "Paper", "p2", "Paper", "CITES_PAPER"),
                        GraphEdge("p3", "Paper", "p1", "Paper", "CITES_PAPER")])
    assert sorted(store.citing_neighbors(["p1"])) == ["p2", "p3"]


def test_query_vector_reaches_neighbor_search(monkeypatch):
    import memento.graph.query.graph_retrieval as gr
    seen = {}

    def fake_neighbors(store, srcs, qv, client, limit):
        seen["qv"] = qv
        return [{"chunk_id": "9", "source": "p2", "score": 0.9}]

    class Store:
        def expand_via_references(self, ids, limit=50): return []
        def close(self): pass

    monkeypatch.setattr(gr, "_neighbor_chunks_by_vector", fake_neighbors)
    rows = gr.graph_candidates("q", dense_chunk_ids=["1"], dense_sources=["p1"], store=Store(),
                               enabled=True, streams="structural", query_vector=[0.1, 0.2])
    assert seen["qv"] == [0.1, 0.2] and rows[0]["chunk_id"] == "9"


@needs_neo4j
def test_references_follow_dense_rank_before_the_limit(store):
    from memento.graph.store import GraphEdge, GraphNode
    n = 30
    dense = [f"d{i:02d}" for i in range(n)]
    store.upsert_nodes([GraphNode(c, "Chunk", {"source": "p", "type": "text"}) for c in dense]
                       + [GraphNode(f"r{i:02d}", "Chunk", {"source": "p", "type": "image"}) for i in range(n)])
    store.upsert_edges([GraphEdge(f"d{i:02d}", "Chunk", f"r{i:02d}", "Chunk", "REFERENCES") for i in range(n)])
    rows = store.expand_via_references(list(reversed(dense)), limit=3)
    assert [r["chunk_id"] for r in rows] == ["r29", "r28", "r27"]
    assert [r["via_chunk"] for r in rows] == ["d29", "d28", "d27"]


@needs_neo4j
def test_references_dedupe_target_keeping_best_dense_rank(store):
    from memento.graph.store import GraphEdge, GraphNode
    store.upsert_nodes([GraphNode(c, "Chunk", {"source": "p", "type": "text"}) for c in ("a", "b", "c", "d", "X", "Y")])
    store.upsert_edges([GraphEdge(s, "Chunk", "X", "Chunk", "REFERENCES") for s in ("a", "b", "c")]
                       + [GraphEdge("d", "Chunk", "Y", "Chunk", "REFERENCES")])
    rows = store.expand_via_references(["b", "a", "c", "d"], limit=2)
    assert [(r["chunk_id"], r["via_chunk"]) for r in rows] == [("X", "b"), ("Y", "d")]


def test_tied_entity_scores_are_broken_by_query_similarity(monkeypatch):
    import memento.graph.query.graph_retrieval as gr
    asked = {}

    class Store:
        def subgraph_chunks(self, seed_ids, max_hops=2, limit=50, max_df=None):
            asked["limit"] = limit
            return [{"chunk_id": str(i), "source": "p", "score": 0.0, "via": ["hub"]} for i in range(1, 6)][:limit]
        def expand_via_references(self, ids, limit=50): return []
        def close(self): pass

    class Pt:
        def __init__(self, i, s): self.id, self.score = i, s

    class Client:
        def query_points(self, collection, **kw):
            sims = {1: 0.1, 2: 0.2, 3: 0.9, 4: 0.5, 5: 0.3}
            ids = kw["query_filter"].must[0].has_id
            return type("R", (), {"points": [Pt(i, sims[i]) for i in ids]})

    monkeypatch.setattr(gr, "_build_seeds", lambda *a, **k: {"hub": 1.0})
    rows = gr.graph_candidates("q", store=Store(), enabled=True, streams="entity", traversal="bfs",
                               limit=2, query_vector=[0.1], qdrant_client=Client())
    assert [r["chunk_id"] for r in rows] == ["3", "4"]
    assert asked["limit"] > 2
