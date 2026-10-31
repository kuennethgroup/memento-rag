# tests/graph/test_rq2_p2_meta.py
import os

import pytest


@pytest.mark.skipif(os.getenv("NEO4J_TEST_URI") is None, reason="set NEO4J_TEST_URI=bolt://localhost:7690")
def test_meta_roundtrip():
    from memento.graph.store import Neo4jStore
    s = Neo4jStore(uri=os.environ["NEO4J_TEST_URI"])
    s._run("MATCH (n) DETACH DELETE n")
    assert s.get_meta("build_id") is None
    s.set_meta("build_id", "b1")
    s.set_meta("build_id", "b2")
    assert s.get_meta("build_id") == "b2"
    s._run("MATCH (n) DETACH DELETE n")
    s.close()
