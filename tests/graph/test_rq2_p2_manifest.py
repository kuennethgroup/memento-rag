# tests/graph/test_rq2_p2_manifest.py
from memento.graph.build.manifest import make_manifest


def test_manifest_totals_and_stable_id():
    s = [{"paper_id": "a", "n_chunks": 3, "n_entities": 2, "n_edges_dropped": 0, "n_llm_calls": 4},
         {"paper_id": "b", "n_chunks": 5, "n_entities": 1, "n_edges_dropped": 1, "n_llm_calls": 0},
         {"paper_id": "c", "failed": True}]
    m1 = make_manifest(domain="ceramics", args={"papers_file": "x"}, summaries=s)
    m2 = make_manifest(domain="ceramics", args={"papers_file": "x"}, summaries=s)
    assert m1["totals"]["n_chunks"] == 8 and m1["totals"]["n_edges_dropped"] == 1
    assert m1["n_papers"] == 2 and m1["failed"] == ["c"]
    assert m1["manifest_id"] == m2["manifest_id"] and "commit" in m1["git"]
