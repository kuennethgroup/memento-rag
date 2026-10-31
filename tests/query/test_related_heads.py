# Best passage of every document, one Qdrant call.
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from memento.retrieval.search import related_heads


def test_one_best_passage_per_document_across_the_library():
    c = QdrantClient(":memory:")
    c.create_collection("lib", vectors_config={"dense": VectorParams(size=3, distance=Distance.COSINE)})
    pts = [("A", [1, 0, 0], "a best"), ("A", [0.9, 0.1, 0], "a second"), ("B", [0.7, 0.3, 0], "b best"),
           ("B", [0, 1, 0], "b far"), ("C", [0, 0, 1], "c only")]
    c.upsert("lib", [PointStruct(id=i, vector={"dense": v}, payload={"source": s, "text": t, "page": 0, "type": "text"})
                     for i, (s, v, t) in enumerate(pts)])
    heads = related_heads(c, "lib", [1, 0, 0], None, groups=50)
    assert [h["source"] for h in heads] == ["A", "B", "C"]                  # every document once, best first
    assert [h["text"] for h in heads] == ["a best", "b best", "c only"]
    assert [h["source"] for h in related_heads(c, "lib", [1, 0, 0], None, groups=2)] == ["A", "B"]
