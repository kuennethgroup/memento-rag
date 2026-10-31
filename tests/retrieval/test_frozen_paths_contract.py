# The RQ2/RQ3 evaluation path (memento/retrieval/retrieve.py, memento/retrieval/generate.py) is frozen: what it sends to Qdrant, the
# embedder, the reranker and the answer model, and what it returns, must stay as recorded when its imports change.
# Recording fakes at those boundaries only. Run through scripts/test_offline.sh.
import dataclasses
import sys
import types

import pytest

from _snapshots import check_snapshot

PAYLOADS = [
    {"source": "paper_a", "page": 0, "type": "text", "text": "Hardness reaches 12 GPa.", "bbox": [0, 10, 100, 20]},
    {"source": "paper_a", "page": 0, "type": "text", "text": "Samples were annealed.", "bbox": [0, 30, 100, 40]},
    {"source": "paper_b", "page": 2, "type": "table", "text": "Table 1 hardness", "caption": "Table 1",
     "bbox": [0, 5, 100, 50]},
    {"source": "paper_b", "page": 3, "type": "image", "text": "Figure 2 microstructure", "caption": "Figure 2",
     "img_path": "/nonexistent/fig2.png", "bbox": [0, 0, 50, 50]},
]


def _dump(x):
    return x.model_dump(exclude_none=True) if hasattr(x, "model_dump") else x


class RecordingQdrant:
    log: list = []

    def __init__(self, *a, **k):
        RecordingQdrant.log.append({"op": "client", "kwargs": sorted(k)})

    def _points(self, n):
        return [types.SimpleNamespace(id=1000 + i, payload=dict(p), score=round(0.9 - i / 10, 3))
                for i, p in enumerate(PAYLOADS[:n])]

    def query_points(self, **kw):
        rec = {k: ([_dump(p) for p in v] if k == "prefetch" else _dump(v)) for k, v in kw.items()}
        RecordingQdrant.log.append({"op": "query_points", **rec})
        return types.SimpleNamespace(points=self._points(min(kw.get("limit", 10), len(PAYLOADS))))

    def retrieve(self, **kw):
        RecordingQdrant.log.append({"op": "retrieve", **{k: _dump(v) for k, v in kw.items()}})
        return []

    def scroll(self, **kw):
        RecordingQdrant.log.append({"op": "scroll", **{k: _dump(v) for k, v in kw.items()}})
        f = kw["scroll_filter"].must
        src, page = f[0].match.value, f[1].match.value
        pts = [types.SimpleNamespace(id=1000 + i, payload=dict(p)) for i, p in enumerate(PAYLOADS)
               if p["source"] == src and p["page"] == page]
        return pts, None


@pytest.fixture
def fakes(monkeypatch):
    import memento.ingestion.embed as sparse
    import memento.common.model_servers as model_servers
    from qdrant_client.models import SparseVector
    calls = []
    RecordingQdrant.log = []

    def vl_embed(url, model, instruction, text=None, image=None, timeout=None):
        calls.append({"op": "embed", "url": url, "model": model, "instruction": instruction, "text": text,
                      "image": image, "timeout": timeout})
        return [0.1, 0.2, 0.3]

    def rerank(url, model, query, docs, instruction, top_n=None, **k):
        calls.append({"op": "rerank", "url": url, "model": model, "query": query, "docs": docs,
                      "instruction": instruction, "top_n": top_n, "extra": sorted(k)})
        order = list(range(len(docs)))[::-1][:top_n]          # reversed: the hits must take the reranker's order
        return [(i, round(1.0 - j / 10, 3)) for j, i in enumerate(order)]

    def sparse_encode_query(text):
        calls.append({"op": "sparse", "text": text})
        return SparseVector(indices=[1, 2], values=[1.0, 1.0])

    monkeypatch.setattr(model_servers, "vl_embed", vl_embed)
    monkeypatch.setattr(model_servers, "rerank", rerank)
    monkeypatch.setattr(model_servers, "image_data_url", lambda path: f"data:image/png;base64,<{path}>")
    monkeypatch.setattr(sparse, "sparse_encode_query", sparse_encode_query)
    return calls


def _hit(h) -> dict:
    return dataclasses.asdict(h)


CASES = {
    "paper_filter_dense": dict(paper_id="paper_a", hybrid=False),
    "hybrid_on": dict(hybrid=True),
    "hybrid_off": dict(hybrid=False),
    "expand_neighbors": dict(hybrid=False, expand=True),
    "no_reranker": dict(hybrid=False, use_reranker=False),
}


def test_frozen_retrieve_contract(fakes):
    from memento.retrieval import retrieve as r
    got = {}
    for name, kw in CASES.items():
        fakes.clear()
        RecordingQdrant.log = []
        hits, timing = r.retrieve("hardness of annealed samples", retrieve_k=4, rerank_top=3,
                                  client=RecordingQdrant(), **kw)
        got[name] = {"hits": [_hit(h) for h in hits], "timing_keys": sorted(timing), "calls": list(fakes),
                     "qdrant": RecordingQdrant.log}
    check_snapshot("frozen_retrieve_contract", got)


class RecordingOpenAI:
    log: list = []

    def __init__(self, *a, **k):
        RecordingOpenAI.log.append({"op": "client", "base_url": k.get("base_url"), "api_key_set": bool(k.get("api_key"))})
        self.chat = self
        self.completions = self

    def create(self, **kw):
        RecordingOpenAI.log.append({"op": "create", **kw})
        msg = types.SimpleNamespace(content="Hardness is 12 GPa [1].", reasoning_content="thought")
        usage = types.SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)], usage=usage, model="fake-model")


def test_frozen_generate_contract(fakes, monkeypatch):
    from memento.retrieval import generate as g
    retrieve_mod = sys.modules[g.retrieve.__module__]      # generate imports retrieve one of two ways
    monkeypatch.setattr(retrieve_mod, "QdrantClient", RecordingQdrant)
    monkeypatch.setattr(g, "OpenAI", RecordingOpenAI)
    got = {}
    for name, kw in {"default": dict(), "think_no_expand": dict(think=True, expand=False)}.items():
        fakes.clear()
        RecordingQdrant.log, RecordingOpenAI.log = [], []
        res = g.generate("hardness of annealed samples", retrieve_k=4, rerank_top=4, validate=False, **kw)
        got[name] = {"answer": res.answer, "thinking": res.thinking, "model": res.model, "usage": res.usage,
                     "hits": [_hit(h) for h in res.hits], "timing_keys": sorted(res.timing),
                     "calls": list(fakes), "qdrant": RecordingQdrant.log, "llm": RecordingOpenAI.log}
    check_snapshot("frozen_generate_contract", got)
