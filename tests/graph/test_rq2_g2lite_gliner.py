# tests/graph/test_rq2_g2lite_gliner.py — G2-lite-gliner rung: GLiNER entities through the LLM fusion path
import sys

sys.argv = [sys.argv[0], "--domain", "polymer-engineering"]
from memento.config.cli_domain import apply_domain_arg  # noqa: E402
apply_domain_arg()
import pytest  # noqa: E402

from memento.graph.build.build_graph import _embeddable_chunks, build_paper  # noqa: E402

ITEMS = [{"type": "text", "text": "Alumina and zirconia composites were sintered by spark plasma sintering "
                                  "and tested for fracture toughness at 1450°C.", "page_idx": 0}]


class FakeStore:
    def __init__(self):
        self.nodes, self.edges = [], []

    def clear_paper(self, pid):
        pass

    def upsert_nodes(self, nodes):
        self.nodes += list(nodes)
        return len(self.nodes)

    def upsert_edges(self, edges):
        self.edges += list(edges)
        return len(self.edges)


def _chunk_id(tmp_path):
    return _embeddable_chunks(ITEMS, tmp_path, "X")[0]["chunk_id"]


def _mentioned(store):
    return {e.end_id for e in store.edges if e.type == "MENTIONS"}


def test_gliner_entities_are_fused_and_filtered(tmp_path):
    cid = _chunk_id(tmp_path)
    gl = {cid: {"entities": [{"name": "grain boundary sliding", "type": "Process"},
                             {"name": "1450°C", "type": "Condition"},          # bare value: junk filter
                             {"name": "composites", "type": "NotAType"}],      # unknown type: dropped
                "relations": []}}
    s = FakeStore()
    out = build_paper("X", store=s, items=ITEMS, paper_dir=tmp_path, use_llm=False, gliner=gl)
    base = FakeStore()
    build_paper("X", store=base, items=ITEMS, paper_dir=tmp_path, use_llm=False)
    added = _mentioned(s) - _mentioned(base)
    assert added == {"grain_boundary_sliding"}
    assert out["n_relations"] == 0 and out["n_llm_calls"] == 0
    assert _mentioned(base) <= _mentioned(s)          # registry matches are kept


def test_gliner_requires_an_entry_for_every_chunk(tmp_path):
    with pytest.raises(RuntimeError, match="no GLiNER entry"):
        build_paper("X", store=FakeStore(), items=ITEMS, paper_dir=tmp_path, use_llm=False, gliner={})


def test_gliner_refuses_llm_mode(tmp_path):
    with pytest.raises(ValueError):
        build_paper("X", store=FakeStore(), items=ITEMS, paper_dir=tmp_path, use_llm=True,
                    gliner={_chunk_id(tmp_path): {"entities": [], "relations": []}})


def test_no_llm_build_does_not_overwrite_the_llm_debug_file(tmp_path):
    build_paper("X", store=FakeStore(), items=ITEMS, paper_dir=tmp_path, use_llm=False)
    assert not (tmp_path / "X_graph_debug.json").exists()
