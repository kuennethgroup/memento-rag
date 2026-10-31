# tests/graph/test_rq2_build_graph_enumeration.py
import sys
sys.argv = [sys.argv[0], "--domain", "ceramics"]
from memento.config.cli_domain import apply_domain_arg  # noqa: E402
apply_domain_arg()
from memento.graph.build.build_graph import _embeddable_chunks  # noqa: E402
from memento.ingestion.embed import enumerate_embeddable  # noqa: E402


def _items(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "a.jpg").write_bytes(b"x")
    return [
        {"type": "text", "text": "Title", "text_level": 1, "page_idx": 0},
        {"type": "text", "text": "Alumina was sintered.", "page_idx": 0},
        {"type": "header", "text": "Journal X", "page_idx": 0},
        {"type": "image", "img_path": "images/a.jpg", "image_caption": ["Fig. 1 SEM"], "page_idx": 1},
        {"type": "image", "img_path": "images/missing.jpg", "content": "mermaid junk", "page_idx": 1},
        {"type": "table", "table_caption": ["Table 1"], "table_body": "<tr>..</tr>", "page_idx": 2},
        {"type": "text", "text": "No page index here."},
    ]


def test_graph_chunk_ids_equal_pipeline_ids(tmp_path):
    items = _items(tmp_path)
    g = _embeddable_chunks(items, tmp_path, "paperX")
    p = enumerate_embeddable(items, tmp_path, "paperX", generate_captions=False)
    assert [c["chunk_id"] for c in g] == [r["chunk_id"] for r in p]
    assert {"chunk_id", "item_idx", "item", "type", "disp_text", "extract_text",
            "section", "page", "page_end"} <= set(g[0])
