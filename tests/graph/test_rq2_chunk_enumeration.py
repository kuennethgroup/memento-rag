# tests/graph/test_rq2_chunk_enumeration.py
import sys
from pathlib import Path

sys.argv = [sys.argv[0], "--domain", "ceramics"]
from memento.config.cli_domain import apply_domain_arg  # noqa: E402
apply_domain_arg()
from memento.ingestion.embed import chunk_point_id  # noqa: E402
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


def test_enumeration_matches_point_id_formula(tmp_path):
    rows = enumerate_embeddable(_items(tmp_path), tmp_path, "paperX", generate_captions=False)
    assert [r["item"]["type"] for r in rows] == ["text", "image", "table", "text"]
    for done, r in enumerate(rows):
        assert r["done"] == done
        p = r["payload"]
        assert r["chunk_id"] == str(chunk_point_id("paperX", p.get("page", 0), p["type"], done))


def test_image_without_file_or_text_is_skipped(tmp_path):
    rows = enumerate_embeddable(_items(tmp_path), tmp_path, "paperX", generate_captions=False)
    assert all(r["item"].get("img_path") != "images/missing.jpg" for r in rows)


def test_enumeration_missing_page_idx_uses_minus_one(tmp_path):
    rows = enumerate_embeddable(_items(tmp_path), tmp_path, "paperX", generate_captions=False)
    last = rows[-1]
    assert last["payload"]["page"] == -1
    assert last["chunk_id"] == str(chunk_point_id("paperX", -1, "text", 3))
