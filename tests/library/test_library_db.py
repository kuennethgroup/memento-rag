# The document table and its one write path.
import json
import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from memento.config.paths import PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT))
from memento.library import answers as ac  # noqa: E402
from memento.library import documents as db  # noqa: E402
from memento.library import documents as ll  # noqa: E402


@pytest.fixture
def engine():
    e = create_engine("sqlite://")
    db.ensure_tables(e)
    return e


def entry(src, title=None):
    return {"source": src, "title": title or src, "parsed_title": title, "pdf_title": None}


def exported(root):
    # the list of collection "col" (its own file; the single-library file is the default library's only)
    return [(d["source"], d["title"]) for d in ll.read_library(root, "col")["documents"]]


def test_add_list_export(engine, tmp_path):
    db.add_document(engine, collection="col", entry=entry("b", "Beta paper"), content_hash="h1", uploaded_by="u",
                    output_root=tmp_path)
    db.add_document(engine, collection="col", entry=entry("a", "alpha paper"), content_hash="h2", uploaded_by="u",
                    output_root=tmp_path)
    assert [r["source"] for r in db.list_documents(engine, "col")] == ["a", "b"]           # A–Z by title
    assert exported(tmp_path) == [("a", "alpha paper"), ("b", "Beta paper")]
    assert db.get_document(engine, "col", "b")["content_hash"] == "h1"
    assert db.get_document(engine, "col", "zzz") is None


def test_add_document_updates_hash_and_key_changes(engine, tmp_path):
    for h in ("h1", "h2"):                          # the same document uploaded again with changed content
        db.add_document(engine, collection="col", entry=entry("a"), content_hash=h, uploaded_by="u",
                        output_root=tmp_path)
    rows = db.list_documents(engine, "col")
    assert len(rows) == 1 and rows[0]["content_hash"] == "h2"


def test_remove_document_drops_rows_cache_and_export(engine, tmp_path):
    for src in ("a", "b"):
        db.add_document(engine, collection="col", entry=entry(src), content_hash="h", uploaded_by="u",
                        output_root=tmp_path)
    f = dict(collection="col", source="a", content_hash="h", action="summary", language="en", model="m", mode="whole",
             prompt_version="p")
    ac.put(engine, ac.cache_key(**f), fields=f, answer="x", citations=[], generated_by="m")
    db.remove_document(engine, collection="col", source="a", output_root=tmp_path)
    assert [r["source"] for r in db.list_documents(engine, "col")] == ["b"]
    assert exported(tmp_path) == [("b", "b")]
    assert ac.get(engine, ac.cache_key(**f)) is None


def test_reset_library_scoped_to_collection(engine, tmp_path):
    db.add_document(engine, collection="col", entry=entry("a"), content_hash="h", uploaded_by="u", output_root=tmp_path)
    db.add_document(engine, collection="other", entry=entry("z"), content_hash="h", uploaded_by="u", output_root=None)
    removed = db.reset_library(engine, collection="col", output_root=tmp_path)
    assert [r["source"] for r in removed] == ["a"]
    assert db.list_documents(engine, "col") == [] and [r["source"] for r in db.list_documents(engine, "other")] == ["z"]
    assert exported(tmp_path) == []


def test_document_entry_titles(tmp_path):
    out_dir, in_dir = tmp_path / "out", tmp_path / "in"
    for src, title in (("p1", r"Impact of cycling on $\mathrm{HfO_2}$ based devices"),
                       ("p2", "Stabilization of ferroelectric properties in")):
        d = out_dir / src / "vlm"
        d.mkdir(parents=True)
        (d / f"{src}_content_list.json").write_text(json.dumps([{"type": "text", "text_level": 1, "text": title,
                                                                 "page_idx": 0}]))
    e1 = db.document_entry("p1", in_dir, out_dir, {})
    assert e1["title"] == "Impact of cycling on HfO2 based devices" and e1["parsed_title"].startswith("Impact")
    assert db.document_entry("p2", in_dir, out_dir, {})["title"] == "p2"                   # cut title → file name


def test_has_documents_and_rebuild(engine, tmp_path):
    assert db.has_documents(engine, "col") is False
    manifest = {"documents": {"a": {"content_hash": "ha", "uploaded_by": "u1", "collection": "col"},
                              "b": {"source_hash": "hb"}}}
    n = db.rebuild_documents(engine, collection="col", sources=["b", "a"], in_dir=tmp_path, out_dir=tmp_path,
                             section_hints={}, manifest=manifest, output_root=tmp_path)
    rows = {r["source"]: r for r in db.list_documents(engine, "col")}
    assert n == 2 and db.has_documents(engine, "col")
    assert (rows["a"]["content_hash"], rows["a"]["uploaded_by"]) == ("ha", "u1")
    assert (rows["b"]["content_hash"], rows["b"]["uploaded_by"]) == ("hb", "cli")          # fallback hash, server
    assert exported(tmp_path) == [("a", "a"), ("b", "b")]


def test_manifest_meta_prefers_content_hash():
    m = {"documents": {"x": {"content_hash": "c", "source_hash": "s", "uploaded_by": "u"}, "y": {"source_hash": "s"}}}
    assert db.manifest_meta(m, "x") == ("c", "u") and db.manifest_meta(m, "y") == ("s", "cli")
    assert db.manifest_meta(m, "missing") == (None, "cli")


def test_removing_a_document_removes_its_open_questions(engine, tmp_path):
    # gap ledger step 5: a deleted document no longer raises questions; questions others raise stay
    from memento.research_gaps import ledger as gl
    gl.ensure_tables(engine)
    m = dict(point_id="p", page=0, signal="s", claim="c", passage="t", judge="supported")
    only_a = gl.add_gap(engine, "col", "Only A?", "limitation")
    gl.add_mention(engine, only_a, "col", source="a", **m)
    both = gl.add_gap(engine, "col", "Both?", "limitation")
    gl.add_mention(engine, both, "col", source="a", **m)
    gl.add_mention(engine, both, "col", source="b", **m)
    db.remove_document(engine, collection="col", source="a", output_root=tmp_path)
    assert [g["statement"] for g in gl.list_gaps(engine, "col")] == ["Both?"]
    assert gl.list_gaps(engine, "col")[0]["documents"] == ["b"]


def test_resetting_a_library_empties_its_gap_ledger(engine, tmp_path):
    from memento.research_gaps import ledger as gl
    gl.ensure_tables(engine)
    gl.add_gap(engine, "col", "Q?", "limitation")
    gl.add_gap(engine, "other", "Kept?", "limitation")
    db.reset_library(engine, collection="col", output_root=tmp_path)
    assert gl.list_gaps(engine, "col") == [] and len(gl.list_gaps(engine, "other")) == 1
