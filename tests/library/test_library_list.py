# The start-screen picker's library list and the shared title rules.
import json
import sys
from pathlib import Path
from memento.config.paths import PROJECT_ROOT
from memento.query.scope import documents

ROOT = PROJECT_ROOT
sys.path.insert(0, str(ROOT))
from memento.library import documents as ll  # noqa: E402


def _parsed(out_dir, src, title):
    d = out_dir / src / "vlm"
    d.mkdir(parents=True)
    (d / f"{src}_content_list.json").write_text(json.dumps(
        [{"type": "text", "text_level": 1, "text": title, "page_idx": 0}]))


def test_entries_titles_sorted_and_fallbacks(tmp_path):
    out_dir, in_dir = tmp_path / "out", tmp_path / "in"
    _parsed(out_dir, "zeta_paper", "A study of stabilization in hafnia thin films")
    _parsed(out_dir, "Impact of field cycling on hafnia", "Impact of Field Cycling on Hafnia")   # title = file name
    _parsed(out_dir, "notes", "1 Experimenteller Teil")                                          # not a title
    e = ll.library_entries(["zeta_paper", "Impact of field cycling on hafnia", "notes"], in_dir, out_dir, {})
    assert e == [{"source": "zeta_paper", "title": "A study of stabilization in hafnia thin films"},
                 {"source": "Impact of field cycling on hafnia", "title": "Impact of field cycling on hafnia"},
                 {"source": "notes", "title": "notes"}]


def test_write_read_and_add(tmp_path):
    assert ll.add_to_library(tmp_path, [{"source": "x", "title": "X"}]) is False      # no list yet: left to the pipe
    assert not ll.library_path(tmp_path).exists()
    ll.write_library(tmp_path, [{"source": "b", "title": "Beta"}, {"source": "a", "title": "alpha"}])
    assert [d["source"] for d in ll.read_library(tmp_path)["documents"]] == ["a", "b"]
    assert ll.add_to_library(tmp_path, [{"source": "c", "title": "Gamma"}, {"source": "b", "title": "Beta 2"}])
    docs = ll.read_library(tmp_path)["documents"]
    assert docs == [{"source": "a", "title": "alpha"}, {"source": "b", "title": "Beta 2"}, {"source": "c", "title": "Gamma"}]
    assert not list(ll.library_path(tmp_path).parent.glob("*.tmp"))


def test_read_broken_is_none(tmp_path):
    ll.library_path(tmp_path).parent.mkdir(parents=True)
    ll.library_path(tmp_path).write_text("{broken")
    assert ll.read_library(tmp_path) is None


def test_picker_titles_plain_text_and_cut_titles():
    assert ll.plain_title(r"Impact of field cycling on $\mathrm{HfO_2}$ based devices") == \
        "Impact of field cycling on HfO2 based devices"
    assert ll.plain_title(r"energy variations in ferroelectric $HfO_{2}$ from calculations") == \
        "energy variations in ferroelectric HfO2 from calculations"
    assert ll.plausible_title("Stabilization of ferroelectric properties in", {}) is None     # cut at the line end
    assert ll.plausible_title("Einfluss der Verarbeitung auf die", {}) is None
    assert ll.plausible_title("Characteristics of low-energy phases of hafnia", {})


def test_query_pipe_exports_the_table_when_the_list_file_differs(tmp_path, monkeypatch):
    from sqlalchemy import create_engine
    from memento.library import documents as db
    from _owui_loader import load_function
    qp = load_function(ROOT / "openwebui/functions/query_pipe.py", "memento_query")
    e = create_engine("sqlite://")
    db.ensure_tables(e)
    monkeypatch.setattr(qp, "_db_engine", lambda: e)
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path))
    for s in ("a", "b"):
        db.add_document(e, collection="col", entry={"source": s, "title": s.upper()}, content_hash="h",
                        uploaded_by="u", output_root=None)
    # the list of collection "col" is its own file
    documents.publish_library("col", {"a", "b"}, qp._db_engine)                          # no file yet → exported from the table
    assert [d["title"] for d in ll.read_library(tmp_path, "col")["documents"]] == ["A", "B"]
    ino = ll.library_path(tmp_path, "col").stat().st_ino
    documents.publish_library("col", {"a", "b"}, qp._db_engine)                          # the same set → left alone
    assert ll.library_path(tmp_path, "col").stat().st_ino == ino
    ll.write_library(tmp_path, [{"source": "P", "title": "P"}], "col")    # changed elsewhere → re-exported
    documents.publish_library("col", {"a", "b"}, qp._db_engine)
    assert [d["source"] for d in ll.read_library(tmp_path, "col")["documents"]] == ["a", "b"]


def test_cover_page_lines_are_not_titles():
    # a repository postprint's cover page is no title; the PDF's own title is the right one
    for cover in ("Dieses Dokument ist eine Zweitveröffentlichung (Postprint)",
                  "This is the accepted manuscript of the following article",
                  "Author manuscript; available in PMC 2020 January 01",
                  "Downloaded from the institutional repository of the university"):
        assert ll.plausible_title(cover, {}) is None, cover
    pdf = "On the Origin of the Large Remanent Polarization in La:HfO₂"
    assert ll.best_title("Dieses Dokument ist eine Zweitveröffentlichung (Postprint)", pdf, {}) == pdf
    assert ll.plausible_title("Accepted manuscripts in materials science: a survey of pre-print servers", {})  # a real title


def test_same_name_ignores_formula_notation():
    assert ll.same_name("On the Origin of the Large Remanent Polarization in La:HfO₂",
                        "On the Origin of the Large Remanent Polarization in LaHfO2")
    assert ll.same_name(ll.plain_title("Impact of field cycling on $\\mathrm{HfO_2}$ based non-volatile memory devices"),
                        "Impact of field cycling on HfO2 based non-volatile memory devices")
    assert not ll.same_name("polyBERT: a chemical language model to enable fully machine-driven polymer informatics",
                            "PolyBERT")
