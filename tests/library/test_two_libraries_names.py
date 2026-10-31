# One name allocator across libraries. Both libraries' documents share one file-server namespace (the file server finds
# a source by its name anywhere under INPUT_ROOT), so a new document never takes a name another library or another file
# already has. At each candidate name (the file's own name, then __<tag>, then __<tag>-<uploader>, then -2, -3 …) this
# uploader's own document in this library is reused (unchanged or a new version); anything else holding the name is
# skipped.
import copy
import sys
from pathlib import Path

import pytest
from memento.config.paths import PROJECT_ROOT

ROOT = PROJECT_ROOT
sys.path.insert(0, str(ROOT))
from memento.ingestion import library_index as li  # noqa: E402

COMP, POLY = "memento-comp-mat-rag", "memento-polymer-rag"


@pytest.fixture
def files(tmp_path):
    def make(name, content):
        p = tmp_path / f"{len(list(tmp_path.iterdir()))}_{name}"
        p.write_text(content)
        return name, p
    return make


def run(items, *, manifest, collection=POLY, uploader="alex-id", tag="Alex", library=(), taken=()):
    before = copy.deepcopy(manifest)
    out = li.decide(items, collection=collection, uploader=uploader, tag=tag, manifest=manifest,
                    in_library=lambda s: s in set(library), taken=lambda s: s in set(taken))
    assert manifest == before                      # decide() never writes; record() does
    return [(d.action, d.stem) for d in out]


def entry(collection, h, by="someone"):
    return {"collection": collection, "content_hash": h, "uploaded_by": by}


def test_a_name_held_by_the_other_library_is_not_taken(files):
    m = {"documents": {"report": entry(COMP, "hc")}}
    assert run([files("report.pdf", "new text")], manifest=m) == [("renamed", "report__alex")]


def test_an_occupied_suffixed_name_in_the_other_library_is_skipped_too(files):
    m = {"documents": {"report": entry(COMP, "h1"), "report__alex": entry(COMP, "h2")}}
    assert run([files("report.pdf", "x")], manifest=m) == [("renamed", "report__alex-alexid")]


def test_numeric_suffixes_follow_when_every_tagged_name_is_taken(files):
    m = {"documents": {"report": entry(COMP, "h1"), "report__alex": entry(COMP, "h2"),
                       "report__alex-alexid": entry(COMP, "h3")}}
    assert run([files("report.pdf", "x")], manifest=m, taken={"report-2"}) == [("renamed", "report-3")]


def test_two_users_with_the_same_tag_get_their_own_names(files):
    m = {"documents": {"report": entry(POLY, "h0", by="owner"), "report__alex": entry(POLY, "h1", by="other-id")}}
    out = run([files("report.pdf", "x")], manifest=m, library={"report", "report__alex"})
    assert out == [("renamed", "report__alex-alexid")]


def test_same_named_files_in_one_batch(files):
    out = run([files("report.pdf", "one"), files("report.pdf", "two")], manifest={"documents": {}})
    assert out == [("new", "report"), ("new", "report-2")]


def test_a_reupload_and_a_changed_version_in_the_second_library(files):
    name, path = files("report.pdf", "v1")
    h = li.file_hash(path)
    m = {"documents": {"report": entry(COMP, "hc"), "report__alex": entry(POLY, h, by="alex-id")}}
    # the same content again under its original name: already in the library as report__alex (today's rule)
    assert run([(name, path)], manifest=m, library={"report__alex"}) == [("duplicate", "report__alex")]
    assert run([files("report.pdf", "v2")], manifest=m, library={"report__alex"}) == [("changed", "report__alex")]


def test_r5_a_changed_upload_under_its_original_name_updates_the_generated_name(files):
    # A (default library) owns "report"; another user's "report.pdf" became "report__alex"; a changed "report.pdf" later
    # updates that document instead of adding another one
    m = {"documents": {"report": entry(COMP, "hc", by="a-id"), "report__alex": entry(POLY, "h-old", by="alex-id")}}
    out = run([files("report.pdf", "changed text")], manifest=m, library={"report__alex"})
    assert out == [("changed", "report__alex")]


def test_a_file_of_another_corpus_on_disk_is_not_taken(files):
    # a ceramics document with this name in the file server's namespace, in no manifest
    assert run([files("Alumina sintering.pdf", "x")], manifest={"documents": {}},
               taken={"Alumina sintering"}) == [("renamed", "Alumina sintering__alex")]


def test_this_librarys_failed_document_is_retried_under_its_own_name(files):
    # parsed before but never stored (no vectors): its file is on disk, its manifest entry is this library's
    m = {"documents": {"thesis": entry(POLY, "h-failed", by="alex-id")}}
    assert run([files("thesis.pdf", "x")], manifest=m, taken={"thesis"}) == [("new", "thesis")]


def test_todays_rules_in_one_library_are_unchanged(files):
    m = {"documents": {"report": entry(POLY, "h0", by="owner")}}
    name, path = files("report.pdf", "same")
    m["documents"]["report"]["content_hash"] = li.file_hash(path)
    assert run([(name, path)], manifest=m, library={"report"}) == [("unchanged", "report")]   # same content
    assert run([files("report.pdf", "other")], manifest=m, library={"report"}) == [("renamed", "report__alex")]
    m["documents"]["report"]["uploaded_by"] = "alex-id"
    assert run([files("report.pdf", "other2")], manifest=m, library={"report"}) == [("changed", "report")]


def test_without_a_namespace_function_the_manifest_still_counts(files):
    m = {"documents": {"report": entry(COMP, "hc")}}
    out = li.decide([files("report.pdf", "x")], collection=POLY, uploader="u", tag="Alex", manifest=m,
                    in_library=lambda s: False)
    assert [(d.action, d.stem) for d in out] == [("renamed", "report__alex")]


def test_the_namespace_holds_every_stem_under_the_input_root_and_the_staged_links(tmp_path):
    inp, out = tmp_path / "in", tmp_path / "out"
    (inp / "Ceramics" / "a").mkdir(parents=True)
    (inp / "Ceramics" / "a" / "Alumina.pdf").write_text("x")
    (inp / "PolymerEngineering").mkdir(parents=True)
    (inp / "PolymerEngineering" / "Foam.docx").write_text("x")
    (out / "_staged").mkdir(parents=True)
    (out / "_staged" / "b__Lebenslauf.doc").write_text("x")
    names = li.namespace(inp, out)
    assert {"Alumina", "Foam", "b__Lebenslauf"} <= names
    # a command-line run's own files are not "taken" by themselves
    assert "Foam" not in li.namespace(inp, out, exclude=[inp / "PolymerEngineering" / "Foam.docx"])


def test_both_upload_paths_pass_the_namespace():
    for path in ("memento/ingestion/upload/document.py", "memento/ingestion/run_pipeline.py"):
        src = (ROOT / path).read_text()
        i = src.index("library_index.decide(")
        assert "taken=" in src[i:i + 400], path
