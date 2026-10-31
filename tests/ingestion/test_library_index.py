# One decision for every ingestion path: duplicates, versions, name clashes.
import json
import sys
from pathlib import Path

import pytest
from memento.config.paths import PROJECT_ROOT

ROOT = PROJECT_ROOT
sys.path.insert(0, str(ROOT))

from memento.ingestion import library_index as li  # noqa: E402

C = "memento-comp-mat-rag"


def _file(tmp_path, name, content):
    p = tmp_path / f"{len(list(tmp_path.iterdir()))}_{name}"
    p.write_bytes(content.encode())
    return (name, p)


def _decide(items, manifest, library, uploader="u-anna", tag="Anna Schmidt", collection=C):
    return li.decide(items, collection=collection, uploader=uploader, tag=tag, manifest=manifest,
                     in_library=lambda s: s in library)


def _doc(content, uploader=None, collection=C):
    import hashlib
    meta = {"content_hash": hashlib.sha256(content.encode()).hexdigest(), "collection": collection}
    if uploader:
        meta["uploaded_by"] = uploader
    return meta


def test_new_file(tmp_path):
    [d] = _decide([_file(tmp_path, "paper.pdf", "A")], {}, set())
    assert (d.action, d.stem, d.filename) == ("new", "paper", "paper.pdf")


def test_same_name_same_content_is_unchanged(tmp_path):
    m = {"documents": {"paper": _doc("A", "u-anna")}}
    [d] = _decide([_file(tmp_path, "paper.pdf", "A")], m, {"paper"}, uploader="u-ben", tag="Ben")
    assert d.action == "unchanged" and d.stem == "paper"          # someone else's identical copy: nothing to do


def test_same_name_new_content_same_person_is_a_new_version(tmp_path):
    m = {"documents": {"paper": _doc("A", "u-anna")}}
    [d] = _decide([_file(tmp_path, "paper.pdf", "B")], m, {"paper"})
    assert (d.action, d.stem) == ("changed", "paper")


def test_same_name_new_content_other_person_is_renamed(tmp_path):
    m = {"documents": {"paper": _doc("A", "u-anna")}}
    [d] = _decide([_file(tmp_path, "paper.pdf", "B")], m, {"paper"}, uploader="u-ben", tag="Ben Müller")
    assert (d.action, d.filename) == ("renamed", "paper__ben-muller.pdf")


def test_renamed_again_by_the_same_person_is_a_version_of_their_copy(tmp_path):
    m = {"documents": {"paper": _doc("A", "u-anna"), "paper__ben": _doc("B", "u-ben")}}
    [d] = _decide([_file(tmp_path, "paper.pdf", "C")], m, {"paper", "paper__ben"}, uploader="u-ben", tag="Ben")
    assert (d.action, d.stem) == ("changed", "paper__ben")
    [d] = _decide([_file(tmp_path, "paper.pdf", "B")], m, {"paper", "paper__ben"}, uploader="u-ben", tag="Ben")
    assert (d.action, d.stem) == ("duplicate", "paper__ben")          # "already in the library as paper__ben"


def test_two_people_with_the_same_name_tag_do_not_collide(tmp_path):
    m = {"documents": {"paper": _doc("A", "u-anna"), "paper__ben": _doc("B", "u-ben1")}}
    [d] = _decide([_file(tmp_path, "paper.pdf", "C")], m, {"paper", "paper__ben"}, uploader="u-ben2", tag="Ben")
    assert d.action == "renamed" and d.stem.startswith("paper__ben-") and d.stem != "paper__ben"


def test_identical_content_under_another_name_is_a_duplicate(tmp_path):
    m = {"documents": {"paper": _doc("A", "u-anna")}}
    [d] = _decide([_file(tmp_path, "paper (1).pdf", "A")], m, {"paper"}, uploader="u-ben", tag="Ben")
    assert (d.action, d.stem) == ("duplicate", "paper")


def test_a_deleted_or_failed_document_does_not_block_a_new_upload(tmp_path):
    m = {"documents": {"paper": _doc("A", "u-anna")}}                 # manifest entry, but no vectors
    [d] = _decide([_file(tmp_path, "copy.pdf", "A")], m, set())
    assert (d.action, d.stem) == ("new", "copy")
    [d] = _decide([_file(tmp_path, "paper.pdf", "B")], m, set(), uploader="u-ben", tag="Ben")
    assert (d.action, d.stem) == ("new", "paper")


def test_collections_are_separate(tmp_path):
    m = {"documents": {"paper": _doc("A", "u-anna", collection="memento-ceramics-rag")}}
    [d] = _decide([_file(tmp_path, "other.pdf", "A")], m, {"paper"})
    assert d.action == "new"                                           # same content in ANOTHER library: not a dup


def test_documents_indexed_before_hashes_existed_are_adopted_not_reindexed(tmp_path):
    # command-line documents of the first form: in Qdrant, no hash, no uploader
    m = {"documents": {"paper": {"vector_count": 40, "collection": C, "ingested_by": "run_pipeline"}}}
    [d] = _decide([_file(tmp_path, "paper.pdf", "A")], m, {"paper"}, uploader=li.SERVER_UPLOADER, tag="cli")
    assert d.action == "unchanged"
    li.record(m, [d], collection=C, uploader=li.SERVER_UPLOADER)
    assert m["documents"]["paper"]["content_hash"] == d.content_hash and "uploaded_by" not in m["documents"]["paper"]


def test_a_chat_upload_does_not_replace_a_server_side_document(tmp_path):
    m = {"documents": {"paper": _doc("A")}}                            # no uploader recorded: the server's
    [d] = _decide([_file(tmp_path, "paper.pdf", "B")], m, {"paper"}, uploader="u-anna", tag="Anna")
    assert (d.action, d.stem) == ("renamed", "paper__anna")
    [d] = _decide([_file(tmp_path, "paper.pdf", "B")], m, {"paper"}, uploader=li.SERVER_UPLOADER, tag="cli")
    assert d.action == "changed"                                       # the command line updates its own document


def test_within_one_batch(tmp_path):
    items = [_file(tmp_path, "a.pdf", "A"), _file(tmp_path, "b.pdf", "A"), _file(tmp_path, "a.pdf", "B")]
    d = _decide(items, {}, set())
    assert [(x.action, x.stem) for x in d] == [("new", "a"), ("duplicate", "a"), ("new", "a-2")]


def test_record_sets_owner_for_new_names_only(tmp_path):
    m = {"documents": {"paper": _doc("A", "u-anna")}}
    ds = _decide([_file(tmp_path, "paper.pdf", "B"), _file(tmp_path, "x.pdf", "X"), _file(tmp_path, "y.pdf", "A")],
                 m, {"paper"}, uploader="u-ben", tag="Ben")
    li.record(m, ds, collection=C, uploader="u-ben")
    docs = m["documents"]
    assert docs["paper"]["uploaded_by"] == "u-anna"                    # untouched
    assert docs["paper__ben"]["uploaded_by"] == "u-ben" and docs["x"]["uploaded_by"] == "u-ben"
    assert "y" not in docs                                             # duplicate: nothing recorded


def test_person_tag():
    assert li.person_tag("Jana Müller") == "jana-muller"
    assert li.person_tag("  ") == "user" and li.person_tag("王") == "user"


def test_locked_manifest_round_trip_and_refuses_to_clobber_a_broken_file(tmp_path):
    p = tmp_path / "manifest.json"
    with li.locked_manifest(p) as m:
        m["documents"]["a"] = {"x": 1}
    assert json.loads(p.read_text())["documents"] == {"a": {"x": 1}}
    p.write_text("{broken")
    with pytest.raises(ValueError):
        with li.locked_manifest(p):
            pass
    assert p.read_text() == "{broken"


def test_the_same_file_attached_twice_in_one_upload_is_stored_once(tmp_path):
    # one upload holding the same file twice under the same name (Open WebUI re-sends an earlier message's attachment)
    # stores it once
    items = [_file(tmp_path, "paper.pdf", "A"), _file(tmp_path, "paper.pdf", "A")]
    d = _decide(items, {}, set())
    assert [(x.action, x.stem) for x in d] == [("new", "paper"), ("duplicate", "paper")]
    m = {"documents": {"paper": _doc("A", "u-anna")}}
    d = _decide([_file(tmp_path, "paper.pdf", "B"), _file(tmp_path, "paper.pdf", "B")], m, {"paper"},
                uploader="u-ben", tag="Ben")
    assert [(x.action, x.stem) for x in d] == [("renamed", "paper__ben"), ("duplicate", "paper__ben")]
