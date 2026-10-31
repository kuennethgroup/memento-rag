# Processed chat attachments per library. Library Upload remembers which Open WebUI attachment ids it already added, so
# a file resent with every message is not added twice. With two libraries the list is per library: an attachment added
# to one library can still be added to the other. The default library's list is manifest["seen_file_ids"]; another
# library's is manifest["seen_file_ids_by_collection"][<collection>]. A reset clears only its own library's list.
import json
import sys
import threading
from pathlib import Path
from memento.config.paths import PROJECT_ROOT
from memento.ingestion.upload import document

ROOT = PROJECT_ROOT
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from memento.ingestion import library_index as li  # noqa: E402
from test_two_libraries_worker import INGEST, load_copy  # noqa: E402

COMP, POLY = "memento-comp-mat-rag", "memento-polymer-rag"


def test_todays_library_keeps_the_old_list_and_another_library_has_its_own():
    m = {"seen_file_ids": ["old"], "documents": {}}
    li.add_seen_ids(m, COMP, {"a1"})
    li.add_seen_ids(m, POLY, {"p1", "a1"})
    assert m["seen_file_ids"] == ["a1", "old"]
    assert m["seen_file_ids_by_collection"] == {POLY: ["a1", "p1"]}
    assert li.seen_ids(m, COMP) == {"old", "a1"} and li.seen_ids(m, POLY) == {"a1", "p1"}
    assert li.seen_ids({"documents": {}}, POLY) == set() and li.seen_ids({}, COMP) == set()


def test_clearing_one_library_keeps_the_other():
    m = {"seen_file_ids": ["a1"], "seen_file_ids_by_collection": {POLY: ["p1"], "memento-x-rag": ["x1"]}}
    li.clear_seen_ids(m, POLY)
    assert m["seen_file_ids"] == ["a1"] and m["seen_file_ids_by_collection"] == {"memento-x-rag": ["x1"]}
    li.clear_seen_ids(m, COMP)
    assert m["seen_file_ids"] == [] and m["seen_file_ids_by_collection"] == {"memento-x-rag": ["x1"]}


def _copy(monkeypatch, tmp_path, collection):
    ip = load_copy(INGEST, "memento_ingestion", monkeypatch)
    monkeypatch.setattr(document, "MANIFEST_FILE", tmp_path / "manifest.json")
    return document.Ingester(ip.Pipe.Valves(COLLECTION=collection, OUTPUT_ROOT=str(tmp_path / "out")))


def test_one_attachment_can_be_added_to_each_library_once(monkeypatch, tmp_path):
    a, b = _copy(monkeypatch, tmp_path, COMP), _copy(monkeypatch, tmp_path, POLY)
    a.save_seen_ids(a.load_seen_ids() | {"f1"})                  # added in A
    assert "f1" in a.load_seen_ids()                               # A again: skipped
    assert "f1" not in b.load_seen_ids()                           # B: still added
    b.save_seen_ids(b.load_seen_ids() | {"f1"})
    assert "f1" in b.load_seen_ids() and "f1" in a.load_seen_ids()
    m = json.loads((tmp_path / "manifest.json").read_text())
    assert m["seen_file_ids"] == ["f1"] and m["seen_file_ids_by_collection"] == {POLY: ["f1"]}


def test_two_libraries_saving_at_once_lose_nothing(monkeypatch, tmp_path):
    a, b = _copy(monkeypatch, tmp_path, COMP), _copy(monkeypatch, tmp_path, POLY)

    def save(pipe, prefix):
        for i in range(40):
            pipe.save_seen_ids({f"{prefix}{i}"})
    ts = [threading.Thread(target=save, args=(a, "a")), threading.Thread(target=save, args=(b, "b"))]
    [t.start() for t in ts]
    [t.join(60) for t in ts]
    assert a.load_seen_ids() == {f"a{i}" for i in range(40)}
    assert b.load_seen_ids() == {f"b{i}" for i in range(40)}


def test_the_seen_ids_are_saved_in_one_locked_step():
    # load-then-save would let another writer's entries in between be overwritten; the save is one locked block
    src = (ROOT / "memento/ingestion/upload/document.py").read_text()
    body = src[src.index("    def save_seen_ids("):]
    body = body[:body.index("\n    def ", 10)]
    assert "locked_manifest(" in body and "self.load_manifest(" not in body
