# The chat pipe and the command line apply the same library decision.
import hashlib
import json
import sys
import types
from pathlib import Path
from memento.config.paths import PROJECT_ROOT
from _owui_loader import load_function
from memento.ingestion.upload import document
from memento.ingestion.upload import workflow
from memento.ingestion import library_index

ROOT = PROJECT_ROOT
sys.path.insert(0, str(ROOT))

ip = load_function(ROOT / "openwebui/functions/ingestion_pipe.py", "memento_ingestion")

C = "memento-comp-mat-rag"


def _pipe(tmp_path, monkeypatch, library: set):
    lib_dir = tmp_path / "library"
    lib_dir.mkdir()
    monkeypatch.setattr(document, "MANIFEST_FILE", tmp_path / "manifest.json")
    p = document.Ingester(ip.Pipe().valves)
    p.valves.COLLECTION = C
    monkeypatch.setattr(p, "input_dir", lambda: lib_dir)
    monkeypatch.setattr(p, "in_library", lambda stem: stem in library)
    processed = []

    def fake_process(docs):
        processed.extend(docs)
        library.update(d.stem for d in docs)          # "indexed"
        yield "processed\n"
    monkeypatch.setattr(workflow, "process_docs", lambda ing, *a_, **k_: (fake_process)(*a_, **k_))
    return p, lib_dir, processed


def _upload(tmp_path, name, content, fid):
    src = tmp_path / f"owui_{fid}_{name}"                  # OpenWebUI's stored copy
    src.write_bytes(content)
    return {"id": fid, "name": name, "file": {"path": str(src)}}


def test_second_person_same_name_other_content_is_kept_apart(tmp_path, monkeypatch):
    library = set()
    p, lib_dir, processed = _pipe(tmp_path, monkeypatch, library)
    anna, ben = {"id": "u-anna", "name": "Anna Schmidt"}, {"id": "u-ben", "name": "Ben Müller"}
    out = "".join(workflow.handle_uploads(p, [_upload(tmp_path, "paper.pdf", b"ANNA", "f1")], anna))
    assert (lib_dir / "paper.pdf").read_bytes() == b"ANNA" and "processed" in out
    out = "".join(workflow.handle_uploads(p, [_upload(tmp_path, "paper.pdf", b"BEN", "f2")], ben))
    assert (lib_dir / "paper.pdf").read_bytes() == b"ANNA"                       # Anna's document untouched
    assert (lib_dir / "paper__ben-muller.pdf").read_bytes() == b"BEN"
    assert "added as **paper__ben-muller.pdf**" in out
    man = json.loads((tmp_path / "manifest.json").read_text())["documents"]
    assert man["paper"]["uploaded_by"] == "u-anna" and man["paper__ben-muller"]["uploaded_by"] == "u-ben"


def test_same_person_new_version_replaces_and_copy_under_other_name_is_skipped(tmp_path, monkeypatch):
    library = set()
    p, lib_dir, processed = _pipe(tmp_path, monkeypatch, library)
    anna = {"id": "u-anna", "name": "Anna"}
    "".join(workflow.handle_uploads(p, [_upload(tmp_path, "paper.pdf", b"V1", "f1")], anna))
    "".join(workflow.handle_uploads(p, [_upload(tmp_path, "paper.pdf", b"V2", "f2")], anna))
    assert (lib_dir / "paper.pdf").read_bytes() == b"V2"                         # a new version
    processed.clear()
    out = "".join(workflow.handle_uploads(p, [_upload(tmp_path, "paper (1).pdf", b"V2", "f3")], {"id": "u-ben", "name": "B"}))
    assert "already in the library as **paper**" in out and not processed
    assert not (lib_dir / "paper (1).pdf").exists()


def test_folder_path_uses_the_server_identity_and_replaces_changed_files(tmp_path, monkeypatch):
    library = {"paper"}
    p, lib_dir, processed = _pipe(tmp_path, monkeypatch, library)
    (lib_dir / "paper.pdf").write_bytes(b"OLD")
    with library_index.locked_manifest(document.MANIFEST_FILE) as m:        # indexed from the command line earlier
        m["documents"]["paper"] = {"content_hash": hashlib.sha256(b"OLD").hexdigest(), "collection": C}
    src = tmp_path / "incoming"
    src.mkdir()
    (src / "paper.pdf").write_bytes(b"NEW")
    decisions = p.decide([("paper.pdf", src / "paper.pdf")], library_index.SERVER_UPLOADER, "server")
    assert [d.action for d in decisions] == ["changed"]                  # server-side document, server-side update


# ── command line ──────────────────────────────────────────────────────────────────────────────────────────────
def _cli(tmp_path, monkeypatch, library: set, manifest: dict):
    import memento.ingestion.run_pipeline as rp
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(rp, "_MANIFEST_FILE", tmp_path / "manifest.json")
    monkeypatch.setattr(rp, "OUTPUT_DIR", tmp_path / "out")
    monkeypatch.setattr(rp, "COLLECTION", C)
    monkeypatch.setattr(rp, "already_ingested", lambda stem, client: stem in library)
    monkeypatch.setattr(rp, "CHANGED_STEMS", set())
    return rp


def test_cli_documents_indexed_before_hashes_existed_are_not_reindexed(tmp_path, monkeypatch):
    docs = []
    for i in range(3):
        f = tmp_path / f"paper{i}.pdf"
        f.write_bytes(f"P{i}".encode())
        docs.append(f)
    legacy = {"documents": {f"paper{i}": {"vector_count": 10, "collection": C, "ingested_by": "run_pipeline"}
                            for i in range(3)}}
    rp = _cli(tmp_path, monkeypatch, {"paper0", "paper1", "paper2"}, legacy)
    assert rp.plan_library(docs, client=None) == docs and rp.CHANGED_STEMS == set()
    man = json.loads((tmp_path / "manifest.json").read_text())["documents"]
    assert all(man[f"paper{i}"]["content_hash"] for i in range(3))            # hashes adopted, nothing else


def test_cli_renames_via_staged_symlink_drops_duplicates_and_marks_changes(tmp_path, monkeypatch):
    h = lambda b: hashlib.sha256(b).hexdigest()
    manifest = {"documents": {
        "paper": {"content_hash": h(b"ANNA"), "collection": C, "uploaded_by": "u-anna"},     # a chat upload
        "notes": {"content_hash": h(b"OLD"), "collection": C},                                # server-side
    }}
    rp = _cli(tmp_path, monkeypatch, {"paper", "notes"}, manifest)
    folder = tmp_path / "in"
    folder.mkdir()
    for name, content in (("paper.pdf", b"OTHER"), ("notes.pdf", b"NEW"), ("copy.pdf", b"ANNA")):
        (folder / name).write_bytes(content)
    kept = rp.plan_library([folder / "paper.pdf", folder / "notes.pdf", folder / "copy.pdf"], client=None)
    assert [p.name for p in kept] == ["paper__server.pdf", "notes.pdf"]           # copy.pdf = Anna's paper: dropped
    assert kept[0].is_symlink() and kept[0].resolve() == (folder / "paper.pdf").resolve()
    assert (folder / "paper.pdf").exists() and not (folder / "paper__server.pdf").exists()   # operator's file as is
    assert rp.CHANGED_STEMS == {"notes"}
