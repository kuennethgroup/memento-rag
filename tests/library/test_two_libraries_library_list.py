# One start-screen document list per library. Each library's list is OUTPUT_ROOT/_runtime/library/<collection>.json; the
# default library's list is also written to _runtime/library.json, which the file server's /library serves. The file
# server serves a library's list as /library?collection=<c>. The shared runtime files are written through temporary
# names unique per thread: both copies run in one process.
import json
import sys
import threading
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from memento.config.paths import PROJECT_ROOT
from memento.query.scope import documents

ROOT = PROJECT_ROOT
sys.path.insert(0, str(ROOT))
from memento.library import documents as ldb  # noqa: E402
from memento.library import documents as ll  # noqa: E402
from test_file_server_view import client  # noqa: E402,F401

COMP, POLY = "memento-comp-mat-rag", "memento-polymer-rag"


def test_today_s_collection_is_the_one_behind_the_old_file():
    assert ll.LEGACY_COLLECTION == COMP


def test_each_collection_has_its_own_file(tmp_path):
    assert ll.library_path(tmp_path) == tmp_path / "_runtime" / "library.json"                       # as before
    assert ll.library_path(tmp_path, POLY) == tmp_path / "_runtime" / "library" / f"{POLY}.json"
    ll.write_library(tmp_path, [{"source": "p1", "title": "Foams"}], collection=POLY)
    ll.write_library(tmp_path, [{"source": "c1", "title": "DFT"}], collection=COMP)
    assert [d["source"] for d in ll.read_library(tmp_path, POLY)["documents"]] == ["p1"]
    assert [d["source"] for d in ll.read_library(tmp_path, COMP)["documents"]] == ["c1"]
    # today's list also in the old file; the polymer list never
    assert [d["source"] for d in ll.read_library(tmp_path)["documents"]] == ["c1"]
    ll.write_library(tmp_path, [{"source": "p2", "title": "Epoxy"}], collection=POLY)
    assert [d["source"] for d in ll.read_library(tmp_path)["documents"]] == ["c1"]


def test_today_s_list_falls_back_to_the_old_file_until_its_own_exists(tmp_path):
    ll.write_library(tmp_path, [{"source": "c1", "title": "DFT"}])                  # written by today's code
    assert [d["source"] for d in ll.read_library(tmp_path, COMP)["documents"]] == ["c1"]
    assert ll.read_library(tmp_path, POLY) is None                                   # another library: never


def test_unsafe_collection_names_are_refused(tmp_path):
    for bad in ("../x", "a/b", "", "UPPER", "x" * 81):
        with pytest.raises(ValueError):
            ll.library_path(tmp_path, bad)


def test_adding_to_a_list_keeps_to_its_library(tmp_path):
    ll.write_library(tmp_path, [{"source": "p1", "title": "Foams"}], collection=POLY)
    assert ll.add_to_library(tmp_path, [{"source": "p2", "title": "Epoxy"}], collection=POLY)
    assert {d["source"] for d in ll.read_library(tmp_path, POLY)["documents"]} == {"p1", "p2"}
    assert ll.read_library(tmp_path) is None


@pytest.fixture
def engine():
    e = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    ldb.ensure_tables(e)
    return e


def test_the_export_writes_each_collections_own_file(engine, tmp_path):
    ldb.add_document(engine, collection=POLY, entry={"source": "p1", "title": "Foams"}, content_hash="h",
                     uploaded_by="u", output_root=tmp_path)
    ldb.add_document(engine, collection=COMP, entry={"source": "c1", "title": "DFT"}, content_hash="h",
                     uploaded_by="u", output_root=tmp_path)
    assert [d["source"] for d in ll.read_library(tmp_path, POLY)["documents"]] == ["p1"]
    assert [d["source"] for d in ll.read_library(tmp_path, COMP)["documents"]] == ["c1"]
    assert [d["source"] for d in ll.read_library(tmp_path)["documents"]] == ["c1"]


# ── the Query pipe reads its own collection's list ────────────────────────────

def _qp():
    sys.path.insert(0, str(ROOT / "openwebui/functions"))
    import query_pipe
    return query_pipe


def test_a_picked_title_resolves_in_the_copys_own_list(tmp_path, monkeypatch):
    qp = _qp()
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path))
    ll.write_library(tmp_path, [{"source": "c_doc", "title": "Thermal analysis"}], collection=COMP)
    ll.write_library(tmp_path, [{"source": "p_doc", "title": "Thermal analysis"}], collection=POLY)
    q = 'Summarize the document "Thermal analysis"'
    assert documents.picked_document(q, {"p_doc"}, POLY) == ("p_doc", "summary")
    assert documents.picked_document(q, {"c_doc"}, COMP) == ("c_doc", "summary")
    assert documents.picked_document(q, {"p_doc"}, COMP) is None                     # comp-mat's list, polymer sources


def test_two_copies_do_not_re_export_each_others_list(tmp_path, monkeypatch, engine):
    qp = _qp()
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path))
    monkeypatch.setattr(qp, "_db_engine", lambda: engine)
    for coll, src in ((COMP, "c1"), (POLY, "p1")):
        ldb.add_document(engine, collection=coll, entry={"source": src, "title": src.upper()}, content_hash="h",
                         uploaded_by="u", output_root=None)
    exports = []
    real = ldb.export_library
    monkeypatch.setattr(ldb, "export_library", lambda e, c, root: exports.append(c) or real(e, c, root))
    for _ in range(3):                                   # both copies answer, in turn
        documents.publish_library(COMP, {"c1"}, qp._db_engine)
        documents.publish_library(POLY, {"p1"}, qp._db_engine)
    assert exports == [COMP, POLY]                        # each once, then never again


# ── the file server ──────────────────────────────────────────────────────────

def test_the_file_server_serves_each_collections_list(client):
    c, fs = client
    ll.write_library(fs.OUTPUT_DIR, [{"source": "p1", "title": "Foams"}], collection=POLY)
    ll.write_library(fs.OUTPUT_DIR, [{"source": "c1", "title": "DFT"}], collection=COMP)
    r = c.get("/library", params={"collection": POLY})
    assert r.status_code == 200 and [d["source"] for d in r.json()["documents"]] == ["p1"]
    assert r.headers["cache-control"] == "no-store"
    assert [d["source"] for d in c.get("/library").json()["documents"]] == ["c1"]          # the old file: today's
    assert [d["source"] for d in c.get("/library", params={"collection": COMP}).json()["documents"]] == ["c1"]


def test_todays_library_is_served_from_the_old_file_until_its_own_exists(client):
    # read_library's rule: the default copy re-exports nothing while the single-library file is current, so its own file
    # may not exist yet; the picker must not call the library empty then
    c, fs = client
    ll.write_library(fs.OUTPUT_DIR, [{"source": "c1", "title": "DFT"}])                  # the old file only
    r = c.get("/library", params={"collection": COMP})
    assert r.status_code == 200 and [d["source"] for d in r.json()["documents"]] == ["c1"]
    assert c.get("/library", params={"collection": POLY}).status_code == 404             # another library: none
    assert fs.LEGACY_COLLECTION == ll.LEGACY_COLLECTION                                  # one name in both places


@pytest.mark.parametrize("bad", ["..", "../_runtime/library", "a/b", "memento-unknown-rag", "UPPER", "x" * 81])
def test_the_file_server_refuses_unknown_or_unsafe_collections(client, bad):
    c, _ = client
    assert c.get("/library", params={"collection": bad}).status_code == 404


# ── shared runtime files written by two copies in one process (N8) ────────────

def _hammer(write, read, n=200):
    errors, bad = [], []

    def writer(k):
        try:
            for i in range(n):
                write(k, i)
        except Exception as e:          # noqa: BLE001
            errors.append(repr(e))
    stop = threading.Event()

    def reader():
        while not stop.is_set():
            try:
                read()
            except FileNotFoundError:
                pass
            except ValueError as e:
                bad.append(repr(e))
    ws = [threading.Thread(target=writer, args=(k,)) for k in range(2)]
    r = threading.Thread(target=reader)
    r.start()
    [w.start() for w in ws]
    [w.join(60) for w in ws]
    stop.set()
    r.join(10)
    return errors, bad


def test_two_threads_write_the_quota_state_without_a_broken_file(tmp_path):
    from memento.common import cloud_quota as cq
    path = cq.state_path(tmp_path)
    errors, bad = _hammer(lambda k, i: cq.write_state(path, {"n": i, "by": k}, ["m"]),
                          lambda: json.loads(path.read_text()))
    assert errors == [] and bad == [] and json.loads(path.read_text())["cloud_models"] == ["m"]


def test_two_threads_write_a_library_list_without_a_broken_file(tmp_path):
    errors, bad = _hammer(lambda k, i: ll.write_library(tmp_path, [{"source": f"s{k}{i}", "title": "t"}],
                                                        collection=POLY),
                          lambda: json.loads(ll.library_path(tmp_path, POLY).read_text()))
    assert errors == [] and bad == []


def test_two_threads_write_the_model_map_without_a_broken_file(tmp_path):
    from memento.query import models as qm
    variants = qm.load_variants()
    path = tmp_path / "_runtime" / "query_models.json"
    errors, bad = _hammer(lambda k, i: qm.write_model_colours(path, variants), lambda: json.loads(path.read_text()))
    assert errors == [] and bad == []
    assert not list(path.parent.glob("*.tmp"))                                    # no temporary file left behind
