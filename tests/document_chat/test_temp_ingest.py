# Memento Document Chat: attached files read into the temporary store. The library's ingestion keeps its behaviour (hook
# defaults).
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from memento.config.paths import PROJECT_ROOT
from memento.ingestion.upload import document

ROOT = PROJECT_ROOT
sys.path.insert(0, str(ROOT))
from memento.document_chat import intake as ti  # noqa: E402
from memento.ingestion.upload import workflow

CHAT, USER = "3f2b8c1e-6a4d-4e0b-9c7a-1d2e3f4a5b6c", "u1"


@pytest.fixture
def roots(tmp_path):
    return tmp_path / "input", tmp_path / "output"


def _ing(roots, **valves):
    return ti.make_ingester(CHAT, USER, input_root=roots[0], output_root=roots[1], collection="memento-x-rag",
                            valves=valves)


def test_the_library_pipe_keeps_marking_and_adds_nothing(monkeypatch):
    ip = ti.ingestion_module()
    seen = []
    monkeypatch.setattr(document, "set_paper_in_progress", lambda stem, flag: seen.append((stem, flag)))
    p = document.Ingester(ip.Pipe().valves)
    p.mark_in_progress("doc", True)
    assert seen == [("doc", True)] and p.extra_payload() == {}


def test_the_temporary_ingester_has_its_own_folders_and_collection(roots):
    p = _ing(roots, GRAPH_BUILD=True, RESEARCH_GAP_TAG=True, FORCE_REEMBED=True, SKIP_PARSE=True,
             COLLECTION="memento-x-rag", MINERU_URL="http://mineru:1", EMBED_BATCH_SIZE=4)
    assert p.input_dir() == roots[0] / "_temp" / CHAT
    assert p.output_dir() == roots[1] / "_temp"
    assert p.valves.COLLECTION == "memento-x-temp"
    assert not (p.valves.GRAPH_BUILD or p.valves.RESEARCH_GAP_TAG or p.valves.FORCE_REEMBED or p.valves.SKIP_PARSE)
    assert (p.valves.MINERU_URL, p.valves.EMBED_BATCH_SIZE) == ("http://mineru:1", 4)     # saved settings kept
    assert p.extra_payload() == {"chat_id": CHAT, "user_id": USER}


def test_the_temporary_ingester_never_touches_the_library(roots, monkeypatch):
    monkeypatch.setattr(document, "set_paper_in_progress", lambda *a: pytest.fail("library 'being processed' row"))
    p = _ing(roots)
    p.mark_in_progress("x", True)
    p.register_document("x", "h")           # no library row, no warm-up job (would need the database)


def test_a_chat_id_that_is_not_a_saved_chat_is_refused(roots):
    with pytest.raises(ValueError):
        ti.make_ingester("../etc", USER, input_root=roots[0], output_root=roots[1], collection="c-rag", valves={})


def test_temp_collection_name():
    assert ti.temp_collection("memento-comp-mat-rag") == "memento-comp-mat-temp"
    assert ti.temp_collection("custom") == "custom-temp"


class FakeQC:
    def __init__(self, *a, **kw):
        FakeQC.inst = self
        self.points = []

    def count(self, *a, **kw):
        return SimpleNamespace(count=0)

    def upsert(self, collection_name, points):
        self.points += points


def test_every_stored_passage_carries_the_chat_and_the_user(roots, monkeypatch):
    p = _ing(roots)
    stem = "Paper-0123456789ab"
    out = roots[1] / "_temp" / stem / p.out_subdir()
    out.mkdir(parents=True)
    (out / f"{stem}_content_list.json").write_text(json.dumps([
        {"type": "text", "text": "Silicon carbide is stiff.", "page_idx": 0},
        {"type": "text", "text": "It is also hard.", "page_idx": 1}]))
    pdf = p.input_dir() / f"{stem}.pdf"
    pdf.parent.mkdir(parents=True, exist_ok=True)
    pdf.write_bytes(b"%PDF-1.4 fake")
    import qdrant_client
    import memento.common.model_servers as vc
    import memento.ingestion.enrich_parsed_content_list as en
    import memento.ingestion.embed as sp
    monkeypatch.setattr(qdrant_client, "QdrantClient", FakeQC)
    monkeypatch.setattr(vc, "vl_embed_many", lambda url, model, inputs: [[0.1] * 4 for _ in inputs])
    monkeypatch.setattr(sp, "sparse_encode", lambda texts: [SimpleNamespace(indices=[], values=[]) for _ in texts])
    monkeypatch.setattr(en, "enrich_content_list", lambda raw, source_pdf=None: str(raw))
    result = [r for r in p.embed_one(pdf) if isinstance(r, dict)][-1]
    assert result["ok"] and result["vectors"] == 2
    assert all(pt.payload["chat_id"] == CHAT and pt.payload["user_id"] == USER and pt.payload["source"] == stem
               for pt in FakeQC.inst.points)


def test_a_pdf_over_the_page_limit_is_refused_before_reading(roots, monkeypatch):
    monkeypatch.setattr(ti, "PdfReader", lambda path: SimpleNamespace(pages=[0] * 151))
    p = _ing(roots)
    monkeypatch.setattr(p, "parse_one", lambda path: pytest.fail("sent to the parser"))
    src = p.input_dir() / "Big-0123456789ab.pdf"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_bytes(b"%PDF")
    r = ti.read_file(p, src, max_pages=150)
    assert not r["ok"] and r["reason"] == "too_many_pages" and r["pages"] == 151


def test_the_temporary_store_indexes_the_chat_and_the_user():
    made = {}

    class C:
        def collection_exists(self, name):
            return False

        def create_collection(self, name, **kw):
            made["collection"] = name

        def get_collection(self, name):
            return SimpleNamespace(payload_schema={k: 1 for k in made if k != "collection"})

        def create_payload_index(self, collection_name, field_name, field_schema, wait=True):
            made[field_name] = field_schema
    ti.ensure_store(C(), "memento-x-temp")
    assert made["collection"] == "memento-x-temp"
    assert getattr(made["chat_id"], "is_tenant", False) is True and "user_id" in made and "source" in made


def test_a_file_that_fails_half_way_leaves_no_passages(roots, monkeypatch):
    deleted = []

    class Q:
        def __init__(self, *a, **k):
            pass

        def delete(self, collection_name, points_selector):
            deleted.append((collection_name, points_selector.filter.must[0].match.value))
    import qdrant_client
    monkeypatch.setattr(qdrant_client, "QdrantClient", Q)
    monkeypatch.setattr(ti, "ensure_store", lambda client, coll: None)
    p = _ing(roots)
    monkeypatch.setattr(p, "parse_one", lambda path: iter([{"ok": True, "pages": 3}]))
    monkeypatch.setattr(p, "embed_one", lambda path: iter([{"ok": False, "error": "embedding failed: boom"}]))
    src = p.input_dir() / "Doc-0123456789ab.docx"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_bytes(b"x")
    r = ti.read_file(p, src, max_pages=150)
    assert not r["ok"] and r["reason"] == "store_failed"
    assert deleted == [("memento-x-temp", "Doc-0123456789ab")]


# ── progress while reading: the library upload's bar ──

def test_reading_and_storing_are_reported_as_they_go(roots, monkeypatch):
    import qdrant_client
    monkeypatch.setattr(qdrant_client, "QdrantClient", lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr(ti, "ensure_store", lambda client, coll: None)
    p = _ing(roots)
    monkeypatch.setattr(p, "parse_one", lambda path: iter(["⏳ reading... 10s elapsed\n", {"ok": True, "pages": 7}]))
    monkeypatch.setattr(p, "embed_one", lambda path: iter(["💾 Storing... 40% (40/100 passages)\n",
                                                            {"ok": True, "vectors": 100}]))
    src = p.input_dir() / "Doc-0123456789ab.docx"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_bytes(b"x")
    seen = []
    r = ti.read_file(p, src, max_pages=150, progress=lambda phase, **kw: seen.append((phase, kw)))
    assert r["ok"]
    assert seen == [("parse", {"pages": None}), ("embed", {"pages": 7, "pct": 0}), ("embed", {"pct": 40})]


def test_the_reading_bar_is_the_library_uploads_with_files_read():
    from test_upload_counts import _labels
    bar = ti.ReadProgress(2)
    bar.update("parse", pages=10)
    line = bar.line()
    assert line.startswith("Uploading... 0 of 2 ready (1 parsing, 0 storing) · 0 read · 0 unchanged · 0 failed · "
                           "0m elapsed · ")
    assert _labels(line)[1] == "0 of 2 documents read · under 1 min"
    bar.update("embed", pct=50)
    assert bar.line().endswith("(0 parsing, 1 storing) · 0 read · 0 unchanged · 0 failed · 0m elapsed · 40% done")
    bar.finish(True)
    bar.update("parse")
    bar.finish(False)
    assert bar.line() == "Done · 1 of 2 read · 0 unchanged · 1 failed · 0m elapsed"
    assert _labels(bar.line()) == ["Done with errors", "1 of 2 documents read · 1 failed · under 1 min", True]


def test_the_bar_shows_a_new_step_when_a_file_starts_moves_on_or_ends():
    bar = ti.ReadProgress(1)
    steps = [bar.step]
    for change in (lambda: bar.update("parse"), lambda: bar.update("parse", pages=3), lambda: bar.update("embed"),
                   lambda: bar.update("embed", pct=30), lambda: bar.finish(True)):
        change()
        steps.append(bar.step)
    assert steps == [0, 1, 1, 2, 2, 3]


def test_without_a_page_limit_a_long_pdf_is_read(roots, monkeypatch):
    monkeypatch.setattr(ti, "PdfReader", lambda path: SimpleNamespace(pages=[0] * 2000))
    p = _ing(roots)
    sent = []
    monkeypatch.setattr(p, "parse_one", lambda path: sent.append(path) or iter([{"ok": False, "error": "stop"}]))
    src = p.input_dir() / "Book-0123456789ab.pdf"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_bytes(b"%PDF")
    r = ti.read_file(p, src, max_pages=0)
    assert sent and r["reason"] == "parse_failed" and r["pages"] == 2000


def test_the_chat_folder_records_its_owner(roots):
    _ing(roots)
    owner = Path(roots[0]) / "_temp" / CHAT / ti.OWNER_FILE
    assert owner.read_text() == USER
    _ing(roots)                                   # the same account again: unchanged
    assert owner.read_text() == USER


def test_a_chat_folder_of_another_account_is_refused(roots):
    _ing(roots)
    with pytest.raises(ValueError):
        ti.make_ingester(CHAT, "someone-else", input_root=roots[0], output_root=roots[1], collection="c", valves={})
    assert (Path(roots[0]) / "_temp" / CHAT / ti.OWNER_FILE).read_text() == USER


def test_two_accounts_claiming_one_new_folder_at_once_get_one_owner(roots, monkeypatch):
    ti.claim_chat_folder(roots[0], CHAT, USER)
    with monkeypatch.context() as m:       # the second claim passed its "no owner yet" check before the first wrote
        m.setattr(Path, "is_file", lambda self: False)
        m.setattr(Path, "exists", lambda self: False)
        with pytest.raises(ValueError):
            ti.claim_chat_folder(roots[0], CHAT, "someone-else")
    assert (Path(roots[0]) / "_temp" / CHAT / ti.OWNER_FILE).read_text() == USER


def test_parallel_claims_of_the_same_account_all_succeed(roots):
    import threading
    start, errors = threading.Barrier(8), []

    def claim():
        start.wait()
        try:
            ti.claim_chat_folder(roots[0], CHAT, USER)
        except Exception as e:                 # a half-written record read by a parallel claim would land here
            errors.append(e)
    threads = [threading.Thread(target=claim) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    folder = Path(roots[0]) / "_temp" / CHAT
    assert errors == [] and (folder / ti.OWNER_FILE).read_text() == USER
    assert sorted(p.name for p in folder.iterdir()) == [ti.OWNER_FILE]       # no temporary file left
