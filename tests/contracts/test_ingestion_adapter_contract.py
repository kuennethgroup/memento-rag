# Contract of the Library Upload pipe: for each scenario, what the real pipe replies and writes (its text and status
# events, the database lock around the run, Qdrant writes, the manifest, the document list and the stored files)
# against fakes at the service boundaries only: Open WebUI's file records and database, MinerU's HTTP API, the
# embedder and Qdrant. The golden transcripts in tests/fixtures/contracts/ingestion/ pin today's behaviour;
# MEMENTO_UPDATE_CONTRACTS=1 rewrites them after an intended change.
import asyncio
import json
import types
from pathlib import Path

import pytest
from sqlalchemy import create_engine, event, text
from sqlalchemy.pool import StaticPool

from _contract_fakes import FakeLLM, FakeQdrant, SyncFakeLLM, check_golden
from memento.config.paths import PROJECT_ROOT
from _owui_loader import load_function
from memento.ingestion.upload import document

ROOT = PROJECT_ROOT
COLLECTION = "memento-x-rag"
CONTENT = [{"type": "text", "text": "A study of sample hardness", "text_level": 1, "page_idx": 0},
           {"type": "text", "text": "Alice Example, Bob Sample", "page_idx": 0},
           {"type": "text", "text": "1. Introduction", "text_level": 1, "page_idx": 0},
           {"type": "text", "text": "The samples reach a hardness of 12 GPa after heat treatment.", "page_idx": 0},
           {"type": "text", "text": "2. Results", "text_level": 1, "page_idx": 1},
           {"type": "text", "text": "Hardness rises with the treatment temperature.", "page_idx": 1}]


class FakeQdrantStore(FakeQdrant):
    """FakeQdrant plus the writes an upload makes."""

    def get_collections(self):
        return types.SimpleNamespace(collections=[types.SimpleNamespace(name=n) for n in FakeQdrant.collections])

    def create_collection(self, collection_name, **k):
        FakeQdrant.collections.setdefault(collection_name, [])
        FakeQdrant.log.append({"op": "create_collection", "collection": collection_name})

    def get_collection(self, collection_name):
        return types.SimpleNamespace(payload_schema={}, points_count=len(FakeQdrant.collections.get(collection_name, [])))

    def create_payload_index(self, collection_name, field_name, **k):
        FakeQdrant.log.append({"op": "payload_index", "collection": collection_name, "field": field_name})

    def upsert(self, collection_name, points, **k):
        points = list(points)
        FakeQdrant.collections.setdefault(collection_name, []).extend(p.payload for p in points)
        FakeQdrant.log.append({"op": "upsert", "collection": collection_name, "points": len(points),
                               "types": sorted({p.payload.get("type") for p in points}),
                               "payload_keys": sorted({k for p in points for k in p.payload})})

    def delete(self, collection_name, points_selector=None, **k):
        FakeQdrant.log.append({"op": "delete", "collection": collection_name})

    def set_payload(self, collection_name, payload, **k):
        FakeQdrant.log.append({"op": "set_payload", "collection": collection_name, "keys": sorted(payload)})


class FakeMinerU:
    """requests.post / requests.get for MinerU's task API; `fail` makes the task end as failed."""
    fail = False
    log: list = []

    @staticmethod
    def _resp(data, ok=True):
        return types.SimpleNamespace(ok=ok, status_code=200 if ok else 500, json=lambda: data,
                                     raise_for_status=lambda: None, text=json.dumps(data))

    @classmethod
    def post(cls, url, *a, **k):
        cls.log.append({"op": "post", "url": url.replace("http://mineru.test", "")})
        return cls._resp({"task_id": "task-1"})

    @classmethod
    def get(cls, url, *a, **k):
        path = url.replace("http://mineru.test", "")
        cls.log.append({"op": "get", "url": path})
        if path.endswith("/result"):
            return cls._resp({"results": {"doc": {"content_list": json.dumps(CONTENT), "md_content": "# A study",
                                                  "images": {}}}})
        return cls._resp({"status": "failed" if cls.fail else "completed"})


_PDFS: dict = {}


def _pdf(path: Path, text_: str):
    """A two-page PDF; the same text gives the same bytes (a re-sent file is the same file)."""
    if text_ not in _PDFS:
        import fitz
        doc = fitz.open()
        for i in range(2):
            doc.new_page().insert_text((72, 72), f"{text_} page {i + 1}")
        _PDFS[text_] = doc.tobytes()
    path.write_bytes(_PDFS[text_])


@pytest.fixture
def world(tmp_path, monkeypatch):
    import openai
    import requests
    import memento.common.model_servers as vc
    from open_webui.internal import db as owui_db
    from open_webui.models.files import Files
    from open_webui.models.groups import Groups
    from qdrant_client.models import SparseVector
    import memento.ingestion.embed as sparse
    locks = []
    e = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)

    @event.listens_for(e, "connect")
    def _pg_functions(conn, _):
        conn.create_function("pg_try_advisory_lock", 1, lambda k: locks.append(("lock", k)) or 1)
        conn.create_function("pg_advisory_unlock", 1, lambda k: locks.append(("unlock", k)) or 1)
        conn.create_function("now", 0, lambda: "now")
    monkeypatch.setattr(owui_db, "engine", e)
    records = {}

    async def get_file_by_id(fid):
        return records.get(fid)
    monkeypatch.setattr(Files, "get_file_by_id", staticmethod(get_file_by_id))

    async def groups_of(user_id):
        return [types.SimpleNamespace(name="contributors")]
    monkeypatch.setattr(Groups, "get_groups_by_member_id", staticmethod(groups_of))
    monkeypatch.setattr(sparse, "sparse_encode",           # the BM25 model: one index per distinct word
                        lambda texts: [SparseVector(indices=sorted({sum(map(ord, w)) % 1000 for w in s.split()}),
                                                    values=[1.0] * len({sum(map(ord, w)) % 1000 for w in s.split()}))
                                       for s in texts])
    FakeQdrant.collections, FakeQdrant.log, FakeMinerU.log, FakeMinerU.fail = {}, [], [], False
    FakeLLM.calls, FakeLLM.route, FakeLLM.answer = [], {}, "A short description."
    monkeypatch.setattr("qdrant_client.QdrantClient", FakeQdrantStore)
    monkeypatch.setattr(requests, "post", FakeMinerU.post)
    monkeypatch.setattr(requests, "get", FakeMinerU.get)
    monkeypatch.setattr(openai, "AsyncOpenAI", FakeLLM)
    monkeypatch.setattr(openai, "OpenAI", SyncFakeLLM)
    monkeypatch.setattr(vc, "vl_embed", lambda *a, **k: [0.0] * 8)
    ip = load_function(ROOT / "openwebui/functions/ingestion_pipe.py", "memento_ingestion")
    monkeypatch.setattr(document, "MANIFEST_FILE", tmp_path / "manifest.json")
    p = ip.Pipe()
    for k, v in {"DOMAIN_NAME": "generic", "COLLECTION": COLLECTION, "INPUT_ROOT": str(tmp_path / "in"),
                 "OUTPUT_ROOT": str(tmp_path / "out"), "MINERU_URL": "http://mineru.test",
                 "EMBEDDER_URL": "http://embed.test", "QDRANT_URL": "http://qdrant.test", "PARSE_WORKERS": 1,
                 "EMBED_WORKERS": 1, "GRAPH_BUILD": False, "RESEARCH_GAP_TAG": False,
                 "LLM_BASE_URL": "http://llm.test/v1", "LLM_MODEL": "local-model"}.items():
        setattr(p.valves, k, v)
    return types.SimpleNamespace(pipe=p, ip=ip, engine=e, records=records, locks=locks, tmp=tmp_path)


def _upload(w, user, fid, name, content, *, owned=True):
    stored = w.tmp / "owui" / f"{fid}_{name}"
    stored.parent.mkdir(exist_ok=True)
    _pdf(stored, content)
    if owned:
        w.records[fid] = types.SimpleNamespace(path=str(stored), user_id=user, filename=name)
    return {"id": fid, "name": name, "file": {"path": "/forged/path" if not owned else str(stored)}}


def _send(w, user, files, message="", role="user"):
    events = []

    async def emit(e):
        events.append(e)

    async def run():
        return [c async for c in w.pipe.pipe({"messages": [{"role": "user", "content": message}]},
                                             {"id": user, "role": role, "name": user}, files, emit)]
    FakeQdrant.log, FakeMinerU.log, FakeLLM.calls, w.locks[:] = [], [], [], []
    out = asyncio.run(run())
    manifest = json.loads((w.tmp / "manifest.json").read_text()) if (w.tmp / "manifest.json").exists() else {}
    try:
        with w.engine.connect() as c:
            rows = [dict(r._mapping) for r in c.execute(text(
                "SELECT source, title, uploaded_by, collection FROM memento_documents ORDER BY source"))]
    except Exception:
        rows = []
    return {"reply": "".join(c for c in out if isinstance(c, str)),
            "status": [(ev.get("data") or {}).get("description") for ev in events if ev.get("type") == "status"],
            "locks": [op for op, _ in w.locks], "mineru": FakeMinerU.log, "qdrant": FakeQdrant.log,
            "llm_calls": [{k: c[k] for k in ("stream", "messages", "max_tokens")} for c in FakeLLM.calls],
            "manifest": {s: sorted(d) for s, d in (manifest.get("documents") or {}).items()},
            "library_rows": rows,
            "input_files": sorted(str(p.relative_to(w.tmp / "in")) for p in (w.tmp / "in").rglob("*") if p.is_file())}


def test_a_new_document_is_parsed_embedded_and_listed(world):
    check_golden("ingestion", "new_document", _send(world, "u1", [_upload(world, "u1", "f1", "paper.pdf", "First")]))


def test_the_same_file_again_is_a_duplicate(world):
    first = _send(world, "u1", [_upload(world, "u1", "f1", "paper.pdf", "First")])
    again = _send(world, "u1", [_upload(world, "u1", "f2", "paper.pdf", "First")])
    check_golden("ingestion", "duplicate_upload", {"first": first, "again": again})


def test_new_content_under_the_same_name_from_the_same_person_is_a_new_version(world):
    first = _send(world, "u1", [_upload(world, "u1", "f1", "paper.pdf", "First")])
    new = _send(world, "u1", [_upload(world, "u1", "f2", "paper.pdf", "Second")])
    check_golden("ingestion", "new_version_same_person", {"first": first, "new": new})


def test_the_same_name_from_another_person_is_stored_apart(world):
    first = _send(world, "u1", [_upload(world, "u1", "f1", "paper.pdf", "First")])
    other = _send(world, "u2", [_upload(world, "u2", "f2", "paper.pdf", "Other")])
    check_golden("ingestion", "same_name_other_person", {"first": first, "other": other})


def test_a_file_that_is_not_the_uploaders_is_refused(world):
    check_golden("ingestion", "forged_file_refused",
                 _send(world, "u1", [_upload(world, "u1", "f1", "paper.pdf", "First", owned=False)]))


def test_a_failed_parse_registers_nothing(world):
    FakeMinerU.fail = True
    check_golden("ingestion", "parse_failure", _send(world, "u1", [_upload(world, "u1", "f1", "paper.pdf", "First")]))


@pytest.mark.parametrize("message", ["", "   "])
def test_an_admin_message_without_a_path_reads_nothing(world, monkeypatch, tmp_path, message):
    # an admin's typed text is a server path; no text is no path (never the working directory)
    (tmp_path / "cwd").mkdir()
    _pdf(tmp_path / "cwd" / "stray.pdf", "Stray")
    monkeypatch.chdir(tmp_path / "cwd")
    r = _send(world, "admin1", [_upload(world, "admin1", "f1", "paper.pdf", "First", owned=False)], message, "admin")
    assert "unable to access the uploaded file" in r["reply"]
    assert "Type the path of a folder or file on the server" in r["reply"]
    assert r["mineru"] == [] and r["input_files"] == [] and r["manifest"] == {}
