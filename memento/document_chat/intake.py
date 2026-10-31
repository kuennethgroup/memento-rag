"""Document Chat: an attached file read into the temporary store.

The same reading and storing as the library's uploads (memento/ingestion/upload/document.py, its Ingester):
MinerU, enrichment, image captions by the local model, embedding. What differs:
- folders INPUT_ROOT/_temp/<chat id>/ and OUTPUT_ROOT/_temp/<source>/, collection <prefix>-temp;
- every passage carries `chat_id` and `user_id` (the Document Chat search filters on both);
- never a library document: no library list row, no warm-up, no "being processed" row the library's answers would see,
no knowledge graph, no gap tags, no manifest (only document.Ingester's parse_one + embed_one, never workflow.run_batch)."""
from __future__ import annotations

import importlib.util
import os
import re
import secrets
import threading
import time
from pathlib import Path

from pypdf import PdfReader

from memento.config.paths import PROJECT_ROOT
from memento.document_chat.store import valid_chat_id
from memento.ingestion.convert import ensure_supported, NATIVE_FORMATS
from memento.ingestion.upload import document, intake as upload_intake, messages as upload_messages

ROOT = PROJECT_ROOT
# the saved settings of the library's ingestion that apply here too (services, parser, batch size, caption model)
COPY_VALVES = ("DOMAIN_NAME", "MINERU_URL", "EMBEDDER_URL", "QDRANT_URL", "BACKEND", "EMBED_BATCH_SIZE",
               "LLM_BASE_URL", "LLM_MODEL", "LLM_API_KEY_ENV")
_MODULE, _LOCK = None, threading.Lock()


def ingestion_module():
    """openwebui/functions/ingestion_pipe.py as a module (loaded once from disk): its Valves' defaults."""
    global _MODULE
    with _LOCK:
        if _MODULE is None:
            spec = importlib.util.spec_from_file_location("memento_ingestion_pipe_disk",
                                                          ROOT / "openwebui/functions/ingestion_pipe.py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            _MODULE = mod
    return _MODULE


def temp_collection(library_collection: str) -> str:
    """`<prefix>-rag` → `<prefix>-temp`."""
    base = library_collection[:-4] if library_collection.endswith("-rag") else library_collection
    return f"{base}-temp"


def ensure_store(client, collection: str) -> None:
    """The collection (same vectors as the library's) and its indexes: the library's fields, `chat_id` as the tenant
    key, `user_id`."""
    from qdrant_client.models import (Distance, KeywordIndexParams, Modifier, PayloadSchemaType, SparseIndexParams,
                                      SparseVectorParams, VectorParams)
    from memento.ingestion.payload_indexes import ensure_payload_indexes
    if not client.collection_exists(collection):
        client.create_collection(
            collection,
            vectors_config={"dense": VectorParams(size=4096, distance=Distance.COSINE)},
            sparse_vectors_config={"sparse": SparseVectorParams(index=SparseIndexParams(), modifier=Modifier.IDF)},
        )
    ensure_payload_indexes(client, collection)
    have = set((client.get_collection(collection).payload_schema or {}).keys())
    if "chat_id" not in have:
        client.create_payload_index(collection_name=collection, field_name="chat_id",
                                    field_schema=KeywordIndexParams(type="keyword", is_tenant=True), wait=True)
    if "user_id" not in have:
        client.create_payload_index(collection_name=collection, field_name="user_id",
                                    field_schema=PayloadSchemaType.KEYWORD, wait=True)


def drop_passages(client, collection: str, source: str) -> None:
    """Delete one file's passages (a file that failed half-way, or whose reading was interrupted). Best effort."""
    from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchValue
    try:
        client.delete(collection_name=collection, points_selector=FilterSelector(filter=Filter(must=[
            FieldCondition(key="source", match=MatchValue(value=source))])))
    except Exception:
        pass


# The account a chat's folder belongs to (INPUT_ROOT/_temp/<chat id>/.owner): the file server opens the chat's files for
# that account only (services/file_server.py, same name).
OWNER_FILE = ".owner"


def claim_chat_folder(input_root, chat_id: str, user_id: str) -> Path:
    """INPUT_ROOT/_temp/<chat id>/, recorded as user_id's; ValueError if it is recorded as another account's. The
    record is written to a temporary file and linked into place: the link fails if a record exists already, so of two
    accounts claiming a new folder at once only one wins, and a reader never sees half a record."""
    d = Path(input_root) / "_temp" / chat_id
    d.mkdir(parents=True, exist_ok=True)
    owner = d / OWNER_FILE
    tmp = d / f"{OWNER_FILE}.{os.getpid()}.{threading.get_ident()}.{secrets.token_hex(4)}.tmp"
    try:
        tmp.write_text(user_id)
        os.link(tmp, owner)
    except FileExistsError:
        pass
    finally:
        tmp.unlink(missing_ok=True)
    if owner.read_text().strip() != user_id:
        raise ValueError("this chat's folder belongs to another account")
    return d


def make_ingester(chat_id: str, user_id: str, *, input_root, output_root, collection: str, valves: dict):
    """An ingestion pipe for one chat's files. `valves`: the library ingestion's saved settings (only COPY_VALVES
    are taken); `collection`: the library's (the temporary one is derived)."""
    if not valid_chat_id(chat_id) or not user_id:
        raise ValueError("Document Chat needs a saved chat and a user")
    in_dir = claim_chat_folder(input_root, chat_id, user_id)
    out_dir = Path(output_root) / "_temp"

    class TempIngester(document.Ingester):
        def input_dir(self) -> Path:
            return in_dir

        def output_dir(self) -> Path:
            return out_dir

        def mark_in_progress(self, stem, in_progress):        # not a library document
            pass

        def register_document(self, stem, file_hash=None):    # not a library document
            pass

        def extra_payload(self) -> dict:
            return {"chat_id": chat_id, "user_id": user_id}

    # the defaults of Library Upload's Valves (they stay in the Open WebUI function)
    p = TempIngester(ingestion_module().Pipe.Valves())
    for k in COPY_VALVES:
        if (valves or {}).get(k) is not None:
            setattr(p.valves, k, valves[k])
    p.valves.INPUT_ROOT, p.valves.OUTPUT_ROOT = str(input_root), str(output_root)
    p.valves.COLLECTION = temp_collection(collection)
    for flag in ("GRAPH_BUILD", "RESEARCH_GAP_TAG", "FORCE_REEMBED", "FORCE_REPARSE", "SKIP_PARSE", "SKIP_EMBED",
                 "NO_ENRICH"):
        setattr(p.valves, flag, False)
    return p


def _last_dict(gen, on_line=None) -> dict | None:
    out = None
    for item in gen:
        if isinstance(item, dict):
            out = item
        elif on_line is not None:
            on_line(item)
    return out


_STORING_RE = re.compile(r"Storing\.\.\. (\d+)%")


class ReadProgress:
    """A chat's files, read one after another, shown with the library upload's status line
    (messages.batch_status, done_status), so loader.js draws the same bar. Updated by the reading thread, read
    by the status timer; `step` changes when a file starts, moves on to storing or ends."""

    def __init__(self, n: int):
        self.n, self.t_all = n, time.time()
        self.completed = self.read = self.failed = self.step = 0
        self.phase, self.t0, self.pages, self.pct, self.shown = None, 0.0, None, 0, 0

    def update(self, phase: str, *, pages=None, pct=None) -> None:
        """The file being read: "parse" (reading) or "embed" (storing, `pct` done)."""
        if phase != self.phase:
            self.phase, self.step = phase, self.step + 1
            if phase == "parse":
                self.t0, self.pages, self.pct = time.time(), None, 0
        if pages:
            self.pages = pages
        if pct is not None:
            self.pct = pct

    def finish(self, ok: bool) -> None:
        self.completed += 1
        self.read, self.failed = self.read + bool(ok), self.failed + (not ok)
        self.phase, self.step = None, self.step + 1

    def line(self) -> str:
        now = time.time()
        counts = (self.read, 0, self.failed, "read")
        if self.completed >= self.n:
            return upload_messages.done_status(self.completed, self.n, now - self.t_all, counts=counts)
        share = ([upload_messages.doc_progress("parse", elapsed=now - self.t0, pages=self.pages)] if self.phase == "parse" else
                 [upload_messages.doc_progress("embed", store_pct=self.pct)] if self.phase == "embed" else [])
        self.shown = upload_messages.shown_percent(previous=self.shown,
                                       pct=upload_messages.batch_percent(completed=self.completed, n=self.n, in_progress=share))
        return upload_messages.batch_status(self.completed, self.n, int(self.phase == "parse"), int(self.phase == "embed"),
                                (now - self.t_all) / 60, self.shown, counts=counts)


def read_file(p, path: Path, *, max_pages: int, progress=None) -> dict:
    """Read and store one file (already in p.input_dir()). {"ok", "pages", "vectors"} or {"ok": False, "reason":
    unsupported | too_many_pages | parse_failed | store_failed, "error", "pages"}. Blocking: run it in a thread.
    `max_pages` 0: no limit. `progress(phase, pages=, pct=)`: called as reading ("parse") and storing ("embed") go on (ReadProgress.update)."""
    report = progress or (lambda phase, **kw: None)
    if path.suffix.lower() not in upload_intake.SUPPORTED_EXTENSIONS:
        return {"ok": False, "reason": "unsupported", "error": path.suffix, "pages": None}
    pages = None
    if path.suffix.lower() == ".pdf":
        try:
            pages = len(PdfReader(str(path)).pages)
        except Exception:
            pages = None                                      # MinerU decides
    if max_pages and pages and pages > max_pages:
        return {"ok": False, "reason": "too_many_pages", "error": "", "pages": pages}
    report("parse", pages=pages)
    if path.suffix.lower() not in NATIVE_FORMATS:          # older Office formats
        path = ensure_supported([path], p.input_dir() / "_converted")[0]
    parsed = _last_dict(p.parse_one(path)) or {"ok": False, "error": "no result"}
    if not parsed.get("ok"):
        return {"ok": False, "reason": "parse_failed", "error": parsed.get("error", ""), "pages": pages}
    pages = parsed.get("pages") or pages
    if max_pages and pages and pages > max_pages:
        return {"ok": False, "reason": "too_many_pages", "error": "", "pages": pages}
    from qdrant_client import QdrantClient
    client = QdrantClient(url=p.valves.QDRANT_URL, timeout=30)
    ensure_store(client, p.valves.COLLECTION)
    report("embed", pages=pages, pct=0)

    def storing(line: str) -> None:
        m = _STORING_RE.search(line)
        if m:
            report("embed", pct=int(m.group(1)))
    try:
        stored = _last_dict(p.embed_one(path), storing) or {"ok": False, "error": "no result"}
    except Exception as e:
        stored = {"ok": False, "error": repr(e)}
    if not stored.get("ok"):                       # passages stored before the failure must not stay behind
        drop_passages(client, p.valves.COLLECTION, path.stem)
        return {"ok": False, "reason": "store_failed", "error": stored.get("error", ""), "pages": pages}
    return {"ok": True, "pages": pages, "vectors": stored.get("vectors", 0)}
