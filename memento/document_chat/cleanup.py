"""Document Chat's leftovers: the files of chats that no longer exist (their passages, their folders, Open WebUI's
own copy of the upload), and old draft-proposal PDFs and paper cards.
"""
import os
import re
import time
from pathlib import Path

from memento.config.paths import PROJECT_ROOT
from memento.query import library_state


DOC_CHAT_CLEANUP_S = 3600


def existing_chat_ids(engine, candidates: set | None = None) -> set | None:
    """Which of `candidates` (Document Chat's chats) still exist in OpenWebUI's chat table (all chat ids when None);
    None when the table cannot be read (then nothing is deleted)."""
    from sqlalchemy import bindparam, text
    try:
        with engine.connect() as c:
            if candidates is None:
                return {r[0] for r in c.execute(text("SELECT id FROM chat"))}
            if not candidates:
                return set()
            q = text("SELECT id FROM chat WHERE id IN :ids").bindparams(bindparam("ids", expanding=True))
            return {r[0] for r in c.execute(q, {"ids": sorted(candidates)})}
    except Exception:
        return None


def remove_owui_upload(engine, file_id: str, user_id: str, upload_dir=None) -> bool:
    """Open WebUI's own copy of a Document Chat upload (its file row with the extracted text, and the file in its
    uploads folder): Open WebUI keeps both when a chat is deleted. Removed only when it belongs to this user, no chat or
    knowledge base uses it any more, and its path lies inside the uploads folder."""
    from sqlalchemy import text
    from memento.document_chat.store import _under
    if upload_dir is None:
        from open_webui.config import UPLOAD_DIR as upload_dir
    with engine.begin() as c:
        row = c.execute(text("SELECT user_id, path FROM file WHERE id = :i"), {"i": file_id}).first()
        if row is None or row[0] != user_id:
            return False
        for table in ("chat_file", "knowledge_file"):
            try:
                if c.execute(text(f"SELECT 1 FROM {table} WHERE file_id = :i LIMIT 1"), {"i": file_id}).first():
                    return False
            except Exception:
                return False                       # cannot tell: keep it
        path = Path(row[1]) if row[1] else None
        if path is not None and not _under(Path(upload_dir), path):
            return False
        c.execute(text("DELETE FROM file WHERE id = :i AND user_id = :u"), {"i": file_id, "u": user_id})
    if path is not None and path.is_file():
        path.unlink()
    return True


def remove_old_reports(output_root: str, days: int = 30) -> int:
    """Draft-proposal PDFs (OUTPUT_ROOT/_runtime/reports/<24 hex>.pdf) and paper cards (…/papers/<24 hex>.json) older
    than `days`: removed (hourly worker)."""
    cut, n = time.time() - days * 86400, 0
    for sub, ext in (("reports", "pdf"), ("papers", "json")):
        d = Path(output_root) / "_runtime" / sub
        if not d.is_dir():
            continue
        for f in d.glob(f"*.{ext}"):
            try:
                if re.fullmatch(rf"[0-9a-f]{{24}}\.{ext}", f.name) and f.stat().st_mtime < cut:
                    f.unlink()
                    n += 1
            except FileNotFoundError:       # the other library's copy removed it first (both clean this folder)
                continue
    return n


def doc_chat_cleanup(valves, engine, *, function_id: str, db_engine) -> int:
    """Document Chat's files of chats that no longer exist (store.cleanup). Best effort."""
    import logging
    try:
        from qdrant_client import QdrantClient
        from memento.document_chat.intake import temp_collection
        from memento.document_chat import store as td
        td.ensure_tables(engine)
        ingestion_valves = library_state.saved_valves(valves.INGESTION_FUNCTION_ID, db_engine) or {}
        in_root = ingestion_valves.get("INPUT_ROOT") or os.getenv("INPUT_ROOT") or str(PROJECT_ROOT / "data/input")
        out_root = ingestion_valves.get("OUTPUT_ROOT") or os.getenv("OUTPUT_ROOT") or str(PROJECT_ROOT / "data/output")
        coll = temp_collection(valves.COLLECTION)
        with td.lifecycle(engine, coll):           # a second copy's cleaner, shared, for its whole run
            if function_id not in library_state.DEFAULT_IDS and not library_state.copy_state(
                    "query", cached=False, function_id=function_id, db_engine=db_engine).enabled:
                return 0                           # switched off while it waited (removal): nothing
            return td.cleanup(engine, QdrantClient(url=valves.QDRANT_URL, timeout=30), coll, in_root, out_root,
                              existing_chat_ids(engine, td.chat_ids(engine)),
                              remove_upload=lambda fid, uid: remove_owui_upload(engine, fid, uid))
    except Exception as e:
        logging.getLogger("memento.errors").warning("E-docchat-cleanup: %r", e)
        return 0
