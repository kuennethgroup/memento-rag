"""Document Chat on the query side: whether a chat is a Document Chat, reading the files attached to it (once each,
with the upload's progress bar), their names as aliases, the replies when no file is attached or the question is
about the whole library, and the instruction added to the answer prompt.
"""
import asyncio
import os
import re
import time
from pathlib import Path
from urllib.parse import quote

from memento.config.paths import PROJECT_ROOT
from memento.ingestion.upload import intake as upload_intake, messages as upload_messages
from memento.query import library_state, request, texts


# Document Chat answers from the reader's own attached files, but the domain prompt describes a library: the model is
# told the files are not the library, and says what the files do not cover about the files, not about papers found.
DOC_CHAT_ADDENDUM = """

Here the numbered sources are the user's own attached files, not the library. Refer to them as the user's file or \
document (German: Ihre Datei, Ihr Dokument), never as "the library" or as something found in it. When they do not \
cover what was asked, say so about the file or files ("Your document does not mention GANs.", with several files \
"Your documents do not mention GANs."; German: "Ihr Dokument erwähnt GANs nicht.", "Ihre Dokumente erwähnen GANs \
nicht."), not "The papers I found don't mention …"."""


def doc_chat_addendum(dc: bool) -> str:
    return DOC_CHAT_ADDENDUM if dc else ""


# the question names the library or other documents (the router may label a topic question about the attached file
# "library")
_NAMES_LIBRARY_RE = re.compile(
    r"\b(?:librar(?:y|ies)|bibliothek\w*|collection|sammlung|corpus|korpus"
    r"|other\s+(?:documents?|papers?|files?|theses|thesis|studies|books?)"
    r"|andere[nmrs]?\s+(?:dokument\w*|paper\w*|dateien|datei|arbeiten|studien|bücher|buch))\b", re.I)


def doc_chat_library_note(doc_chat: bool, scope: str, question: str) -> str:
    """The whole reply in Document Chat when the question asks about the whole library: the router says so AND the
    question names the library or other documents; anything else is answered from the attached files."""
    if not doc_chat or scope != "library" or not _NAMES_LIBRARY_RE.search(question or ""):
        return ""
    from urllib.parse import quote
    return texts.DOC_CHAT_LIBRARY_NOTE.format(q=quote(question or "", safe=""))


# ── Memento Document Chat ──
# A model of its own (manifold entry "document-chat"): files attached in the chat are read with the library's ingestion
# (memento/document_chat/intake.py) into a temporary store scoped to the chat and the user, and answers come only from them, by
# the local model. Everything that differs from Library Chat is carried by one _DocChat object; without it every path is
# the library's.
DOC_CHAT_ID = "document-chat"


DOC_CHAT_WORKSPACE_ID = "memento-document-chat"


def doc_chat_no_file(question: str) -> str:
    """Document Chat asked a question with no file in the chat: how to attach one, and the same question for the library
    in Library Chat."""
    if not (question or "").strip():
        return texts.DOC_CHAT_NO_FILE_MSG
    return (f"{texts.DOC_CHAT_NO_FILE_MSG}\n\nTo ask the group's library instead, "
            f"[ask it in Library Chat](/?model=memento-query&q={quote(question.strip(), safe='')}).")


DOC_CHAT_REASONS = {
    "unsupported": "this file type is not supported",
    "too_many_pages": "it has {pages} pages, and Document Chat takes files of up to {max} pages",
    "file_limit": "a chat takes up to {max_files} files. Please start a new chat for more",
    "parse_failed": "it could not be read",
    "store_failed": "it could not be stored",
    "missing": "the uploaded file could not be found. Please attach it again",
    "interrupted": "reading it was interrupted. Please attach it again",
}


DOC_CHAT_STALE_S = 1800          # a file still "reading" after this long was interrupted (MinerU's own limit)


DOC_CHAT_STATUS_POLL_S = 0.5     # how often the reading's bar is looked at (shown when it moved on, else every
                                 # ingestion STATUS_EVERY_S)


def is_doc_chat(model_id, workspace_id: str | None = None) -> bool:
    """The chat's model is Memento Document Chat ("memento_query.document-chat", its workspace model, or the bare id;
    `workspace_id`: this copy's own workspace model)."""
    mid = model_id or ""
    return (mid == DOC_CHAT_WORKSPACE_ID or bool(workspace_id and mid == workspace_id)
            or mid.split(".", 1)[-1] == DOC_CHAT_ID)


class DocChat:
    """One Document Chat answer's scope: the temporary collection, a search client limited to this chat and user,
    and the chat's ready files ({source: row}, the names shown in citations and listings). `read_line`: the finished
    bar of the files read with this message, else None."""

    def __init__(self, *, collection: str, chat_id: str, user_id: str, documents: dict, qdrant_url: str,
                 read_line: str | None = None):
        self.collection, self.chat_id, self.user_id = collection, chat_id, user_id
        self.documents, self.read_line = documents, read_line
        self.names = {src: (row or {}).get("name") or src for src, row in documents.items()}
        self._url = qdrant_url

    def client(self, timeout: int = 30):
        """The only Qdrant client of a Document Chat answer (store.ScopedQdrant)."""
        from qdrant_client import QdrantClient
        from memento.document_chat.store import ScopedQdrant
        return ScopedQdrant(QdrantClient(url=self._url, timeout=timeout), self.chat_id, self.user_id,
                            sources=list(self.documents))


def doc_chat_failure(name: str, reason: str, pages=None, max_pages: int = 0, max_files: int = 0) -> str:
    why = DOC_CHAT_REASONS.get(reason, DOC_CHAT_REASONS["parse_failed"]).format(pages=pages, max=max_pages,
                                                                                 max_files=max_files)
    return f"⚠️ **{name}** was not added: {why}.\n\n"


def doc_chat_drop(valves, source: str) -> None:
    """Delete the passages of a file whose reading was interrupted (never searched anyway: not ready)."""
    try:
        from qdrant_client import QdrantClient
        from memento.document_chat.intake import drop_passages, temp_collection
        drop_passages(QdrantClient(url=valves.QDRANT_URL, timeout=30), temp_collection(valves.COLLECTION), source)
    except Exception:
        pass


async def file_record(file_id: str):
    """OpenWebUI's record of an uploaded file (owner, stored path), or None."""
    try:
        from open_webui.models.files import Files
        rec = Files.get_file_by_id(file_id)
        return await rec if asyncio.iscoroutine(rec) else rec
    except Exception:
        return None


def doc_chat_read_one(ti, ing, path, source: str, max_pages: int, progress=None) -> dict:
    """Copy one attached file into the chat's folder under its source name and read it (blocking); `progress` as in
    intake.read_file."""
    import shutil
    if not path or not Path(path).is_file():
        return {"ok": False, "reason": "missing", "error": "", "pages": None}
    dest = ing.input_dir() / f"{source}{Path(path).suffix.lower()}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, dest)
    return ti.read_file(ing, dest, max_pages=max_pages, progress=progress)


def register_doc_chat_names(documents: dict, domain: dict | None) -> None:
    """The attached files' names as aliases, so a question naming "Report.pdf" finds its source ("Report-<token>")."""
    import importlib
    try:
        out_dir = str(importlib.import_module("memento.config.paths").corpus_dirs(domain or {})[1])
    except Exception:
        out_dir = "None"
    for src, row in documents.items():
        name = (row or {}).get("name") or src
        stem = name.rsplit(".", 1)[0] if "." in name else name
        library_state.ALIASES[(out_dir, src)] = list(dict.fromkeys([src, name, stem]))
        library_state.TITLES[(out_dir, src)] = (None, None)


async def doc_chat_read(valves, engine, td, chat_id: str, user_id: str, files: list, status, log, *, function_id: str,
                        db_engine, saved_valves) -> tuple:
    """Claim and read the chat's files not seen before (Open WebUI resends every file of the chat with each
    message). Returns (lines for the reader about files not added, number of files read now, outcome "ok" |
    "not_ready" | "other_library", the finished bar or None). A file is claimed once, so a failure is reported once
    and a file being read by a parallel request is not read twice. The whole read runs in one thread
    (doc_chat_read_sync): a second copy's read holds its library's lifecycle lock until its last write, also when
    the request is cancelled. While it reads, the library upload's progress bar is shown
    (intake.ReadProgress)."""
    loop = asyncio.get_running_loop()

    def record(fid: str):                          # OpenWebUI's file record, read on the server's loop
        return asyncio.run_coroutine_threadsafe(file_record(fid), loop).result(timeout=60)
    progress: dict = {}
    work = asyncio.ensure_future(asyncio.to_thread(doc_chat_read_sync, valves, engine, td, chat_id, user_id,
                                                   files, progress, record, log, function_id=function_id,
                                                   db_engine=db_engine, saved_valves=saved_valves))
    every, step, last = upload_messages.STATUS_EVERY_S, None, 0.0
    while True:
        try:
            finished, _ = await asyncio.wait({work}, timeout=DOC_CHAT_STATUS_POLL_S)
        except asyncio.CancelledError:             # the thread reads on; only the waiting stops
            work.cancel()
            raise
        if finished:
            break
        bar = progress.get("bar")
        if bar is not None and (bar.step != step or time.monotonic() - last >= every):
            step, last = bar.step, time.monotonic()
            await status(bar.line())
    notes, read_now, outcome = work.result()
    bar = progress.get("bar")
    read_line = bar.line() if bar is not None and bar.completed >= bar.n else None
    if read_line:
        await status(read_line)
    return notes, read_now, outcome, read_line


def doc_chat_read_sync(valves, engine, td, chat_id: str, user_id: str, files: list, progress: dict, record,
                       log, *, function_id: str, db_engine, saved_valves) -> tuple:
    from datetime import datetime, timezone
    from memento.document_chat.intake import temp_collection
    coll = temp_collection(valves.COLLECTION)
    with td.lifecycle(engine, coll):           # a second copy's reads, shared; the default copy: no lock
        # admitted with the 10-s cache; removal may have switched the copy off since: check again, fresh
        if function_id not in library_state.DEFAULT_IDS and not library_state.copy_state(
                "query", cached=False, function_id=function_id, db_engine=db_engine).enabled:
            return [], 0, "not_ready"
        notes, max_files, max_pages = [], valves.DOC_CHAT_MAX_FILES, valves.DOC_CHAT_MAX_PAGES
        rows = td.chat_documents(engine, chat_id, user_id, collection=coll)
        now = datetime.now(timezone.utc)
        for r in rows:                         # still "reading" long after: OpenWebUI stopped during the read
            added = r.get("added_at")
            if added is not None and added.tzinfo is None:
                added = added.replace(tzinfo=timezone.utc)
            if (r["status"] == "reading" and added is not None
                    and (now - added).total_seconds() > DOC_CHAT_STALE_S):
                td.set_status(engine, chat_id, r["file_id"], "failed", error="interrupted")
                doc_chat_drop(valves, r["source"])
                notes.append(doc_chat_failure(r["name"], "interrupted"))
                r["status"] = "failed"
        known = {r["file_id"] for r in rows}
        active = sum(1 for r in rows if r["status"] != "failed")
        new = []
        for f in files:
            fid = f.get("id")
            if not fid or fid in known:
                continue
            known.add(fid)
            # only the asker's own uploads, read from the path OpenWebUI stored (the file list comes from the
            # browser)
            rec = record(fid)
            if rec is None or getattr(rec, "user_id", None) != user_id or not getattr(rec, "path", None):
                log.warning("file %s skipped: not an upload of this user", fid)
                continue
            name = f.get("name") or (f.get("file") or {}).get("filename") or "file"
            src = td.claim(engine, chat_id=chat_id, user_id=user_id, file_id=fid, name=name, collection=coll)
            if src is td.OTHER_LIBRARY:        # the other library's first file arrived first
                return notes, 0, "other_library"
            if not src:                        # a parallel request of this chat has it
                continue
            if active >= max_files:
                td.set_status(engine, chat_id, fid, "failed", error="file_limit")
                notes.append(doc_chat_failure(name, "file_limit", max_files=max_files))
                continue
            active += 1
            new.append((fid, name, src, rec.path))
        ingestion_valves = (saved_valves(valves.INGESTION_FUNCTION_ID) or {}) if new or rows else {}
        in_root = ingestion_valves.get("INPUT_ROOT") or os.getenv("INPUT_ROOT") or str(PROJECT_ROOT / "data/input")
        out_root = ingestion_valves.get("OUTPUT_ROOT") or os.getenv("OUTPUT_ROOT") or str(PROJECT_ROOT / "data/output")
        from memento.document_chat import intake as ti
        if not new:
            # files read before chat folders recorded their owner: record it now, so their citations open for the owner
            if rows and (Path(in_root) / "_temp" / chat_id).is_dir():
                try:
                    ti.claim_chat_folder(in_root, chat_id, user_id)
                except ValueError:
                    log.warning("chat %s: its folder is recorded for another account", chat_id)
            return notes, 0, "ok"
        ing = ti.make_ingester(chat_id, user_id, input_root=in_root, output_root=out_root,
                               collection=valves.COLLECTION, valves=ingestion_valves)
        read, bar = 0, ti.ReadProgress(len(new))
        progress["bar"] = bar
        for fid, name, src, path in new:
            bar.update("parse")
            t0 = time.time()
            try:
                result = doc_chat_read_one(ti, ing, path, src, max_pages, progress=bar.update)
            except Exception as e:
                result = {"ok": False, "reason": "parse_failed", "error": repr(e), "pages": None}
            if result.get("ok"):
                td.set_status(engine, chat_id, fid, "ready", pages=result.get("pages"))
                read += 1
            else:
                td.set_status(engine, chat_id, fid, "failed", pages=result.get("pages"),
                              error=f"{result.get('reason')}: {result.get('error', '')}"[:500])
                notes.append(doc_chat_failure(name, result.get("reason", ""), result.get("pages"), max_pages,
                                               max_files))
            bar.finish(bool(result.get("ok")))
            log.info("read %s (%s pages, %s passages) in %.0f s: %s", src, result.get("pages"),
                     result.get("vectors"), time.time() - t0, "ok" if result.get("ok") else result.get("reason"))
        return notes, read, "ok"


async def answer(turn, task=None, *, run):
    """Memento Document Chat: the chat's new files are read into the temporary store (memento/document_chat/intake.py),
    then the question is answered from this chat's ready files only (`run`, service.run, with the chat's DocChat). Background tasks
    read and search nothing."""
    import logging
    log = logging.getLogger("memento.docchat")
    from memento.document_chat import store as td

    status = turn.status                    # the message's own status line

    meta = turn.metadata or {}
    chat_id, user_id = meta.get("chat_id"), (turn.user or {}).get("id")
    last = ((turn.body.get("messages") or [{}])[-1] or {}).get("content")
    if task or (isinstance(last, str) and last.startswith("### Task:") and "<user_query>" not in last):
        # title, tags, follow-up suggestions: answered without files (no file read, no search)
        turn.doc_chat, turn.files = DocChat(collection="", chat_id=chat_id or "-", user_id=user_id or "-", documents={},
                                            qdrant_url=""), None
        async for c in run(turn):
            yield c
        return
    if not td.valid_chat_id(chat_id) or not user_id:
        yield texts.DOC_CHAT_SAVED_CHAT_MSG
        return
    engine = turn.engine()
    from memento.document_chat.intake import temp_collection
    coll = temp_collection(turn.valves.COLLECTION)
    read_line = None
    try:
        await asyncio.to_thread(td.ensure_tables, engine)      # one existence check per Document Chat message
        # a chat belongs to one library: refused before any file is read or searched
        if await asyncio.to_thread(td.chat_collections, engine, chat_id) - {coll}:
            outcome = "other_library"
        else:
            files = list(turn.files or [])
            try:                           # images OpenWebUI moved into the message (as Memento Ingestion)
                files += await upload_intake.message_images(turn.metadata, files, turn.user)
            except Exception as e:
                log.warning("message images not read: %r", e)
            notes, read_now, outcome, read_line = await doc_chat_read(
                turn.valves, engine, td, chat_id, user_id, files, status, log, function_id=turn.function_id,
                db_engine=turn.engine, saved_valves=lambda fid: library_state.saved_valves(fid, turn.engine))
            rows = await asyncio.to_thread(td.chat_documents, engine, chat_id, user_id, collection=coll)
    except Exception as e:
        yield texts.user_error("library", e)
        await status("Done", done=True)
        return
    if outcome != "ok":
        yield texts.DOC_CHAT_OTHER_LIBRARY_MSG if outcome == "other_library" else texts.STUDY_NOT_READY_MSG
        await status("Done", done=True)
        return
    if read_now:
        yield f"_{texts.DOC_CHAT_PRIVACY}_\n\n"
    for line in notes:
        yield line
    ready = {r["source"]: r for r in rows if r["status"] == "ready"}
    if not ready and (notes or any(r["status"] == "reading" for r in rows)):
        if not notes:
            yield texts.DOC_CHAT_READING_MSG
        await status(read_line or "Done", done=True)
        return
    # no file yet: the router still decides; a greeting or a general question is answered, a question about a file
    # gets DOC_CHAT_NO_FILE_MSG (service.run, after the router); nothing is searched
    turn.doc_chat = DocChat(collection=coll, chat_id=chat_id, user_id=user_id, documents=ready,
                            qdrant_url=turn.valves.QDRANT_URL, read_line=read_line)
    register_doc_chat_names(ready, request.domain(turn.valves.DOMAIN_NAME))
    with request.shown_names(turn.doc_chat.names):
        async for c in run(turn):
            yield c
