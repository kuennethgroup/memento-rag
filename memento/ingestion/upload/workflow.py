"""Library Upload's request: who may upload, which files, the server path an admin types, the duplicate and
new-version decisions, the parse and store batch on worker threads, and the status lines while it runs.
"""
import asyncio
import hashlib
import json
import logging
import queue
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import AsyncGenerator, Generator

from memento.config.paths import PROJECT_ROOT
# duplicates, new versions and name clashes: the same decision as run_pipeline.py
from memento.ingestion import library_index
from memento.ingestion.convert import NATIVE_FORMATS, discover_files, ensure_supported, resolve_stem_collisions
from memento.ingestion.upload import document, intake, messages
from memento.library.copies import DEFAULT_IDS
from memento.query import library_state, request
from memento.query.library_state import INGEST_STATUS_TABLE

_log = logging.getLogger("memento.ingestion")



# Ingestion runs are serialized through a Postgres session-level advisory lock on Open WebUI's own database. A second
# run is rejected with a message instead of racing on data/manifest.json and data/input/. The lock works across Open
# WebUI worker processes, and pg_try_advisory_lock doubles as the "is ingestion running?" check in qdrant_pipe.py. The
# key is a fixed value that must match qdrant_pipe.py's and must not collide with other advisory-lock users of this
# database.
INGEST_ADVISORY_LOCK_KEY = 7281990014127733


# Open WebUI consumes a sync-generator pipe with a plain loop on the shared event loop, so every blocking call between
# yields would freeze the whole server for the length of an ingestion run. `run` is therefore async: it runs pipe_sync
# on a background thread and relays each chunk back through an asyncio.Queue.
async def run(body: dict, user: dict, files: list, emit, task, chat_id, metadata, *, function_id: str, valves,
              engine) -> AsyncGenerator:
    """Library Upload's reply to one message; `engine`: a callable returning Open WebUI's database engine."""
    if task == "title_generation":      # OpenWebUI parses {"title": …} out of the reply
        yield json.dumps({"title": messages.upload_title(await intake.chat_file_names(chat_id))})
        return
    if not task and request.continue_request(body):          # "Continue Response": the reply is complete
        return
    if function_id not in DEFAULT_IDS:       # another library's copy: only once it is set up
        state = await asyncio.to_thread(library_state.copy_state, "ingestion", function_id=function_id,
                                        db_engine=engine)
        if not state.enabled:
            logging.getLogger("memento.errors").warning("%s not set up: %s", function_id, state.reason)
            yield messages.STUDY_NOT_READY_MSG
            return
    if not await intake.may_ingest(valves, user):
        yield "Uploading documents is available to contributors only. Please ask the Memento admin for access."
        return
    try:                                     # images OpenWebUI moved into the message (see intake.message_images)
        files = list(files or []) + await intake.message_images(metadata, files, user)
    except Exception as e:
        logging.getLogger("memento.errors").warning("E-ingest-images: %r", e)
    # only the uploader's own files, from the path Open WebUI stored (the list comes from the browser)
    files, refused = await intake.owned_uploads(files, user)
    for name in refused:
        logging.getLogger("memento.errors").warning("E-ingest-not-owned: %s (user %s)", name,
                                                    (user or {}).get("id"))
        yield f"⚠ **{name}**: unable to access the uploaded file, please try attaching it again\n"
    files, earlier = intake.current_message_files(metadata, files)
    ing      = document.Ingester(valves)
    loop     = asyncio.get_event_loop()
    out_q: asyncio.Queue = asyncio.Queue()
    SENTINEL = object()

    def runner():
        try:
            for item in pipe_sync(ing, body, user, files, earlier=earlier):
                loop.call_soon_threadsafe(out_q.put_nowait, item)
        except Exception as e:
            import traceback
            loop.call_soon_threadsafe(
                out_q.put_nowait,
                f"❌ Pipe error: `{type(e).__name__}: {e}`\n```\n{traceback.format_exc()}\n```\n",
            )
        finally:
            loop.call_soon_threadsafe(out_q.put_nowait, SENTINEL)

    threading.Thread(target=runner, daemon=True).start()

    while True:
        item = await out_q.get()
        if item is SENTINEL:
            break
        if isinstance(item, messages.Status):
            if emit:
                await emit(
                    {"type": "status", "data": {"description": item.text, "done": item.done}}
                )
        else:
            yield item


def pipe_sync(ing, body: dict, __user__: dict, __files__: list = [], earlier=()) -> Generator:
    __files__ = __files__ or []

    # Open WebUI sends its background task prompts (follow-up generation, tagging) to the active pipe model:
    # ignored.
    last_msg = body["messages"][-1]["content"] if body.get("messages") else ""
    if isinstance(last_msg, str) and last_msg.startswith("### Task:"):
        return

    if ing.valves.SKIP_PARSE and ing.valves.SKIP_EMBED:
        yield "Both SKIP_PARSE and SKIP_EMBED are on, nothing to do."
        return

    try:
        from open_webui.internal.db import engine
        from sqlalchemy import text
    except Exception as e:
        yield f"❌ Could not reach the database to coordinate ingestion: {e}\n"
        return

    # AUTOCOMMIT: pg_try_advisory_lock/pg_advisory_unlock are session-level,
    # not transactional — autocommit avoids leaving this connection
    # "idle in transaction" for the entire (potentially long) run.
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        got_lock = conn.execute(
            text("SELECT pg_try_advisory_lock(:key)"),
            {"key": INGEST_ADVISORY_LOCK_KEY},
        ).scalar()
        if not got_lock:
            yield (
                "⏳ Another upload is already being processed right now. "
                "Please wait for it to finish, then try again.\n"
            )
            return

        # Self-healing: only one run can hold the lock, so any rows left
        # over here belong to a run that crashed mid-paper without
        # clearing them — start this run with a clean slate.
        try:
            conn.execute(text(
                f"CREATE TABLE IF NOT EXISTS {INGEST_STATUS_TABLE} "
                "(source TEXT PRIMARY KEY, started_at TIMESTAMPTZ NOT NULL DEFAULT now())"
            ))
            conn.execute(text(f"DELETE FROM {INGEST_STATUS_TABLE}"))
        except Exception:
            pass

        try:
            yield from pipe_body(ing, body, __user__, __files__, earlier=earlier)
        finally:
            conn.execute(
                text("SELECT pg_advisory_unlock(:key)"),
                {"key": INGEST_ADVISORY_LOCK_KEY},
            )


def pipe_body(ing, body: dict, __user__: dict, __files__: list, earlier=()) -> Generator:
    # Path 1: supported files attached via paperclip
    seen_ids = ing.load_seen_ids()
    _fname = lambda f: f.get("name") or f.get("file", {}).get("filename", "")
    # files attached to earlier messages that were never added (for Library Chat or Document Chat): say so
    not_added = [f for f in earlier or [] if f.get("id") not in seen_ids
                 and Path(_fname(f)).suffix.lower() in intake.SUPPORTED_EXTENSIONS]
    note = messages.EARLIER_ATTACHMENTS_NOTE if not_added else ""
    all_docs = [
        f for f in __files__
        if (f.get("id") is None or f.get("id") not in seen_ids)
        and Path(_fname(f)).suffix.lower() in intake.SUPPORTED_EXTENSIONS
    ]
    # Members may only add files: typing server paths is the admin's library maintenance (it could pull any folder
    # the server can read into the shared library), and plain text gets a short hint instead. Earlier uploads in the
    # chat are re-sent with every message: not re-listed.
    admin = (__user__ or {}).get("role") == "admin"
    skipped = [(_fname(f), "file type not supported") for f in __files__
               if Path(_fname(f)).suffix.lower() not in intake.SUPPORTED_EXTENSIONS]
    if admin:
        skipped += [(_fname(f), "this upload was already added") for f in __files__
                    if f.get("id") in seen_ids and Path(_fname(f)).suffix.lower() in intake.SUPPORTED_EXTENSIONS]
    for name, why in skipped:
        yield f"⏭ **{name}**: skipped, {why}\n"
    if not admin and not all_docs:
        added_before = any(f.get("id") in seen_ids for f in list(__files__) + list(earlier or []))
        yield messages.ONLY_UPLOADS_MSG + messages.ask_line(messages.last_text(body)) + (messages.EARLIER_UPLOADS_NOTE if added_before else "") + note
        return
    if __files__ and not all_docs:
        yield (f"\nNothing new to add. Supported: {', '.join(sorted(intake.SUPPORTED_EXTENSIONS))}\n")
        return
    if all_docs:
        stems: list = []
        yield from handle_uploads(ing, all_docs, __user__, stems)
        new_ids = {f.get("id") for f in all_docs if f.get("id")}
        if new_ids:
            ing.save_seen_ids(seen_ids | new_ids)
        if messages.last_text(body):        # a request typed with the upload: handed over to Library Chat
            yield messages.ask_line(messages.last_text(body), ing.stored_titles(stems) if len(stems) <= messages.ASK_NAMED_MAX else None,
                           uploaded=True)
        if note:
            yield note
        return

    # Path 2: server-side folder or filename typed in chat
    text = messages.last_text(body)
    if not text.strip():
        yield messages.ADMIN_NO_PATH_MSG
        return

    # Resolve as directory
    folder = None
    for candidate in (
        Path(text),
        Path("/" + text),
        PROJECT_ROOT / text,
    ):
        if candidate.is_dir():
            folder = candidate
            break

    if folder is not None:
        docs = discover_files(folder, intake.SUPPORTED_EXTENSIONS)
        unreadable = sorted(set(discover_files(folder, {p.suffix.lower() for p in folder.rglob("*")
                                                        if p.is_file()})) - set(docs))
        if unreadable:
            names = ", ".join(f"`{p.relative_to(folder)}`" for p in unreadable[:10])
            more = f" and {len(unreadable) - 10} more" if len(unreadable) > 10 else ""
            yield f"⏭ Skipped {len(unreadable)} file(s), file type not supported: {names}{more}\n\n"
        if not docs:
            yield f"No supported files found in `{folder}`\n"
            yield f"Supported: {', '.join(sorted(intake.SUPPORTED_EXTENSIONS))}\n"
            return
    else:
        # Try to resolve as a single filename — try exact name first, then add extensions
        found = None
        candidates = [text] if Path(text).suffix.lower() in intake.SUPPORTED_EXTENSIONS else \
                     [text + ext for ext in intake.SUPPORTED_EXTENSIONS]
        for name in candidates:
            for search_dir in [ing.input_dir()]:
                candidate = search_dir / name
                if candidate.exists():
                    found = candidate
                    break
            if found:
                break
        # For embed-only, file bytes aren't needed — just the stem
        if found is None and ing.valves.SKIP_PARSE:
            stem = Path(text).stem
            paper_dir = ing.find_paper_dir(stem)
            if paper_dir is not None:
                found = ing.input_dir() / text
        if found is None:
            yield f"Not found: `{text}`\n\n"
            yield f"Copy the file to `data/input/` and type the filename, or type a folder path.\n"
            yield f"Supported formats: {', '.join(sorted(intake.SUPPORTED_EXTENSIONS))}\n"
            return
        docs = [found]

    if ing.valves.SKIP_PARSE:
        # Check which files have existing parsed output
        parsed     = [p for p in docs if ing.find_paper_dir(p.stem) is not None]
        not_parsed = [p for p in docs if p not in parsed]
        for p in not_parsed:
            yield f"⚠ `{p.name}`: not parsed yet, run with SKIP_PARSE=False first\n"
        if not parsed:
            return
        docs = parsed

    location = str(folder) if folder else str(docs[0].parent)
    n_nested = sum(1 for p in docs if folder and p.parent != folder) if folder else 0
    nested_note = f" (across {len({p.parent for p in docs})} subdirectories)" if n_nested else ""
    yield f"Found **{len(docs)}** file(s) in `{location}`{nested_note}:\n"
    FILE_LIST_PREVIEW = 10
    preview = docs[:FILE_LIST_PREVIEW]
    rest    = docs[FILE_LIST_PREVIEW:]
    for p in preview:
        display = str(p.relative_to(folder)) if folder else p.name
        yield f"- `{display}`\n"
    if rest:
        yield messages.details_block(
            f"...and {len(rest)} more",
            "".join(f"- `{str(p.relative_to(folder)) if folder else p.name}`\n" for p in rest),
        )

    if ing.valves.SKIP_PARSE:
        yield from run_batch(ing, docs)
        return

    ing.input_dir().mkdir(parents=True, exist_ok=True)
    if folder:
        docs, name_map = resolve_stem_collisions(docs, folder)
    else:
        name_map = {p: p.name for p in docs}
    # server-side paths are library maintenance, like the command line: one identity (library_index)
    flat_docs = []
    copied_count = 0
    decisions = ing.decide([(name_map[src], src) for src in docs], library_index.SERVER_UPLOADER, "server")
    for d in decisions:
        note = ing.decision_line(d)
        if note:
            yield note
        if d.action == "duplicate":
            continue
        dest = ing.input_dir() / d.filename
        if dest.resolve() != d.path.resolve() and (d.action != "unchanged" or not dest.exists()):
            dest.write_bytes(d.path.read_bytes())       # a changed source file now replaces the old copy
            copied_count += 1
        flat_docs.append(dest)
    if copied_count:
        yield f"\nCopied {copied_count} file(s) to the library folder\n"

    needs_convert = [f for f in flat_docs if f.suffix.lower() not in NATIVE_FORMATS]
    if needs_convert:
        yield f"Converting {len(needs_convert)} older Office file(s) to the current Office format...\n"
        flat_docs = ensure_supported(flat_docs, ing.input_dir() / "_converted")
        yield f"**{len(flat_docs)}** file(s) to check\n"

    yield from process_docs(ing, flat_docs)


# ── dispatch resolved files through parse + embed, skipping the MinerU pass for any whose content is unchanged
# since the last successful parse ──
def process_docs(ing, docs: list) -> Generator:
    if not docs:
        return

    doc_meta             = ing.load_manifest().get("documents", {})
    to_parse             = []
    updated              = []   # filenames whose content changed since last parse
    already_parsed       = []   # filenames, for display
    already_parsed_paths = []

    for p in docs:
        if ing.needs_parse(p, doc_meta):
            to_parse.append(p)
            # "changed since last time" only when the file's content differs from the record: document.needs_parse() is
            # also True when there is no parsed output for the current backend yet, which is a configuration change, not
            # an edited file.
            if not ing.valves.FORCE_REPARSE:
                stored_hash = doc_meta.get(p.stem, {}).get("parsed_hash")
                if stored_hash:
                    try:
                        if hashlib.sha256(p.read_bytes()).hexdigest() != stored_hash:
                            updated.append(p.name)
                    except Exception:
                        pass
        else:
            already_parsed.append(p.name)
            already_parsed_paths.append(p)

    for name in updated:
        yield f"- 🔄 **{name}**: this file has changed since last time, re-reading it\n"

    if already_parsed:
        names = ', '.join(f"**{n}**" for n in already_parsed)
        if ing.valves.SKIP_EMBED:
            yield f"⏭ {names}: skipped, read before and unchanged (storing is switched off)\n"
        else:
            yield f"⏭ {names}: read before, not read again\n"

    if to_parse:
        yield from run_batch(ing, to_parse)

    if already_parsed_paths and not ing.valves.SKIP_EMBED:
        if to_parse:
            yield "\n---\n"
        try:
            yield from embed_batch(ing, already_parsed_paths)
        except Exception as e:
            import traceback
            yield (
                "❌ Something went wrong while storing documents.\n"
                + messages.details_block("Error details", f"```\n{traceback.format_exc()}\n```")
            )


# ── handle paperclip uploads ──────────────────────────────────────────────

def handle_uploads(ing, files: list, user: dict | None = None, stems: list | None = None) -> Generator:
    """`stems` (filled): the documents of this upload as stored or already in the library (a question typed with
    them is handed over to Library Chat naming the document)."""
    ing.input_dir().mkdir(parents=True, exist_ok=True)
    stems = stems if stems is not None else []
    docs = []
    accepted = []   # (file name, OpenWebUI's stored copy)

    for f in files:
        filename = f.get("name") or f.get("file", {}).get("filename", "upload.pdf")
        src_path = f.get("file", {}).get("path")
        dest     = ing.input_dir() / filename

        if ing.valves.SKIP_PARSE:
            # Embed-only: check output exists, no need to copy file
            if ing.find_paper_dir(Path(filename).stem) is not None:
                docs.append(dest)
            else:
                yield f"⚠ **{filename}**: not read yet, unable to store\n"
            continue

        if Path(filename).suffix.lower() not in intake.SUPPORTED_EXTENSIONS:
            yield f"⚠ **{filename}**: file type not supported, skipped\n"
            continue

        if not src_path or not Path(src_path).exists():
            yield f"⚠ **{filename}**: unable to access the uploaded file, please try attaching it again\n"
            continue

        accepted.append((filename, Path(src_path)))

    # Decide before writing anything into the library folder, so a same-named upload never overwrites another
    # person's document: duplicate / unchanged / new version / renamed / new (library_index.decide)
    user = user or {}
    for d in (ing.decide(accepted, user.get("id") or "unknown", user.get("name") or user.get("email") or "")
              if accepted else []):
        note = ing.decision_line(d)
        if note:
            yield note
        if d.action == "duplicate":            # already there: the questions to ask about it
            stems.append(d.stem)
            yield messages.ask_links(ing.stored_titles([d.stem]))
            continue
        stems.append(Path(d.filename).stem)
        dest = ing.input_dir() / d.filename
        if d.action != "unchanged" or not dest.exists():
            dest.write_bytes(d.path.read_bytes())
        docs.append(dest)

    if not docs:
        if not stems:                           # nothing stored and nothing already there
            yield "Nothing to process. See the messages above, or try attaching the file(s) again.\n"
        if ing.valves.CLEANUP_CHAT_UPLOAD:
            cleanup_webui_files(files)
        return

    if ing.valves.SKIP_PARSE:
        yield from run_batch(ing, docs)
    else:
        needs_convert = [f for f in docs if f.suffix.lower() not in NATIVE_FORMATS]
        if needs_convert:
            yield f"Converting {len(needs_convert)} older Office file(s) to the current Office format...\n"
            docs = ensure_supported(docs, ing.input_dir() / "_converted")
        yield from process_docs(ing, docs)

    if ing.valves.CLEANUP_CHAT_UPLOAD:
        cleanup_webui_files(files)


# ── delete OpenWebUI upload copies ────────────────────────────────────────

def cleanup_webui_files(files: list):
    # Through Open WebUI's own SQLAlchemy session (bound to its DATABASE_URL, sqlite or Postgres), so the right
    # chat_file/file rows are changed.
    try:
        from open_webui.internal.db import get_db
        from sqlalchemy import text
    except Exception:
        return
    try:
        with get_db() as db:
            for f in files:
                file_id  = f.get("id")
                src_path = f.get("file", {}).get("path")
                if src_path:
                    Path(src_path).unlink(missing_ok=True)
                if file_id:
                    db.execute(text("DELETE FROM file WHERE id = :id"), {"id": file_id})
                    db.execute(text("DELETE FROM chat_file WHERE file_id = :id"), {"id": file_id})
            db.commit()
    except Exception:
        pass


# ── parallel batch (parse + optional embed) ───────────────────────────────

# ── parse one document: structured start/file events and the final result dict go to msg_q ──
# Per-poll progress strings from parse_one are not forwarded (many concurrent workers would interleave them); the
# drain loop (run_batch) knows the batch totals and formats the lines (messages.file_summary_line).
def parse_worker(ing, pdf_path: Path, msg_q: "queue.Queue", results: dict):
    # Everything below must end with msg_q.put(None) (see embed_worker): an uncaught exception would otherwise kill
    # this thread silently and hang the drain loop.
    msg_q.put(("start", "parse", pdf_path))
    try:
        parse_result = None
        for item in ing.parse_one(pdf_path):
            if not isinstance(item, str):
                parse_result = item
        results[pdf_path] = parse_result
    except Exception as e:
        parse_result = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        results[pdf_path] = parse_result

    msg_q.put(("file", "parse", pdf_path, parse_result))
    msg_q.put(None)


# ── embed one document: structured start/file events and the final result dict go to msg_q (per-poll "Storing... %"
# strings are dropped, as in parse_worker) ──
def embed_worker(ing, pdf_path: Path, doc_meta: dict, msg_q: "queue.Queue", results: dict):
    # Everything below must end with msg_q.put(None): that sentinel lets the `while done < n` drain loop end. An
    # uncaught exception (a transient ConnectionError, say) would otherwise kill this thread silently (pool.submit
    # discards the Future's exception) and the batch would wait forever.
    msg_q.put(("start", "embed", pdf_path))
    try:
        s_hash = doc_meta.get(pdf_path.stem, {}).get("source_hash")
        result = None
        for item in ing.embed_one(pdf_path, stored_hash=s_hash):
            if not isinstance(item, str):
                result = item
            else:                            # "Storing... 40% (…)": the bar's progress
                m = re.search(r"Storing\.\.\. (\d+)%", item)
                if m:
                    msg_q.put(("progress", "embed", pdf_path, int(m.group(1))))
        results[pdf_path] = result
    except Exception as e:
        result = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        results[pdf_path] = result

    msg_q.put(("file", "embed", pdf_path, result))
    msg_q.put(None)


def run_batch(ing, pdfs: list) -> Generator:
    n       = len(pdfs)
    t_batch = time.time()

    # Embed-only: no parse phase. A re-embed needs only the document's content_list.json under data/output/; the
    # source file may not be in data/input/, so its path is not touched.
    if ing.valves.SKIP_PARSE:
        if not ing.valves.SKIP_EMBED:
            try:
                yield from embed_batch(ing, pdfs)
            except Exception as e:
                import traceback
                yield (
                    "❌ Something went wrong while storing documents.\n"
                    + messages.details_block("Error details", f"```\n{traceback.format_exc()}\n```")
                )
        else:
            yield f"\n---\n✅ All done: {n} file(s) processed in {time.time() - t_batch:.1f}s\n"
        return

    # Parse phase, pipelined with embed: each file goes to the embed pool as soon as its own parse finishes
    # (parse_then_embed), so parse and embed run on their GPUs at the same time.
    #
    # Smallest file first (size is a cheap proxy for parse cost): small documents fill the pool and finish early, so
    # a large one spends most of its run with little competition. A file's time lost to contention grows with its
    # size, so the small ones take the busiest window.
    pdfs           = sorted(pdfs, key=lambda p: p.stat().st_size)
    parse_workers  = min(ing.valves.PARSE_WORKERS, n)
    pipeline_embed = not ing.valves.SKIP_EMBED
    embed_workers  = min(ing.valves.EMBED_WORKERS, n) if pipeline_embed else 0
    mode = "parse-only" if not pipeline_embed else "parse + embed (pipelined)"
    # "Uploading" for the whole batch; the live status line's "N parsing, M storing" carries the phases.
    yield messages.batch_header(n)
    # run settings go to the server log, not the chat
    _log.info("INGEST START: %d file(s), %s, backend %s, parse workers %d, embed workers %d, collection %s",
              n, mode, ing.valves.BACKEND, parse_workers, embed_workers, ing.valves.COLLECTION)

    manifest = ing.load_manifest()
    doc_meta = manifest.setdefault("documents", {})

    msg_q         = queue.Queue()
    parse_results = {}
    embed_results = {}
    embed_futures = []
    embed_pool    = None

    if pipeline_embed:
        # pre-create the collection once, up front — avoids a 409 race
        # once embeds start firing concurrently with still-running parses
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams, SparseVectorParams, SparseIndexParams, Modifier
            _qc   = QdrantClient(url=ing.valves.QDRANT_URL, timeout=30)
            _cols = [c.name for c in _qc.get_collections().collections]
            if ing.valves.COLLECTION not in _cols:
                _qc.create_collection(
                    ing.valves.COLLECTION,
                    vectors_config={"dense": VectorParams(size=4096, distance=Distance.COSINE)},
                    sparse_vectors_config={"sparse": SparseVectorParams(index=SparseIndexParams(), modifier=Modifier.IDF)},
                )
            ing.ensure_indexes(_qc)
        except Exception as e:
            yield (
                "❌ Unable to connect to the database.\n"
                + messages.details_block("Error details", f"{type(e).__name__}: {e}")
            )
            return
        embed_pool = ThreadPoolExecutor(max_workers=embed_workers)

    def parse_then_embed(pdf_path: Path):
        parse_worker(ing, pdf_path, msg_q, parse_results)
        r = parse_results.get(pdf_path)
        if embed_pool is not None and r and r.get("ok"):
            embed_futures.append(
                embed_pool.submit(embed_worker, ing, pdf_path, doc_meta, msg_q, embed_results)
            )

    # Shared across the drain loop below: these give the transcript its "[k/n]" numbering and the status line its
    # batch-wide count, whichever worker finishes next.
    completed, reading, storing, failed, unchanged = 0, 0, 0, 0, 0
    failed_names: list[str] = []
    doc_lines = messages.DocLines()        # one line per finished document
    bar_verb = "stored" if pipeline_embed else "read"
    counts = lambda: (completed - failed - unchanged, unchanged, failed, bar_verb)   # the bar's counts

    # each document in progress: its phase, when it started reading and its pages, its storing percentage
    phase_of: dict = {}
    read_start: dict = {}
    store_pct: dict = {}
    last_push = [time.time()]
    shown = [0]                                # the percent shown last: the bar never goes back

    def push_status() -> messages.Status:
        now = time.time()
        last_push[0] = now
        shares = []
        for pdf, ph in phase_of.items():
            if ph == "parse":
                t0, pages = read_start.get(pdf, (now, None))
                shares.append(messages.doc_progress("parse", elapsed=now - t0, pages=pages, reading=max(1, reading)))
            elif ph == "between":
                shares.append(messages.READ_SHARE)
            else:
                shares.append(messages.doc_progress("embed", store_pct=store_pct.get(pdf, 0)))
        shown[0] = messages.shown_percent(previous=shown[0],
                                  pct=messages.batch_percent(completed=completed, n=n, in_progress=shares))
        return messages.Status(messages.batch_status(completed, n, reading, storing, (now - t_batch) / 60, shown[0],
                                     counts=counts()))

    with ThreadPoolExecutor(max_workers=parse_workers) as parse_pool:
        parse_futures = [parse_pool.submit(parse_then_embed, p) for p in pdfs]
        while True:
            try:
                msg = msg_q.get(timeout=5)
            except queue.Empty:
                msg = None

            if isinstance(msg, tuple):
                kind = msg[0]
                if kind == "start":
                    _, phase, _pdf = msg
                    reading += (phase == "parse")
                    storing += (phase == "embed")
                    phase_of[_pdf] = phase
                    if phase == "parse":
                        read_start[_pdf] = (time.time(), messages.page_count(_pdf))
                    # no status push here: Open WebUI keeps every status update as a permanent log entry, so pushing
                    # on every worker start and stop floods it. Only push when `completed` changes, below.
                elif kind == "file":
                    _, phase, pdf_path, result = msg
                    ok = bool(result and result.get("ok"))
                    # a successful parse in pipelined mode is not terminal: its embed worker is queued (above) and
                    # emits this file's one transcript line.
                    terminal = phase == "embed" or not (pipeline_embed and ok)
                    if phase == "parse":
                        reading -= 1
                    else:
                        storing -= 1
                    if terminal:
                        phase_of.pop(pdf_path, None)
                    else:                    # read, waiting to be stored
                        phase_of[pdf_path] = "between"
                    if terminal:
                        completed += 1
                        if ok and result.get("skipped"):
                            unchanged += 1
                        if not ok:
                            failed += 1
                            failed_names.append(f"{pdf_path.name} ({phase})")
                            err = (result or {}).get("error", "unknown")
                            _log.error("INGEST FAIL [%s] %s: %s", phase, pdf_path.name, err)
                        pages = (parse_results.get(pdf_path) or {}).get("pages") if phase == "embed" else None
                        out = doc_lines.add(messages.file_summary_line(completed, n, phase, pdf_path.name, result,
                                                               pages=pages), ok)
                        if out:
                            yield out
                        yield push_status()
                elif kind == "progress":
                    _, _phase, _pdf, value = msg
                    store_pct[_pdf] = value
            elif isinstance(msg, str):
                yield msg
            if msg is not None and time.time() - last_push[0] >= messages.STATUS_EVERY_S:
                yield push_status()          # the bar moves inside long documents too

            # Check parse_futures first: while any parse runs, more embed_futures may still be added
            # (parse_then_embed), so the embed_futures snapshot is final only once every parse has finished; and
            # stop only once the queue is empty, since a worker's last messages (its "file" result) can still be
            # queued when its future is done.
            if (all(f.done() for f in parse_futures) and all(f.done() for f in embed_futures)
                    and msg_q.empty()):
                break
            if msg is None:
                # Refresh the elapsed-time status here too, so it does not look stalled while nothing has finished
                # yet.
                yield push_status()
                yield " "  # keepalive: prevents an SSE timeout during long parse/embed

    if embed_pool is not None:
        embed_pool.shutdown(wait=True)

    # commit the manifest for parse and embed together, in one write. `parsed_hash` is what `document.needs_parse`
    # compares on a later run; `source_hash`/`vector_count` mirror what an earlier embed wrote. Merged, so neither phase
    # overwrites the other's keys.
    for pdf_path in pdfs:
        r = parse_results.get(pdf_path)
        if r and r.get("ok"):
            try:
                parsed_hash = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
            except Exception:
                continue
            doc_meta.setdefault(pdf_path.stem, {}).update({
                "parsed_hash": parsed_hash,
                "parsed_at":   time.strftime("%Y-%m-%dT%H:%M:%S"),
            })
    for pdf_path, r in embed_results.items():
        if r and r.get("ok") and not r.get("skipped") and r.get("source_hash"):
            doc_meta.setdefault(pdf_path.stem, {}).update({
                "source_hash":  r["source_hash"],
                "vector_count": r["vectors"],
                "collection":   ing.valves.COLLECTION,
                "embedded_at":  time.strftime("%Y-%m-%dT%H:%M:%S"),
            })
    ing.save_manifest(manifest)

    elapsed   = time.time() - t_batch
    n_skip    = sum(1 for r in embed_results.values() if r and r.get("ok") and r.get("skipped"))
    succeeded = completed - failed - n_skip
    verb      = "stored" if pipeline_embed else "parsed"
    summary   = f"\n---\n✅ Done · {succeeded} {verb}"
    if n_skip:
        summary += f" · {n_skip} skipped (already in the library, unchanged)"
    summary  += f" · {failed} failed"
    took      = messages.took(elapsed)
    summary  += f" · {took}\n"
    total_vecs = sum(r["vectors"] for r in embed_results.values() if r and r.get("ok"))
    _log.info("INGEST DONE: %d succeeded, %d failed, %d passages, %.0fs elapsed", succeeded, failed, total_vecs,
              elapsed)
    yield doc_lines.rest()
    yield summary
    if pipeline_embed and succeeded:
        yield messages.ask_links(ing.stored_titles(
            [Path(p).stem for p, r in embed_results.items() if r and r.get("ok") and not r.get("skipped")]))
    if failed_names:
        _log.warning("INGEST FAILED FILES: %s", "; ".join(failed_names))
        yield f"❌ Failed: {', '.join(failed_names)}\n"
    if succeeded:
        resolve_summary = ing.resolve_cross_paper_citations()
        if resolve_summary:
            yield f"{resolve_summary}\n"
    yield messages.Status(messages.done_status(completed, n, elapsed, counts=counts()), done=True)


# ── embed batch — concurrent workers, one per file ───────────────────────

def embed_batch(ing, pdfs: list) -> Generator:
    if not pdfs:
        return
    n       = len(pdfs)
    workers = min(ing.valves.EMBED_WORKERS, n)
    yield f"\nChecking **{n}** file(s) against the library...\n\n"
    _log.info("EMBED START: %d file(s), embed workers %d, collection %s", n, workers, ing.valves.COLLECTION)

    # pre-create collection once — avoids 409 race when workers all try simultaneously
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams, SparseVectorParams, SparseIndexParams, Modifier
        _qc   = QdrantClient(url=ing.valves.QDRANT_URL, timeout=30)
        _cols = [c.name for c in _qc.get_collections().collections]
        if ing.valves.COLLECTION not in _cols:
            _qc.create_collection(
                ing.valves.COLLECTION,
                vectors_config={"dense": VectorParams(size=4096, distance=Distance.COSINE)},
                sparse_vectors_config={"sparse": SparseVectorParams(index=SparseIndexParams(), modifier=Modifier.IDF)},
            )
        ing.ensure_indexes(_qc)
    except Exception as e:
        yield (
            "❌ Unable to connect to the database.\n"
            + messages.details_block("Error details", f"{type(e).__name__}: {e}")
        )
        return

    manifest = ing.load_manifest()
    doc_meta = manifest.setdefault("documents", {})

    msg_q   = queue.Queue()
    results = {}
    t_batch = time.time()

    def worker(pdf_path: Path):
        embed_worker(ing, pdf_path, doc_meta, msg_q, results)

    completed, storing, failed, unchanged = 0, 0, 0, 0
    failed_names: list[str] = []
    doc_lines = messages.DocLines()
    counts = lambda: (completed - failed - unchanged, unchanged, failed, "stored")   # the bar's counts

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for p in pdfs:
            pool.submit(worker, p)
        done = 0
        while done < n:
            try:
                msg = msg_q.get(timeout=5)
            except queue.Empty:
                mins = (time.time() - t_batch) / 60
                yield messages.Status(messages.batch_status(completed, n, None, storing, mins, counts=counts()))
                yield " "  # keepalive
                continue
            if msg is None:
                done += 1
            elif isinstance(msg, tuple):
                kind = msg[0]
                if kind == "start":
                    storing += 1
                    # no status push here (OpenWebUI keeps every status push as a permanent
                    # log entry, so only push when `completed` changes).
                elif kind == "file":
                    _, phase, pdf_path, result = msg
                    storing -= 1
                    completed += 1
                    ok = bool(result and result.get("ok"))
                    if ok and result.get("skipped"):
                        unchanged += 1
                    if not ok:
                        failed += 1
                        failed_names.append(pdf_path.name)
                    out = doc_lines.add(messages.file_summary_line(completed, n, phase, pdf_path.name, result), ok)
                    if out:
                        yield out
                    mins = (time.time() - t_batch) / 60
                    yield messages.Status(messages.batch_status(completed, n, None, storing, mins, counts=counts()))
            else:
                yield msg

    # commit successful embeds to the manifest (main thread, no locking needed). Merged, so an earlier parse-only
    # run's `parsed_hash`/`parsed_at` stay.
    for pdf_path, r in results.items():
        if r and r.get("ok") and not r.get("skipped") and r.get("source_hash"):
            doc_meta.setdefault(pdf_path.stem, {}).update({
                "source_hash": r["source_hash"],
                "vector_count": r["vectors"],
                "collection": ing.valves.COLLECTION,
                "embedded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            })
    ing.save_manifest(manifest)

    elapsed    = time.time() - t_batch
    total_vecs = sum(r["vectors"] for r in results.values() if r and r.get("ok"))
    n_skip     = sum(1 for r in results.values() if r and r.get("ok") and r.get("skipped"))
    n_stored   = completed - failed - n_skip
    skip_note  = f" · {n_skip} skipped (already in the library, unchanged)" if n_skip else ""
    took       = messages.took(elapsed)
    _log.info("EMBED DONE: %d stored, %d failed, %d passages, %.0fs elapsed", n_stored, failed, total_vecs, elapsed)
    yield doc_lines.rest()
    yield f"\n---\n✅ Done · {n_stored} stored{skip_note} · {failed} failed · {took}\n"
    if n_stored:
        yield messages.ask_links(ing.stored_titles(
            [Path(p).stem for p, r in results.items() if r and r.get("ok") and not r.get("skipped")]))
    if failed_names:
        yield f"❌ Failed: {', '.join(failed_names)}\n"
    if n_stored:
        resolve_summary = ing.resolve_cross_paper_citations()
        if resolve_summary:
            yield f"{resolve_summary}\n"
    yield messages.Status(messages.done_status(completed, n, elapsed, counts=counts()), done=True)
