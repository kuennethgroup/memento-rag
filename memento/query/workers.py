"""Each copy's background worker: its upload warm-up (the start screen's questions asked once per new document while
the model has room, so their answers are stored), the people of documents not read yet, Document Chat's cleanup,
the worker generations, and the copy's runtime files (its model map and its library's list).
"""
import asyncio
import contextlib
import os
import re
import threading
import time
from pathlib import Path

import requests

from memento.config.paths import PROJECT_ROOT
from memento.document_chat import cleanup
from memento.query import checking, generation, library_state, service


PEOPLE_PER_ROUND = 4          # documents the worker reads per round for their people (doc_people)


def fill_people(valves, engine, st: dict, *, function_id: str, db_engine) -> None:
    """The people (authors; supervisors and reviewers) of documents not read yet, a few per round while the model is
    idle (memento/library/people.py): new uploads, command-line ones and older documents alike. Nothing left or the model
    failing: rests a while. Best effort, logged."""
    import logging
    if time.monotonic() < st.get("people_next", 0.0) or not library_state.people_lock(function_id).acquire(blocking=False):
        return
    v = valves
    try:
        metrics = vllm_metrics(getattr(v, "METRICS_URL", None), getattr(v, "METRICS_API_KEY_ENV", None))
        if not warmup_idle(metrics, library_state.in_progress_sources(db_engine), getattr(v, "WARMUP_MAX_RUNNING", None)):
            return
        from memento.library import front_matter as doc_authors, people as doc_people, documents as ldb
        doc_people.ensure_table(engine)
        rows = ldb.list_documents(engine, v.COLLECTION)
        library = {r["source"]: r.get("content_hash") for r in rows}
        fronts, _ = doc_authors.fresh(doc_authors.load(engine, v.COLLECTION, with_hash=True), library)
        if not [s for s in doc_people.todo(engine, v.COLLECTION, library) if fronts.get(s)]:
            st["people_next"] = time.monotonic() + 60
            return
        stored, failed = doc_people.fill(engine, collection=v.COLLECTION, library=library, fronts=fronts,
                                         titles={r["source"]: r.get("title") for r in rows}, client=library_state.people_client(v),
                                         model=v.LLM_MODEL, limit=PEOPLE_PER_ROUND)
        logging.getLogger("memento.query").info("people: %d read for %s, %d failed", stored, v.COLLECTION, failed)
        if failed and not stored:
            st["people_next"] = time.monotonic() + 300                 # the model failing: try again later
    except Exception as e:
        logging.getLogger("memento.errors").warning("E-people: %r", e)
        st["people_next"] = time.monotonic() + 300
    finally:
        library_state.people_lock(function_id).release()


# ── Upload warm-up (memento/library/warmup_queue.py) ──
# After every stored upload the document's summary and research gaps are queued; one background thread asks each picker
# question once through this pipe (the local model, no chat) while the model has room, so the answer is stored under
# exactly the live key. Questions never wait for it.
WARMUP_METRICS_URL = os.getenv("WARMUP_METRICS_URL", "http://127.0.0.1:8090/metrics")   # the local model server


# a warm-up job runs while at most this many requests are running and none is waiting, so one slot always stays free for
# a new question
WARMUP_MAX_RUNNING = int(os.getenv("WARMUP_MAX_RUNNING", "2"))


# a running job counts as crashed (and is handed out again) only after the answer limit + its check + a margin
WARMUP_STALE_S = int(float(os.getenv("ANSWER_TIMEOUT", "480")) + checking.WARMUP_VERIFY_TIMEOUT + 300)


def vllm_metrics(url: str | None = None, key_env: str | None = None) -> str | None:
    """The answer model's vLLM metrics: the copy's METRICS_URL (default: the local model), with a Bearer key only when
    the copy's METRICS_API_KEY_ENV names a set variable."""
    key = os.getenv(key_env.strip(), "") if (key_env or "").strip() else ""
    try:
        r = requests.get(url or WARMUP_METRICS_URL, timeout=2,
                         headers={"Authorization": f"Bearer {key}"} if key else None)
        return r.text if r.ok else None
    except Exception:
        return None


def warmup_idle(metrics: str | None, uploads: set, max_running: int | None = None) -> bool:
    """Room for a warm-up job: no upload in progress (answers are not stored then, so the work would be wasted), and
    the model has at most `max_running` (the copy's WARMUP_MAX_RUNNING) requests running and none waiting (without
    metrics: the upload rule)."""
    if uploads:
        return False
    limit = WARMUP_MAX_RUNNING if max_running is None else max_running

    def value(name):
        m = re.search(rf"^vllm:{name}\{{[^}}]*\}} ([0-9.eE+]+)$", metrics or "", re.M)
        return float(m.group(1)) if m else 0.0
    return value("num_requests_waiting") == 0 and value("num_requests_running") <= limit


def warmup_question(action: str, title: str) -> str:
    """The start-screen picker's question for this action (loader.js), so the stored answer has the live key."""
    t = title.replace('"', "'")
    return f'Summarize the document "{t}"' if action == "summary" else f'What research gaps does the document "{t}" name?'


def warmup_name(engine, collection: str, doc: dict) -> str:
    """The document's name in the warm-up's question, as the picker sends it (loader.js `shared`): its title, or its
    source when another document of this library has the same title, which the pipe could not resolve."""
    from sqlalchemy import func, select
    from memento.library import documents as ldb
    with engine.connect() as c:
        n = c.execute(select(func.count()).select_from(ldb.DOCUMENTS).where(
            ldb.DOCUMENTS.c.collection == collection, ldb.DOCUMENTS.c.title == doc["title"])).scalar()
    return doc["source"] if (n or 0) > 1 else doc["title"]


async def run_warmup_job(valves, job: dict, title: str, *, function_id: str, db_engine) -> dict:
    """The warm-up's outcome as the answer reported it ({result, reason, key}: hit | stored | ineligible | error); no
    report, or an exception, is an error (retried), never success."""
    local = next(v for v in generation.copy_variants(valves) if not v.is_cloud)
    body = {"model": f"{function_id}.{local.id}", "stream": True,          # this copy's own model
            "messages": [{"role": "user", "content": warmup_question(job["action"], title)}]}

    async def collect(event):
        return None
    try:
        # without "Thorough check" (a checked warm-up takes minutes per document): the first reader who asks with the
        # switch on gets a fresh checked answer, which then replaces this one for everyone (`checked`)
        result: dict = {}
        # the copy's own answer path, as the adapter's pipe() takes it: a copy of the Valves, closed when done
        async with contextlib.aclosing(service.handle(
                body, {"id": "memento-warmup", "role": "admin"}, collect, None, {"warmup": True, "warmup_result": result},
                None, function_id=function_id, valves=valves.model_copy(), engine=db_engine)) as stream:
            async for _ in stream:
                pass
        if result.get("result") not in ("hit", "stored", "ineligible", "error"):
            return {"result": "error", "reason": "no result reported", "key": None}
        return {"reason": "", "key": None, **result}
    except Exception as e:
        return {"result": "error", "reason": repr(e), "key": None}


async def warmup_step(valves, engine, *, function_id: str, db_engine) -> str:
    """One round of the worker: busy | empty | skipped | done | failed."""
    import logging
    log = logging.getLogger("memento.query")
    v = valves
    metrics = await asyncio.to_thread(vllm_metrics, getattr(v, "METRICS_URL", None),
                                      getattr(v, "METRICS_API_KEY_ENV", None))
    idle = warmup_idle(metrics, await asyncio.to_thread(library_state.in_progress_sources, db_engine), getattr(v, "WARMUP_MAX_RUNNING", None))
    if not idle:
        return "busy"
    from memento.library import documents as ldb, warmup_queue as wq
    # only this copy's library: another copy's worker runs those
    job = await asyncio.to_thread(wq.claim_next, engine, WARMUP_STALE_S, collection=v.COLLECTION)
    if job is None:
        return "empty"
    doc = (await asyncio.to_thread(ldb.get_document, engine, job["collection"], job["source"])
           if job["collection"] == valves.COLLECTION else None)
    if doc is None:                     # another library's job (the deployment switched), or the document is gone
        await asyncio.to_thread(wq.finish, engine, job, True, "document gone", outcome="skipped")
        log.info("warmup: skipped %s %s", job["action"], job["source"])
        return "skipped"
    log.info("warmup: start %s %s", job["action"], job["source"])
    t0 = time.monotonic()
    out = await run_warmup_job(valves, job, await asyncio.to_thread(warmup_name, engine, job["collection"], doc),
                               function_id=function_id, db_engine=db_engine)
    res = out["result"]
    if res in ("hit", "stored"):
        await asyncio.to_thread(wq.finish, engine, job, True, outcome="done", cache_key=out.get("key"))
    elif res == "ineligible":            # final: never retried; the reason is kept
        await asyncio.to_thread(wq.finish, engine, job, False, out.get("reason") or "", outcome="ineligible")
    else:
        await asyncio.to_thread(wq.finish, engine, job, False, out.get("reason") or "error")
    status = {"hit": "done", "stored": "done", "ineligible": "ineligible"}.get(res, "failed")
    log.info("warmup: %s %s %s (%.0f s)%s", status, job["action"], job["source"], time.monotonic() - t0,
             "" if status == "done" else f" {(out.get('reason') or '')[:200]}")
    return "done" if status == "done" else "failed" if status == "failed" else "skipped"


def warmup_cycle(st: dict, *, function_id: str, valves_class, db_engine) -> float:
    """One round of this copy's worker (`st` carries its state between rounds); the pause before the next. A copy other
    than the default works only while it is enabled (memento/library/copies.py: installed, every setting saved, active;
    checked every round, so saving the Valves or switching the function on or off takes effect without a restart);
    becoming enabled publishes its runtime files. The worker runs outside a request, where Open WebUI would pass the
    Valves: they are read saved, every round."""
    import logging
    log = logging.getLogger("memento.query")
    if function_id not in library_state.DEFAULT_IDS:
        state = library_state.copy_state("query", cached=False, function_id=function_id, db_engine=db_engine)
        if not state.enabled:
            if st.get("enabled") is not False:
                log.info("warmup: %s waits: %s", function_id, state.reason or "not enabled")
            st["enabled"] = False
            return 15
        if st.get("enabled") is not True:
            log.info("warmup: %s enabled", function_id)
            try:
                publish_runtime(valves_from(valves_class, library_state.saved_valves(function_id, db_engine)),
                                function_id=function_id, db_engine=db_engine)
            except Exception as e:
                logging.getLogger("memento.errors").warning("E-publish-runtime: %r", e)
        st["enabled"] = True
    saved = library_state.saved_valves(function_id, db_engine)
    if saved is None:
        return 15
    valves = valves_from(valves_class, saved)
    engine = db_engine()
    if not st.get("ready"):
        from memento.library import documents as ldb
        ldb.ensure_tables(engine)
        library_state.fill_front_matter(valves, engine)            # once per start: author lists the library lacks
        st["ready"] = True
    if time.monotonic() >= st.get("next_cleanup", 0.0):      # Document Chat: files of deleted chats (hourly)
        st["next_cleanup"] = time.monotonic() + cleanup.DOC_CHAT_CLEANUP_S
        cleanup.doc_chat_cleanup(valves, engine, function_id=function_id, db_engine=db_engine)
        try:
            cleanup.remove_old_reports(os.getenv("OUTPUT_ROOT") or str(PROJECT_ROOT / "data/output"))
        except Exception as e:
            logging.getLogger("memento.errors").warning("E-reports-cleanup: %r", e)
    fill_people(valves, engine, st, function_id=function_id, db_engine=db_engine)   # authors, supervisors of unread documents
    step = asyncio.run(warmup_step(valves, engine, function_id=function_id, db_engine=db_engine))
    return 2 if step in ("done", "failed", "skipped") else 15


def warmup_loop(gen: int, *, function_id: str, valves_class, db_engine) -> None:
    import logging
    log = logging.getLogger("memento.query")
    log.info("warmup: worker %s %d started", function_id, gen)
    st: dict = {}
    while gen_current(function_id, gen):                      # a code update of this copy starts a newer worker
        pause = 15
        try:
            pause = warmup_cycle(st, function_id=function_id, valves_class=valves_class, db_engine=db_engine)
        except Exception as e:
            logging.getLogger("memento.errors").warning("E-warmup: %r", e)
        time.sleep(pause)
    log.info("warmup: worker %s %d stopped (newer code loaded)", function_id, gen)


def next_gen(own_id: str) -> int:
    """A new worker generation for this copy. The counter is one dict per process, keyed by function id, so two copies'
    workers never stop each other."""
    import builtins
    gens = getattr(builtins, "_memento_warmup_gen", None)
    if not isinstance(gens, dict):
        gens = {}
        builtins._memento_warmup_gen = gens
    gens[own_id] = gens.get(own_id, 0) + 1
    return gens[own_id]


def gen_current(own_id: str, gen: int) -> bool:
    import builtins
    gens = getattr(builtins, "_memento_warmup_gen", None)
    return isinstance(gens, dict) and gens.get(own_id) == gen


def start(function_id: str, valves_class, db_engine) -> None:
    """The copy's worker (its Valves class and database from the adapter); the default copies publish their model map
    at once, another copy once its worker finds it enabled (warmup_cycle)."""
    if function_id in library_state.DEFAULT_IDS:
        publish_model_colours(None, function_id, valves_class=valves_class, db_engine=db_engine)
    # what the worker imports later, imported here in the loading thread: a first import in the worker while Open
    # WebUI loads the next function can end in an import deadlock, and Open WebUI then switches that function off
    import qdrant_client.http.models  # noqa: F401
    import qdrant_client.models  # noqa: F401
    from memento.library import documents, front_matter, people, warmup_queue  # noqa: F401
    from memento.query import models  # noqa: F401
    gen = next_gen(function_id)
    threading.Thread(target=warmup_loop, args=(gen,), daemon=True, name=f"memento-warmup-{function_id}-{gen}",
                     kwargs=dict(function_id=function_id, valves_class=valves_class, db_engine=db_engine)).start()


def valves_from(valves_class, saved: dict | None):
    """A copy's Valves from saved values (unknown keys and None left out), else the defaults."""
    fields = valves_class.model_fields
    return valves_class(**{k: v for k, v in (saved or {}).items() if v is not None and k in fields})


def publish_model_colours(valves, function_id: str, *, valves_class=None, db_engine=None) -> None:
    """This copy's entries of OUTPUT_ROOT/_runtime/query_models.json (retrieval/query_models.merge_model_map): its
    models with their colours, its Document Chat and Library Upload, its collection and model ids, for the chat box, the
    model dropdown and the links (loader.js reads it as /files/models). `valves`: the copy's settings (None: its saved
    Valves, read here, so a failure is logged, never raised while the pipe loads). A new model's colour shows once Open
    WebUI reloads this pipe."""
    root = os.getenv("OUTPUT_ROOT")
    if not root:
        return
    try:
        from memento.query.models import map_entries, merge_model_map
        copy_valves = valves if valves is not None else valves_from(
            valves_class, library_state.saved_valves(function_id, db_engine))
        v = copy_valves
        merge_model_map(Path(root) / "_runtime" / "query_models.json", function_id,
                        map_entries(generation.copy_variants(copy_valves), base=v.QUERY_MODEL_ID, collection=v.COLLECTION,
                                    siblings={"query": v.QUERY_MODEL_ID, "upload": v.UPLOAD_MODEL_ID,
                                              "doc": v.DOC_CHAT_MODEL_ID}))
    except Exception as e:
        import logging
        logging.getLogger("memento.errors").warning("E-model-colours: %r", e)


def publish_runtime(valves, *, function_id: str, engine=None, db_engine=None) -> None:
    """A copy's runtime files: its model map entries and its library's start-screen list (after setup, and a second
    copy's worker once it is enabled)."""
    publish_model_colours(valves, function_id)
    root = os.getenv("OUTPUT_ROOT")
    if root:
        from memento.library import documents as ldb
        e = engine if engine is not None else db_engine()
        ldb.ensure_tables(e)
        ldb.export_library(e, valves.COLLECTION, root)
