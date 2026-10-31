#!/usr/bin/env python3
# scripts/update_gap_ledger.py: gap ledger step 5, keeping the ledger current while the library changes. Every --every
# seconds (default 60):
#   1. waits until the local model has room (vLLM: nothing waiting, at most WARMUP_MAX_RUNNING running) and no upload
#      is in progress;
#   2. documents the library no longer has: their questions removed (deletions through Memento already do this), and
#      question vectors without a question removed from <collection>-gaps;
#   3. new or changed documents: retrieval/gap_update.process_document (its questions, the existing questions checked
#      against it, the new questions' library and literature checks; 2 model calls at a time; a per-document lock, so
#      a report that processes a pending document itself is never duplicated). Never two updaters (a lock file).
#   ( set -a; source config/.env; set +a; nohup setsid .venv/bin/python scripts/update_gap_ledger.py \
#       > logs/gap_updater.log 2>&1 & )          --once: one round; GAP_LEDGER_UPDATES=false: does nothing
import argparse
import fcntl
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "openwebui/functions"))
sys.path.insert(0, str(ROOT / "scripts"))
PY = str(ROOT / ".venv/bin/python")


def log(*a):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), *a, flush=True)


def model_has_room(V) -> bool:
    """Room on this copy's own answer model (its Valves: METRICS_URL, a key only when METRICS_API_KEY_ENV names a set
    variable, WARMUP_MAX_RUNNING)."""
    import requests
    name = (getattr(V, "METRICS_API_KEY_ENV", "") or "").strip()
    key = os.getenv(name, "") if name else ""
    try:
        text = requests.get(V.METRICS_URL, timeout=5,
                            headers={"Authorization": f"Bearer {key}"} if key else None).text
    except Exception:
        return False

    def value(name):
        return sum(float(l.rsplit(" ", 1)[1]) for l in text.splitlines() if l.startswith(name + "{"))
    return value("vllm:num_requests_waiting") == 0 and value("vllm:num_requests_running") <= V.WARMUP_MAX_RUNNING


def upload_in_progress(engine) -> bool:
    from sqlalchemy import text
    try:
        with engine.connect() as c:
            return bool(c.execute(text("SELECT count(*) FROM memento_ingestion_status")).scalar())
    except Exception:
        return False


def other_gap_job_running() -> bool:
    """A gap script started by hand (a full build, a check, the OpenAlex top-up): this round waits."""
    r = subprocess.run(["pgrep", "-f", r"scripts/(build_gap_ledger|check_gap_ledger|check_gap_literature)\.py"],
                       capture_output=True, text=True)
    return bool(r.stdout.strip())


def load_valves(engine, function_id: str):
    """The copy's Valves: its saved values over the defaults; None when it is not installed (today's copy: the
    defaults, as before)."""
    import query_pipe as qp
    from build_gap_ledger import saved_valves
    saved = saved_valves(engine, function_id)
    if saved is None and function_id != "memento_query":
        return None
    V = qp.Pipe.Valves()
    for k, v in (saved or {}).items():
        if v is not None and hasattr(V, k):
            setattr(V, k, v)
    return V


def take_lock(lock_dir, collection: str):
    """The collection's updater lock (an open file), or None when another updater of this library holds it."""
    lock = open(Path(lock_dir) / f".gap_updater.{collection}.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        lock.close()
        return None
    return lock


def still_ours(engine, function_id: str, collection: str) -> str | None:
    """None while the copy is enabled and serves `collection`; else why this updater stops."""
    from memento.library import copies as cs
    state = cs.copy_state(engine, function_id, "query")
    if not state.enabled:
        return f"{function_id} no longer enabled: {state.reason}"
    V = load_valves(engine, function_id)
    if V is None or V.COLLECTION != collection:
        return f"{function_id} now names another collection ({V and V.COLLECTION})"
    return None


def one_round(engine, V) -> str:
    from qdrant_client import QdrantClient
    from qdrant_client.models import PointIdsList
    from memento.research_gaps import ledger as gl
    from memento.library import documents as ldb
    coll = V.COLLECTION
    gl.ensure_tables(engine)
    todo = gl.pending_documents(ldb.list_documents(engine, coll), gl.ledger_documents(engine, coll))
    qc = QdrantClient(url=V.QDRANT_URL, timeout=60)
    for src in todo["removed"]:
        log(f"removed from the library: {src}")
        gl.delete_for_source(engine, coll, src)
    gaps_coll = f"{coll}-gaps"
    if qc.collection_exists(gaps_coll):
        have = {g["id"] for g in gl.list_gaps(engine, coll)}
        stale, offset = [], None
        while True:
            pts, offset = qc.scroll(gaps_coll, limit=1000, offset=offset, with_payload=False, with_vectors=False)
            stale += [p.id for p in pts if int(p.id) not in have]
            if offset is None:
                break
        if stale:
            qc.delete(gaps_coll, PointIdsList(points=stale))
            log(f"question vectors without a question removed: {len(stale)}")
    if not todo["new"]:
        return "nothing new"
    from memento.research_gaps import update as gu
    done = 0
    for src in todo["new"]:
        if not model_has_room(V) or upload_in_progress(engine):
            return "busy"
        if gu.in_progress(engine, coll, src):
            continue                                     # a report is processing it right now
        log(f"new or changed document: {src}")
        r = gu.process_document(make_ctx(engine, V, workers=2), src, owner="updater")
        done += r is not None
    return f"updated {done}"


def make_ctx(engine, V, workers: int):
    """The processing context from the Query pipe's saved valves (as the scripts read them)."""
    from openai import OpenAI
    from qdrant_client import QdrantClient
    import query_pipe as qp
    from memento.query import request
    from memento.research_gaps import update as gu
    dom = request.domain(V.DOMAIN_NAME)
    return gu.Ctx(engine=engine, qdrant=QdrantClient(url=V.QDRANT_URL, timeout=60),
                  llm_client=OpenAI(base_url=V.LLM_BASE_URL, api_key=request.llm_key(V), timeout=600),
                  model=V.LLM_MODEL, collection=V.COLLECTION, embedder_url=V.EMBEDDER_URL,
                  embedder_model=V.EMBEDDER_MODEL, reranker_url=V.RERANKER_URL,
                  instruction=dom.get("instruction") or request.FALLBACK_INSTRUCTION, retrieve_k=V.RETRIEVE_K,
                  rerank_instruction=request.rerank_instruction(dom),
                  hybrid=V.HYBRID_ENABLED, hybrid_prefetch_k=V.HYBRID_PREFETCH_K, workers=workers,
                  s2_key=os.environ.get("S2_API_KEY", ""), oa_mailto=os.environ.get("OPENALEX_MAILTO", ""),
                  oa_key=os.environ.get("OPENALEX_API_KEY", ""), log=log)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=60)
    ap.add_argument("--once", action="store_true")
    # the copy whose library this updater keeps current (default: the default copy)
    ap.add_argument("--function-id", default="memento_query")
    ap.add_argument("--lock-dir", default=str(ROOT / "logs"))
    args = ap.parse_args(argv)
    if os.getenv("GAP_LEDGER_UPDATES", "true").strip().lower() in ("0", "false", "no", "off"):
        log("GAP_LEDGER_UPDATES is off")
        return 0
    from sqlalchemy import create_engine
    from memento.library import copies as cs
    engine = create_engine(os.environ["DATABASE_URL"])
    fid = args.function_id
    state = cs.copy_state(engine, fid, "query")
    if not state.enabled:                    # never today's settings for a copy that is not set up
        log(f"{fid} not enabled ({state.reason}): nothing to do")
        return 0
    V = load_valves(engine, fid)
    coll = V.COLLECTION
    lock = take_lock(args.lock_dir, coll)
    if lock is None:
        log(f"another updater of {coll} is running")
        return 0
    log(f"gap ledger updater of {coll} ({fid}) started (every {args.every} s)")
    try:
        while True:
            try:
                why = still_ours(engine, fid, coll)
                if why:
                    log(f"stopped: {why}")
                    return 0
                V = load_valves(engine, fid)         # this round's saved settings: a change applies at once
                if model_has_room(V) and not upload_in_progress(engine) and not other_gap_job_running():
                    result = one_round(engine, V)
                    if result != "nothing new":
                        log(result)
            except Exception as e:
                log(f"round failed: {e!r}")
            if args.once:
                return 0
            time.sleep(args.every)
    finally:
        lock.close()


if __name__ == "__main__":
    sys.exit(main())
