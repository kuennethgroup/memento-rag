"""The library as the query path reads it: documents mid-(re)embed, the document list (load, rebuild, the title caches),
whether a copy is set up, the people client and lock, and the Qdrant query helper.
"""
import os
import threading
import time

from memento.config.paths import PROJECT_ROOT
from memento.library.copies import DEFAULT_IDS
from memento.query import request


# Other names of each document for the no-graph path: the title on its first page and the title stored in the PDF when
# it is a real one (memento/library/documents.py, shared with the start-screen picker's list). Read once per document, kept
# for the life of the process.
ALIASES: dict = {}


TITLES: dict = {}          # (out_dir, source) → (first-page title, PDF title), each None when missing


LIB_TITLES: dict = {}      # (out_dir, source) → the library's own title (memento_documents.title, as the picker lists
                            # it)


# ── Hide documents mid-(re)embed ──
# Library Upload lists a document here while it replaces its vectors, so a query never sees it half-written.
INGEST_STATUS_TABLE = "memento_ingestion_status"


def in_progress_sources(db_engine) -> set:
    try:
        engine = db_engine()           # one access path (the command line passes its own, memento/retrieval/pipe_cli.py)
        from sqlalchemy import text
        with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            rows = conn.execute(text(f"SELECT source FROM {INGEST_STATUS_TABLE}")).fetchall()
            return {r[0] for r in rows}
    except Exception:
        # the table may not exist yet (no ingestion has run): nothing in progress
        return set()


# ── The library's documents (memento/library/documents.py) ──
# One row per document of each collection in `memento_documents` (Open WebUI's Postgres), written by every upload,
# command-line upload, deletion and library reset through one write path.
_TABLES_READY: set = set()


# ── Two copies in one Open WebUI: is this copy set up? ──
# The default ids (DEFAULT_IDS) always are (nothing is read). Another copy works only when memento/library/copies.py says
# it is enabled; requests read that through a 10 s cache, the worker every round.
STATE_CACHE_S = 10.0


_STATE_CACHE: dict = {}


_state_clock = time.monotonic


def copy_state(role: str = "query", *, function_id: str, db_engine, cached: bool = True):
    """(configured, enabled, reason) of the copy `function_id`."""
    from memento.library import copies as cs
    fid = function_id
    if fid in DEFAULT_IDS:
        return cs.CopyState(True, True, "")
    key = (fid, role)
    hit = _STATE_CACHE.get(key)
    if cached and hit is not None and _state_clock() - hit[0] < STATE_CACHE_S:
        return hit[1]
    state = cs.copy_state(db_engine(), fid, role)
    _STATE_CACHE[key] = (_state_clock(), state)
    return state


def load_documents(collection: str, domain: dict | None, db_engine) -> dict | None:
    """{source: row} of this collection, or None when the table cannot be read (the caller then scrolls Qdrant).
    Fills the title caches from the rows, so names in questions match without reading every document's files."""
    import importlib
    try:
        from memento.library import documents as ldb
        engine = db_engine()
        if id(engine) not in _TABLES_READY:
            ldb.ensure_tables(engine)
            _TABLES_READY.add(id(engine))
        rows = {r["source"]: r for r in ldb.list_documents(engine, collection)}
    except Exception:
        return None
    try:
        out_dir = str(importlib.import_module("memento.config.paths").corpus_dirs(domain or {})[1])
    except Exception:
        out_dir = "None"
    for src, r in rows.items():
        key = (out_dir, src)
        LIB_TITLES[key] = r.get("title")              # each read: a title changed in the library shows at once
        if key not in ALIASES:
            TITLES[key] = (r.get("parsed_title"), r.get("pdf_title"))
            ALIASES[key] = [src] + [n for n in TITLES[key] if n]
    return rows


def rebuild_documents(collection: str, domain: dict | None, read_sources, db_engine) -> None:
    """The list is empty for this collection (a new library, or after a reset): rows for every document found in Qdrant
    (`read_sources`: the scroll, read again under the collection's lock), fingerprints from data/manifest.json. Best
    effort; the next question tries again."""
    import importlib
    import json as _json
    import logging
    try:
        from memento.library import documents as ldb
        in_dir, out_dir = importlib.import_module("memento.config.paths").corpus_dirs(domain or {})
        try:
            manifest = _json.loads((PROJECT_ROOT / "data/manifest.json").read_text())
        except Exception:
            manifest = {}
        n = ldb.rebuild_documents(db_engine(), collection=collection, sources=read_sources, in_dir=in_dir,
                                  out_dir=out_dir, section_hints=(domain or {}).get("section_hints") or {},
                                  manifest=manifest, output_root=os.getenv("OUTPUT_ROOT"))
        logging.getLogger("memento.query").info("documents: list rebuilt for %s, %d documents", collection, n)
    except Exception as e:
        logging.getLogger("memento.errors").warning("E-documents-rebuild: %r", e)


def fill_front_matter(valves, engine) -> None:
    """The library's missing author lists, read from the parsed text when the worker starts (doc_authors.fill_missing).
    Best effort, logged."""
    import importlib
    import logging
    try:
        from memento.library import front_matter as doc_authors
        out_dir = importlib.import_module("memento.config.paths").corpus_dirs(request.domain(valves.DOMAIN_NAME) or {})[1]
        stored, missing = doc_authors.fill_missing(engine, collection=valves.COLLECTION, out_dir=out_dir)
        if stored or missing:
            logging.getLogger("memento.query").info("author lists: %d filled for %s; no parsed text for %d: %s",
                                                    stored, valves.COLLECTION, len(missing), missing[:5])
    except Exception as e:
        logging.getLogger("memento.errors").warning("E-front-fill: %r", e)


# one of each copy's workers reads at a time, so two never read and store the same documents
_PEOPLE_LOCKS: dict = {}
_PEOPLE_LOCKS_GUARD = threading.Lock()


def people_lock(function_id: str) -> threading.Lock:
    """The copy's own lock for reading people (one per copy, as each copy's module had its own)."""
    with _PEOPLE_LOCKS_GUARD:
        return _PEOPLE_LOCKS.setdefault(function_id, threading.Lock())


def people_client(valves):
    """This copy's local answer model, synchronous (the worker's thread), for reading title pages."""
    from openai import OpenAI
    return OpenAI(base_url=valves.LLM_BASE_URL, api_key=request.llm_key(valves) or "none", timeout=60, max_retries=0)


def with_exclusions(filt, excluded_sources: set):
    """Add a must_not source-in(...) clause to `filt` (or create one) for
    papers currently mid-(re)embed. Returns `filt` unchanged if nothing to
    exclude."""
    if not excluded_sources:
        return filt
    from qdrant_client.models import Filter, FieldCondition, MatchAny
    excl = FieldCondition(key="source", match=MatchAny(any=list(excluded_sources)))
    if filt is None:
        return Filter(must_not=[excl])
    return Filter(must=filt.must, should=filt.should, must_not=list(filt.must_not or []) + [excl])


def qdrant_query(client, collection: str, qvec: list[float], query_text: str,
                  query_filter, limit: int, hybrid: bool, hybrid_prefetch_k: int):
    """Dense or hybrid (dense + BM25, RRF) Qdrant search; hybrid needs Valves.HYBRID_ENABLED. A thin wrapper around
    memento/retrieval/search.py:qdrant_query, shared with the evaluation so the two cannot drift apart: change search.py, not
    this."""
    from memento.retrieval import search
    return search.qdrant_query(client, collection, qvec, query_text, query_filter, limit, hybrid, hybrid_prefetch_k)


def exclude_keywords_section(filt):
    """Add a must_not section == "Keywords" clause: bare 1-3 word fragments from a paper's "Keywords:" line
    (enrich_parsed_content_list.py), never useful as a source."""
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    excl = FieldCondition(key="section", match=MatchValue(value="Keywords"))
    if filt is None:
        return Filter(must_not=[excl])
    return Filter(must=filt.must, should=filt.should, must_not=list(filt.must_not or []) + [excl])


def collection_status(client, collection: str) -> str:
    """"ok" | "empty" (no collection yet, or no documents in it)."""
    if not client.collection_exists(collection):
        return "empty"
    return "ok" if client.count(collection, exact=False).count > 0 else "empty"


def ordered_passages(client, collection: str, source: str) -> list:
    """Every passage of one document in reading order. Point ids are sha256(source, page, type, index in document order)
    (memento.ingestion.embed.chunk_point_id), so the index is recovered by matching the ids.
    Document order runs page by page, so only a window of indices per page is hashed. A passage that matches none goes
    last, by page."""
    import hashlib
    from collections import Counter
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    points, offset = [], None
    while True:
        batch, offset = client.scroll(
            collection_name=collection,
            scroll_filter=Filter(must=[FieldCondition(key="source", match=MatchValue(value=source))]),
            with_payload=True, with_vectors=False, limit=1000, offset=offset)
        points += batch
        if offset is None:
            break
    per_page = Counter((p.payload or {}).get("page", 0) for p in points)
    start, n = {}, 0
    for page in sorted(per_page):
        start[page] = n
        n += per_page[page]
    def find(p, indices):
        pl = p.payload or {}
        page, typ = pl.get("page", 0), pl.get("type", "")
        return next((i for i in indices
                     if int(hashlib.sha256(f"{source}_{page}_{typ}_{i}".encode()).hexdigest(), 16) % (2 ** 63)
                     == int(p.id)), None)

    order = {}
    for p in points:
        page = (p.payload or {}).get("page", 0)
        i = find(p, range(start[page], start[page] + per_page[page]))
        if i is None:          # a missing or extra passage shifts the windows: search the whole document
            i = find(p, range(n + 50))
        if i is not None:
            order[p.id] = i
    from memento.retrieval.search import point_to_hit
    ordered = sorted(points, key=lambda p: (order.get(p.id, 10 ** 9), (p.payload or {}).get("page", 0)))
    return [point_to_hit(p.payload or {}, 0.0) for p in ordered if (p.payload or {}).get("text", "").strip()]


def saved_valves(function_id: str, db_engine) -> dict | None:
    """A function's saved Valves (the linked ingestion copy's for Document Chat's reading; a copy's own for its worker),
    None when they cannot be read."""
    import json as _json
    from sqlalchemy import text
    try:
        with db_engine().connect() as c:
            v = c.execute(text("SELECT valves FROM function WHERE id = :id"), {"id": function_id}).scalar()
        v = _json.loads(v) if isinstance(v, str) else v
        return v if isinstance(v, dict) else None
    except Exception:
        return None
