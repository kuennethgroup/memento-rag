#!/usr/bin/env python3
"""RAG ingestion from the command line (dense embeddings, plus a BM25 "sparse" vector on every chunk, used when
HYBRID_ENABLED is set at query time). MinerU parsing and embedding are pipelined: while one document is embedded, the
next is parsed.

python memento/ingestion/run_pipeline.py --all [--force] [--skip-parse] [--reset]
python memento/ingestion/run_pipeline.py --pdf paper.pdf

--backend / --collection select the MinerU backend and the target collection (how the comparison collections are built,
with the backend as the only difference):
python memento/ingestion/run_pipeline.py --all --backend pipeline --collection memento-rag-ocr
python memento/ingestion/run_pipeline.py --all --backend hybrid-engine --collection memento-rag-hybrid

--no-enrich embeds MinerU's raw content_list.json (no figure/table/equation context, no section assignment, no fragment
filtering), the enrichment ablation:
python memento/ingestion/run_pipeline.py --all --no-enrich --collection memento-rag-noenrich"""
import argparse
import concurrent.futures
import hashlib
import json
import logging
import logging.handlers
import os
import queue as _queue
import subprocess
import sys
import threading
import time
import requests
from dataclasses import dataclass, field
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue, FilterSelector,
    SparseVectorParams, SparseIndexParams, Modifier,
)

from memento.config.cli_domain import apply_domain_arg, domain_banner  # noqa: E402
from memento.config.paths import PROJECT_ROOT
# Required on the CLI; as a library (build_graph, OpenWebUI's ingestion_pipe)
# the caller has already set DOMAIN_NAME, and a SystemExit would escape it.
apply_domain_arg(required=__name__ == "__main__")
from memento.ingestion.embed import chunk_point_id
from memento.ingestion.embed import enumerate_embeddable
from memento.config.settings import DOMAIN, COLLECTION, INPUT_DIR, OUTPUT_DIR, \
    EMBEDDER_URL, QDRANT_URL, EMBED_DIM, EMBED_BATCH_SIZE, PARSE_WORKERS, \
    MINERU_BACKEND as BACKEND, MINERU_BACKEND_FOLDER as BACKEND_FOLDER, \
    MINERU_OUT_SUBDIR, MINERU_API_URL, GRAPH_BUILD_ENABLED, \
    CONTEXTUALIZE_ENABLED, CONTEXT_BM25, GRAPH_ENRICH_COMBINED, \
    GAP_TAG_ENABLED
from memento.ingestion.convert import ensure_supported, ALL_SUPPORTED, discover_files, resolve_stem_collisions
from memento.ingestion import library_index
from memento.ingestion.enrich_parsed_content_list import enrich_content_list
from memento.ingestion.contextualize import contextualize_items
from memento.ingestion.embed import sparse_encode
# memento.graph.build.llm_enrich is imported lazily (only with GRAPH_BUILD_ENABLED, off by default): it needs the optional graph
# package.

# ── Config ────────────────────────────────────────────────────────────────────
DENSE_VECTOR_NAME  = 'dense'
SPARSE_VECTOR_NAME = 'sparse'  # defined on the collection (as Library Upload)
                                # but not yet populated by either ingestion path

# MinerU backend, output folder and API URL come from memento.config.settings, so this pipeline and the Library Upload pipe
# (Valves.BACKEND defaults to the same MINERU_BACKEND) parse into the same output folder


LOG_DIR  = PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "pipeline.log"

_FMT = logging.Formatter(
    fmt="%(asctime)s  %(levelname)-5s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Module-level logger — console only until main() attaches the file handler
log = logging.getLogger("pipeline")
log.setLevel(logging.DEBUG)
if not log.handlers:
    _ch = logging.StreamHandler(sys.stdout)
    _ch.setLevel(logging.INFO)
    _ch.setFormatter(_FMT)
    log.addHandler(_ch)


def _attach_file_handler():
    """Add a rotating file handler. Called once from main() at startup."""
    LOG_DIR.mkdir(exist_ok=True)
    fh = logging.handlers.RotatingFileHandler(
        LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(_FMT)
    log.addHandler(fh)
    log.info(f"Logging to {LOG_FILE}")


# ── Metrics ───────────────────────────────────────────────────────────────────
@dataclass
class PaperMetrics:
    paper_id:       str
    n_raw:          int   = 0
    n_skipped:      int   = 0
    n_embedded:     int   = 0
    n_vectors:      int   = 0

    t_parse_s:      float = 0.0
    t_json_s:       float = 0.0
    t_build_s:      float = 0.0
    t_embed_wall_s: float = 0.0
    t_embed_api_s:  float = 0.0
    t_qdrant_s:     float = 0.0
    t_total_s:      float = 0.0   # embed+qdrant only (parse overlaps in pipelined mode)

    @property
    def http_overhead_s(self) -> float:
        return max(0.0, self.t_embed_wall_s - self.t_embed_api_s)

    @property
    def vectors_per_sec(self) -> float:
        return self.n_vectors / self.t_embed_wall_s if self.t_embed_wall_s > 0 else 0.0


def fmt(s: float) -> str:
    if s < 60:
        return f"{s:.2f}s"
    m, sec = divmod(int(s), 60)
    return f"{m}m {sec}s"


# ── Embedder API ──────────────────────────────────────────────────────────────
def check_embedder(url: str) -> dict:
    try:
        r = requests.get(f"{url}/v1/models", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.error(f"Embedder (vLLM) not reachable at {url}: {e}")
        sys.exit(1)


def call_embedder(inputs: list[dict], url: str, text_only: bool = False) -> tuple[list, float]:
    """text_only routes to the text-only Qwen3-Embedding-8B server (its own
    raw Instruct/Query format) — the --text-only ablation never carries images."""
    import time as _time
    from concurrent.futures import ThreadPoolExecutor
    from memento.config.settings import EMBEDDER_MODEL, TEXT_EMBEDDER_MODEL
    from memento.common.model_servers import vl_embed_many, text_embed

    t0 = _time.perf_counter()
    if text_only:
        with ThreadPoolExecutor(max_workers=min(16, len(inputs))) as pool:
            futures = [pool.submit(text_embed, url, TEXT_EMBEDDER_MODEL,
                                   inp["instruction"], inp["text"]) for inp in inputs]
            embeddings = [f.result() for f in futures]
    else:
        embeddings = vl_embed_many(url, EMBEDDER_MODEL, inputs)
    return embeddings, _time.perf_counter() - t0


# ── Idempotency ───────────────────────────────────────────────────────────────
def already_ingested(paper_id: str, client: QdrantClient) -> bool:
    try:
        return client.count(
            collection_name=COLLECTION,
            count_filter=Filter(
                must=[FieldCondition(key="source", match=MatchValue(value=paper_id))]
            ),
        ).count > 0
    except Exception:
        return False


def delete_paper(paper_id: str, client: QdrantClient):
    client.delete(
        collection_name=COLLECTION,
        points_selector=FilterSelector(
            filter=Filter(
                must=[FieldCondition(key="source", match=MatchValue(value=paper_id))]
            )
        ),
    )
    log.info(f"[force] Deleted existing vectors for '{paper_id}'")


# ── Find parsed output dir ────────────────────────────────────────────────────
_OFFICE_SUBDIR = "office"


def find_paper_dir_in(output_dir: Path, stem: str, subdir: str) -> Path | None:
    """Parsed-output dir for one document: the backend subdir, else office/
    (Office-converted .doc/.docx/.pptx/.xls/.xlsx are parsed into "office"
    rather than vlm/auto/hybrid_auto), else any subdir with a content list."""
    paper_dir = output_dir / stem / subdir
    if paper_dir.exists():
        return paper_dir
    office_dir = output_dir / stem / _OFFICE_SUBDIR
    if office_dir.exists() and list(office_dir.glob('*content_list*.json')):
        return office_dir
    parent = output_dir / stem
    if parent.exists():
        for sub in sorted(parent.iterdir()):
            if sub.is_dir() and list(sub.glob('*content_list*.json')):
                log.debug(f"Found parsed output at: {sub}")
                return sub
    return None


def find_paper_dir(pdf_path: Path) -> Path | None:
    return find_paper_dir_in(OUTPUT_DIR, pdf_path.stem, MINERU_OUT_SUBDIR)


# ── Manifest (shared idempotency record with Library Upload) ──────────────
_MANIFEST_FILE = PROJECT_ROOT / "data" / "manifest.json"
_manifest_lock = threading.Lock()


def _record_in_manifest(paper_id: str, n_vectors: int) -> None:
    """Both ingestion paths must leave the same record, or a paper ingested by
    one is re-parsed by the other (Library Upload checks this file, not Qdrant)."""
    try:
        # file-locked read-modify-write, shared with the chat pipe (library_index.locked_manifest)
        with _manifest_lock, library_index.locked_manifest(_MANIFEST_FILE) as man:
            man["documents"].setdefault(paper_id, {}).update({
                "vector_count": n_vectors,
                "collection": COLLECTION,
                "embedded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "ingested_by": "run_pipeline",
            })
    except Exception as e:
        log.warning(f"manifest update failed for {paper_id}: {e!r}")


# ── Which files to add, under which name (library_index — the same decision as the chat pipe) ─────────────────
CHANGED_STEMS: set[str] = set()     # documents whose file changed: re-indexed without --force


def plan_library(docs: list[Path], client: QdrantClient) -> list[Path]:
    """Duplicates are dropped, a document renamed because of a name clash gets a symlink under its new name in
    OUTPUT_DIR/_staged (the operator's files are never renamed), changed documents go into CHANGED_STEMS."""
    with library_index.locked_manifest(_MANIFEST_FILE) as man:
        # the file server's namespace over every corpus, without this run's own files
        from memento.config.paths import roots
        names = library_index.namespace(*roots(), exclude=docs)
        decisions = library_index.decide(
            [(p.name, p) for p in docs], collection=COLLECTION, uploader=library_index.SERVER_UPLOADER,
            tag="server", manifest=man, in_library=lambda stem: already_ingested(stem, client),
            taken=names.__contains__)
        library_index.record(man, decisions, collection=COLLECTION, uploader=library_index.SERVER_UPLOADER)
    keep = []
    for d in decisions:
        if d.action == "duplicate":
            log.info(f"[{d.path.stem}] identical to '{d.stem}', already in the library — skipped")
            continue
        path = d.path
        if d.action == "renamed":
            staged = OUTPUT_DIR / "_staged"
            staged.mkdir(parents=True, exist_ok=True)
            path = staged / d.filename
            if path.is_symlink() or path.exists():
                path.unlink()
            path.symlink_to(d.path.resolve())
            log.info(f"[{d.name}] a different document with this name is already in the library — added as "
                     f"'{d.filename}'")
        if d.action == "changed":
            CHANGED_STEMS.add(d.stem)
            log.info(f"[{d.stem}] the file changed — the new version replaces the stored one")
        keep.append(path)
    return keep


# ── The library's document list (memento/library/documents.py, memento_documents) ──
# Library Chat lists the documents of its collection from one table in Open WebUI's Postgres (DATABASE_URL): the library
# listing, the start-screen picker and finding a document by name. This is the command line's side of the one write
# path. Only when Library Chat serves this collection, and only into a list that has been built: an empty list is
# rebuilt from the whole collection, since a first row alone would hide every other document.
_doc_list: dict = {}          # cached per run: engine, served collection


def _add_to_document_list(paper_id: str, client=None) -> None:
    """The stored document's row in the served library's list, then its warm-up (summary and research gaps), exactly as
    Library Upload does (Ingester.register_document): an empty list (after a library reset) is built whole from
    the collection, since a first row alone would hide every other document. Only for the collection Library
    Chat serves; best effort."""
    url = os.getenv("DATABASE_URL", "")
    if not url:
        return
    try:
        from sqlalchemy import create_engine, text
        from memento.library import documents as ldb, warmup_queue
        if "engine" not in _doc_list:
            _doc_list["engine"] = create_engine(url)
            # the query copy that serves this collection (--function-id, else the one copy whose saved COLLECTION is
            # this run's); encrypted Valves or several copies: an error, never a guess
            from memento.library import copies as copy_state
            try:
                fid = copy_state.find_copy(_doc_list["engine"], COLLECTION, FUNCTION_ID)
            except ValueError as e:
                log.error(f"document list: {e} — not updated")
                fid = None
            _doc_list["served"] = COLLECTION if fid else None
            if not fid:
                log.info(f"document list: no Memento Query copy serves {COLLECTION!r} — not updated")
        if _doc_list["served"] != COLLECTION:
            return
        engine = _doc_list["engine"]
        ldb.ensure_tables(engine)
        from memento.config.paths import roots
        try:
            manifest = json.loads(_MANIFEST_FILE.read_text())
        except Exception:
            manifest = {}
        hints = DOMAIN.get("section_hints") or {}
        if ldb.has_documents(engine, COLLECTION):
            h, who = ldb.manifest_meta(manifest, paper_id)
            ldb.add_document(engine, collection=COLLECTION,
                             entry=ldb.document_entry(paper_id, INPUT_DIR, OUTPUT_DIR, hints),
                             content_hash=h, uploaded_by=who, output_root=roots()[1])
            from memento.library import front_matter as doc_authors      # front matter for "documents by <person>"
            doc_authors.store_front(engine, collection=COLLECTION, source=paper_id, out_dir=OUTPUT_DIR,
                                    content_hash=h)
        else:
            if client is None:
                client = QdrantClient(url=QDRANT_URL, timeout=30)
            def read_sources():
                # the collection's documents as they are now, read again under the collection's lock. No seed: a removed
                # upload must stay out; a live upload's vectors are written (wait=True) before it is registered, so it
                # is in the scroll
                sources, offset = set(), None
                while True:
                    points, offset = client.scroll(collection_name=COLLECTION, with_payload=["source"],
                                                   with_vectors=False, limit=1000, offset=offset)
                    sources.update((pt.payload or {}).get("source") for pt in points
                                   if (pt.payload or {}).get("source"))
                    if offset is None:
                        return sources
            n = ldb.rebuild_documents(engine, collection=COLLECTION, sources=read_sources, in_dir=INPUT_DIR,
                                      out_dir=OUTPUT_DIR, section_hints=hints, manifest=manifest,
                                      output_root=roots()[1])
            log.info(f"document list: built for {COLLECTION}, {n} documents")
        # its summary and research gaps, written in the background by Memento Query while the model is idle
        warmup_queue.enqueue(engine, COLLECTION, paper_id, ["summary", "gaps"])
    except Exception as e:
        log.warning(f"document list update failed for {paper_id}: {e!r}")


# ── Ingest a single paper (embed + upsert) ────────────────────────────────────
def ingest_paper(paper_dir: Path, paper_id: str,
                 embedder_url: str, client: QdrantClient,
                 enrichment_enabled: bool = True, text_only: bool = False,
                 source_pdf: Path | None = None) -> PaperMetrics:
    # source_pdf: the parsed PDF; the enrichment crops merged multi-panel figures from its pages

    m = PaperMetrics(paper_id=paper_id)
    t_paper = time.perf_counter()

    t0 = time.perf_counter()
    matches = [p for p in paper_dir.glob('*_content_list.json')
               if 'enriched' not in p.name and 'v2' not in p.name]
    if not matches:
        matches = [p for p in paper_dir.glob('*_content_list_v2.json') if 'enriched' not in p.name]
    if not matches:
        log.error(f"No content_list.json in {paper_dir}")
        return m
    raw_path = matches[0]

    if enrichment_enabled:
        log.info(f"[{paper_id}] Enriching parsed content...")
        content_path = enrich_content_list(raw_path, source_pdf=source_pdf)
    else:
        # Enrichment ablation (--no-enrich / config.NOENRICH_COLLECTION): embed MinerU's raw content_list.json as is: no
        # context grounding (enriched_context), no section assignment (build_embed_input tolerates missing fields), no
        # fragment filtering (low_info is never set).
        log.info(f"[{paper_id}] Enrichment disabled (--no-enrich) — using raw parsed content")
        content_path = raw_path

    data = json.load(open(content_path))
    m.t_json_s = time.perf_counter() - t0
    m.n_raw = len(data)

    # Ingestion-time LLM enrichment (all off by default), in three shapes: CONTEXTUALIZE + GRAPH_BUILD (+
    # GRAPH_ENRICH_COMBINED): one batched pass writes both the situating context and the per-chunk entity/relation
    # extraction (item['llm_extraction'], read by build_graph.py), far fewer calls than separate passes; CONTEXTUALIZE
    # only: the plain contextualize pass (the same prompt as the memento-rag-context ablation arm); GRAPH_BUILD only (+
    # combined): an entities-only batched pass. Best effort: degrades to no context / per-chunk extraction.
    if CONTEXTUALIZE_ENABLED and GRAPH_BUILD_ENABLED and GRAPH_ENRICH_COMBINED:
        from memento.graph.build.llm_enrich import llm_enrich_items
        log.info(f"[{paper_id}] Combined LLM enrichment (context + graph extraction)...")
        llm_enrich_items(data, want_context=True, want_entities=True, images_dir=paper_dir, label=paper_id)
    elif CONTEXTUALIZE_ENABLED:
        # Cache the LLM situating contexts next to the content list: without it
        # every re-embed regenerates (nondeterministic) contexts.
        ctx_cache = content_path.with_name(content_path.stem + '_chunk_context.json')
        cached = json.load(open(ctx_cache)) if ctx_cache.exists() else None
        if cached is not None and len(cached) == len(data):
            log.info(f"[{paper_id}] Reusing cached situating context ({ctx_cache.name})")
            for item, ctx in zip(data, cached):
                if ctx:
                    item['chunk_context'] = ctx
        else:
            log.info(f"[{paper_id}] Generating contextual-retrieval situating context...")
            contextualize_items(data)
            contexts = [item.get('chunk_context') for item in data]
            n_ctx = sum(1 for c in contexts if c)
            log.info(f"[{paper_id}] Situating context for {n_ctx}/{len(data)} items")
            if n_ctx:
                with open(ctx_cache, 'w') as f:
                    json.dump(contexts, f, ensure_ascii=False)
    elif GRAPH_BUILD_ENABLED and GRAPH_ENRICH_COMBINED:
        from memento.graph.build.llm_enrich import llm_enrich_items
        log.info(f"[{paper_id}] Batched graph entity/relation extraction...")
        llm_enrich_items(data, want_context=False, want_entities=True, images_dir=paper_dir, label=paper_id)

    t0 = time.perf_counter()
    rows = enumerate_embeddable(data, paper_dir, paper_id, text_only=text_only,
                                enrichment_enabled=enrichment_enabled)
    pairs = [(r["embed_input"], r["payload"]) for r in rows]
    m.n_skipped = len(data) - len(pairs)
    m.t_build_s = time.perf_counter() - t0
    m.n_embedded = len(pairs)

    print(f"  Items  : {m.n_raw} raw  →  {m.n_skipped} skipped  →  {m.n_embedded} to embed")
    print(f"  Timing : JSON {fmt(m.t_json_s)}  |  build {fmt(m.t_build_s)}")

    if not pairs:
        return m

    total_batches = (len(pairs) + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE
    n_vectors = 0

    for i in range(0, len(pairs), EMBED_BATCH_SIZE):
        batch        = pairs[i : i + EMBED_BATCH_SIZE]
        embed_inputs = [p[0] for p in batch]
        metadatas    = [p[1] for p in batch]
        batch_num    = i // EMBED_BATCH_SIZE + 1

        t_wall = time.perf_counter()
        dense_vecs, api_s = call_embedder(embed_inputs, embedder_url, text_only=text_only)
        wall_s = time.perf_counter() - t_wall
        m.t_embed_wall_s += wall_s
        m.t_embed_api_s  += api_s

        print(f"  Batch  {batch_num}/{total_batches}: "
              f"{len(batch)} items  |  "
              f"dense GPU {api_s:.3f}s  |  "
              f"wall {wall_s:.3f}s")

        # BM25 sparse vectors from the displayed text for text and table items (their own words; more prose would dilute
        # the exact-term signal) and for images (enriched_context + caption). Equations display bare LaTeX, which has no
        # queryable terms, so they use `equation_context` (the enriched_context folded into embed_text) when present.
        # Hybrid search is gated at query time (HYBRID_ENABLED), but filling the sparse slot here keeps new uploads in
        # step with the stored collections. CONTEXT_BM25 (off) is the "contextual BM25" ablation arm, encoding the
        # contextualized embed_text instead; off because BM25's value is exact-term rescue, which context prose dilutes.
        def _sparse_text(meta: dict) -> str:
            if meta.get('type') == 'equation' and meta.get('equation_context'):
                return f"{meta['equation_context']}\n{meta.get('text') or ''}"
            return meta.get('text') or ''

        if CONTEXT_BM25:
            sparse_texts = [(ei.get('text') or _sparse_text(meta)) for ei, meta in batch]
        else:
            sparse_texts = [_sparse_text(meta) for _, meta in batch]
        sparse_vecs = sparse_encode(sparse_texts)

        points = []
        for j, ((dense_vec, meta), sparse_vec) in enumerate(zip(zip(dense_vecs, metadatas), sparse_vecs)):
            # the payload schema of Library Upload; Library Chat groups and filters citations by "source"
            payload = {
                'source':      paper_id,
                'type':        meta.get('type'),
                'text':        meta.get('text'),
                'page':        meta.get('page', 0),
                'img_path':    meta.get('img_path'),
                'section':     meta.get('section'),
                'section_raw': meta.get('section_raw'),
                'bbox':        meta.get('bbox'),
            }
            # a paragraph merged across a page break cites "pp. N–M" (as Library Upload stores it)
            if meta.get('page_end') is not None:
                payload['page_end'] = meta['page_end']
            # the authors' text the answer model reads under an equation / a table (as the upload pipe)
            for key in ('equation_context', 'table_context'):
                if meta.get(key):
                    payload[key] = meta[key]
            if meta.get('mineru_text'):
                payload['mineru_text'] = meta['mineru_text']
            if meta.get('caption_source'):
                payload['caption_source'] = meta['caption_source']
            # the same point id as Library Upload: `done` is the 0-based index over `pairs` in
            # document order, matching build_graph.py's _chunk_id(), so graph Chunk nodes line up with Qdrant points
            # whichever ingestion path ran
            done = i + j
            point_id = chunk_point_id(
                paper_id, meta.get('page', 0), meta.get('type', ''), done)
            vector = {DENSE_VECTOR_NAME: dense_vec}
            # Qdrant rejects a vector update with zero nonzero indices — short
            # stopword-only fragments can legitimately encode to an empty BM25
            # vector; just omit the sparse slot for those (dense-only fallback).
            if sparse_vec.indices:
                vector[SPARSE_VECTOR_NAME] = sparse_vec
            points.append(PointStruct(
                id=point_id,
                vector=vector,
                payload=payload,
            ))

        t0 = time.perf_counter()
        client.upsert(collection_name=COLLECTION, points=points)
        m.t_qdrant_s += time.perf_counter() - t0
        n_vectors    += len(points)

    m.n_vectors = n_vectors
    m.t_total_s = time.perf_counter() - t_paper

    if n_vectors > 0:
        _record_in_manifest(paper_id, n_vectors)
        _add_to_document_list(paper_id, client=client)

    collection_total = client.count(collection_name=COLLECTION).count
    print(f"  ✓ {m.n_vectors} vectors  |  "
          f"dense {fmt(m.t_embed_api_s)} GPU  |  "
          f"qdrant {fmt(m.t_qdrant_s)}  |  "
          f"embed total {fmt(m.t_total_s)}")
    print(f"  Collection total: {collection_total}")

    # ── materialize the knowledge graph (GRAPH_BUILD) ──
    # Best effort and off the critical path: the Chunk ids above are the same deterministic hashes build_graph.py
    # computes from `data`, so graph nodes match the Qdrant points. A failure (Neo4j unreachable, extraction error)
    # never fails ingestion; the graph can be rebuilt with `build_graph.py --all`.
    if GRAPH_BUILD_ENABLED and n_vectors > 0:
        try:
            from memento.graph.store import get_store
            from memento.graph.build.build_graph import build_paper

            with get_store() as gstore:
                gstore.ensure_schema()
                build_paper(paper_id, store=gstore, items=data, paper_dir=paper_dir, use_llm=True)
        except Exception as e:
            log.warning(f"[{paper_id}] Knowledge-graph update failed (vectors stored OK): {e}")

    # ── Gap tagging (best-effort, after upsert) ──────────────────────────
    if GAP_TAG_ENABLED and n_vectors > 0:
        try:
            from memento.config.settings import REASONING_URL
            from memento.research_gaps.tagging import GAP_TAG_TYPES, tag_paper_chunks

            text_indices = [(done, meta)
                           for done, (_ei, meta) in enumerate(pairs)
                           if meta.get("type", "text") in GAP_TAG_TYPES
                           and (meta.get("text") or "").strip()]
            chunks = [{"text": meta["text"], "section": meta.get("section", "")}
                      for _done, meta in text_indices]
            if chunks:
                log.info(f"[{paper_id}] Gap tagging {len(chunks)} text chunks...")
                tagged = tag_paper_chunks(chunks, REASONING_URL)
                for ci, (done, meta) in enumerate(text_indices):
                    pid = chunk_point_id(paper_id, meta.get("page", 0),
                                         meta.get("type", ""), done)
                    if ci in tagged:
                        client.set_payload(COLLECTION, {
                            "gap_relevant": True,
                            "gap_signal": tagged[ci]["gap_signal"],
                            "gap_type": tagged[ci]["gap_type"],
                        }, [pid])
                    else:
                        client.set_payload(COLLECTION, {"gap_relevant": False}, [pid])
                log.info(f"[{paper_id}] Gap tagged: {len(tagged)}/{len(chunks)} chunks marked as gap-relevant")
        except Exception as e:
            log.warning(f"[{paper_id}] Gap tagging failed (vectors stored OK): {e}")

    return m


# ── MinerU parse helper ───────────────────────────────────────────────────────
def _mineru_parse(pdf_path: Path, backend: str, api_url: str,
                  timeout_s: float = 3600.0) -> tuple[bool, str]:
    """Run the mineru CLI to parse a PDF via the running MinerU server.
    Output is written to OUTPUT_DIR/<stem>/<backend_subdir>/ by the CLI.
    Returns (success, error_message).
    """
    mineru_bin = str(Path(sys.executable).parent / 'mineru')
    if not Path(mineru_bin).exists():
        mineru_bin = 'mineru'
    cmd = [mineru_bin, '-p', str(pdf_path), '-b', backend,
           '-o', str(OUTPUT_DIR), '--api-url', api_url]
    if backend == 'hybrid-engine':
        # medium (the CLI default) disables image/chart analysis — explicit
        # high keeps it, matching vlm-engine's always-on behavior.
        cmd += ['--effort', 'high']
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return False, f"timed out after {timeout_s:.0f}s"
    except FileNotFoundError:
        return False, "mineru CLI not found — pip install mineru==3.4.0"

    if result.returncode != 0:
        return False, (result.stderr or result.stdout).strip()[:400]
    return True, ''


# ── Pipelined run: parse thread feeds embed (main thread) ─────────────────────
def run_pipelined(pdfs: list[Path], args, embedder_url: str,
                  client: QdrantClient, embed_workers: int = 1) -> tuple[list, list, list]:
    """A parse thread calls MinerU for each document and puts results on a queue; the main thread submits embed + upsert
    to an embed pool as soon as each document's own parse finishes (as Library Upload's run_batch), so small files do
    not wait behind a slow one.

    embed_workers=1 (default) embeds one document at a time; raising it adds document-level concurrency, like
    Library Upload's EMBED_WORKERS Valve. Measure before raising it: a single shared embedder model may not gain."""
    parse_concurrency = getattr(args, 'parse_workers', 1)
    # Unlimited queue — parallel parse may complete multiple docs before embed catches up
    parse_q: _queue.Queue = _queue.Queue(maxsize=0)

    def _parse_one(pdf_path: Path):
        paper_id = pdf_path.stem

        replace = args.force or paper_id in CHANGED_STEMS   # a changed file is re-indexed, as --force (library_index)
        if not replace and not args.reset and already_ingested(paper_id, client):
            return ('skip', paper_id, 0.0, None)

        if replace and already_ingested(paper_id, client):
            delete_paper(paper_id, client)

        log.info(f"[parse] {paper_id} — submitting")
        t0 = time.perf_counter()
        ok, err = _mineru_parse(pdf_path, BACKEND, MINERU_API_URL)
        elapsed = time.perf_counter() - t0

        if not ok:
            log.error(f"[parse] {paper_id} — FAILED ({fmt(elapsed)}): {err}")
            return ('error', paper_id, elapsed, None)

        paper_dir = find_paper_dir(pdf_path)
        if paper_dir is None:
            log.error(f"[parse] {paper_id} — output not found: "
                       f"{OUTPUT_DIR / pdf_path.stem / MINERU_OUT_SUBDIR}")
            return ('error', paper_id, elapsed, None)

        log.info(f"[parse] {paper_id} — done in {fmt(elapsed)}")
        return ('ok', paper_id, elapsed, paper_dir)

    def parse_worker():
        if parse_concurrency > 1:
            log.info(f"[parse] parallel mode — {parse_concurrency} concurrent submissions")
            with concurrent.futures.ThreadPoolExecutor(max_workers=parse_concurrency) as pool:
                futures = [pool.submit(_parse_one, p) for p in pdfs]
                for future in concurrent.futures.as_completed(futures):
                    parse_q.put(future.result())
        else:
            for pdf_path in pdfs:
                parse_q.put(_parse_one(pdf_path))
        parse_q.put(None)  # sentinel

    thread = threading.Thread(target=parse_worker, daemon=True)
    thread.start()

    all_metrics:  list[PaperMetrics] = []
    skipped_list: list[str]          = []
    failed:       list[str]          = []

    pdf_by_id = {p.stem: p for p in pdfs}

    def _embed_one(paper_id: str, paper_dir: Path, parse_elapsed: float):
        m = ingest_paper(paper_dir, paper_id, embedder_url, client,
                         enrichment_enabled=not args.no_enrich,
                         text_only=getattr(args, 'text_only', False),
                         source_pdf=pdf_by_id.get(paper_id))
        m.t_parse_s = parse_elapsed
        return paper_id, m

    embed_futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=embed_workers) as embed_pool:
        while True:
            item = parse_q.get()
            if item is None:
                break

            status, paper_id, parse_elapsed, paper_dir = item
            log.info(f"{'─'*60}")
            log.info(f"Paper: {paper_id}")

            if status == 'skip':
                log.info(f"[{paper_id}] Already in Qdrant — skipping  (--force to re-ingest)")
                skipped_list.append(paper_id)
                continue

            if status == 'error':
                log.error(f"[{paper_id}] Parse failed — skipping embed")
                failed.append(paper_id)
                continue

            embed_futures.append(embed_pool.submit(_embed_one, paper_id, paper_dir, parse_elapsed))

        thread.join()
        for future in concurrent.futures.as_completed(embed_futures):
            paper_id, m = future.result()
            if m.n_vectors == 0:
                failed.append(paper_id)
            else:
                all_metrics.append(m)

    return all_metrics, skipped_list, failed


# ── Sequential/concurrent run (--skip-parse: nothing to pipeline against) ─────
def run_sequential(pdfs: list[Path], args, embedder_url: str,
                   client: QdrantClient, embed_workers: int = 1) -> tuple[list, list, list]:
    """embed_workers=1 (default) keeps the in-order behaviour; see run_pipelined() on raising it."""
    all_metrics:  list[PaperMetrics] = []
    skipped_list: list[str]          = []
    failed:       list[str]          = []

    def _embed_one(pdf_path: Path):
        paper_id = pdf_path.stem
        log.info(f"{'─'*60}")
        log.info(f"Paper: {paper_id}")

        if paper_id in CHANGED_STEMS and not args.force:
            log.warning(f"[{paper_id}] The file changed — run without --skip-parse to read the new version; "
                        "the stored version is kept")
            return ('skip', paper_id, None)

        if not args.force and not args.reset and already_ingested(paper_id, client):
            log.info(f"[{paper_id}] Already in Qdrant — skipping  (--force to re-ingest)")
            return ('skip', paper_id, None)

        if args.force and already_ingested(paper_id, client):
            delete_paper(paper_id, client)

        paper_dir = find_paper_dir(pdf_path)
        if paper_dir is None:
            log.warning(f"[{paper_id}] No parsed output found — run without --skip-parse first")
            return ('error', paper_id, None)

        m = ingest_paper(paper_dir, paper_id, embedder_url, client,
                         enrichment_enabled=not args.no_enrich,
                         text_only=getattr(args, 'text_only', False),
                         source_pdf=pdf_path)
        return ('error' if m.n_vectors == 0 else 'ok', paper_id, m)

    def _collect(status: str, paper_id: str, m):
        if status == 'skip':
            skipped_list.append(paper_id)
        elif status == 'error':
            failed.append(paper_id)
        else:
            all_metrics.append(m)

    if embed_workers <= 1:
        for pdf_path in pdfs:
            _collect(*_embed_one(pdf_path))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=embed_workers) as pool:
            futures = [pool.submit(_embed_one, p) for p in pdfs]
            for future in concurrent.futures.as_completed(futures):
                _collect(*future.result())

    return all_metrics, skipped_list, failed


# ── Metrics summary table ─────────────────────────────────────────────────────
def print_metrics_summary(all_metrics: list[PaperMetrics], total_elapsed: float,
                          pipelined: bool):
    if not all_metrics:
        return

    mode = "pipelined parse+embed" if pipelined else "sequential (skip-parse)"
    print(f"\n{'═'*88}")
    print(f"  TIMING BREAKDOWN  [{mode}]")
    print(f"{'═'*88}")

    col = "{:<24} {:>6} {:>8} {:>7} {:>9} {:>7} {:>7} {:>7}"
    print(col.format("Paper", "Vecs", "Parse", "JSON", "Dense GPU",
                     "HTTP▲", "Qdrant", "Embed"))
    print("─" * 88)

    t_parse = t_json = t_embed = t_http = t_qdrant = t_total = 0.0
    n_vecs = 0

    for m in all_metrics:
        print(col.format(
            m.paper_id[:24], m.n_vectors,
            fmt(m.t_parse_s), fmt(m.t_json_s), fmt(m.t_embed_api_s),
            fmt(m.http_overhead_s), fmt(m.t_qdrant_s),
            fmt(m.t_total_s),
        ))
        t_parse  += m.t_parse_s
        t_json   += m.t_json_s
        t_embed  += m.t_embed_api_s
        t_http   += m.http_overhead_s
        t_qdrant += m.t_qdrant_s
        t_total  += m.t_total_s
        n_vecs   += m.n_vectors

    print("─" * 88)
    print(col.format(
        f"TOTAL ({len(all_metrics)} papers)", n_vecs,
        fmt(t_parse), fmt(t_json), fmt(t_embed),
        fmt(t_http), fmt(t_qdrant), fmt(t_total),
    ))

    pct = lambda t: f"{t/total_elapsed*100:.1f}%" if total_elapsed > 0 else "—"
    print(f"\n  Wall-clock total    {fmt(total_elapsed)}")
    if pipelined:
        print(f"  ├─ MinerU parse     {fmt(t_parse):>8}  ({pct(t_parse)})  [overlapped]")
    print(f"  ├─ Dense GPU embed  {fmt(t_embed):>8}  ({pct(t_embed)})")
    print(f"  ├─ HTTP overhead    {fmt(t_http):>8}  ({pct(t_http)})")
    print(f"  ├─ Qdrant writes    {fmt(t_qdrant):>8}  ({pct(t_qdrant)})")
    print(f"  └─ JSON + build     {fmt(t_json):>8}  ({pct(t_json)})")
    if n_vecs > 0 and total_elapsed > 0:
        print(f"\n  Throughput  {n_vecs/total_elapsed:.1f} vec/s  |  "
              f"{len(all_metrics)/(total_elapsed/60):.2f} papers/min")
    print(f"{'═'*88}")


# ── Main ──────────────────────────────────────────────────────────────────────
FUNCTION_ID = None      # --function-id: the query copy whose start-screen list a run updates


def main():
    global BACKEND, MINERU_OUT_SUBDIR, COLLECTION, INPUT_DIR, OUTPUT_DIR, FUNCTION_ID
    parser = argparse.ArgumentParser(
        description='RAG ingestion pipeline — dense embeddings, pipelined parse+embed'
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--pdf', help='Single PDF filename in input/')
    group.add_argument('--all', action='store_true', help='Process all PDFs in input dir')
    parser.add_argument('--domain', required=True, help='Corpus domain profile (config/domains/<name>.yaml), e.g. ceramics | polymer-engineering')
    parser.add_argument('--reset',        action='store_true',
                        help='Wipe the Qdrant collection and re-ingest everything')
    parser.add_argument('--skip-parse',   action='store_true',
                        help='Skip MinerU — use existing parsed output (disables pipelining)')
    parser.add_argument('--parse-only',   action='store_true',
                        help='Parse with MinerU only — skip embed and Qdrant upsert')
    parser.add_argument('--force',        action='store_true',
                        help='Re-ingest papers already in Qdrant')
    parser.add_argument('--embedder-url',   default=EMBEDDER_URL,
                        help=f'Embedder API base URL (default: {EMBEDDER_URL})')
    parser.add_argument('--parse-workers',  type=int, default=PARSE_WORKERS,
                        help=f'Concurrent MinerU parse submissions (default: {PARSE_WORKERS} — empirically tuned to a 40 GB GPU ceiling, retune for other VRAM sizes)')
    parser.add_argument('--embed-workers',  type=int, default=1,
                        help='Concurrent papers embedded at once (default: 1, i.e. sequential — '
                             'like Library Upload\'s EMBED_WORKERS Valve, but raising '
                             'this is not free throughput — measure on your hardware first.')
    parser.add_argument('--backend', default=BACKEND,
                        choices=list(BACKEND_FOLDER.keys()),
                        help=f'MinerU backend (default: {BACKEND})')
    parser.add_argument('--collection', default=COLLECTION,
                        help=f'Qdrant collection name (default: {COLLECTION}) — '
                             f'override to ingest a backend into a separate collection, '
                             f'e.g. hybrid-engine output into memento.config.settings.HYBRID_ENGINE_COLLECTION')
    parser.add_argument('--input-dir', type=Path, default=None,
                        help='Override INPUT_DIR for this run (e.g. '
                             'data/input/PolymerEngineering to scope to a subdirectory)')
    parser.add_argument('--output-dir', type=Path, default=None,
                        help='Override OUTPUT_DIR for this run (e.g. '
                             'data/output/Ceramics to mirror input structure)')
    parser.add_argument('--no-enrich', action='store_true',
                        help='Skip enrich_content_list() — embed MinerU\'s raw '
                             'content_list.json as-is (no figure/table/equation '
                             'context grounding, no section assignment, no '
                             'junk-fragment filtering). ')
    parser.add_argument('--no-images', action='store_true',
                        help='Exclude standalone image files (png/jpg/tiff/bmp/webp) '
                             'from discovery — only parse documents (pdf/docx/pptx/…)')
    parser.add_argument('--text-only', action='store_true',
                        help='RQ1 embedding-model ablation (config.TEXT_COLLECTION): route '
                             'image/chart items through the same text-only fallback used when '
                             'an image file is missing (caption/context only, image bytes '
                             'never sent) — required for text-only embedder backends whose '
                             '/embed schema has no image field and no vision capability.')
    parser.add_argument('--function-id', default=None,
                        help='the Memento Query copy whose start-screen list this run updates (default: the one '
                             'whose saved COLLECTION is this run\'s)')
    args = parser.parse_args()
    FUNCTION_ID = args.function_id

    domain_banner()
    BACKEND = args.backend
    MINERU_OUT_SUBDIR = BACKEND_FOLDER[BACKEND]
    COLLECTION = args.collection

    if args.input_dir is not None:
        INPUT_DIR = args.input_dir.resolve()
        if not INPUT_DIR.is_dir():
            print(f"[ERROR] --input-dir {INPUT_DIR} is not a directory.")
            sys.exit(1)

    if args.output_dir is not None:
        OUTPUT_DIR = args.output_dir.resolve()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    _attach_file_handler()
    embedder_url = args.embedder_url

    if args.pdf:
        pdf_path = INPUT_DIR / args.pdf
        if not pdf_path.exists():
            print(f"[ERROR] {pdf_path} not found.")
            sys.exit(1)
        docs = [pdf_path]
    else:
        if not INPUT_DIR.is_dir():
            print(f"[ERROR] documents folder {INPUT_DIR} does not exist — it is INPUT_ROOT/<corpus_dir> "
                  f"(INPUT_ROOT in config/.env, corpus_dir in config/domains/<domain>.yaml)")
            sys.exit(1)
        docs = discover_files(INPUT_DIR, include_images=not args.no_images)
        if not docs:
            print(f"[ERROR] No supported files in {INPUT_DIR}/")
            sys.exit(1)
        n_nested = sum(1 for p in docs if p.parent != INPUT_DIR)
        if n_nested:
            log.info(f"Found {len(docs)} files across "
                     f"{len({p.parent for p in docs})} directories")

        docs, _ = resolve_stem_collisions(
            docs, INPUT_DIR, staging_dir=OUTPUT_DIR / "_staged", logger=log,
        )
    # duplicates, new versions and name clashes against what the library already holds — the same decision as
    # the chat (library_index); original files, before any conversion
    if not args.parse_only:
        docs = plan_library(docs, QdrantClient(url=QDRANT_URL, timeout=60))
    pdfs = ensure_supported(docs, OUTPUT_DIR / "_converted")

    # ── Parse-only mode ───────────────────────────────────────────────────────
    if args.parse_only:
        n_workers = args.parse_workers
        mode = f"parallel x{n_workers}" if n_workers > 1 else "sequential"
        log.info(f"{'='*60}")
        log.info(f"RAG Pipeline — parse only ({len(pdfs)} PDF(s), {mode})")
        log.info(f"Backend: {BACKEND}  →  {MINERU_OUT_SUBDIR}/")
        log.info(f"{'='*60}")
        failed = []

        def _parse_one_standalone(pdf_path: Path):
            paper_id = pdf_path.stem
            log.info(f"[parse] {paper_id} — submitting")
            t0 = time.perf_counter()
            ok, err = _mineru_parse(pdf_path, BACKEND, MINERU_API_URL)
            elapsed = time.perf_counter() - t0
            if not ok:
                log.error(f"[parse] {paper_id} — FAILED ({fmt(elapsed)}): {err}")
                return (paper_id, False)
            paper_dir = find_paper_dir(pdf_path)
            if paper_dir is not None:
                log.info(f"[parse] {paper_id} — done in {fmt(elapsed)}  →  {paper_dir}")
                return (paper_id, True)
            log.error(f"[parse] {paper_id} — output not found: "
                       f"{OUTPUT_DIR / pdf_path.stem / MINERU_OUT_SUBDIR}")
            return (paper_id, False)

        if n_workers > 1:
            log.info(f"[parse] submitting {len(pdfs)} PDFs with {n_workers} concurrent workers")
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = [pool.submit(_parse_one_standalone, p) for p in pdfs]
                for future in concurrent.futures.as_completed(futures):
                    paper_id, ok = future.result()
                    if not ok:
                        failed.append(paper_id)
        else:
            for pdf_path in pdfs:
                paper_id, ok = _parse_one_standalone(pdf_path)
                if not ok:
                    failed.append(paper_id)

        log.info(f"{'='*60}")
        log.info(f"Parse complete — {len(pdfs)-len(failed)}/{len(pdfs)} succeeded")
        if failed:
            log.error(f"Failed: {', '.join(failed)}")
            sys.exit(1)
        sys.exit(0)

    pipelined = not args.skip_parse

    print(f"{'='*60}")
    print(f"RAG Pipeline — {len(pdfs)} PDF(s)")
    print(f"Collection: {COLLECTION}")
    print(f"Mode: {'pipelined parse+embed' if pipelined else 'embed-only (skip-parse)'}")
    print(f"{'='*60}")

    info = check_embedder(embedder_url)
    gpu_mem = info.get('memory', {}).get('allocated_gb', '?')
    print(f"[Embedder] ✓  GPU {info.get('gpu')}  |  "
          f"{gpu_mem} GB allocated  |  "
          f"loaded in {info.get('load_time_s', 0):.1f}s")

    client = QdrantClient(url=QDRANT_URL)
    try:
        client.get_collections()
    except Exception as e:
        print(f"[ERROR] Cannot reach Qdrant at {QDRANT_URL}: {e}")
        sys.exit(1)
    print(f"[Qdrant]   ✓  {QDRANT_URL}")

    if args.reset and client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
        print(f"[Qdrant] Collection '{COLLECTION}' wiped")

    if not client.collection_exists(COLLECTION):
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config={
                DENSE_VECTOR_NAME: VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
            },
            sparse_vectors_config={
                # modifier=IDF: FastEmbed's BM25 model (memento/ingestion/embed.py) emits TF-only document values and 1.0 for
                # queries, expecting the IDF from Qdrant's index-side scoring; without it a rare term would weigh the
                # same as a common one.
                SPARSE_VECTOR_NAME: SparseVectorParams(index=SparseIndexParams(), modifier=Modifier.IDF),
            },
        )
        print(f"[Qdrant] Created collection '{COLLECTION}' (named dense + sparse, BM25 IDF-modified)")
    else:
        n = client.count(collection_name=COLLECTION).count
        print(f"[Qdrant] Collection '{COLLECTION}' — {n} existing vectors")
    from memento.ingestion.payload_indexes import ensure_payload_indexes      # filters by document stay fast at any size
    made = ensure_payload_indexes(client, COLLECTION)
    if made:
        print(f"[Qdrant] Indexed fields: {', '.join(made)}")

    pipeline_start = time.perf_counter()

    if pipelined:
        all_metrics, skipped_list, failed = run_pipelined(pdfs, args, embedder_url, client,
                                                          embed_workers=args.embed_workers)
    else:
        all_metrics, skipped_list, failed = run_sequential(pdfs, args, embedder_url, client,
                                                           embed_workers=args.embed_workers)

    total_elapsed = time.perf_counter() - pipeline_start

    print(f"\n{'='*60}")
    print(f"Pipeline complete in {fmt(total_elapsed)}")
    print(f"  Processed : {len(all_metrics)}/{len(pdfs)}")
    print(f"  Skipped   : {len(skipped_list)} (already in Qdrant)")
    print(f"  Failed    : {len(failed)}")
    print(f"  Vectors   : {sum(m.n_vectors for m in all_metrics)}")
    print(f"  Collection: {client.count(collection_name=COLLECTION).count} total")
    if failed:
        print(f"  Failed on : {', '.join(failed)}")

    # Cross-paper citation resolution (dual graph): once per batch, not per document, since it fuzzy-matches titles
    # against every document's Reference nodes in Neo4j. Skipped unless GRAPH_BUILD is on.
    if GRAPH_BUILD_ENABLED and all_metrics:
        try:
            from memento.graph.store import get_store
            from memento.graph.build.build_graph import resolve_cross_paper_citations
            with get_store() as gstore:
                summary = resolve_cross_paper_citations(gstore)
            print(f"  Graph resolve: {summary['n_resolved']}/{summary['n_references']} "
                  f"references matched to in-corpus papers, {summary['n_edges']} edges")
        except Exception as e:
            print(f"[WARN] Cross-paper citation resolution failed (papers' vectors stored OK): {e}")

    print_metrics_summary(all_metrics, total_elapsed, pipelined)


if __name__ == '__main__':
    main()
