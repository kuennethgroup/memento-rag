"""One library's documents: where they are, and one document's steps (parse with MinerU, store its passages in
Qdrant, register it in the library). Library Upload makes one Ingester per request; Document Chat's temporary
ingester (memento/document_chat/intake.py) changes where the files are and skips the library's records.
"""
import base64
import hashlib
import json
import logging
import os
import time
from pathlib import Path

import requests
from pypdf import PdfReader

from memento.config.paths import PROJECT_ROOT
# duplicates, new versions and name clashes: the same decision as run_pipeline.py
from memento.ingestion import library_index
# the chunk-type sets of run_pipeline.py (memento/ingestion/embed.py), so a document embeds the same whichever path
# parsed it. Headings (text_level items) are not chunks of their own: enrich_content_list() (assign_sections) folds them
# into each item's `section` and `section_raw`. aside_text, page_footnote and footer are watermark and licence
# boilerplate: skipped.
from memento.ingestion.embed import IMAGE_TYPES, SKIP_TYPES, TABLE_TYPES, TEXT_TYPES
from memento.ingestion.library_index import MANIFEST_FILE
from memento.ingestion.upload import intake
from memento.query.library_state import INGEST_STATUS_TABLE
from memento.query.request import llm_key

_log = logging.getLogger("memento.ingestion")

# the library's document list: memento/library/documents.py (memento_documents), through Ingester.register_document.
# The embedding instruction comes from memento.config.settings (shared with run_pipeline.py), loaded where it is needed.

_LEGACY_IDS   = PROJECT_ROOT / "data/.seen_file_ids.json"
POLL_INTERVAL = 3
TIMEOUT       = 1800
TERMINAL      = {"completed", "done", "success", "failed", "error"}


# Documents being (re)embedded, so Library Chat can hide them: a document mid-rewrite has its old vectors deleted
# before the new ones are written. Only one run holds the advisory lock, so a fresh run truncates this table: leftover
# rows come from a crashed run.
def set_paper_in_progress(stem: str, in_progress: bool):
    """Mark/unmark `stem` as mid-(re)embed in INGEST_STATUS_TABLE, so Library Chat hides it from retrieval. Best
    effort: a database hiccup here must never fail ingestion."""
    try:
        from open_webui.internal.db import engine
        from sqlalchemy import text
        with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            conn.execute(text(
                f"CREATE TABLE IF NOT EXISTS {INGEST_STATUS_TABLE} "
                "(source TEXT PRIMARY KEY, started_at TIMESTAMPTZ NOT NULL DEFAULT now())"
            ))
            if in_progress:
                conn.execute(
                    text(
                        f"INSERT INTO {INGEST_STATUS_TABLE} (source) VALUES (:source) "
                        "ON CONFLICT (source) DO UPDATE SET started_at = now()"
                    ),
                    {"source": stem},
                )
            else:
                conn.execute(
                    text(f"DELETE FROM {INGEST_STATUS_TABLE} WHERE source = :source"),
                    {"source": stem},
                )
    except Exception:
        pass


class Ingester:
    """Where one library's documents are (its Valves' domain, INPUT_ROOT and OUTPUT_ROOT) and one document's steps.
    Document Chat's temporary ingester overrides input_dir, output_dir, mark_in_progress, register_document and
    extra_payload."""

    def __init__(self, valves):
        self.valves = valves

    def corpus_dirs(self) -> tuple[Path, Path]:
        """(documents folder, parsed-output folder) of the Valve's domain."""
        import importlib
        paths = importlib.import_module("memento.config.paths")
        return paths.corpus_dirs(self.valves.DOMAIN_NAME, self.valves.INPUT_ROOT, self.valves.OUTPUT_ROOT)

    def input_dir(self) -> Path:
        d = self.corpus_dirs()[0]
        if getattr(self, "_logged_dirs", None) != (self.valves.DOMAIN_NAME, str(d)):
            self._logged_dirs = (self.valves.DOMAIN_NAME, str(d))
            _log.info("Corpus %s: documents %s | parsed output %s", self.valves.DOMAIN_NAME, d, self.output_dir())
        return d

    def output_dir(self) -> Path:
        return self.corpus_dirs()[1]

    # ── MinerU output subfolder for the configured backend ───────────────────
    def out_subdir(self) -> str:
        try:
            import importlib
            MINERU_BACKEND_FOLDER = importlib.import_module("memento.config.settings").MINERU_BACKEND_FOLDER
            return MINERU_BACKEND_FOLDER.get(self.valves.BACKEND, "vlm_auto")
        except Exception:
            return "vlm"

    def find_paper_dir(self, stem: str) -> Path | None:
        """Locate parsed output for *stem*, falling back across subfolders.

        MinerU's office parser writes to ``<stem>/office/`` instead of the backend folder (``vlm/``); this mirrors
        ``run_pipeline.find_paper_dir``. Searches the domain's corpus folder (OUTPUT_ROOT/<corpus_dir>/<stem>/) first,
        then the flat layout (OUTPUT_ROOT/<stem>/), never another corpus's folder."""
        for base in (self.output_dir() / stem, Path(self.valves.OUTPUT_ROOT) / stem):
            primary = base / self.out_subdir()
            if primary.exists() and list(primary.glob("*content_list*.json")):
                return primary
            if base.is_dir():
                for sub in base.iterdir():
                    if sub.is_dir() and list(sub.glob("*content_list*.json")):
                        return sub
        return None

    # ── manifest ──

    def load_manifest(self) -> dict:
        try:
            return json.loads(MANIFEST_FILE.read_text())
        except Exception:
            m: dict = {"seen_file_ids": [], "documents": {}}
            # one-time migration from legacy files
            try:
                m["seen_file_ids"] = json.loads(_LEGACY_IDS.read_text())
            except Exception:
                pass
            for hf in list(self.output_dir().glob("*/hybrid_auto/.embed_hash")) + \
                       list(Path(self.valves.OUTPUT_ROOT).glob("*/hybrid_auto/.embed_hash")):
                stem = hf.parent.parent.name
                try:
                    m["documents"][stem] = {"source_hash": hf.read_text().strip()}
                except Exception:
                    pass
            return m

    def save_manifest(self, manifest: dict):
        """Merge into the manifest under its file lock: the command line may have written in the meantime, so
        re-read and add this run's entries instead of overwriting the whole file."""
        with library_index.locked_manifest(MANIFEST_FILE) as m:
            m["seen_file_ids"] = sorted(set(m["seen_file_ids"]) | set(manifest.get("seen_file_ids", [])))
            for stem, meta in manifest.get("documents", {}).items():
                m["documents"].setdefault(stem, {}).update(meta)

    def in_library(self, stem: str) -> bool:
        """Does the collection hold vectors for this document? (A manifest entry alone does not count.)"""
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            return QdrantClient(url=self.valves.QDRANT_URL, timeout=30).count(
                self.valves.COLLECTION, exact=True,
                count_filter=Filter(must=[FieldCondition(key="source", match=MatchValue(value=stem))])).count > 0
        except Exception:
            return False            # collection not created yet

    def decide(self, items: list, uploader: str, tag: str) -> list:
        with library_index.locked_manifest(MANIFEST_FILE) as m:
            # the file server's namespace over every corpus: a new name is free in both libraries
            names = library_index.namespace(self.valves.INPUT_ROOT, self.valves.OUTPUT_ROOT,
                                            exclude=[p for _n, p in items])
            decisions = library_index.decide(items, collection=self.valves.COLLECTION, uploader=uploader, tag=tag,
                                             manifest=m, in_library=self.in_library, taken=names.__contains__)
            library_index.record(m, decisions, collection=self.valves.COLLECTION, uploader=uploader)
        return decisions

    @staticmethod
    def decision_line(d) -> str | None:
        if d.action == "duplicate":
            return f"⏭ **{d.name}**: already in the library as **{d.stem}**, not added again\n"
        if d.action == "renamed":
            return (f"📄 **{d.name}**: a different document with this name is already in the library, "
                    f"so this one is added as **{d.filename}**\n")
        return None

    def ensure_indexes(self, qc) -> None:
        """Indexes on the fields every filter uses (memento/ingestion/payload_indexes.py); best effort, never blocks an upload.
        Indexes on the fields every filter uses (memento/ingestion/payload_indexes.py); best effort, never blocks an upload."""
        try:
            self.corpus_dirs()
            from memento.ingestion.payload_indexes import ensure_payload_indexes
            made = ensure_payload_indexes(qc, self.valves.COLLECTION)
            if made:
                _log.info("Qdrant: indexed %s in %s", ", ".join(made), self.valves.COLLECTION)
        except Exception as e:
            _log.warning("Qdrant indexes not created: %r", e)

    def mark_in_progress(self, stem: str, in_progress: bool) -> None:
        """Hide / show `stem` for the Query pipe while its passages are rewritten. A hook: Document Chat's own files
        (memento/document_chat/intake.py) are not library documents and skip it."""
        set_paper_in_progress(stem, in_progress)

    def extra_payload(self) -> dict:
        """Fields added to every stored passage: none for the library; Document Chat adds its chat and user."""
        return {}

    def stored_titles(self, stems: list) -> list:
        """The titles the picker shows for these documents, for the links after an upload; the file name when a title
        cannot be read. Best effort."""
        try:
            import importlib
            in_dir, out_dir = self.corpus_dirs()
            from memento.library import documents as ldb
            hints = importlib.import_module("memento.config.domain_loader").load_domain(
                self.valves.DOMAIN_NAME).get("section_hints") or {}
            return [ldb.document_entry(s, in_dir, out_dir, hints).get("title") or s for s in stems]
        except Exception:
            return list(stems)

    def register_document(self, stem: str, file_hash: str | None = None) -> None:
        """Add a stored document to the library's list (memento/library/documents.py): its row, then the picker's list. An
        empty list is built whole from the collection instead, since a first row alone would hide every other document.
        Best effort; never fails an upload."""
        try:
            import importlib
            in_dir, out_dir = self.corpus_dirs()
            from open_webui.internal.db import engine
            from memento.library import documents as ldb, warmup_queue
            ldb.ensure_tables(engine)
            hints = importlib.import_module("memento.config.domain_loader").load_domain(
                self.valves.DOMAIN_NAME).get("section_hints") or {}
            manifest = self.load_manifest()
            if ldb.has_documents(engine, self.valves.COLLECTION):
                h, who = ldb.manifest_meta(manifest, stem)
                ldb.add_document(engine, collection=self.valves.COLLECTION,
                                 entry=ldb.document_entry(stem, in_dir, out_dir, hints), content_hash=h or file_hash,
                                 uploaded_by=who, output_root=self.valves.OUTPUT_ROOT)
                # its front matter, for "documents by <person>" (memento/library/front_matter.py)
                from memento.library import front_matter as doc_authors
                doc_authors.store_front(engine, collection=self.valves.COLLECTION, source=stem, out_dir=out_dir,
                                        content_hash=h or file_hash)
                # its summary and research gaps, written in the background while the model is idle
                warmup_queue.enqueue(engine, self.valves.COLLECTION, stem, ["summary", "gaps"])
                return
            from qdrant_client import QdrantClient
            qc = QdrantClient(url=self.valves.QDRANT_URL, timeout=30)

            def read_sources():
                # the collection's documents as they are now, read again under the collection's lock. No seed: a removed
                # upload must stay out; a live upload's vectors are written (wait=True) before it is registered, so it
                # is in the scroll
                sources, offset = set(), None
                while True:
                    points, offset = qc.scroll(collection_name=self.valves.COLLECTION, with_payload=["source"],
                                               with_vectors=False, limit=1000, offset=offset)
                    sources.update((pt.payload or {}).get("source") for pt in points
                                   if (pt.payload or {}).get("source"))
                    if offset is None:
                        return sources
            n = ldb.rebuild_documents(engine, collection=self.valves.COLLECTION, sources=read_sources, in_dir=in_dir,
                                      out_dir=out_dir, section_hints=hints, manifest=manifest,
                                      output_root=self.valves.OUTPUT_ROOT)
            _log.info("library list: built for %s, %d documents", self.valves.COLLECTION, n)
            warmup_queue.enqueue(engine, self.valves.COLLECTION, stem, ["summary", "gaps"])
        except Exception as e:
            _log.warning("library list not updated for %s: %r", stem, e)

    def resolve_cross_paper_citations(self) -> str | None:
        """Dual graph: run once per batch, not per document, since it fuzzy-matches titles against every document's
        Reference nodes in Neo4j. No-op unless Valves.GRAPH_BUILD is on. Returns a one-line summary, or None if skipped
        or failed (best effort, never blocks the batch)."""
        if not self.valves.GRAPH_BUILD:
            return None
        try:
            from memento.graph.store import get_store
            from memento.graph.build.build_graph import resolve_cross_paper_citations
            with get_store() as gstore:
                s = resolve_cross_paper_citations(gstore)
            return (f"Graph resolve: {s['n_resolved']}/{s['n_references']} references "
                    f"matched to documents in the library, {s['n_edges']} edges")
        except Exception as e:
            return f"⚠ Cross-document citation resolution failed (the documents are stored): {e}"

    def load_seen_ids(self) -> set:
        """This library's processed attachment ids (one list per library)."""
        return library_index.seen_ids(self.load_manifest(), self.valves.COLLECTION)

    def save_seen_ids(self, ids: set):
        """Add to this library's list, in one locked read-modify-write of the manifest."""
        with library_index.locked_manifest(MANIFEST_FILE) as m:
            library_index.add_seen_ids(m, self.valves.COLLECTION, ids)

    # ── parse-time dedup (mirrors the embed-time source_hash check below) ──
    # `parsed_hash` is written to the manifest right after a successful parse (see workflow.run_batch), apart from `source_hash`
    # (written after a successful embed).

    def needs_parse(self, path: Path, doc_meta: dict) -> bool:
        if self.valves.FORCE_REPARSE:
            return True
        paper_dir = self.find_paper_dir(path.stem)
        if paper_dir is None:
            return True
        cl = paper_dir / f"{path.stem}_content_list.json"
        if not cl.exists():
            return True
        # parsed_hash: written by this pipe after a parse; content_hash: by library_index for either path (the command
        # line writes no parsed_hash) — the same bytes for every format that needs no conversion
        meta = doc_meta.get(path.stem, {})
        stored_hash = meta.get("parsed_hash") or meta.get("content_hash")
        if not stored_hash:
            return True
        try:
            return hashlib.sha256(path.read_bytes()).hexdigest() != stored_hash
        except Exception:
            return True

    # ── embed one document: embedder + Qdrant upsert ──
    # Yields progress strings, then one final result dict.

    def embed_one(self, pdf_path: Path, stored_hash: str | None = None):
        from qdrant_client import QdrantClient
        from qdrant_client.models import PointStruct

        PROGRESS_EVERY = 100  # report every N chunks

        t0      = time.time()
        stem    = pdf_path.stem
        out     = self.find_paper_dir(stem)
        if out is None:
            yield {"ok": False, "error": "Not read yet, unable to store"}
            return
        notices: list[str] = []  # folded into the final result dict, shown in the file's collapsible line

        raw_cl = out / f"{stem}_content_list.json"
        if not raw_cl.exists():
            yield {"ok": False, "error": "Not read yet, unable to store"}
            return

        # ── embed decision first (file hash + vector count): enrichment belongs to embedding, so it must not run for
        # documents about to be skipped as already embedded ──
        try:
            client = QdrantClient(url=self.valves.QDRANT_URL, timeout=30)
        except Exception as e:
            yield {"ok": False, "error": f"Qdrant connection failed: {e}"}
            return

        try:
            file_hash = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
        except Exception:
            file_hash = None

        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue, FilterSelector
            existing = client.count(
                self.valves.COLLECTION,
                count_filter=Filter(must=[FieldCondition(key="source", match=MatchValue(value=stem))]),
                exact=True,
            )
        except Exception:
            existing = None  # collection not yet created — fine, proceed

        if (existing is not None and existing.count > 0
                and not self.valves.FORCE_REEMBED
                and file_hash and stored_hash == file_hash):
            self.register_document(stem, file_hash)
            yield {"ok": True, "vectors": 0, "type_counts": {}, "elapsed": 0.0, "skipped": True, "source_hash": file_hash}
            return

        # ── this document will be embedded: run enrichment now (figure-label and reference linking, caption recovery,
        # section assignment), so annotations always follow the current code ──
        if self.valves.NO_ENRICH:
            # Enrichment ablation (run_pipeline.py's --no-enrich): embed MinerU's raw content_list.json as is: no
            # context grounding for figures, tables and equations, no section assignment (the embed loop tolerates
            # missing fields), no junk-fragment filtering.
            cl_path = raw_cl
        else:
            try:
                from memento.ingestion.enrich_parsed_content_list import enrich_content_list
                # the parsed PDF: merged multi-panel figures are cropped from its pages
                cl_path = Path(enrich_content_list(
                    raw_cl, source_pdf=pdf_path if pdf_path.suffix.lower() == ".pdf" and pdf_path.is_file() else None))
            except Exception as e:
                logging.getLogger("memento.ingest").warning("%s: enrichment failed, previous version used: %r", stem, e)
                cl_path = out / f"{stem}_content_list_enriched.json"
                if not cl_path.exists():
                    cl_path = raw_cl

        from memento.ingestion.embed import sparse_encode
        # the instruction of the DOMAIN_NAME Valve's profile, as Library Chat loads it; memento.config.settings.DOMAIN
        # is the process's .env domain and may differ
        import importlib
        EMBED_INSTRUCTION = importlib.import_module("memento.config.domain_loader").load_domain(
            self.valves.DOMAIN_NAME)["instruction"]

        items      = json.loads(cl_path.read_text())
        images_dir = out / "images"

        pairs = []  # (embed_input, metadata, category)
        for item in items:
            if item.get("text_level"):
                # headings are not embedded as near-empty chunks: enrich_content_list() folded them into each item's
                # section/section_raw
                continue

            t = item.get("type", "")
            if t in SKIP_TYPES:
                continue
            if item.get("low_info"):
                continue
            if t in TEXT_TYPES:
                text = (item.get("text") or "").strip()
                if not text:
                    continue
                embed_text = text
                sparse_text = text
                if t == "equation":
                    # Bare LaTeX has no words a query would use: the author's surrounding sentences (enriched_context)
                    # are folded in, so an equation is found by what it represents. The displayed text stays the
                    # equation. BM25 (sparse_text, the pair's 7th element) uses the same text, as run_pipeline.py does.
                    ref_text = (item.get("enriched_context") or "").strip()
                    if ref_text:
                        embed_text = "\n".join(filter(None, [ref_text, text]))
                        sparse_text = embed_text
                        item["equation_context"] = ref_text      # stored: the answer model reads it
                pairs.append(({"instruction": EMBED_INSTRUCTION, "text": embed_text}, item, "text", text, None, None, sparse_text))
            elif t in TABLE_TYPES:
                # table items carry table_caption/table_body, not text/content (as run_pipeline.py's build_embed_input)
                caption  = " ".join(item.get("table_caption") or [])
                body     = item.get("table_body") or ""
                combined = f"{caption}\n{body}".strip()
                if not combined:
                    continue
                # the passages citing the table by its label go first in the dense vector and to the answer model
                # (memento/ingestion/embed.py); BM25 and the displayed/cited text stay caption + body
                from memento.ingestion.embed import table_context, table_embed_text
                context = table_context(item)
                if context:
                    item["table_context"] = context
                pairs.append(({"instruction": EMBED_INSTRUCTION, "text": table_embed_text(context, combined)}, item,
                              "table", combined, None, None, combined))
            elif t in IMAGE_TYPES:
                img_name    = item.get("img_path") or item.get("image_path") or ""
                img_path    = images_dir / Path(img_name).name if img_name else None
                caption     = " ".join(
                    item.get("image_caption") or [] if t == "image"
                    else item.get("chart_caption") or []
                ).strip()
                vlm_content = (item.get("content") or "").strip()
                enriched    = (item.get("enriched_context") or "").strip()
                # disp_text is stored as the chunk's `text`: shown in citations and given to the LLM as context.
                # MinerU's own figure reconstruction (`vlm_content`) is kept out of it and out of the embedded text
                # (stored apart as `mineru_text`, for display only): on complex charts it can invent values that would
                # look like real data once embedded. Only the author's own text (enriched_context, caption) is
                # searchable. Mirrors run_pipeline.py.
                disp_text   = "\n".join(filter(None, [enriched, caption]))
                full_text   = disp_text
                img_ref     = str(img_path) if (img_path and img_path.is_file()) else None
                # VLM captioning fallback: image exists but has no author text
                if not disp_text and img_path and img_path.is_file() and not self.valves.NO_ENRICH:
                    try:
                        # saved per document (vlm_captions.json), shared with run_pipeline.py: a re-upload reuses
                        # the caption instead of generating a different one
                        from memento.ingestion.caption_image import cached_caption
                        generated = cached_caption(
                            out, img_name,
                            base_url=self.valves.LLM_BASE_URL,
                            model=self.valves.LLM_MODEL,
                            api_key=llm_key(self.valves),
                        )
                        if generated:
                            disp_text = generated
                            full_text = generated
                            item["caption_source"] = "vlm"
                    except ImportError:
                        pass
                if img_path and img_path.is_file():
                    ei = {"instruction": EMBED_INSTRUCTION, "image": str(img_path)}
                    if full_text:
                        ei["text"] = full_text
                    pairs.append((ei, item, "image", disp_text, img_ref, vlm_content or None, disp_text))
                elif full_text:
                    pairs.append(({"instruction": EMBED_INSTRUCTION, "text": full_text}, item, "image", disp_text, img_ref, vlm_content or None, disp_text))

        if not pairs:
            # nothing embeddable — existing vectors (if any) were left intact
            yield {"ok": False, "error": "No readable content was found in this document"}
            return

        total = len(pairs)

        # hidden from Library Chat until embedding succeeds: it loses its old vectors and gains new ones in
        # batches, so a concurrent query must not see it half-written
        self.mark_in_progress(stem, True)

        # old vectors are deleted only now, with the new chunks ready, so a broken or empty content list cannot wipe a
        # document
        if existing is not None and existing.count > 0:
            client.delete(
                collection_name=self.valves.COLLECTION,
                points_selector=FilterSelector(filter=Filter(must=[
                    FieldCondition(key="source", match=MatchValue(value=stem))
                ])),
            )
            if self.valves.FORCE_REEMBED:
                notices.append(f"🗑 Storing from scratch ({existing.count:,} old entries removed)")
            elif stored_hash:
                notices.append(f"🔄 File changed, re-storing ({existing.count:,} old entries removed)")
            else:
                notices.append(f"🔄 Re-storing ({existing.count:,} old entries removed)")

        batch_size  = self.valves.EMBED_BATCH_SIZE
        extra       = self.extra_payload()
        type_counts: dict[str, int] = {}
        done        = 0

        try:
            for i in range(0, total, batch_size):
                batch        = pairs[i:i + batch_size]
                embed_inputs = [p[0] for p in batch]
                from memento.common.model_servers import vl_embed_many
                dense_vecs = vl_embed_many(
                    self.valves.EMBEDDER_URL,
                    os.getenv("EMBEDDER_MODEL", "Qwen3-VL-Embedding-8B"),
                    embed_inputs,
                )
                # BM25 sparse vectors from the displayed text for every type except equations (bare LaTeX), which use
                # the pair's 7th element instead. Mirrors run_pipeline.py, so both ingestion paths fill the `sparse`
                # slot the same way.
                sparse_vecs = sparse_encode([b[6] or "" for b in batch])

                upsert_pts = []
                for (_, meta, cat, disp_text, img_ref, mineru_text, _sparse_text), vec, sparse_vec in zip(batch, dense_vecs, sparse_vecs):
                    kind = meta.get("type") or cat
                    type_counts[kind] = type_counts.get(kind, 0) + 1
                    payload = {
                        "source":   stem,
                        "type":     meta.get("type"),
                        "text":     disp_text,
                        "page":     meta.get("page_idx", 0),
                        "img_path": img_ref,
                        "section":  meta.get("section"),
                        "section_raw": meta.get("section_raw"),
                        # bbox enables document-order sorting (page + y) for
                        # the query pipe's verbatim section mode
                        "bbox":     meta.get("bbox"),
                    }
                    if meta.get("page_end") is not None:
                        payload["page_end"] = meta["page_end"]
                    # the authors' text the answer model reads under an equation / a table (as run_pipeline.py)
                    for key in ("equation_context", "table_context"):
                        if meta.get(key):
                            payload[key] = meta[key]
                    if mineru_text:
                        payload["mineru_text"] = mineru_text
                    if meta.get("caption_source"):
                        payload["caption_source"] = meta["caption_source"]
                    payload.update(extra)
                    point_vector = {"dense": vec}
                    # Qdrant rejects a sparse vector without nonzero entries; stopword-only fragments can encode to one,
                    # so they go without (dense-only at query time).
                    if sparse_vec.indices:
                        point_vector["sparse"] = sparse_vec
                    upsert_pts.append(PointStruct(
                        id      = int(hashlib.sha256(f"{stem}_{meta.get('page_idx',0)}_{meta.get('type','')}_{done}".encode()).hexdigest(), 16) % (2**63),
                        vector  = point_vector,
                        payload = payload,
                    ))
                    done += 1

                # upsert at once, overlapping with the next GPU batch
                client.upsert(collection_name=self.valves.COLLECTION, points=upsert_pts)

                # progress every PROGRESS_EVERY chunks
                prev = done - len(batch)
                if prev // PROGRESS_EVERY < done // PROGRESS_EVERY:
                    pct = done * 100 // total
                    yield f"💾 Storing... {pct}% ({done}/{total} passages)\n"

        except Exception as e:
            yield {"ok": False, "error": f"embedding failed: {e}"}
            return

        # shown and registered only after the graph and the gap tags below: the gap updater, the Limitations warm-up and
        # a "Draft a proposal" click wait while a document is hidden or unlisted, so none of them reads it mid-tagging

        # ── materialize the knowledge graph (Valves.GRAPH_BUILD) ──
        # Best effort and off the critical path: the Chunk ids above are the same deterministic hashes build_graph.py
        # computes from `items`, so graph nodes match the Qdrant points. A failure (Neo4j unreachable, extraction error)
        # never fails ingestion; the graph can be rebuilt with `build_graph.py --all`.
        if self.valves.GRAPH_BUILD:
            try:
                from memento.graph.store import get_store
                from memento.graph.build.build_graph import build_paper

                with get_store() as gstore:
                    gstore.ensure_schema()
                    build_paper(stem, store=gstore, items=items, paper_dir=out, use_llm=True)
            except Exception as e:
                logging.getLogger("memento.ingest").warning("%s: knowledge-graph update failed (stored OK): %r", stem, e)

        # ── Gap tagging (best-effort, after upsert) ────────────────────────────
        if self.valves.RESEARCH_GAP_TAG and done > 0:
            try:
                from memento.research_gaps.tagging import GAP_TAG_TYPES, tag_paper_chunks

                text_items = [(d, item) for d, (ei, item, cat, *_rest) in enumerate(pairs)
                              if item.get("type") in GAP_TAG_TYPES and (item.get("text") or "").strip()]
                chunks = [{"text": item["text"], "section": item.get("section", "")}
                          for _d, item in text_items]
                if chunks:
                    tagged = tag_paper_chunks(
                        chunks,
                        base_url=self.valves.LLM_BASE_URL,
                        model=self.valves.LLM_MODEL,
                        api_key=llm_key(self.valves),
                    )
                    for ci, (d, item) in enumerate(text_items):
                        pid = int(hashlib.sha256(
                            f"{stem}_{item.get('page_idx', 0)}_{item.get('type', '')}_{d}".encode()
                        ).hexdigest(), 16) % (2 ** 63)
                        if ci in tagged:
                            client.set_payload(self.valves.COLLECTION, {
                                "gap_relevant": True,
                                "gap_signal": tagged[ci]["gap_signal"],
                                "gap_type": tagged[ci]["gap_type"],
                            }, [pid])
                        else:
                            client.set_payload(self.valves.COLLECTION, {"gap_relevant": False}, [pid])
                    logging.getLogger("memento.ingest").info("%s: gap tagged %d/%d passages", stem, len(tagged), len(chunks))
            except Exception as e:
                logging.getLogger("memento.ingest").warning("%s: gap tagging failed (stored OK): %r", stem, e)

        # cleared only on success: on failure the document stays hidden until the next run's truncate, as it may have
        # partial or no vectors
        self.mark_in_progress(stem, False)
        self.register_document(stem, file_hash)

        yield {"ok": True, "vectors": done, "type_counts": type_counts, "elapsed": time.time() - t0,
               "source_hash": file_hash, "notices": notices}

    # ── parse one document with MinerU ──
    # Yields progress lines every PROGRESS_INTERVAL s, then one final result dict.

    def parse_one(self, pdf_path: Path):
        PROGRESS_INTERVAL = 10  # seconds between progress ticks

        t0 = time.time()

        # the page count up front: MinerU's status endpoint reports no progress percentage, so this is the only size
        # signal during the wait
        page_count = None
        if pdf_path.suffix.lower() == ".pdf":
            try:
                page_count = len(PdfReader(str(pdf_path)).pages)
            except Exception:
                pass

        try:
            mime = intake.MIME_TYPES.get(pdf_path.suffix.lower(), "application/octet-stream")
            with open(pdf_path, "rb") as f:
                resp = requests.post(
                    f"{self.valves.MINERU_URL}/tasks",
                    data={"backend":             self.valves.BACKEND,
                          "return_content_list": "true",
                          "return_md":           "true",
                          "return_images":       "true",
                          # On, as for run_pipeline.py's mineru CLI and every evaluated corpus: besides describing
                          # figures (mineru_text, never searched or shown to the model), it turns pictured tables into
                          # tables and pictured formulas into equations (mineru_vl_utils post_process).
                          "image_analysis":      "true"},
                    files={"files": (pdf_path.name, f, mime)},
                    timeout=30,
                )
            resp.raise_for_status()
            task_id = resp.json()["task_id"]
        except Exception as e:
            yield {"ok": False, "error": f"submit failed: {e}"}
            return

        deadline       = time.time() + TIMEOUT
        status         = "unknown"
        last_progress  = time.time()
        poll_fail_n    = 0
        POLL_FAIL_MAX  = 5  # consecutive transient poll errors tolerated before giving up

        while time.time() < deadline:
            try:
                r    = requests.get(f"{self.valves.MINERU_URL}/tasks/{task_id}", timeout=10)
                data = r.json() if r.ok else {}
            except Exception as e:
                # transient network blips are common with many concurrent pollers over a long run; give up only after
                # several in a row
                poll_fail_n += 1
                if poll_fail_n > POLL_FAIL_MAX:
                    yield {"ok": False, "error": f"status poll failed {poll_fail_n}x: {type(e).__name__}: {e}"}
                    return
                time.sleep(POLL_INTERVAL)
                continue

            poll_fail_n = 0
            status = data.get("status", "error")
            if status in TERMINAL:
                break

            now = time.time()
            if now - last_progress >= PROGRESS_INTERVAL:
                elapsed = now - t0
                pct     = data.get("progress")
                size    = f" · {page_count} pages" if page_count is not None else ""
                if pct is not None:
                    yield f"⏳ reading... {elapsed:.0f}s · {pct}%{size}\n"
                else:
                    yield f"⏳ reading... {elapsed:.0f}s elapsed{size}\n"
                last_progress = now

            time.sleep(POLL_INTERVAL)
        else:
            yield {"ok": False, "error": f"MinerU status: timeout (last status: {status})"}
            return

        if status in ("failed", "error", "timeout"):
            yield {"ok": False, "error": f"MinerU status: {status}"}
            return

        try:
            full        = requests.get(
                f"{self.valves.MINERU_URL}/tasks/{task_id}/result", timeout=60
            ).json()
            file_result = list(full.get("results", {}).values())[0]
        except Exception as e:
            yield {"ok": False, "error": f"fetch failed: {e}"}
            return

        stem       = pdf_path.stem
        out_dir    = self.output_dir() / stem / self.out_subdir()
        images_dir = out_dir / "images"
        out_dir.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(exist_ok=True)

        cl_raw  = file_result.get("content_list", "[]")
        items   = json.loads(cl_raw) if isinstance(cl_raw, str) else cl_raw
        cl_path = out_dir / f"{stem}_content_list.json"
        cl_path.write_text(json.dumps(items, ensure_ascii=False, indent=2))
        (out_dir / f"{stem}.md").write_text(file_result.get("md_content", ""))
        for fname, data_uri in file_result.get("images", {}).items():
            img_path = images_dir / fname
            raw = data_uri.split(",", 1)[1] if "," in data_uri else data_uri
            img_path.write_bytes(base64.b64decode(raw))

        # enrichment is not run here: embed_one runs it at embed time

        pages  = max((it.get("page_idx", 0) for it in items), default=0) + 1
        counts: dict[str, int] = {}
        for it in items:
            t = it.get("type", "unknown")
            counts[t] = counts.get(t, 0) + 1

        yield {
            "ok":      True,
            "elapsed": time.time() - t0,
            "pages":   pages,
            "chunks":  len(items),
            "counts":  counts,
        }
