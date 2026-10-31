"""`memento_documents` (Open WebUI's Postgres): one row per document of a collection, with its display title, the raw
titles the Query pipe matches names against, and its content fingerprint. Every upload, command-line upload, deletion
and library reset goes through add_document / remove_document / reset_library, which also rewrite the picker's
library.json and drop a removed document's stored answers, so listing, picker and title lookup are current at once.
SQLAlchemy Core: Postgres live, SQLite in tests.

The documents' titles (the first-page title or the PDF title, cleaned; shared by the Query pipe's listing and the
picker) are derived here too, and the list file OUTPUT_ROOT/_runtime/library.json = {"updated", "documents":
[{"source", "title"}]} is written atomically; the file server hands it to the picker.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, MetaData, String, Table, Text, delete, insert, select

from memento.config.domain_loader import default_library_collection
from memento.library import answers as answer_cache, locks as db_locks, warmup_queue

import glob
import json
import re
import unicodedata
from pathlib import Path


METADATA = MetaData()
DOCUMENTS = Table(
    "memento_documents", METADATA,
    Column("collection", String(200), primary_key=True),
    Column("source", String(500), primary_key=True),
    Column("title", Text, nullable=False),
    Column("parsed_title", Text),
    Column("pdf_title", Text),
    Column("content_hash", String(128)),
    Column("uploaded_by", String(200)),
    Column("added_at", DateTime(timezone=True)),
)


def ensure_tables(engine) -> None:
    METADATA.create_all(engine, tables=[DOCUMENTS])
    answer_cache.ensure_table(engine)
    warmup_queue.ensure_table(engine)


def document_entry(source: str, in_dir, out_dir, section_hints: dict | None) -> dict:
    """{source, title, parsed_title, pdf_title}: the display title by the rules below (plain text; the file
    name when there is no better one), plus the raw titles for matching names in questions."""
    parsed, pdf = raw_titles(source, in_dir, out_dir)
    title = best_title(parsed, pdf, section_hints)
    title = plain_title(title) if title else None
    shown = source if not title or same_name(title, source) else title
    return {"source": source, "title": shown, "parsed_title": parsed, "pdf_title": pdf}


def _rows(conn, collection: str) -> list[dict]:
    rows = conn.execute(select(DOCUMENTS).where(DOCUMENTS.c.collection == collection)).mappings().all()
    return sorted((dict(r) for r in rows), key=lambda r: (r["title"].casefold(), r["source"]))


def list_documents(engine, collection: str) -> list[dict]:
    with engine.connect() as c:
        return _rows(c, collection)


def get_document(engine, collection: str, source: str) -> dict | None:
    with engine.connect() as c:
        row = c.execute(select(DOCUMENTS).where(DOCUMENTS.c.collection == collection,
                                                DOCUMENTS.c.source == source)).mappings().first()
    return dict(row) if row else None


def export_library(engine, collection: str, output_root) -> None:
    """The picker's list of this library (OUTPUT_ROOT/_runtime/library/<collection>.json; the default library also
    writes the single-library library.json) from the table."""
    if not output_root:
        return
    write_library(output_root, [{"source": r["source"], "title": r["title"]}
                                             for r in list_documents(engine, collection)], collection)


def add_document(engine, *, collection: str, entry: dict, content_hash: str | None, uploaded_by: str | None,
                 output_root) -> None:
    """Insert or replace one document's row (a changed upload replaces its fingerprint), then the picker's list."""
    with engine.begin() as c:
        db_locks.document(c, collection, entry["source"])             # coordinated with removal, reset and stores
        old = c.execute(select(DOCUMENTS.c.added_at).where(DOCUMENTS.c.collection == collection,
                                                           DOCUMENTS.c.source == entry["source"])).scalar()
        c.execute(delete(DOCUMENTS).where(DOCUMENTS.c.collection == collection,
                                          DOCUMENTS.c.source == entry["source"]))
        c.execute(insert(DOCUMENTS).values(
            collection=collection, source=entry["source"], title=entry["title"],
            parsed_title=entry.get("parsed_title"), pdf_title=entry.get("pdf_title"), content_hash=content_hash,
            uploaded_by=uploaded_by, added_at=old or datetime.now(timezone.utc)))
    export_library(engine, collection, output_root)


def remove_document(engine, *, collection: str, source: str, output_root) -> list[int]:
    """Delete one document's row, its stored answers and its open questions in the gap ledger (questions no other
    document raises go; their ids are returned, for their vectors in <collection>-gaps), then the picker's list."""
    # the row, its stored answers and its queued jobs in one transaction under the document's lock, so no stored answer
    # is written in between; the gap ledger keeps its own (its updater also removes a vanished document's questions)
    with engine.begin() as c:
        db_locks.document(c, collection, source)
        c.execute(delete(DOCUMENTS).where(DOCUMENTS.c.collection == collection, DOCUMENTS.c.source == source))
        answer_cache.delete_for_source(engine, collection, source, conn=c)
        warmup_queue.delete_for_source(engine, collection, source, conn=c)
    from memento.research_gaps import ledger as gap_ledger
    gap_ledger.ensure_tables(engine)
    orphans = gap_ledger.delete_for_source(engine, collection, source)
    export_library(engine, collection, output_root)
    return orphans


SERVER_UPLOADER = "cli"        # memento.ingestion.library_index.SERVER_UPLOADER: documents with no recorded uploader


def manifest_meta(manifest: dict, source: str) -> tuple[str | None, str]:
    """(content fingerprint, uploader) of one document from data/manifest.json: library_index's content_hash, else
    the embed's source_hash; the server identity when no uploader is recorded."""
    meta = (manifest.get("documents") or {}).get(source) or {}
    return meta.get("content_hash") or meta.get("source_hash"), meta.get("uploaded_by") or SERVER_UPLOADER


def has_documents(engine, collection: str) -> bool:
    """Has this collection's list been built? An empty list is rebuilt from the whole collection (rebuild_documents),
    so a single upload must not start one: it would hide every other document."""
    with engine.connect() as c:
        return c.execute(select(DOCUMENTS.c.source).where(DOCUMENTS.c.collection == collection).limit(1)).first() \
            is not None


def versions(engine, collection: str) -> dict:
    """{source: fingerprint} of a collection's rows: the baseline a refresh compares against (R4)."""
    with engine.connect() as c:
        return {r.source: r.content_hash for r in c.execute(select(DOCUMENTS.c.source, DOCUMENTS.c.content_hash)
                                                            .where(DOCUMENTS.c.collection == collection))}


def rebuild_documents(engine, *, collection: str, sources, in_dir, out_dir, section_hints, manifest: dict,
                      output_root, refresh: bool = False, baseline: dict | None = None) -> int:
    """Rows for the collection's documents (titles from the parsed output, fingerprints from the manifest), written in
    one transaction, then the picker's list. `sources`: the documents, or a function that returns them (the vector
    store's sources); a function is called again under the collection's lock, so a document removed after the first read
    is left out (removal deletes the vectors before the row). Default (the list is empty: a new library, the first
    upload): rows already present are kept (an upload that finished meanwhile). refresh=True (the migration): the rows
    the rebuild found are replaced by fresh ones, fingerprints included; a row added or replaced meanwhile (a newer
    upload) stays. `baseline` (versions()): the rows as the caller found them before it read its metadata; without it,
    as found here."""
    read = sources if callable(sources) else (lambda: sources)
    now = datetime.now(timezone.utc)
    start = baseline if baseline is not None else versions(engine, collection)   # R4: a refresh replaces only these

    def build(src):
        e = document_entry(src, in_dir, out_dir, section_hints)
        h, who = manifest_meta(manifest, src)
        return dict(collection=collection, source=src, title=e["title"], parsed_title=e["parsed_title"],
                    pdf_title=e["pdf_title"], content_hash=h, uploaded_by=who, added_at=now)

    built = {src: build(src) for src in sorted(set(read()))}      # titles read before the lock (files)
    with engine.begin() as c:
        db_locks.collection(c, collection)                          # no upload, removal or store in between
        current = set(read()) if callable(sources) else set(built)
        for src in sorted(current - set(built)):                    # appeared meanwhile: only these read here
            built[src] = build(src)
        rows = [built[s] for s in sorted(current)]
        have = {r.source: r.content_hash for r in c.execute(select(DOCUMENTS.c.source, DOCUMENTS.c.content_hash)
                                                            .where(DOCUMENTS.c.collection == collection))}
        if refresh:
            # rows unchanged since the start are replaced (or dropped when their document is gone); a row added or
            # replaced meanwhile is a newer upload than the metadata read before the lock, and stays
            newer = {s for s, h in have.items() if s not in start or start[s] != h}
            stale = [s for s in have if s not in newer]
            if stale:
                c.execute(delete(DOCUMENTS).where(DOCUMENTS.c.collection == collection, DOCUMENTS.c.source.in_(stale)))
            rows = [r for r in rows if r["source"] not in newer]
        else:
            rows = [r for r in rows if r["source"] not in have]
        if rows:
            c.execute(insert(DOCUMENTS), rows)
    # each document's front matter, for "documents by <person>": the first upload into an empty library comes here
    # instead of add_document's path, which stores it per upload. Best effort, as there.
    from memento.library import front_matter as doc_authors
    for r in rows:
        doc_authors.store_front(engine, collection=collection, source=r["source"], out_dir=out_dir,
                                content_hash=r["content_hash"])
    export_library(engine, collection, output_root)
    return len(rows)


def reset_library(engine, *, collection: str, output_root) -> list[dict]:
    """Empty one collection's list (and its stored answers); returns the removed rows (the setup saves them)."""
    with engine.begin() as c:
        db_locks.collection(c, collection)                            # no upload, removal or store in between
        removed = _rows(c, collection)
        c.execute(delete(DOCUMENTS).where(DOCUMENTS.c.collection == collection))
        c.execute(delete(answer_cache.ANSWERS).where(answer_cache.ANSWERS.c.collection == collection))
        c.execute(delete(warmup_queue.JOBS).where(warmup_queue.JOBS.c.collection == collection))
    from memento.research_gaps import ledger as gap_ledger                  # its open questions too (gap ledger)
    gap_ledger.ensure_tables(engine)
    gap_ledger.clear_library(engine, collection)
    export_library(engine, collection, output_root)
    return removed


# ── Titles and the library list file ──────────────────────────────────────────
# a title cut off at its line end
_DANGLING = {"a", "an", "and", "at", "by", "for", "from", "in", "of", "on", "or", "the", "to", "with",
             "bei", "das", "der", "die", "für", "im", "mit", "und", "von", "zum", "zur"}
# a repository's or publisher's cover page, not the title ("This document is a postprint …"); at the start of the line,
# where such lines begin
_COVER_PAGE = re.compile(r"^(dieses dokument|this (is the|document is)|this article (is|has been)|downloaded from|"
                         r"author'?s? manuscript|accepted manuscript(?!s)|postprint|preprint|terms of use|"
                         r"nutzungsbedingungen)|zweitveröffentlichung", re.I)
_JUNK_PDF_TITLE = re.compile(r"^(microsoft\s|untitled|document\d*$)|\.(pdf|docx?|pptx?|tex)$|^\S+\s+\d+\.\.\d+", re.I)


def raw_titles(source: str, in_dir, out_dir) -> tuple[str | None, str | None]:
    """(first-page title, PDF title) of one document, each None when missing. The first-page title is the first
    page-1 heading of the parsed output (cleaned by memento.ingestion.extract_metadata, as the graph build does); the PDF
    title only when it is a real one (publisher PDFs; not "cm2c02991 1..8" or "Microsoft Word - …")."""
    parsed = pdf_title = None
    if out_dir is None:
        return None, None
    try:
        # the name as text: a file name, not a glob pattern
        lists = sorted((Path(out_dir) / source).glob(f"*/{glob.escape(source)}_content_list.json"),
                       key=lambda q: q.parent.name != "vlm")
        if lists:
            from memento.ingestion.extract_metadata import extract_paper_metadata
            parsed = extract_paper_metadata(json.loads(lists[0].read_text())).get("title") or None
    except Exception:
        pass
    try:
        pdf = Path(in_dir) / f"{source}.pdf"
        if pdf.exists():
            import fitz
            from memento.common.pdf import PYMUPDF
            with PYMUPDF, fitz.open(pdf) as d:
                t = ((d.metadata or {}).get("title") or "").strip()
            if len(t.split()) >= 4 and not _JUNK_PDF_TITLE.search(t):
                pdf_title = t
    except Exception:
        pass
    return parsed, pdf_title


def plausible_title(title: str | None, section_hints: dict | None) -> str | None:
    """A title candidate fit to show as the document's title, else None. The first-page heading is sometimes a chapter
    heading, a funding note or cut off."""
    t = re.sub(r"\*+", "", title or "").strip()
    words = t.split()
    if not 4 <= len(words) <= 40 or re.match(r"\d", t) or t.endswith((":", ",", ";", "-", "–")):
        return None
    if words[-1].lower() in _DANGLING or _COVER_PAGE.search(t):
        return None
    low = t.lower()
    if len(words) <= 6 and any(w in low for trig in (section_hints or {}).values() for w in trig):
        return None                                    # a short section heading ("Ergebnisse und Diskussion …")
    return t


def best_title(parsed: str | None, pdf: str | None, section_hints: dict | None) -> str | None:
    return plausible_title(parsed, section_hints) or plausible_title(pdf, section_hints)


def same_name(title: str, source: str) -> bool:
    """The file name IS the title (maybe cut short or re-punctuated), not a short name."""
    # NFKC: formula notation counts as the same name (subscript digits as plain digits)
    norm = lambda x: re.sub(r"\W+", "", unicodedata.normalize("NFKC", x or "").lower())
    nt, ns = norm(title), norm(source)
    return bool(nt) and (nt == ns or (min(len(nt), len(ns)) / max(len(nt), len(ns)) > 0.8
                                      and (nt.startswith(ns) or ns.startswith(nt))))


def plain_title(title: str) -> str:
    """A title as plain text for the picker: "$\\mathrm{X_2}$" → "X2" (MinerU writes formulas in titles as LaTeX)."""
    t = re.sub(r"\\(?:mathrm|text|textrm|mathit|mathbf|rm|bf)\s*\{([^{}]*)\}", r"\1", title)
    t = re.sub(r"[$_^{}]", "", t)
    return re.sub(r"\s+", " ", t).strip()


def library_entries(sources, in_dir, out_dir, section_hints: dict | None) -> list[dict]:
    """[{"source", "title"}] sorted A–Z by title; title = the file name when there is no better one."""
    out = []
    for src in sources:
        title = best_title(*raw_titles(src, in_dir, out_dir), section_hints)
        title = plain_title(title) if title else None
        out.append({"source": src, "title": src if not title or same_name(title, src) else title})
    return sorted(out, key=lambda e: (e["title"].casefold(), e["source"]))


# Two copies in one Open WebUI: each library's list is _runtime/library/<collection>.json. The default library's list is
# also written to _runtime/library.json, which the browser script and the file server's /library serve; another
# library's list never goes there.
LEGACY_COLLECTION = default_library_collection()
_COLLECTION_RE = re.compile(r"[a-z0-9-]{1,80}")


def valid_collection(collection) -> bool:
    return isinstance(collection, str) and bool(_COLLECTION_RE.fullmatch(collection))


def library_path(output_root, collection: str | None = None) -> Path:
    """The old file (collection None) or a library's own; a collection name that is not a plain name: ValueError."""
    if collection is None:
        return Path(output_root) / "_runtime" / "library.json"
    if not valid_collection(collection):
        raise ValueError(f"not a collection name: {collection!r}")
    return Path(output_root) / "_runtime" / "library" / f"{collection}.json"


def _read(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) and isinstance(data.get("documents"), list) else None
    except (OSError, ValueError):
        return None


def read_library(output_root, collection: str | None = None) -> dict | None:
    """A library's list (the single-library file for None); the default library reads that file until its own exists."""
    data = _read(library_path(output_root, collection))
    if data is None and collection == LEGACY_COLLECTION:
        data = _read(library_path(output_root))
    return data


def write_library(output_root, entries: list[dict], collection: str | None = None) -> None:
    """Written atomically (temp file, then rename) so the picker never reads half a file; today's library also to
    the old file."""
    from memento.common.atomic_file import write_text_atomic
    docs = sorted(entries, key=lambda e: (e["title"].casefold(), e["source"]))
    text = json.dumps({"updated": datetime.now(timezone.utc).isoformat(timespec="seconds"), "documents": docs},
                      ensure_ascii=False)
    write_text_atomic(library_path(output_root, collection), text)
    if collection == LEGACY_COLLECTION:
        write_text_atomic(library_path(output_root), text)


def add_to_library(output_root, entries: list[dict], collection: str | None = None) -> bool:
    """Add (or retitle) documents in an existing list; False when there is no list yet (the Query pipe writes the
    whole list at its next question)."""
    data = read_library(output_root, collection)
    if data is None:
        return False
    by_src = {e["source"]: e for e in data["documents"] if isinstance(e, dict) and e.get("source")}
    by_src.update({e["source"]: e for e in entries})
    write_library(output_root, list(by_src.values()), collection)
    return True
