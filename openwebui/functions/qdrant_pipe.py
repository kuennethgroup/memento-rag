"""
title: Qdrant Manager
author: rangan
version: 1.6.2
type: pipe
requirements: qdrant-client
"""

import os
from pydantic import BaseModel
from typing import Generator

# The library's document list (memento/library/documents.py, memento_documents): a deleted document or collection leaves the
# list (and its stored answers) at once, through the same write path as uploads.
def _library_db():
    import sys
    root = os.getenv("MEMENTO_DIR", "")
    if root not in sys.path:
        sys.path.insert(0, root)
    from open_webui.internal.db import engine
    from memento.library import documents as library_db
    library_db.ensure_tables(engine)
    return library_db, engine


def _remove_paper_from_table(collection: str, source: str):
    """Best-effort: a DB hiccup here must never fail a delete command."""
    try:
        ldb, engine = _library_db()
        ldb.remove_document(engine, collection=collection, source=source, output_root=os.getenv("OUTPUT_ROOT"))
    except Exception:
        pass


def _clear_papers_table(collections: list):
    try:
        ldb, engine = _library_db()
        for c in collections:
            ldb.reset_library(engine, collection=c, output_root=os.getenv("OUTPUT_ROOT"))
    except Exception:
        pass


HELP_TEXT = """\
**Qdrant Manager**

> ⚠ **Prerequisite**: Qdrant must be running before using any command.
> Start it with: `podman-compose --env-file config/.env up -d qdrant`

| Command | Description |
|---|---|
| `list` | List all collections with full stats |
| `info <name>` | Detailed breakdown for a collection |
| `count <name>` | Point count for a collection |
| `papers <name> [page]` | List source documents in a collection, paginated (25/page) |
| `chunks <name> <source> [page]` | Show chunks from a specific paper, paginated |
| `chunk <id>` | Show full detail for a single chunk by ID |
| `create <name>` | Create collection (4096-d dense + sparse, cosine) |
| `delete paper <name> <source>` | Delete all chunks from a specific paper (asks for confirm) |
| `delete paper <name> <source> confirm` | Confirm deletion of all chunks from a paper |
| `delete chunk <id>` | Delete a single chunk by ID |
| `delete <name> [confirm]` | Delete a collection and its document list (asks for confirm) |
| `delete all confirm` | Delete ALL collections |
| `help` | Show this message |
"""

CMD_REF = "\n\n---\n`list` · `info <name>` · `count <name>` · `papers <name> [page]` · `chunks <name> <source> [page]` · `chunk <id>` · `create <name>` · `delete paper <name> <source> confirm` · `delete chunk <id>` · `delete <name>` · `delete all confirm` · `help`"

CHUNKS_PAGE_SIZE = 25

STATUS_EMOJI = {"green": "🟢", "yellow": "🟡", "red": "🔴"}

# Must match INGEST_ADVISORY_LOCK_KEY in ingestion_pipe.py: the Postgres advisory lock an ingestion run holds.
# pg_try_advisory_lock returns at once (true if it was free, then released again), so this is a non-blocking "is
# ingestion running?" check.
INGEST_ADVISORY_LOCK_KEY = 7281990014127733


def _ingestion_in_progress() -> bool:
    try:
        from open_webui.internal.db import engine
        from sqlalchemy import text
        with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            free = conn.execute(
                text("SELECT pg_try_advisory_lock(:key)"),
                {"key": INGEST_ADVISORY_LOCK_KEY},
            ).scalar()
            if free:
                conn.execute(
                    text("SELECT pg_advisory_unlock(:key)"),
                    {"key": INGEST_ADVISORY_LOCK_KEY},
                )
                return False
            return True
    except Exception:
        # best-effort — never block Qdrant Manager on this check
        return False


def _status(info) -> str:
    # Safely extracts the value whether it's an Enum or a plain string
    s = str(getattr(info.status, "value", info.status)).lower()
    return f"{STATUS_EMOJI.get(s, '⚪')} {s}"


def _vec_info(cfg):
    """Returns (dense_rows, sparse_names) from CollectionConfig."""
    if not cfg:
        return [], []
    params     = getattr(cfg, "params", None)
    vec_cfg    = getattr(params, "vectors", {}) if params else {}
    sparse_cfg = getattr(params, "sparse_vectors", {}) if params else {}

    dense_rows = []
    if isinstance(vec_cfg, dict):
        for vname, vcfg in vec_cfg.items():
            dim      = getattr(vcfg, "size", "?")
            dist_raw = getattr(vcfg, "distance", "?")
            dist     = str(getattr(dist_raw, "value", dist_raw)).split(".")[-1].capitalize()
            dense_rows.append((vname, dim, dist))
    elif vec_cfg:
        dim      = getattr(vec_cfg, "size", "?")
        dist_raw = getattr(vec_cfg, "distance", "?")
        dist     = str(getattr(dist_raw, "value", dist_raw)).split(".")[-1].capitalize()
        dense_rows.append(("(default)", dim, dist))

    sparse_names = list(sparse_cfg.keys()) if isinstance(sparse_cfg, dict) else []
    return dense_rows, sparse_names


class Pipe:
    class Valves(BaseModel):
        QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")

    def __init__(self):
        self.type   = "pipe"
        self.name   = "Qdrant Manager"
        self.valves = self.Valves()

    def pipe(self, body: dict, __user__: dict) -> Generator:
        if __user__.get("role") != "admin":
            yield "Qdrant Manager is restricted to admins only."
            return

        raw = body.get("messages", [{}])[-1].get("content", "")
        if isinstance(raw, str) and raw.startswith("### Task:"):
            return
        raw_strip = raw.strip() if isinstance(raw, str) else ""
        cmd = raw_strip.lower()

        if not cmd or cmd == "help":
            yield HELP_TEXT
            return

        if _ingestion_in_progress():
            yield "⚠️ An upload is in progress — results below may be incomplete.\n\n"

        from qdrant_client import QdrantClient
        from qdrant_client.models import (Distance, FieldCondition, Filter,
                                          MatchValue, Modifier, SparseIndexParams,
                                          SparseVectorParams, VectorParams)

        qc = QdrantClient(url=self.valves.QDRANT_URL, timeout=30)
        try:
            all_cols = qc.get_collections()
        except Exception as e:
            yield (
                f"❌ **Qdrant is not reachable** at `{self.valves.QDRANT_URL}`\n\n"
                f"Start it with: `podman-compose --env-file config/.env up -d qdrant`\n\n"
                f"*{type(e).__name__}: {e}*"
            ) + CMD_REF
            return

        # ── list ──────────────────────────────────────────────────────────────
        if cmd == "list":
            cols = all_cols.collections
            if not cols:
                yield "No collections found." + CMD_REF
                return
            yield f"**{len(cols)} collection(s):**\n\n"
            yield "| Collection | Points | Dim | Distance | Sparse | Status |\n|---|---|---|---|---|---|\n"
            for c in cols:
                try:
                    info             = qc.get_collection(c.name)
                    count            = info.points_count or 0
                    dense_rows, sprs = _vec_info(info.config)
                    dim  = dense_rows[0][1] if dense_rows else "?"
                    dist = dense_rows[0][2] if dense_rows else "?"
                    spr  = "✓" if sprs else "✗"
                    st   = _status(info)
                except Exception:
                    count, dim, dist, spr, st = "?", "?", "?", "?", "?"
                yield f"| `{c.name}` | {count} | {dim} | {dist} | {spr} | {st} |\n"
            yield CMD_REF
            return

        # ── info <name> ───────────────────────────────────────────────────────
        if cmd.startswith("info "):
            name = cmd[5:].strip()
            try:
                info = qc.get_collection(name)
            except Exception as e:
                yield f"❌ Collection `{name}` not found: {e}" + CMD_REF
                return

            count      = info.points_count or 0
            indexed    = getattr(info, "indexed_vectors_count", None) or 0
            segments   = getattr(info, "segments_count", None)
            index_note = f"{indexed:,} HNSW indexed" if indexed else "brute-force (< 20k threshold)"
            dense_rows, sparse_names = _vec_info(info.config)

            yield f"**`{name}`**\n\n"

            yield "**Overview**\n"
            yield f"- Points: **{count:,}** · {index_note}\n"
            if segments is not None:
                yield f"- Segments: {segments}\n"
            yield f"- Status: {_status(info)}\n\n"

            if dense_rows:
                yield "**Dense vectors**\n"
                yield "| Name | Dim | Distance |\n|---|---|---|\n"
                for vname, dim, dist in dense_rows:
                    yield f"| `{vname}` | {dim} | {dist} |\n"
                yield "\n"

            if sparse_names:
                yield f"**Sparse vectors:** {', '.join(f'`{n}`' for n in sparse_names)}\n\n"

            # payload type breakdown
            yield "**Payload breakdown**\n"
            yield "| Type | Count |\n|---|---|\n"
            for t in ["text", "table", "image"]:
                try:
                    r = qc.count(
                        name,
                        count_filter=Filter(must=[
                            FieldCondition(key="type", match=MatchValue(value=t))
                        ]), # type: ignore
                        exact=True,
                    )
                    yield f"| {t} | {r.count:,} |\n"
                except Exception:
                    yield f"| {t} | ? |\n"

            yield CMD_REF
            return

        # ── count <name> ──────────────────────────────────────────────────────
        if cmd.startswith("count "):
            name = cmd[6:].strip()
            try:
                info  = qc.get_collection(name)
                count = info.points_count or 0
                yield f"`{name}`: **{count:,}** points" + CMD_REF
            except Exception as e:
                yield f"❌ Collection `{name}` not found: {e}" + CMD_REF
            return

        # ── delete paper <name> <source> [confirm] ───────────────────────────
        if cmd.startswith("delete paper "):
            parts = raw_strip.split(None, 3)  # ["delete", "paper", "<col>", "<source> [confirm]"]
            if len(parts) < 4:
                yield "Usage: `delete paper <collection> <source>`" + CMD_REF
                return
            name       = parts[2]
            rest       = parts[3]
            confirmed  = rest.lower().endswith(" confirm")
            source     = rest[:-8].strip() if confirmed else rest.strip()

            try:
                qc.get_collection(name)
            except Exception as e:
                yield f"❌ Collection `{name}` not found: {e}" + CMD_REF
                return

            count_result = qc.count(
                name,
                count_filter=Filter(must=[
                    FieldCondition(key="source", match=MatchValue(value=source))
                ]), # type: ignore
                exact=True,
            )
            n = count_result.count

            if n == 0:
                yield (
                    f"No chunks found for source `{source}` in `{name}`.\n\n"
                    f"_Check available sources with `papers {name}`._"
                ) + CMD_REF
                return

            if not confirmed:
                yield (
                    f"⚠ This will permanently delete **{n} chunk(s)** from `{source}` in `{name}`.\n\n"
                    f"To confirm, run:\n`delete paper {name} {source} confirm`"
                ) + CMD_REF
                return

            from qdrant_client.models import FilterSelector
            qc.delete(
                name,
                points_selector=FilterSelector(filter=Filter(must=[
                    FieldCondition(key="source", match=MatchValue(value=source))
                ])), # type: ignore
            )
            _remove_paper_from_table(name, source)
            yield f"✅ Deleted **{n} chunk(s)** from `{source}` in `{name}`." + CMD_REF
            return

        # ── delete chunk <id> ─────────────────────────────────────────────────
        if cmd.startswith("delete chunk "):
            raw_id = cmd[13:].strip()
            try:
                point_id = int(raw_id)
            except ValueError:
                yield f"❌ Invalid chunk ID `{raw_id}` — must be a number." + CMD_REF
                return

            from qdrant_client.models import PointIdsList
            deleted_from = None
            for c in all_cols.collections:
                try:
                    pts = qc.retrieve(c.name, ids=[point_id], with_payload=False, with_vectors=False)
                    if pts:
                        qc.delete(c.name, points_selector=PointIdsList(points=[point_id]))
                        deleted_from = c.name
                        break
                except Exception:
                    continue

            if deleted_from:
                yield f"✅ Deleted chunk `{point_id}` from `{deleted_from}`." + CMD_REF
            else:
                yield f"❌ No chunk found with ID `{point_id}`." + CMD_REF
            return

        # ── delete all confirm ────────────────────────────────────────────────
        if cmd == "delete all confirm":
            cols = all_cols.collections
            if not cols:
                yield "No collections to delete." + CMD_REF
                return
            deleted = []
            for c in cols:
                try:
                    qc.delete_collection(c.name)
                    deleted.append(c.name)
                except Exception as e:
                    yield f"❌ Failed to delete `{c.name}`: {e}\n"
            if deleted:
                _clear_papers_table(deleted)
                yield f"✅ Deleted **{len(deleted)}** collection(s): {', '.join(f'`{n}`' for n in deleted)}"
            yield CMD_REF
            return

        if cmd in ("delete all", "delete all "):
            yield "⚠ This will delete ALL collections. Type `delete all confirm` to proceed." + CMD_REF
            return

        # ── delete <name> [confirm] ───────────────────────────────────────────
        if cmd.startswith("delete "):
            rest = cmd[7:].strip()
            confirmed = rest.endswith(" confirm")
            name = rest[:-8].strip() if confirmed else rest
            if name not in {c.name for c in all_cols.collections}:
                yield f"❌ No collection `{name}`." + CMD_REF
                return
            if not confirmed:
                yield (f"⚠ This will delete the collection `{name}` and its document list. "
                       f"Type `delete {name} confirm` to proceed.") + CMD_REF
                return
            try:
                qc.delete_collection(name)
                _clear_papers_table([name])
                yield f"✅ Deleted `{name}`" + CMD_REF
            except Exception as e:
                yield f"❌ Failed to delete `{name}`: {e}" + CMD_REF
            return

        # ── create <name> ─────────────────────────────────────────────────────
        if cmd.startswith("create "):
            name = cmd[7:].strip()
            if not name:
                yield "Usage: `create <collection-name>`" + CMD_REF
                return
            existing = [c.name for c in all_cols.collections]
            if name in existing:
                yield f"⚠ Collection `{name}` already exists. Use `info {name}` to inspect it." + CMD_REF
                return
            try:
                qc.create_collection(
                    name,
                    vectors_config={
                        "dense": VectorParams(size=4096, distance=Distance.COSINE)
                    },
                    sparse_vectors_config={
                        "sparse": SparseVectorParams(index=SparseIndexParams(), modifier=Modifier.IDF)
                    },
                )
                yield f"✅ Created `{name}` — 4096-d dense (cosine) + sparse (BM25, IDF-modified)" + CMD_REF
            except Exception as e:
                yield f"❌ Failed to create `{name}`: {e}" + CMD_REF
            return

        # ── papers <name> [page] ─────────────────────────────────────────────
        if cmd.startswith("papers "):
            parts = raw_strip.split()
            page = 1
            if len(parts) >= 3 and parts[-1].isdigit():
                page = max(1, int(parts[-1]))
                name = " ".join(parts[1:-1])
            else:
                name = " ".join(parts[1:])

            try:
                qc.get_collection(name)
            except Exception as e:
                yield f"❌ Collection `{name}` not found: {e}" + CMD_REF
                return

            from collections import defaultdict
            source_counts: dict = defaultdict(lambda: defaultdict(int))
            offset = None
            total_chunks = 0
            while True:
                batch, offset = qc.scroll(
                    name, limit=1000, offset=offset,
                    with_payload=["source", "type"], with_vectors=False,
                )
                for p in batch:
                    pl = p.payload or {}
                    src = pl.get("source", "(unknown)")
                    typ = pl.get("type", "(unknown)")
                    source_counts[src][typ] += 1
                    total_chunks += 1
                if offset is None:
                    break

            if not source_counts:
                yield f"No points in `{name}`." + CMD_REF
                return

            sorted_sources = sorted(source_counts.keys())
            n_sources      = len(sorted_sources)
            total_pages    = (n_sources + CHUNKS_PAGE_SIZE - 1) // CHUNKS_PAGE_SIZE
            start          = (page - 1) * CHUNKS_PAGE_SIZE

            if start >= n_sources:
                yield (
                    f"Page {page} is out of range — `{name}` has {n_sources} paper(s) "
                    f"across {total_pages} page(s) of {CHUNKS_PAGE_SIZE}."
                ) + CMD_REF
                return

            page_sources = sorted_sources[start:start + CHUNKS_PAGE_SIZE]

            yield (
                f"**`{name}`** — {total_chunks:,} chunks across **{n_sources}** paper(s) "
                f"(page {page}/{total_pages}, showing {start + 1}–{start + len(page_sources)}):\n\n"
            )
            yield "| # | Source | text | table | image | chart | Total |\n|---|---|---|---|---|---|---|\n"
            for i, src in enumerate(page_sources, start + 1):
                counts    = source_counts[src]
                row_total = sum(counts.values())
                yield (
                    f"| {i} | `{src}` | {counts.get('text', 0)} | {counts.get('table', 0)} "
                    f"| {counts.get('image', 0)} | {counts.get('chart', 0)} | {row_total} |\n"
                )

            nav = f"\n_Use `chunks {name} <source>` to inspect a paper._"
            if total_pages > 1:
                nav += f" Page {page}/{total_pages}."
                if page < total_pages:
                    nav += f" `papers {name} {page + 1}` for next."
                if page > 1:
                    nav += f" `papers {name} {page - 1}` for prev."
            yield nav
            yield CMD_REF
            return

        # ── chunks <name> <source> [page] ──────────────────────────────────────
        if cmd.startswith("chunks "):
            parts = raw_strip.split(None, 3)
            if len(parts) < 3:
                yield "Usage: `chunks <collection> <source> [page]`" + CMD_REF
                return
            name, source = parts[1], parts[2]
            page_arg = parts[3] if len(parts) > 3 else "1"
            try:
                page = max(1, int(page_arg))
            except ValueError:
                yield f"Invalid page number `{page_arg}` — must be an integer." + CMD_REF
                return

            try:
                qc.get_collection(name)
            except Exception as e:
                yield f"❌ Collection `{name}` not found: {e}" + CMD_REF
                return

            all_pts, _ = qc.scroll(
                name,
                scroll_filter=Filter(must=[
                    FieldCondition(key="source", match=MatchValue(value=source))
                ]), # type: ignore
                limit=10000,
                with_payload=True,
                with_vectors=False,
            )

            if not all_pts:
                yield (
                    f"No chunks found for source `{source}` in `{name}`.\n\n"
                    f"_Check available sources with `papers {name}`._"
                ) + CMD_REF
                return

            # payload can be None for some points, guard against that
            all_pts.sort(key=lambda p: ((p.payload or {}).get("page", 0), (p.payload or {}).get("type", "")))

            total = len(all_pts)
            total_pages = (total + CHUNKS_PAGE_SIZE - 1) // CHUNKS_PAGE_SIZE
            start = (page - 1) * CHUNKS_PAGE_SIZE
            if start >= total:
                yield (
                    f"Page {page} is out of range — `{source}` in `{name}` has "
                    f"{total} chunks across {total_pages} page(s) of {CHUNKS_PAGE_SIZE}."
                ) + CMD_REF
                return
            page_pts = all_pts[start:start + CHUNKS_PAGE_SIZE]

            yield (
                f"**{total} chunks** from `{source}` in `{name}` "
                f"(page {page}/{total_pages}, showing {start + 1}-{start + len(page_pts)}):\n\n"
            )
            yield "| # | ID | Page | Type | Section | Text preview | img |\n|---|---|---|---|---|---|---|\n"
            for i, pt in enumerate(page_pts, start + 1):
                pl   = pt.payload or {}
                page_no = pl.get("page", "?")
                typ  = pl.get("type", "?")
                section = pl.get("section") or "—"
                raw_text = (pl.get("text") or "").replace("\n", " ")
                preview  = raw_text[:80] + ("…" if len(raw_text) > 80 else "")
                if typ in ("image", "chart"):
                    img = "✓" if pl.get("img_path") else "⚠ missing"
                else:
                    img = "—"
                yield f"| {i} | `{pt.id}` | {page_no} | {typ} | {section} | {preview} | {img} |\n"
            yield "\n_Use `chunk <id>` to see full detail for any row._"
            if page < total_pages:
                yield f" Use `chunks {name} {source} {page + 1}` for the next page."
            yield CMD_REF
            return

        # ── chunk <id> ────────────────────────────────────────────────────────
        if cmd.startswith("chunk "):
            raw_id = cmd[6:].strip()
            try:
                point_id = int(raw_id)
            except ValueError:
                yield f"❌ Invalid chunk ID `{raw_id}` — must be a number." + CMD_REF
                return

            # Search across all collections for this ID
            found_col = None
            found_pt  = None
            for c in all_cols.collections:
                try:
                    pts = qc.retrieve(c.name, ids=[point_id], with_payload=True, with_vectors=False)
                    if pts:
                        found_col = c.name
                        found_pt  = pts[0]
                        break
                except Exception:
                    continue

            if not found_pt:
                yield f"❌ No chunk found with ID `{point_id}`." + CMD_REF
                return

            pl = found_pt.payload or {}
            yield f"**Chunk `{point_id}`** — collection `{found_col}`\n\n"
            yield f"| Field | Value |\n|---|---|\n"
            yield f"| source | `{pl.get('source', '—')}` |\n"
            yield f"| page | {pl.get('page', '—')} |\n"
            yield f"| type | {pl.get('type', '—')} |\n"
            yield f"| section | {pl.get('section') or '—'} |\n"
            img_path = pl.get("img_path")
            yield f"| img_path | `{img_path}` |\n" if img_path else "| img_path | — |\n"

            yield f"\n**Full text:**\n\n{pl.get('text', '(empty)')}\n"

            mineru_text = pl.get("mineru_text")
            if mineru_text:
                yield (
                    "\n**MinerU reconstruction** "
                    "(MinerU's own VLM-generated description of this figure — e.g. a "
                    "Mermaid flowchart for diagram-type images. Kept separate from "
                    "`text` since it can be inaccurate; embedded for searchability "
                    "only. Pending evaluation/filter criteria for when to surface it):"
                    f"\n\n{mineru_text}\n"
                )

            if img_path:
                import base64, os
                if os.path.exists(img_path):
                    data = base64.b64encode(open(img_path, "rb").read()).decode()
                    yield f"\n![chunk {point_id}](data:image/jpeg;base64,{data})\n"
                else:
                    yield f"\n⚠ Image file not found on disk: `{img_path}`\n"

            yield CMD_REF
            return

        # ── unknown ───────────────────────────────────────────────────────────
        yield f"Unknown command: `{cmd}`" + CMD_REF