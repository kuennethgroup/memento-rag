"""Document Chat's own files: one row per file attached in a chat (chat, user, Open WebUI file id → Memento source name,
original name, pages, status), and the scoped Qdrant client every Document Chat answer searches through. The passages
live in one collection per deployment (`<collection_prefix>-temp`); the scope (chat id and user id) is added to every
filter, so one person's file never reaches another person or another chat. SQLAlchemy Core: Open WebUI's Postgres live,
SQLite in tests.

Two copies in one Open WebUI: each row records its library's temporary collection, so each copy reads and cleans only
its own rows; rows without one (NULL) belong to the default copy. A chat belongs to one library. A second copy's file
reads and cleaner hold its library's lifecycle lock (memento/library/locks.py) until their last write; removing that copy
moves its rows into an archive table (and restore back) under that lock, exclusively, with one transfer rule (`archive`,
`restore`)."""
from __future__ import annotations

import inspect as pyinspect
import logging
import re
import secrets
import shutil
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import (Column, DateTime, Integer, MetaData, String, Table, Text, and_, delete, insert, inspect,
                        or_, select, text, update)
from sqlalchemy.exc import IntegrityError, OperationalError

_log = logging.getLogger("memento.docchat")
METADATA = MetaData()
TEMP_DOCS = Table(
    "memento_temp_documents", METADATA,
    Column("chat_id", String(64), primary_key=True),
    Column("file_id", String(128), primary_key=True),      # OpenWebUI's file id
    Column("user_id", String(128), nullable=False, index=True),
    Column("source", String(120), nullable=False, unique=True),   # the passages' `source`, folder names
    Column("name", Text, nullable=False),                   # the file name as attached (shown in citations)
    Column("pages", Integer),
    Column("status", String(10), nullable=False),           # reading | ready | failed
    Column("error", Text),
    Column("added_at", DateTime(timezone=True)),
    Column("collection", String(120), nullable=True),       # the library's temporary collection; NULL: the default
                                                            # copy's
)
# a second copy's rows while that copy is removed (setup_openwebui.py remove-study), restored when it is set up again
ARCHIVE = Table("memento_temp_documents_study", METADATA,
                *(Column(c.name, c.type, primary_key=c.primary_key, nullable=c.nullable, unique=c.unique,
                         index=c.index) for c in TEMP_DOCS.columns))
TABLES = [TEMP_DOCS, ARCHIVE]
# the only temporary collection before the cutover: rows without a collection belong to it
DEFAULT_TEMP_COLLECTION = "memento-comp-mat-temp"


class _OtherLibrary:
    def __repr__(self):
        return "OTHER_LIBRARY"


OTHER_LIBRARY = _OtherLibrary()       # claim(): this chat already has files in another library's Document Chat

_UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
SOURCE_RE = r"[A-Za-z0-9_.-]{1,87}-[0-9a-f]{12}"


def ensure_tables(engine) -> None:
    """The tables, and the `collection` column on an older table (added, nullable: its rows stay)."""
    METADATA.create_all(engine, tables=TABLES)
    cols = {c["name"] for c in inspect(engine).get_columns(TEMP_DOCS.name)}
    if "collection" not in cols:
        with engine.begin() as c:
            if engine.dialect.name == "postgresql":
                c.execute(text(f"ALTER TABLE {TEMP_DOCS.name} ADD COLUMN IF NOT EXISTS collection VARCHAR(120)"))
            else:
                c.execute(text(f"ALTER TABLE {TEMP_DOCS.name} ADD COLUMN collection VARCHAR(120)"))


def _of(table, collection: str):
    """The rows of one library: its collection, plus the NULL rows when it is today's."""
    if collection == DEFAULT_TEMP_COLLECTION:
        return or_(table.c.collection == collection, table.c.collection.is_(None))
    return table.c.collection == collection


def valid_chat_id(chat_id) -> bool:
    """A saved OpenWebUI chat (a UUID). OpenWebUI's temporary chats ("local…") are never stored, so their files
    could never be deleted with the chat: Document Chat refuses them."""
    return isinstance(chat_id, str) and bool(_UUID_RE.fullmatch(chat_id))


def safe_source(name: str) -> str:
    """A unique source name for an attached file: its stem reduced to safe characters plus a random 12-hex token
    (folder names and citation links; cannot be guessed)."""
    stem = name.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    stem = stem.rsplit(".", 1)[0] if "." in stem.strip(".") else stem
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_.-")[:80] or "file"
    return f"{stem}-{secrets.token_hex(6)}"


def _now():
    return datetime.now(timezone.utc)


def claim(engine, *, chat_id: str, user_id: str, file_id: str, name: str,
          collection: str = DEFAULT_TEMP_COLLECTION):
    """Register a file of a chat as being read; its new source name, or None when this chat already has it (a file
    is resent with every message of the chat, and two requests may arrive at once: the primary key decides), or
    OTHER_LIBRARY when the chat already has files in another library (checked under the chat's lock, so the first
    files of one chat in two libraries are claimed one after the other)."""
    from memento.library import locks as db_locks
    src = safe_source(name)
    try:
        with engine.begin() as c:
            db_locks.temp_chat(c, chat_id)
            if _collections_of(c, chat_id) - {collection}:
                return OTHER_LIBRARY
            c.execute(insert(TEMP_DOCS).values(chat_id=chat_id, file_id=file_id, user_id=user_id, source=src,
                                               name=name, status="reading", added_at=_now(), collection=collection))
    except IntegrityError:
        return None
    return src


def _collections_of(conn, chat_id: str) -> set:
    rows = conn.execute(select(TEMP_DOCS.c.collection).where(TEMP_DOCS.c.chat_id == chat_id).distinct()).all()
    return {r[0] or DEFAULT_TEMP_COLLECTION for r in rows}


def chat_collections(engine, chat_id: str) -> set:
    """The libraries (temporary collections) this chat has files in, any user's; NULL counts as the default copy's."""
    with engine.connect() as c:
        return _collections_of(c, chat_id)


def set_status(engine, chat_id: str, file_id: str, status: str, *, pages: int | None = None,
               error: str | None = None) -> None:
    vals = {"status": status, "error": error}
    if pages is not None:
        vals["pages"] = pages
    with engine.begin() as c:
        c.execute(update(TEMP_DOCS).where(TEMP_DOCS.c.chat_id == chat_id, TEMP_DOCS.c.file_id == file_id)
                  .values(**vals))


def chat_documents(engine, chat_id: str, user_id: str, *, collection: str = DEFAULT_TEMP_COLLECTION) -> list[dict]:
    """This chat's files of this user in this library, oldest first."""
    with engine.connect() as c:
        rows = c.execute(select(TEMP_DOCS).where(TEMP_DOCS.c.chat_id == chat_id, TEMP_DOCS.c.user_id == user_id,
                                                 _of(TEMP_DOCS, collection))
                         .order_by(TEMP_DOCS.c.added_at, TEMP_DOCS.c.file_id)).mappings().all()
    return [dict(r) for r in rows]


def chat_ids(engine) -> set:
    """The chats that have Document Chat files."""
    with engine.connect() as c:
        return {r[0] for r in c.execute(select(TEMP_DOCS.c.chat_id).distinct())}


def orphans(engine, existing_chat_ids: set, *, collection: str = DEFAULT_TEMP_COLLECTION) -> list[dict]:
    """This library's rows whose chat no longer exists (deleted chat or account)."""
    with engine.connect() as c:
        rows = c.execute(select(TEMP_DOCS).where(_of(TEMP_DOCS, collection))).mappings().all()
    return [dict(r) for r in rows if r["chat_id"] not in existing_chat_ids]


def remove_rows(engine, chat_id: str, *, collection: str = DEFAULT_TEMP_COLLECTION) -> None:
    with engine.begin() as c:
        c.execute(delete(TEMP_DOCS).where(TEMP_DOCS.c.chat_id == chat_id, _of(TEMP_DOCS, collection)))


def chat_has_rows(engine, chat_id: str) -> bool:
    """Any row of this chat left, in any library."""
    with engine.connect() as c:
        return c.execute(select(TEMP_DOCS.c.file_id).where(TEMP_DOCS.c.chat_id == chat_id).limit(1)).first() is not None


# ── the scoped search client ──────────────────────────────────────────────────

_FILTER_ARG = {"scroll": "scroll_filter", "count": "count_filter", "query_points": "query_filter",
               "query_points_groups": "query_filter"}
_ALLOWED = set(_FILTER_ARG) | {"collection_exists"}


class ScopedQdrant:
    """A Qdrant client that only ever sees one chat's passages of one user (of its ready files): `chat_id` AND
    `user_id` (AND `source` in the ready files) conditions are
    added to every filter (and to every prefetch of a query), the caller's own filter nested unchanged. Only the
    methods the answer path uses exist; anything else (writes, lookups by id, other searches) raises, so a code path
    that is not scoped fails instead of leaking."""

    def __init__(self, client, chat_id: str, user_id: str, sources: list | None = None):
        """sources: the chat's READY files; passages of a file still being read, or of one that failed half-way,
        are then never found (None = no source condition, for tests of the chat and user scope alone)."""
        if not chat_id or not user_id:
            raise ValueError("a Document Chat scope needs a chat id and a user id")
        if sources is not None and not sources:
            raise ValueError("a Document Chat scope with sources needs at least one")
        self._client, self._chat, self._user = client, chat_id, user_id
        self._sources = sorted(sources) if sources is not None else None

    def _scope(self, f):
        from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue
        must = [FieldCondition(key="chat_id", match=MatchValue(value=self._chat)),
                FieldCondition(key="user_id", match=MatchValue(value=self._user))]
        if self._sources is not None:
            must.append(FieldCondition(key="source", match=MatchAny(any=self._sources)))
        return Filter(must=must + ([f] if f is not None else []))

    def _scope_prefetch(self, p):
        if p is None:
            return None
        if isinstance(p, list):
            return [self._scope_prefetch(x) for x in p]
        return p.model_copy(update={"filter": self._scope(p.filter), "prefetch": self._scope_prefetch(p.prefetch)})

    def __getattr__(self, name):
        if name not in _ALLOWED:
            raise PermissionError(f"Document Chat search client: '{name}' is not allowed")
        method = getattr(self._client, name)
        if name == "collection_exists":
            return method

        def scoped(*args, **kwargs):
            bound = pyinspect.signature(method).bind_partial(*args, **kwargs)
            arg = _FILTER_ARG[name]
            bound.arguments[arg] = self._scope(bound.arguments.get(arg))
            if "prefetch" in bound.arguments:
                bound.arguments["prefetch"] = self._scope_prefetch(bound.arguments["prefetch"])
            return method(*bound.args, **bound.kwargs)
        return scoped


# ── clean-up: chats that no longer exist (Open WebUI has no "chat deleted" hook) ──

def _under(root: Path, path: Path) -> bool:
    """`path` lies strictly inside `root` (both resolved): the only paths the clean-up may remove."""
    try:
        root, path = Path(root).resolve(), Path(path).resolve()
    except Exception:
        return False
    return path != root and root in path.parents


def _remove_tree(root: Path, path: Path) -> None:
    if _under(root, path) and path.exists():
        shutil.rmtree(path)


def cleanup(engine, client, collection: str, input_root, output_root, existing_chat_ids: set | None,
            remove_upload=None) -> int:
    """Remove the passages, folders and rows of every chat that no longer exists; the number of chats removed.
    existing_chat_ids None (the chat list could not be read) → nothing is removed. A chat whose passages cannot be
    deleted keeps its rows (and folders), so the next run tries again. remove_upload(file_id, user_id): removes
    OpenWebUI's own copy of each upload (OpenWebUI keeps it when a chat is deleted)."""
    from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchValue
    if existing_chat_ids is None:
        return 0
    in_root, out_root = Path(input_root) / "_temp", Path(output_root) / "_temp"
    by_chat: dict[str, list[dict]] = {}
    for r in orphans(engine, existing_chat_ids, collection=collection):      # this library's rows only
        by_chat.setdefault(r["chat_id"], []).append(r)
    removed = 0
    for chat_id, rows in by_chat.items():
        try:
            client.delete(collection_name=collection, points_selector=FilterSelector(filter=Filter(must=[
                FieldCondition(key="chat_id", match=MatchValue(value=chat_id))])))
        except Exception as e:
            if "not found" not in str(e).lower():        # no collection yet: nothing stored, rows can go
                _log.warning("temp clean-up: passages of %s not deleted: %r", chat_id, e)
                continue
        for r in rows:
            if re.fullmatch(SOURCE_RE, r["source"] or ""):
                _remove_tree(out_root, out_root / r["source"])
            if remove_upload is not None:
                try:
                    remove_upload(r["file_id"], r["user_id"])
                except Exception as e:
                    _log.warning("temp clean-up: upload %s not removed: %r", r["file_id"], e)
        remove_rows(engine, chat_id, collection=collection)
        # the chat's folder of attachments: once no library has a row of this chat left (legacy data only)
        if valid_chat_id(chat_id) and not chat_has_rows(engine, chat_id):
            _remove_tree(in_root, in_root / chat_id)
        removed += 1
        _log.info("temp clean-up: chat %s removed (%d files)", chat_id, len(rows))
    return removed


# ── two copies: the lifecycle lock, archive and restore ──

class LockTimeout(RuntimeError):
    """Removal or restore waited longer than its limit for running Document Chat reads or cleaners."""


class TransferConflict(RuntimeError):
    """A row under the same (chat, file) key exists on the other side with different content: nothing was moved."""

    def __init__(self, conflicts: list):
        self.conflicts = conflicts            # [(row being moved, row already there)]
        super().__init__(f"{len(conflicts)} row(s) differ under the same chat and file: nothing moved")


@contextmanager
def lifecycle(engine, collection: str):
    """`with lifecycle(engine, collection):` a second copy's lifecycle lock held SHARED (memento/library/locks.py), for a
    Document Chat file read or a cleaner, until the block ends. The default copy takes none: removal and restore only
    ever move a second copy's rows."""
    if collection == DEFAULT_TEMP_COLLECTION:
        yield
        return
    from memento.library import locks as db_locks
    with db_locks.temp_lifecycle_shared(engine, collection):
        yield


def _delete_row(conn, table, row: dict) -> None:
    conn.execute(delete(table).where(table.c.chat_id == row["chat_id"], table.c.file_id == row["file_id"]))


def _transfer(conn, source, target, collection: str) -> dict:
    """Move one library's rows from `source` to `target` in the caller's transaction, by (chat, file): a row not in
    `target` is copied with every column and then deleted from `source`; an identical row already in `target` only
    deleted (a repeated run); a different one is a conflict: TransferConflict, and the caller's transaction rolls back,
    so nothing on either side changes."""
    moved, identical, conflicts = 0, 0, []
    rows = conn.execute(select(source).where(_of(source, collection))
                        .order_by(source.c.chat_id, source.c.file_id)).mappings().all()
    for r in (dict(x) for x in rows):
        there = conn.execute(select(target).where(target.c.chat_id == r["chat_id"],
                                                  target.c.file_id == r["file_id"])).mappings().first()
        if there is None:
            conn.execute(insert(target).values(**r))
            moved += 1
        elif dict(there) == r:
            identical += 1
        else:
            conflicts.append((r, dict(there)))
            continue
        _delete_row(conn, source, r)
    if conflicts:
        raise TransferConflict(conflicts)
    return {"moved": moved, "identical": identical}


def _locked_transfer(engine, source, target, collection: str, timeout_s: float, before_move=None) -> dict:
    from memento.library import locks as db_locks
    try:
        with engine.begin() as c:
            db_locks.temp_lifecycle_exclusive(c, collection, timeout_s)
            if before_move is not None:
                before_move([dict(r) for r in c.execute(select(source).where(_of(source, collection))
                                                         .order_by(source.c.chat_id, source.c.file_id)).mappings()])
            return _transfer(c, source, target, collection)
    except OperationalError as e:
        if "lock timeout" in str(e).lower() or "LockNotAvailable" in type(getattr(e, "orig", e)).__name__:
            raise LockTimeout(f"{collection}: Document Chat reads or cleaners still running after {timeout_s:.0f} s") \
                from e
        raise


def archive(engine, collection: str, *, timeout_s: float, before_move=None) -> dict:
    """A second copy's Document Chat rows into the archive table, all or nothing, once every running read and cleaner of
    it has finished (the exclusive lifecycle lock, at most `timeout_s`). `before_move(rows)` runs inside the lock,
    before anything moves (the caller's authoritative backup). Vectors and files are not touched."""
    if collection == DEFAULT_TEMP_COLLECTION:
        raise ValueError("today's library is never archived")
    ensure_tables(engine)
    return _locked_transfer(engine, TEMP_DOCS, ARCHIVE, collection, timeout_s, before_move)


def restore(engine, collection: str, *, timeout_s: float) -> dict:
    """The archived rows of a second copy back into the live table, by the same rule. With nothing archived for it:
    nothing happens and no lock is taken (so setting it up again never makes a read wait)."""
    if collection == DEFAULT_TEMP_COLLECTION:
        raise ValueError("today's library is never archived")
    ensure_tables(engine)
    with engine.connect() as c:
        if c.execute(select(ARCHIVE.c.file_id).where(_of(ARCHIVE, collection)).limit(1)).first() is None:
            return {"moved": 0, "identical": 0}
    return _locked_transfer(engine, ARCHIVE, TEMP_DOCS, collection, timeout_s)
