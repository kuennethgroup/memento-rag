# memento/library/locks.py: advisory locks for the library's writes
"""Transaction-scoped PostgreSQL advisory locks in the two-number key form, a separate key space from the upload's
single-number session lock (INGEST_ADVISORY_LOCK_KEY). Order, everywhere: the upload's session lock (if any), then the
collection lock, then the document's lock, so no two writers wait on each other in a cycle.
- per document (NS_DOCUMENT, key(collection, source)): with the collection lock SHARED first; taken by adding or
  replacing a document's row, removing it, storing an answer and writing its author list;
- per collection (NS_COLLECTION, key(collection)) EXCLUSIVE: a reset or rebuild of the whole list.
Never wait for one of these on a second connection while the same thread holds an open transaction with one: PostgreSQL
cannot see that cycle. SQLite (tests): no-ops."""
from __future__ import annotations

import zlib

from sqlalchemy import text

NS_COLLECTION = 731001
NS_DOCUMENT = 731002


def key(*parts: str) -> int:
    """A stable signed 32-bit key (CRC32; PostgreSQL's hashtext is internal and may change)."""
    v = zlib.crc32("\x1f".join(parts).encode())
    return v - 2 ** 32 if v >= 2 ** 31 else v


def _pg(conn) -> bool:
    return conn.dialect.name == "postgresql"


def document(conn, collection: str, source: str) -> None:
    """The collection lock shared, then the document's lock, until this transaction ends."""
    if not _pg(conn):
        return
    conn.execute(text("SELECT pg_advisory_xact_lock_shared(:ns, :k)"), {"ns": NS_COLLECTION, "k": key(collection)})
    conn.execute(text("SELECT pg_advisory_xact_lock(:ns, :k)"), {"ns": NS_DOCUMENT, "k": key(collection, source)})


def collection(conn, name: str) -> None:
    """The collection lock exclusive, until this transaction ends: no document of it changes meanwhile."""
    if not _pg(conn):
        return
    conn.execute(text("SELECT pg_advisory_xact_lock(:ns, :k)"), {"ns": NS_COLLECTION, "k": key(name)})


# ── Document Chat with two copies in one Open WebUI ──
# - per temporary collection (NS_TEMP_LIFECYCLE, key(collection)): a second copy's Document Chat file read and its
# cleaner hold it SHARED, in a session lock on their own connection, until their last write; removal and restore of that
# copy's rows take it EXCLUSIVE, transaction-scoped, under a lock_timeout. A waiting exclusive request queues every
# later shared one behind it. Taken before any other lock of the holder (outermost).
# - per chat (NS_TEMP_CHAT, key(chat_id)) inside the transaction that claims a chat's file: the first files of one chat
# in two libraries are claimed one after the other. Inside the lifecycle lock, never the other way round.
NS_TEMP_LIFECYCLE = 731003
NS_TEMP_CHAT = 731004


def temp_chat(conn, chat_id: str) -> None:
    """The chat's lock until this transaction ends."""
    if not _pg(conn):
        return
    conn.execute(text("SELECT pg_advisory_xact_lock(:ns, :k)"), {"ns": NS_TEMP_CHAT, "k": key(chat_id)})


class temp_lifecycle_shared:
    """`with temp_lifecycle_shared(engine, collection):` the collection's lifecycle lock SHARED, on a connection of its
    own (blocking), released when the block ends, also on an error. A connection whose unlock fails is discarded, so the
    lock never stays behind on a pooled connection. SQLite: nothing."""

    def __init__(self, engine, collection: str):
        self.engine, self.collection, self.conn = engine, collection, None

    def __enter__(self):
        if self.engine.dialect.name != "postgresql":
            return self
        self.conn = self.engine.connect()
        try:
            self.conn.execute(text("SELECT pg_advisory_lock_shared(:ns, :k)"),
                              {"ns": NS_TEMP_LIFECYCLE, "k": key(self.collection)})
            self.conn.commit()                      # a session lock: it outlives this (empty) transaction
        except BaseException:
            self.conn.invalidate()
            self.conn.close()
            self.conn = None
            raise
        return self

    def __exit__(self, *exc):
        conn, self.conn = self.conn, None
        if conn is None:
            return False
        try:
            conn.rollback()
            conn.execute(text("SELECT pg_advisory_unlock_shared(:ns, :k)"),
                         {"ns": NS_TEMP_LIFECYCLE, "k": key(self.collection)})
            conn.commit()
        except Exception:
            conn.invalidate()                       # closing the physical connection releases its session locks
        finally:
            conn.close()
        return False


def temp_lifecycle_exclusive(conn, collection: str, timeout_s: float) -> None:
    """The collection's lifecycle lock EXCLUSIVE until this transaction ends, waiting at most `timeout_s` for every
    shared holder (then PostgreSQL cancels the statement: LockNotAvailable). SQLite: nothing."""
    if not _pg(conn):
        return
    conn.execute(text(f"SET LOCAL lock_timeout = '{max(1, int(timeout_s))}s'"))
    conn.execute(text("SELECT pg_advisory_xact_lock(:ns, :k)"), {"ns": NS_TEMP_LIFECYCLE, "k": key(collection)})
