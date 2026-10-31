# A throwaway PostgreSQL database for the lock tests. SQLite has no advisory locks, so the two-library lock rules are
# tested on a throwaway PostgreSQL server (MEMENTO_TEST_PG_URL): this creates `memento_2lib_test` (refusing one that
# already exists, which is never taken over or dropped), yields an engine on it, and drops only the database it
# created. Open WebUI's own database is never connected to: the server's maintenance database `postgres` issues
# CREATE / DROP DATABASE.
import os

import pytest

NAME = "memento_2lib_test"


def scratch_engine():
    """A generator for a pytest fixture: the engine of a fresh scratch database, dropped afterwards."""
    from sqlalchemy import create_engine, text
    from sqlalchemy.engine import make_url
    url = os.getenv("MEMENTO_TEST_PG_URL", "")
    if not url.startswith("postgres"):
        pytest.skip("needs a throwaway PostgreSQL (MEMENTO_TEST_PG_URL)")
    server = make_url(url).set(database="postgres")
    admin = create_engine(server, isolation_level="AUTOCOMMIT", pool_pre_ping=True)
    created = False
    try:
        with admin.connect() as c:
            if c.execute(text("SELECT 1 FROM pg_database WHERE datname = :n"), {"n": NAME}).first():
                pytest.fail(f"database {NAME} already exists: refusing to use or drop it")
            c.execute(text(f'CREATE DATABASE "{NAME}"'))
            created = True
        engine = create_engine(server.set(database=NAME), pool_size=12, max_overflow=4)
        try:
            yield engine
        finally:
            engine.dispose()
    finally:
        if created:
            with admin.connect() as c:
                c.execute(text("SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                               "WHERE datname = :n AND pid <> pg_backend_pid()"), {"n": NAME})
                c.execute(text(f'DROP DATABASE IF EXISTS "{NAME}"'))
        admin.dispose()
