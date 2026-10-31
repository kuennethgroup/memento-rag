"""Shared fixtures for the Memento test suite. Tests that need a database run only against throwaway servers named by
MEMENTO_TEST_PG_URL, NEO4J_TEST_URI and MEMENTO_TEST_QDRANT_URL (never the configured ones) and skip without them.

Usage: pytest tests/ (all), -m unit (no services), -v (verbose)."""

import os
os.environ.setdefault("REPLAY_CHARS_PER_S", "0")   # finished answers at once in tests (typed out live)
import pytest
from pathlib import Path

# The suite is written against the polymer-engineering domain profile. config/.env names whatever domain the deployment
# serves, and memento.config.settings reads it once at import, so the tests' domain is pinned before anything imports config.
os.environ["DOMAIN_NAME"] = "polymer-engineering"
# the setup tests check the Valves built from the local model server and the member group, whatever config/.env or
# the shell says
# (the frozen evaluation path, memento/retrieval/generate.py, reads REASONING_URL too: its own default keeps its recording)
os.environ["REASONING_URL"] = "http://localhost:8023"
os.environ["MEMENTO_MEMBER_GROUP"] = "members"

_env_file = Path(__file__).parent.parent / "config" / ".env"
if _env_file.exists():
    from dotenv import load_dotenv
    load_dotenv(_env_file)      # does not override DOMAIN_NAME set above

# Runtime state files (OUTPUT_ROOT/_runtime/: the picker's library.json, the cloud quota) are written by the pipes at
# every question, and config/.env points OUTPUT_ROOT at the live data folder. Every test gets its own OUTPUT_ROOT; tests
# that need one set it.
@pytest.fixture(autouse=True)
def _private_output_root(tmp_path, monkeypatch):
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "output"))


# The pipes reach Open WebUI's database through open_webui.internal.db.engine (the live database, DATABASE_URL); the
# document list and the answer cache write there (memento/library/documents.py). Every test gets its own in-memory database
# instead.
#
# Importing Open WebUI's config deletes every file in its served static folder and copies them back from its build
# folder, which the tests do not find: a test run would leave the live server without loader.js / custom.css. The tests
# get their own static folder.
import tempfile
os.environ["STATIC_DIR"] = tempfile.mkdtemp(prefix="memento-test-static-")
try:
    import open_webui.internal.db as _owui_db
except Exception:                                   # OpenWebUI not installed: nothing to isolate
    _owui_db = None


# The query package's shared caches (memento/query/request.py, library_state.py) start empty in every test: a domain
# profile, a copy's state, a title or an engine id from another test must not leak into the next.
@pytest.fixture(autouse=True)
def _fresh_query_caches(monkeypatch):
    from memento.query import library_state, request
    monkeypatch.setattr(request, "_DOMAINS", {})
    for name in ("ALIASES", "TITLES", "LIB_TITLES", "_STATE_CACHE"):
        monkeypatch.setattr(library_state, name, {})
    monkeypatch.setattr(library_state, "_TABLES_READY", set())


@pytest.fixture(autouse=True)
def _private_database(monkeypatch):
    if _owui_db is not None:
        from sqlalchemy import create_engine
        from sqlalchemy.pool import StaticPool
        # one in-memory database for every thread (pipes write from worker threads, as with Postgres live)
        monkeypatch.setattr(_owui_db, "engine", create_engine("sqlite://", poolclass=StaticPool,
                                                              connect_args={"check_same_thread": False}))


@pytest.fixture
def throwaway_qdrant():
    """A client of the throwaway Qdrant named by MEMENTO_TEST_QDRANT_URL; skips without it."""
    url = os.getenv("MEMENTO_TEST_QDRANT_URL")
    if not url:
        pytest.skip("needs a throwaway Qdrant (MEMENTO_TEST_QDRANT_URL)")
    from qdrant_client import QdrantClient
    return QdrantClient(url=url, timeout=10)
