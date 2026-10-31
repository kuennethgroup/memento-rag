# Deleting a whole collection asks first, as `delete paper` and `delete all` do, and takes the collection's document list
# with it. The other delete commands stay as they are.
from types import SimpleNamespace

import pytest
import qdrant_client

from _owui_loader import load_function
from memento.config.paths import PROJECT_ROOT

COLLECTION = "memento-generic-rag"


class FakeQdrant:
    deleted: list = []
    fail = False

    def __init__(self, url=None, timeout=None):
        pass

    def get_collections(self):
        return SimpleNamespace(collections=[SimpleNamespace(name=COLLECTION), SimpleNamespace(name="memento-generic-temp")])

    def delete_collection(self, name):
        if FakeQdrant.fail:
            raise RuntimeError("no connection")
        FakeQdrant.deleted.append(name)


@pytest.fixture
def run(monkeypatch):
    FakeQdrant.deleted, FakeQdrant.fail = [], False
    mod = load_function(PROJECT_ROOT / "openwebui" / "functions" / "qdrant_pipe.py", "qdrant_manager")
    cleared = []
    monkeypatch.setattr(qdrant_client, "QdrantClient", FakeQdrant)
    monkeypatch.setattr(mod, "_ingestion_in_progress", lambda: False)
    monkeypatch.setattr(mod, "_clear_papers_table", cleared.extend)

    def send(command):
        body = {"messages": [{"role": "user", "content": command}]}
        return "".join(mod.Pipe().pipe(body, {"role": "admin"})), cleared
    return send


def test_a_collection_is_not_deleted_without_confirm(run):
    text, cleared = run(f"delete {COLLECTION}")
    assert FakeQdrant.deleted == [] and cleared == []
    assert f"delete {COLLECTION} confirm" in text


def test_confirm_deletes_the_collection_and_its_document_list(run):
    text, cleared = run(f"delete {COLLECTION} confirm")
    assert FakeQdrant.deleted == [COLLECTION] and cleared == [COLLECTION]
    assert "Deleted" in text


@pytest.mark.parametrize("command", ["delete no-such-collection", "delete no-such-collection confirm", "delete paper",
                                     "delete chunk"])
def test_a_name_that_is_no_collection_deletes_nothing(run, command):
    text, cleared = run(command)
    assert FakeQdrant.deleted == [] and cleared == []
    assert "No collection" in text


def test_a_failed_delete_keeps_the_document_list(run):
    FakeQdrant.fail = True
    text, cleared = run(f"delete {COLLECTION} confirm")
    assert cleared == [] and "Failed" in text


def test_the_help_shows_the_confirm_step():
    for path in ("openwebui/functions/qdrant_pipe.py", "openwebui/README.md"):
        assert "delete <name> [confirm]" in (PROJECT_ROOT / path).read_text() or \
               "`delete <name> confirm`" in (PROJECT_ROOT / path).read_text(), path
