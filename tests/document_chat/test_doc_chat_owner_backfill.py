# A chat whose files were read before chat folders recorded their owner gets the record at its next message (no new
# file needed), so its earlier citations open again for its owner; a folder recorded for another account is left as is.
import contextlib
import logging
from types import SimpleNamespace

from memento.document_chat import chat, intake

CHAT = "3f2b8c1e-6a4d-4e0b-9c7a-1d2e3f4a5b6c"


class FakeStore:
    OTHER_LIBRARY = object()

    def __init__(self, rows):
        self.rows = rows

    def lifecycle(self, engine, coll):
        return contextlib.nullcontext()

    def chat_documents(self, engine, chat_id, user_id, collection=None):
        return self.rows


def _read(tmp_path, rows, user="alice"):
    valves = SimpleNamespace(COLLECTION="memento-x-rag", DOC_CHAT_MAX_FILES=5, DOC_CHAT_MAX_PAGES=50,
                             INGESTION_FUNCTION_ID="memento_ingestion")
    return chat.doc_chat_read_sync(valves, None, FakeStore(rows), CHAT, user, [], {}, None, logging.getLogger("t"),
                                   function_id="memento_query", db_engine=None,
                                   saved_valves=lambda fid: {"INPUT_ROOT": str(tmp_path / "in"),
                                                             "OUTPUT_ROOT": str(tmp_path / "out")})


def test_an_earlier_chat_folder_gets_its_owner_at_the_next_message(tmp_path):
    folder = tmp_path / "in" / "_temp" / CHAT
    folder.mkdir(parents=True)
    (folder / "Report-0123456789ab.pdf").write_bytes(b"%PDF")
    assert _read(tmp_path, [{"status": "ready", "file_id": "f1", "added_at": None}]) == ([], 0, "ok")
    assert (folder / intake.OWNER_FILE).read_text() == "alice"


def test_a_chat_without_files_gets_no_folder(tmp_path):
    _read(tmp_path, [])
    assert not (tmp_path / "in" / "_temp" / CHAT).exists()


def test_a_folder_recorded_for_another_account_is_left_as_is(tmp_path):
    folder = tmp_path / "in" / "_temp" / CHAT
    folder.mkdir(parents=True)
    (folder / intake.OWNER_FILE).write_text("bob")
    assert _read(tmp_path, [{"status": "ready", "file_id": "f1", "added_at": None}]) == ([], 0, "ok")
    assert (folder / intake.OWNER_FILE).read_text() == "bob"
