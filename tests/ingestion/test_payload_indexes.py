# The passage-field indexes.
import sys
from pathlib import Path
from memento.config.paths import PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT))
from memento.ingestion.payload_indexes import PAYLOAD_INDEXES, ensure_payload_indexes  # noqa: E402


class FakeClient:
    def __init__(self, have):
        self.have, self.made = dict.fromkeys(have), []

    def get_collection(self, name):
        return type("C", (), {"payload_schema": self.have})()

    def create_payload_index(self, collection_name, field_name, field_schema, wait):
        self.made.append(field_name)


def test_creates_only_missing_indexes():
    c = FakeClient(["source"])
    assert ensure_payload_indexes(c, "col") == ["section", "type", "gap_relevant", "page"] == c.made
    assert ensure_payload_indexes(FakeClient(PAYLOAD_INDEXES), "col") == []


def test_every_filtered_field_is_indexed():
    # the fields the code filters on (grep FieldCondition(key=...)): source, section, type, gap_relevant, page
    import re
    root = PROJECT_ROOT
    used = set()
    for f in list((root / "openwebui/functions").glob("*.py")) + list((root / "retrieval").glob("*.py")):
        used |= set(re.findall(r'FieldCondition\(\s*key="([a-z_]+)"', f.read_text()))
    # chat_id / user_id: Memento Document Chat's temporary collection only, indexed by
    # memento/document_chat/intake.ensure_store
    # (tests/document_chat/test_temp_ingest.py::test_the_temporary_store_indexes_the_chat_and_the_user)
    allowed = set(PAYLOAD_INDEXES) | {"chat_id", "user_id"}
    assert used <= allowed, used - allowed
