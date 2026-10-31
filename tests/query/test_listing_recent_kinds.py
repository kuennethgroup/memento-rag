# "Which documents were uploaded most recently?" lists the newest uploads with their dates; "Are there any theses?" is a
# question about a kind of document, not the plain list.
from datetime import datetime, timezone

import pytest

import openwebui.functions.query_pipe as qp
from memento.query import router as qr
from memento.query.scope import documents
from memento.query.flows import library_listing


@pytest.mark.parametrize("q, recent", [("Which documents were uploaded most recently?", True),
                                       ("What are the latest documents?", True), ("Show the newest uploads", True),
                                       ("Welche Dokumente wurden zuletzt hochgeladen?", True),
                                       ("Which documents are in the library?", False)])
def test_recent_wording(q, recent):
    assert documents.wants_recent(q) is recent


def test_recent_listing():
    rows = {"A": {"added_at": datetime(2026, 10, 9, 20, 31, tzinfo=timezone.utc)},
            "B": {"added_at": datetime(2026, 10, 7, 11, 38, tzinfo=timezone.utc)},
            "C": {"added_at": None}}
    out = library_listing.recent_listing(rows, {"A": "Ax: A Platform", "B": "Thesis", "C": "Old"}, {"A": "u/A", "B": "u/B"}, n=2)
    assert out.startswith("The most recently added documents:\n\n")
    assert "1. [Ax: A Platform](u/A) · 9 Oct 2026" in out and "2. [Thesis](u/B) · 7 Oct 2026" in out
    assert "Old" not in out
    de = library_listing.recent_listing(rows, {"A": "Ax", "B": "Thesis"}, {}, n=1, de=True)
    assert de.startswith("Zuletzt hinzugefügte Dokumente:\n\n") and "1. Ax · 9 Oct 2026" in de


def test_kinds_of_documents_are_not_the_plain_list():
    p = qr.ROUTER_SYSTEM_PROMPT
    assert "theses" in p and "NOT corpus_listing" in p
