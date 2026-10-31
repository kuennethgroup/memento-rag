# "The X paper" is the document a question is about, even when the question also names a method that is another
# document's short name.
import pytest

import openwebui.functions.query_pipe as qp
from memento.query.scope import documents

DIELECTRICS = "AI-assisted discovery of high-temperature dielectrics for energy storage"


@pytest.mark.parametrize("q, want", [
    ("The dielectrics paper also uses polyBERT, right?", ["the dielectrics paper"]),
    ("Does the battery electrode paper use polyBERT?", ["the battery electrode paper"]),
    ("Summarize the polyBERT paper", ["the polyBERT paper"]),
    ("What does the LaHfO2 study report?", ["the LaHfO2 study"]),
    ("Nutzt die Dielektrika-Arbeit polyBERT?", ["die Dielektrika-Arbeit"]),
    ("Which papers use polyBERT?", []),
    ("What is polyBERT?", []),
    ("Summarize the first paper", []),
    ("What does the original polyBERT paper say?", ["the polyBERT paper"]),
    ("Is that the same study?", []),
])
def test_document_phrases(q, want):
    assert documents.document_phrases(q) == want


def _fake_resolver(table):
    return lambda q, sources, dom=None: next((v for k, v in table.items() if k in q.lower()), None)


def test_the_described_document_wins_over_a_method_name(monkeypatch):
    monkeypatch.setattr(documents, "_resolve_paper", _fake_resolver({"dielectrics": DIELECTRICS, "polybert": "PolyBERT"}))
    assert documents.resolve_reference("The dielectrics paper also uses polyBERT, right?", [], None) == DIELECTRICS


def test_a_described_document_that_is_not_found_does_not_fall_back_to_the_method(monkeypatch):
    monkeypatch.setattr(documents, "_resolve_paper", _fake_resolver({"polybert": "PolyBERT"}))
    assert documents.resolve_reference("Does the battery electrode paper use polyBERT?", [], None) is None


def test_without_a_phrase_nothing_changes(monkeypatch):
    monkeypatch.setattr(documents, "_resolve_paper", _fake_resolver({"polybert": "PolyBERT"}))
    assert documents.resolve_reference("What does PolyBERT report about its training data?", [], None) == "PolyBERT"
    assert documents.resolve_reference("Summarize the polyBERT paper", [], None) == "PolyBERT"
