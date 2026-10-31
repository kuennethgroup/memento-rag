# "Documents from <person>" asked in everyday words lists the person's documents (own, supervised, possibly theirs);
# it is not a topic search within their own documents. A topic in the question still narrows it.
import pytest

from memento.query.flows.library_listing import author_topic

A = "Holger Rückert"


@pytest.mark.parametrize("question", [
    "Does this library contain documents from Holger Rückert?",
    "Does the library contain anything by Holger Rückert?",
    "Is there anything by Holger Rückert?",
    "What has Holger Rückert published?",
    "Has Holger Rückert written anything?",
    "Are any of these documents from Holger Rückert?",
    "Which documents include Holger Rückert?",
    "Find everything from Holger Rückert",
    "List all documents of Holger Rückert in this library",
    "Do we have work from Holger Rückert?",
    "Enthält die Bibliothek Arbeiten von Holger Rückert?",
    "Was hat Holger Rückert veröffentlicht?",
    "Gibt es etwas von Holger Rückert?",
    "Hat Holger Rückert etwas geschrieben?",
    "Welche Arbeiten in dieser Bibliothek sind von Holger Rückert?",
    "Haben wir Dokumente von Holger Rückert?",
])
def test_everyday_wording_is_a_listing(question):
    assert not author_topic(question, A)


@pytest.mark.parametrize("question", [
    "Which of Holger Rückert's papers are about foams?",
    "Does this library contain documents from Holger Rückert on bead foams?",
    "What has Holger Rückert published on epoxy toughening?",
    "Welche Arbeiten von Holger Rückert behandeln Partikelschäume?",
    "Which documents by Holger Rückert include DSC measurements?",
])
def test_a_topic_still_narrows_the_search(question):
    assert author_topic(question, A)
