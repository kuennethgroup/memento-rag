# The library list with a link per document, and in German for a German question.
import openwebui.functions.query_pipe as qp
from memento.query.flows import library_listing

NAMES = [f"Doc {i:02d}" for i in range(25)]


def test_each_title_is_a_link():
    links = {n: f"https://files/view/{n.replace(' ', '%20')}/0" for n in NAMES}
    text = library_listing.listing_markdown(NAMES, links=links)
    assert "1. [Doc 00](https://files/view/Doc%2000/0)" in text
    assert "25. [Doc 24](https://files/view/Doc%2024/0)" in text                 # in the collapsed rest too


def test_brackets_in_titles_are_escaped():
    text = library_listing.listing_markdown(["A [x] title"], links={"A [x] title": "https://f/view/a/0"})
    assert "1. [A \\[x\\] title](https://f/view/a/0)" in text


def test_without_links_nothing_changes():
    assert library_listing.listing_markdown(NAMES) == library_listing.listing_markdown(NAMES, links=None)
    assert "1. Doc 00" in library_listing.listing_markdown(NAMES)


def test_german_listing():
    assert library_listing.listing_intro(28, {}, de=True) == "Die Bibliothek enthält derzeit **28** Dokumente:\n\n"
    assert library_listing.listing_intro(1, {}, de=True) == "Die Bibliothek enthält derzeit **1** Dokument:\n\n"
    text = library_listing.listing_markdown(NAMES, de=True)
    assert "<summary>5 weitere anzeigen (21–25 von 25)</summary>" in text
    many = library_listing.listing_markdown([f"D{i}" for i in range(130)], de=True)
    assert "Alle 130 anzeigen" in many
    assert library_listing.listing_intro(28, {}) == "The library currently contains **28** documents:\n\n"     # English unchanged
