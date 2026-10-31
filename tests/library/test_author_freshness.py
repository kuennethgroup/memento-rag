# Polite author requests and stale author data: the polite words are generic ("Please list all papers by X" is an author
# question), a stored author list counts only when its fingerprint is known and equal to the library's, and a write
# never replaces the list of a newer version.
import pytest
from sqlalchemy import create_engine, text

import openwebui.functions.query_pipe as qp
from memento.library import front_matter as da
from memento.query.flows import library_listing

X = "Christopher Brueckner"


@pytest.mark.parametrize("q", [f"Please list all papers by {X}", f"Bitte zeige alle Arbeiten von {X}",
                               f"Could you list the documents by {X}?", f"Würdest du mir die Arbeiten von {X} zeigen?",
                               f"Kindly list papers by {X}", f"Könntest du bitte alle Paper von {X} auflisten?",
                               f"Would you show me the works of {X}, please?"])
def test_polite_requests_are_no_topic(q):
    assert not library_listing.author_topic(q, X)


def test_a_real_topic_still_counts():
    assert library_listing.author_topic(f"Please list papers by {X} about hafnia", X)
    assert library_listing.author_topic(f"Bitte zeige die Arbeiten von {X} zu Polymeren", X)


# ── freshness ────────────────────────────────────────────────────────────────────────────────────────────────────────

def test_only_a_known_matching_fingerprint_is_current():
    rows = {"A": ("front A", "h1"), "B": ("front B", "h1"), "C": ("front C", None), "D": ("front D", "h1")}
    library = {"A": "h1", "B": "h2", "C": "h1", "D": None, "E": "h1"}
    current, unverified = da.fresh(rows, library)
    assert current == {"A": "front A"}
    assert unverified == {"B", "C", "D", "E"}       # replaced; no stored hash; no library hash; no stored list


# ── writing ──────────────────────────────────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def engine():
    e = create_engine("sqlite://")
    with e.begin() as c:
        c.execute(text("CREATE TABLE memento_documents (collection VARCHAR(200), source VARCHAR(500), "
                       "content_hash VARCHAR(128), PRIMARY KEY (collection, source))"))
    da.ensure_table(e)
    return e


def _library(engine, h):
    with engine.begin() as c:
        c.execute(text("DELETE FROM memento_documents WHERE collection='c' AND source='Doc'"))
        c.execute(text("INSERT INTO memento_documents VALUES ('c', 'Doc', :h)"), {"h": h})


def test_a_write_for_the_current_version_is_stored(engine):
    _library(engine, "h1")
    assert da.put(engine, collection="c", source="Doc", content_hash="h1", front="Alice Jones") is True
    assert da.load(engine, "c", with_hash=True) == {"Doc": ("Alice Jones", "h1")}


def test_an_old_successful_write_after_a_newer_one_is_ignored(engine):
    _library(engine, "h2")                                       # the newer version is in the library …
    da.put(engine, collection="c", source="Doc", content_hash="h2", front="Bob New")      # … with its author list
    assert da.put(engine, collection="c", source="Doc", content_hash="h1", front="Alice Old") is False   # late, old job
    assert da.load(engine, "c", with_hash=True) == {"Doc": ("Bob New", "h2")}


# a write needs the library's row with a known, equal fingerprint; anything else leaves the stored list as it is (no
# overwrite of a newer list, no orphan list for a removed document)

def test_a_job_without_a_fingerprint_cannot_overwrite_the_current_list(engine):
    _library(engine, "h2")
    da.put(engine, collection="c", source="Doc", content_hash="h2", front="Bob New")
    assert da.put(engine, collection="c", source="Doc", content_hash=None, front="Alice Old") is False
    assert da.load(engine, "c", with_hash=True) == {"Doc": ("Bob New", "h2")}


def test_a_library_row_without_a_fingerprint_gets_no_list(engine):
    _library(engine, None)
    assert da.put(engine, collection="c", source="Doc", content_hash="h1", front="Alice Jones") is False
    assert da.load(engine, "c") == {}


def test_a_removed_document_gets_no_list(engine):
    assert da.put(engine, collection="c", source="Gone", content_hash="h1", front="Old author") is False
    assert da.load(engine, "c") == {}


def test_without_a_library_table_nothing_is_written():
    e = create_engine("sqlite://")
    da.ensure_table(e)
    assert da.put(e, collection="c", source="Doc", content_hash="h1", front="Alice Jones") is False


def test_store_front_reports_what_was_stored(engine, tmp_path):
    (tmp_path / "Doc" / "vlm").mkdir(parents=True)
    (tmp_path / "Doc" / "vlm" / "Doc.md").write_text("# T\n\nAlice Jones\n")
    _library(engine, "h2")
    assert da.store_front(engine, collection="c", source="Doc", out_dir=tmp_path, content_hash="h1") is False
    assert da.store_front(engine, collection="c", source="Doc", out_dir=tmp_path, content_hash="h2") is True


def test_the_pipe_checks_freshness():
    import inspect
    src = inspect.getsource(library_listing.person_reply)
    assert "doc_authors.fresh(" in src and "with_hash=True" in src
