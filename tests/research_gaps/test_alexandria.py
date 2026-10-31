# The offline literature index (memento/research_gaps/literature.py): search by keywords, results in the same shape as the online
# services (memento/research_gaps/literature.Work).
import sqlite3
import sys
from pathlib import Path

import pytest
from memento.config.paths import PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT))
from memento.research_gaps import literature as al  # noqa: E402


@pytest.fixture
def index(tmp_path):
    p = tmp_path / "abstracts.sqlite"
    db = sqlite3.connect(p)
    db.executescript("""
        CREATE TABLE works (id INTEGER PRIMARY KEY, doi TEXT, title TEXT, abstract TEXT, journal TEXT, issn TEXT,
                            publisher TEXT);
        CREATE VIRTUAL TABLE works_fts USING fts5(title, abstract, content='works', content_rowid='id',
                                                  tokenize='porter unicode61');""")
    rows = [("10.1/a", "Lanthanum doped hafnia", "La doping raises the remanent polarization of HfO2 films.", "J1"),
            ("10.1/b", "Polymer dielectrics", "High temperature capacitors from polyimides.", "J2"),
            ("10.1/c", "Hafnia phases", "Ferroelectric orthorhombic phase in hafnia thin films.", "J1")]
    for k, (doi, t, a, j) in enumerate(rows, start=1):
        db.execute("INSERT INTO works VALUES (?,?,?,?,?,?,?)", (k, doi, t, a, j, "0000-0000", "wiley"))
    db.execute("INSERT INTO works_fts (rowid, title, abstract) SELECT id, title, abstract FROM works")
    db.commit()
    return p


def test_the_best_matches_come_first_in_the_online_services_shape(index):
    works = al.search("lanthanum hafnia remanent polarization", path=index, n=5)
    assert [w.doi for w in works][:2] == ["10.1/a", "10.1/c"] and "10.1/b" not in [w.doi for w in works]
    w = works[0]
    assert (w.source, w.title, w.url, w.year) == ("alexandria", "Lanthanum doped hafnia", "https://doi.org/10.1/a", None)
    assert w.abstract.startswith("La doping") and w.work_id == "10.1/a"


def test_search_syntax_never_breaks_the_query(index):
    for q in ['mu* "Tc" (ambient): NOT OR AND -x', "electron-phonon", "", "***"]:
        al.search(q, path=index)                         # no exception


def test_a_missing_index_is_no_source(tmp_path):
    assert al.search("hafnia", path=tmp_path / "none.sqlite") == []
    assert al.available(tmp_path / "none.sqlite") is False


def test_the_journal_of_a_doi(index):
    # paper cards: the journal where the local index has it; DOIs are case-insensitive
    assert al.journal("10.1/a", path=index) == "J1" and al.journal("10.1/A", path=index) == "J1"
    assert al.journal("https://doi.org/10.1/b", path=index) == "J2"
    assert al.journal("10.9/none", path=index) == "" and al.journal("", path=index) == ""


def test_no_index_no_journal(tmp_path):
    assert al.journal("10.1/a", path=tmp_path / "missing.sqlite") == ""
