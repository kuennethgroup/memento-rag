# The gap ledger's storage.
import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from memento.config.paths import PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT))
from memento.research_gaps import ledger as gl  # noqa: E402


@pytest.fixture
def engine():
    e = create_engine("sqlite://")
    gl.ensure_tables(e)
    return e


def _mention(src, pid="p1", page=0):
    return dict(source=src, point_id=pid, page=page, signal="tag line", claim="X is not examined.",
                passage="passage text", judge="supported", judge_reason="states it")


def test_a_gap_is_stated_by_documents_and_counts_each_document_once(engine):
    gid = gl.add_gap(engine, "col", "How does X affect Y?", "limitation")
    gl.add_mention(engine, gid, "col", **_mention("A", "p1"))
    gl.add_mention(engine, gid, "col", **_mention("A", "p2", 3))      # same document, second passage
    gl.add_mention(engine, gid, "col", **_mention("B", "p9"))
    gaps = gl.list_gaps(engine, "col")
    assert len(gaps) == 1 and gaps[0]["statement"] == "How does X affect Y?" and gaps[0]["kind"] == "limitation"
    assert gaps[0]["documents"] == ["A", "B"] and len(gaps[0]["mentions"]) == 3


def test_other_libraries_are_kept_apart(engine):
    gl.add_mention(engine, gl.add_gap(engine, "col", "Q1?", "uncertainty"), "col", **_mention("A"))
    gl.add_mention(engine, gl.add_gap(engine, "other", "Q2?", "uncertainty"), "other", **_mention("A"))
    assert [g["statement"] for g in gl.list_gaps(engine, "col")] == ["Q1?"]


def test_deleting_a_document_removes_its_mentions_and_gaps_nobody_states_any_more(engine):
    g1 = gl.add_gap(engine, "col", "Only A states this?", "limitation")
    g2 = gl.add_gap(engine, "col", "A and B state this?", "future_work")
    gl.add_mention(engine, g1, "col", **_mention("A"))
    gl.add_mention(engine, g2, "col", **_mention("A"))
    gl.add_mention(engine, g2, "col", **_mention("B"))
    gl.add_scope_only(engine, "col", "A", point_id="p3", page=1, signal="s", claim="c", reason="study scope only")
    gl.add_dropped(engine, "col", "A", point_id="p4", page=1, question="Q?", claim="c", reason="unsupported: ...")
    gl.set_document(engine, "col", "A", content_hash="h", status="done", n_questions=2, n_scope=1, n_dropped=1)
    orphans = gl.delete_for_source(engine, "col", "A")
    assert orphans == [g1]                                                         # vectors of these go too
    gaps = gl.list_gaps(engine, "col")
    assert [g["statement"] for g in gaps] == ["A and B state this?"] and gaps[0]["documents"] == ["B"]
    assert gl.scope_only(engine, "col") == [] and gl.dropped(engine, "col") == []
    assert gl.get_document(engine, "col", "A") is None


def test_document_status_says_what_is_done_and_with_which_content(engine):
    assert gl.get_document(engine, "col", "A") is None
    gl.set_document(engine, "col", "A", content_hash="h1", status="done", n_questions=4, n_scope=2, n_dropped=1)
    d = gl.get_document(engine, "col", "A")
    assert d["status"] == "done" and d["content_hash"] == "h1" and (d["n_questions"], d["n_scope"], d["n_dropped"]) == (4, 2, 1)
    gl.set_document(engine, "col", "A", content_hash="h2", status="failed", error="model down")
    d = gl.get_document(engine, "col", "A")
    assert d["status"] == "failed" and d["content_hash"] == "h2" and d["error"] == "model down"


def test_reset_of_one_library(engine):
    gl.add_mention(engine, gl.add_gap(engine, "col", "Q?", "limitation"), "col", **_mention("A"))
    gl.add_mention(engine, gl.add_gap(engine, "other", "R?", "limitation"), "other", **_mention("A"))
    gl.set_document(engine, "col", "A", content_hash="h", status="done")
    assert gl.clear_library(engine, "col") >= 1
    assert gl.list_gaps(engine, "col") == [] and gl.get_document(engine, "col", "A") is None
    assert len(gl.list_gaps(engine, "other")) == 1


def test_each_mention_keeps_its_own_question_for_auditing_merges(engine):
    gid = gl.add_gap(engine, "col", "How does X affect Y?", "limitation")
    gl.add_mention(engine, gid, "col", question="Does X change Y in thin films?", **_mention("B"))
    assert gl.list_gaps(engine, "col")[0]["mentions"][0]["question"] == "Does X change Y in thin films?"


def test_an_existing_mentions_table_gets_the_question_column():
    from sqlalchemy import create_engine, text
    e = create_engine("sqlite://")
    with e.begin() as c:                                                           # the table in its first form
        c.execute(text("CREATE TABLE memento_gap_mentions (id INTEGER PRIMARY KEY, gap_id INTEGER NOT NULL, "
                       "collection VARCHAR(200) NOT NULL, source VARCHAR(500) NOT NULL, point_id VARCHAR(64) NOT NULL, "
                       "page INTEGER, signal TEXT, claim TEXT NOT NULL, passage TEXT NOT NULL, judge VARCHAR(20) NOT NULL, "
                       "judge_reason TEXT, created_at TIMESTAMP)"))
    gl.ensure_tables(e)
    gl.ensure_tables(e)
    gid = gl.add_gap(e, "col", "Q?", "limitation")
    gl.add_mention(e, gid, "col", question="q?", **_mention("A"))
    assert gl.list_gaps(e, "col")[0]["mentions"][0]["question"] == "q?"


def test_related_gaps_form_themes_and_go_with_their_gaps(engine):
    ids = [gl.add_gap(engine, "col", f"Q{k}?", "limitation") for k in range(5)]
    for gid, src in zip(ids, "ABCDE"):
        gl.add_mention(engine, gid, "col", **_mention(src))
    gl.add_relation(engine, "col", ids[0], ids[1])
    gl.add_relation(engine, "col", ids[1], ids[2])
    gl.add_relation(engine, "col", ids[1], ids[0])                                 # duplicate, other direction
    themes = gl.themes(engine, "col")
    assert sorted(map(sorted, themes)) == [sorted(ids[:3])]                         # singletons are no theme
    assert gl.related(engine, "col", ids[1]) == sorted([ids[0], ids[2]])
    gl.delete_for_source(engine, "col", "B")                                       # gap ids[1] has no mention left
    assert gl.related(engine, "col", ids[0]) == [] and gl.themes(engine, "col") == []


def test_themes_are_formed_only_within_a_selection(engine):
    # groups are only built among the questions one report covers (chaining "related" links over the whole library joins
    # many questions through a few hub questions)
    ids = [gl.add_gap(engine, "col", f"Q{k}?", "limitation") for k in range(5)]
    for gid, src in zip(ids, "ABCDE"):
        gl.add_mention(engine, gid, "col", **_mention(src))
    for a, b in ((0, 1), (1, 2), (2, 3), (3, 4)):                                   # one long chain
        gl.add_relation(engine, "col", ids[a], ids[b])
    assert gl.themes(engine, "col", within={ids[0], ids[1], ids[3], ids[4]}) == [sorted([ids[0], ids[1]]),
                                                                                  sorted([ids[3], ids[4]])]


def test_library_check_results_are_stored_with_their_evidence(engine):
    gid = gl.add_gap(engine, "col", "Q?", "limitation")
    gl.add_mention(engine, gid, "col", **_mention("A"))
    gl.set_library_status(engine, gid, status="partly", docs_checked=24,
                          evidence=[dict(source="B", point_id="p7", page=2, relation="partly", reason="other material"),
                                    dict(source="C", point_id="p9", page=0, relation="near_miss", reason="off topic")])
    g = gl.list_gaps(engine, "col")[0]
    assert g["library_status"] == "partly" and g["library_docs_checked"] == 24 and g["library_checked_at"] is not None
    ev = gl.evidence(engine, gid)
    assert [(e["source"], e["relation"]) for e in ev] == [("B", "partly"), ("C", "near_miss")]
    gl.set_library_status(engine, gid, status="open", docs_checked=24, evidence=[])   # a redo replaces the evidence
    assert gl.evidence(engine, gid) == []
    gl.delete_for_source(engine, "col", "B")                                         # nothing breaks


def test_a_literature_check_is_stored_with_its_papers(engine):
    gid = gl.add_gap(engine, "col", "Q?", "limitation")
    ev = [dict(origin="openalex", source="W1", title="Wake-up", year=2021, doi="10.1/x", url="https://doi.org/10.1/x",
               relation="addresses", reason="reports it")]
    gl.set_web_status(engine, gid, status="possibly_addressed", query="hafnia wake-up",
                      sources=["semantic_scholar", "openalex"], evidence=ev)
    g = gl.get_gap(engine, gid)
    assert (g["web_status"], g["web_query"], g["web_sources"]) == ("possibly_addressed", "hafnia wake-up",
                                                                   "openalex,semantic_scholar")
    assert g["web_checked_at"] is not None
    (e,) = gl.evidence(engine, gid, origin="openalex")
    assert (e["title"], e["year"], e["doi"], e["relation"]) == ("Wake-up", 2021, "10.1/x", "addresses")
    gl.set_web_status(engine, gid, status="nothing_found", query="q2", sources=["semantic_scholar"], evidence=[])
    assert gl.evidence(engine, gid, origin="openalex") == [] and gl.get_gap(engine, gid)["web_status"] == "nothing_found"


def test_web_evidence_does_not_touch_library_evidence(engine):
    gid = gl.add_gap(engine, "col", "Q?", "limitation")
    gl.set_library_status(engine, gid, status="partly", docs_checked=3,
                          evidence=[dict(source="Doc", point_id="1", page=0, relation="partly", reason="r")])
    gl.set_web_status(engine, gid, status="nothing_found", query="q", sources=["semantic_scholar"], evidence=[])
    assert len(gl.evidence(engine, gid, origin="library")) == 1


def test_old_gaps_tables_get_the_new_columns():
    from sqlalchemy import text
    e = create_engine("sqlite://")
    with e.begin() as c:
        c.execute(text("CREATE TABLE memento_gaps (id INTEGER PRIMARY KEY, collection TEXT, statement TEXT, kind TEXT, "
                       "created_at TIMESTAMP, library_status TEXT, library_checked_at TIMESTAMP, "
                       "library_docs_checked INTEGER, web_status TEXT, web_checked_at TIMESTAMP)"))
    gl.ensure_tables(e)
    from sqlalchemy import inspect
    cols = {c["name"] for c in inspect(e).get_columns("memento_gaps")}
    assert {"web_query", "web_sources"} <= cols


def test_a_papers_abstract_is_stored_as_the_text_its_verdict_was_made_from(engine):
    gid = gl.add_gap(engine, "col", "Q?", "limitation")
    gl.set_web_status(engine, gid, status="partly", query="q", sources=["openalex"],
                      evidence=[dict(origin="openalex", source="W1", title="T", relation="partly", reason="r",
                                     text="The abstract.")])
    assert gl.evidence(engine, gid, origin="openalex")[0]["text"] == "The abstract."


def test_missing_abstracts_are_filled_in_by_work_id(engine):
    gid = gl.add_gap(engine, "col", "Q?", "limitation")
    gl.set_web_status(engine, gid, status="partly", query="q", sources=["openalex", "semantic_scholar"],
                      evidence=[dict(origin="openalex", source="W1", title="Same title", relation="partly", reason="r"),
                                dict(origin="semantic_scholar", source="p9", title="Other", relation="near_miss",
                                     reason="r"),
                                dict(origin="semantic_scholar", source="p0", title="Unknown", relation="partly",
                                     reason="r")])
    n = gl.fill_web_text(engine, {"W1": "Abstract one.", "p9": "Abstract nine.", "zz": "unused"})
    texts = {e["source"]: e["text"] for e in gl.evidence(engine, gid)}
    assert n == 2 and texts == {"W1": "Abstract one.", "p9": "Abstract nine.", "p0": None}


def test_related_counts_per_question(engine):
    a, b, c = (gl.add_gap(engine, "col", f"Q{i}?", "limitation") for i in range(3))
    gl.add_relation(engine, "col", a, b)
    gl.add_relation(engine, "col", a, c)
    gl.add_relation(engine, "other", a, b)
    assert gl.related_counts(engine, "col") == {a: 2, b: 1, c: 1}


def test_a_new_documents_evidence_is_added_and_near_misses_go_when_no_longer_open(engine):
    gid = gl.add_gap(engine, "col", "Q?", "limitation")
    gl.set_library_status(engine, gid, status="open", docs_checked=3,
                          evidence=[dict(source="D", point_id="9", page=0, relation="near_miss", reason="r")])
    gl.add_library_evidence(engine, gid, status="partly", new_docs=1,
                            evidence=[dict(source="New", point_id="5", page=2, relation="partly", reason="part")])
    g = gl.get_gap(engine, gid)
    assert (g["library_status"], g["library_docs_checked"]) == ("partly", 4)
    assert [(e["source"], e["relation"]) for e in gl.evidence(engine, gid, origin="library")] == [("New", "partly")]
    gl.add_library_evidence(engine, gid, status="partly", new_docs=1, evidence=[])        # nothing new
    assert gl.get_gap(engine, gid)["library_docs_checked"] == 5 and len(gl.evidence(engine, gid)) == 1


def test_what_the_updater_has_to_do():
    library = [{"source": "A", "content_hash": "h1"}, {"source": "B", "content_hash": "h2"},
               {"source": "C", "content_hash": "h3"}, {"source": "D", "content_hash": "h4"}]
    ledger = [{"source": "A", "content_hash": "h1", "status": "done"},
              {"source": "B", "content_hash": "old", "status": "done"},        # re-uploaded with new content
              {"source": "D", "content_hash": "h4", "status": "failed"},       # failed before: not retried forever
              {"source": "X", "content_hash": "hx", "status": "done"}]         # deleted from the library
    todo = gl.pending_documents(library, ledger)
    assert todo == {"new": ["B", "C"], "removed": ["X"]}


def test_deleting_a_document_takes_back_its_evidence_and_its_count(engine):
    # a deleted document no longer "addresses" a question and is no longer counted among the documents checked
    m = dict(point_id="p", page=0, signal="s", claim="c", passage="t", judge="supported")
    g1 = gl.add_gap(engine, "col", "Answered only by New?", "limitation")
    gl.add_mention(engine, g1, "col", source="A", **m)
    gl.set_library_status(engine, g1, status="partly", docs_checked=3,
                          evidence=[dict(source="B", point_id="1", page=0, relation="partly", reason="r")])
    gl.add_library_evidence(engine, g1, status="addressed", new_docs=1,
                            evidence=[dict(source="New", point_id="9", page=0, relation="addresses", reason="r")])
    g2 = gl.add_gap(engine, "col", "Untouched?", "limitation")
    gl.add_mention(engine, g2, "col", source="A", **m)
    gl.set_library_status(engine, g2, status="open", docs_checked=4, evidence=[])
    g3 = gl.add_gap(engine, "col", "Raised by New and C?", "limitation")
    gl.add_mention(engine, g3, "col", source="New", **m)
    gl.add_mention(engine, g3, "col", source="C", **m)
    gl.set_library_status(engine, g3, status="open", docs_checked=3, evidence=[])
    gl.set_document(engine, "col", "New", content_hash="h", status="done")     # the ledger had processed it
    gl.delete_for_source(engine, "col", "New")
    a, b, c = gl.get_gap(engine, g1), gl.get_gap(engine, g2), gl.get_gap(engine, g3)
    assert (a["library_status"], a["library_docs_checked"]) == ("partly", 3)      # back to what B says
    assert (b["library_status"], b["library_docs_checked"]) == ("open", 3)        # counted it: one fewer
    assert c["library_docs_checked"] == 3                                         # raised it: never counted it


def test_clearing_a_document_never_processed_changes_no_counts(engine):
    # extracting a new document first clears its (empty) ledger data; that must not count it off any question
    g = gl.add_gap(engine, "col", "Q?", "limitation")
    gl.add_mention(engine, g, "col", source="A", point_id="p", page=0, signal="s", claim="c", passage="t",
                   judge="supported")
    gl.set_library_status(engine, g, status="open", docs_checked=24, evidence=[])
    gl.delete_for_source(engine, "col", "Brand new")                          # no ledger row: never counted
    assert gl.get_gap(engine, g)["library_docs_checked"] == 24
    gl.set_document(engine, "col", "Processed", content_hash="h", status="done")
    gl.delete_for_source(engine, "col", "Processed")                          # was counted: counted off
    assert gl.get_gap(engine, g)["library_docs_checked"] == 23


def test_offline_papers_are_stored_and_replaced_like_online_ones(engine):
    gid = gl.add_gap(engine, "col", "Q?", "limitation")
    gl.set_web_status(engine, gid, status="partly", query="q", sources=["alexandria", "openalex"],
                      evidence=[dict(origin="alexandria", source="10.1/x", title="T", relation="partly", reason="r",
                                     text="abs")])
    assert [e["origin"] for e in gl.evidence(engine, gid)] == ["alexandria"]
    gl.set_web_status(engine, gid, status="nothing_found", query="q", sources=["openalex"], evidence=[])
    assert gl.evidence(engine, gid) == []


def test_the_evidence_of_many_questions_is_read_at_once(engine):
    # a report reads the evidence of all its questions at once (one at a time grows with the library)
    a, b, c = (gl.add_gap(engine, "lib", f"Q{k}?", "limitation") for k in range(3))
    gl.set_library_status(engine, a, status="partly", docs_checked=2,
                          evidence=[{"source": "D1", "point_id": "1", "page": 0, "relation": "partly", "reason": "r"}])
    gl.set_library_status(engine, b, status="addressed", docs_checked=2,
                          evidence=[{"source": "D2", "point_id": "2", "page": 1, "relation": "addresses", "reason": "s"},
                                    {"source": "D3", "point_id": "3", "page": 2, "relation": "partly", "reason": "t"}])
    many = gl.evidence_many(engine, [a, b, c])
    assert many == {a: gl.evidence(engine, a), b: gl.evidence(engine, b), c: []}
    assert gl.evidence_many(engine, []) == {}
