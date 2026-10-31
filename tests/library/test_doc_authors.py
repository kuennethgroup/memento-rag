# Which documents name a person as an author. The name is matched against each document's front matter, so an author
# deep in a long author list (an initial, an affiliation letter glued to the name) is found where the meaning-based
# search does not reach.
import pytest
from sqlalchemy import create_engine

from memento.library import front_matter as da
from memento.config.paths import PROJECT_ROOT
from _owui_loader import load_function
from memento.query.flows import library_listing, lists
from memento.query import search_step, workers

# a real front matter, as parsed
FIELD_CYCLING = ("# Impact of field cycling on $\\mathrm{HfO_2}$ based non-volatile memory devices\n\nU. Hoelzer "
                 "$^{a,\\dagger}$ , M. Kovačić $^{a}$ , E. D. Hartley $^{f}$ , S. Wisniewski $^{d}$ , R. van Dalen $^{d}$ , "
                 "A. Kessler $^{e}$ , C. Brückner $^{e}$ , T. Novakowski $^{a,c}$ $^{a}$ NaMLab gGmbH, 01197 Dresden")
SINGLE_DOMAIN = ("# Evidence of single domain switching in Hafnium Oxide based FeFETs\n\nH. Begovica, M. Kovacica, "
                 "S. Grübera, S. Lewandowskyd, J. Grüberb, P. Jankowskib, C. Vogta, T. Brandta, A. Kesslere, "
                 "C. Brücknere, R. van Dalend, U. Hölzera, T. Novakowskia,c\n"
                 # real affiliation lines: a letter glued to a name counts only with its affiliation
                 "aNaMLab gGMBH, bFraunhofer IPMS-CNT, cIHM TU-Dresden, dGLOBALFOUNDRIES, Dresden, Germany,\n"
                 "eHAW-München, München, Germany; fDEIB, Politecnico di Milano, Milan, Italy")
CHARACTERISTICS = ("# Characteristics of Low-Energy Phases of Hafnia and Zirconia\n\nLuis Moreira Santos, Richard "
                   "Gassner, Christopher Brueckner, and Alfred Kessler\\*\n\nA systematic simulation study")
BOTORCH = "# BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization\n\nMaximilian Brenner, Brian Holloway"


@pytest.mark.parametrize("front,full", [(FIELD_CYCLING, "possible_identity"), (SINGLE_DOMAIN, "possible_identity"),
                                        (CHARACTERISTICS, "author")])
def test_every_form_of_the_name_matches(front, full):
    # an initial only is possibly the same person, not certainly the author
    assert da.match_person(front, "Christopher Brueckner") == full
    assert da.match_person(front, "Christopher Brückner") == full
    assert da.names_person(front, "Brueckner")                                   # the surname asked: it decides


def test_other_people_and_initials_do_not_match():
    assert not da.names_person(BOTORCH, "Christopher Brueckner")
    assert da.match_person(SINGLE_DOMAIN, "J. Grüber") == "possible_identity"
    assert da.match_person(SINGLE_DOMAIN, "S. Grueber") == "possible_identity"
    assert da.match_person(SINGLE_DOMAIN, "K. Grüber") == "no"                  # a shared surname, other initials
    assert not da.names_person(CHARACTERISTICS, "Jonas Weidner")
    assert not da.names_person(CHARACTERISTICS, "")


def test_front_matter_is_the_head_before_the_references():
    md = "# Title\n\nA. Author, B. Author\n\n![](images/x.jpg)\n\nBody text.\n\n# References\n\n[1] C. Brückner, cited."
    front = da.front_matter(md)
    assert "A. Author" in front and "images/x.jpg" not in front and "cited" not in front
    assert len(da.front_matter("x " * 5000)) <= da.FRONT_CHARS


def _library_rows(engine, rows):
    """The library's rows the author lists belong to: a list is stored only for the version in the library (I3)."""
    from memento.library import documents as ldb
    ldb.ensure_tables(engine)
    for src, h in rows:
        ldb.add_document(engine, collection="c", entry={"source": src, "title": src}, content_hash=h, uploaded_by="u",
                         output_root=None)


def test_documents_by_a_person_from_the_stored_front_matter():
    engine = create_engine("sqlite://")
    da.ensure_table(engine)
    docs = [("Impact of field cycling", FIELD_CYCLING), ("Evidence", SINGLE_DOMAIN),
            ("Characteristics", CHARACTERISTICS), ("BoTorch", BOTORCH)]
    _library_rows(engine, [(src, "h-" + src) for src, _ in docs])
    for src, front in docs:
        da.put(engine, collection="c", source=src, content_hash="h-" + src, front=front)
    fronts = da.load(engine, "c")
    assert set(fronts) == {"Impact of field cycling", "Evidence", "Characteristics", "BoTorch"}
    assert da.documents_by(fronts, "Christopher Brueckner") == ["Characteristics"]
    assert da.classify(fronts, "Christopher Brueckner")["possible_identity"] == ["Evidence", "Impact of field cycling"]
    _library_rows(engine, [("BoTorch", "h2")])                                                       # replaced
    da.put(engine, collection="c", source="BoTorch", content_hash="h2", front="# BoTorch\n\nC. Brueckner")
    assert "BoTorch" in da.documents_by(da.load(engine, "c"), "Brueckner")


def test_front_matter_stops_at_the_abstract():
    # a model named after a person in an abstract is not an author
    md = ("# A computational study of hafnia-based ferroelectric memories\n\nA. Kessler, C. Brückner\n\n"
          "Abstract\n\nWe use the Landau-Khalatnikov model.\n\n# 1. Introduction\n\nText.")
    front = da.front_matter(md)
    assert "Brückner" in front and "Landau" not in front
    assert not da.names_person(front, "Landau") and da.match_person(front, "Christopher Brueckner") == "possible_identity"


# the router names the person whose documents are asked for (additive field)
def test_router_author_field_is_coerced():
    from memento.query.router import RouterResponse
    assert RouterResponse(author="  Christopher Brueckner ").author == "Christopher Brueckner"
    for empty in ("", "none", "None", "null", None, 3):
        assert RouterResponse(author=empty).author == ""
    assert len(RouterResponse(author="x" * 500).author) <= 100
    assert RouterResponse().author == ""


def test_router_prompt_asks_for_the_author_and_keeps_methods_out():
    from memento.query.router import ROUTER_SYSTEM_PROMPT as P
    assert '7. "author"' in P and '"author": ""' in P
    assert "Landau-Khalatnikov" in P and "is not an author" in P


# ── the Query pipe's answer for "documents by a person" ──
import openwebui.functions.query_pipe as qp
from memento.ingestion.upload import document


@pytest.mark.parametrize("q", ["Are there any documents from Christopher Brueckner?", "Gibt es Paper von Alfred Kessler?",
                               "List the publications by Thomas Novakowski", "Which papers did Ravi Ramaswami co-author?",
                               "How many papers did Christopher Brückner write?", "what can you find regarding jonas weidner?"])
def test_a_plain_request_for_a_persons_documents_has_no_topic(q):
    person = {"Are": "Christopher Brueckner", "Gib": "Alfred Kessler", "Lis": "Thomas Novakowski", "Whi": "Ravi Ramaswami",
              "How": "Christopher Brückner", "wha": "jonas weidner"}[q[:3]]
    assert not library_listing.author_topic(q, person)


def test_a_topic_beyond_the_person_is_seen():
    assert library_listing.author_topic("Which of Brueckner's papers are about hafnia?", "Brueckner")
    assert library_listing.author_topic("Welche Arbeiten von Alfred Kessler behandeln Dotierung?", "Alfred Kessler")


def test_author_listing_in_both_languages():
    links = {"A": "/files/view/A/0", "B": "/files/view/B/0"}
    en = library_listing.author_listing("Christopher Brueckner", ["B", "A"], links, de=False, missing=0)
    assert en.startswith("**2 documents** in the library name Christopher Brueckner as an author:")
    assert "[A](/files/view/A/0)" in en and en.index("[A]") < en.index("[B]")
    one = library_listing.author_listing("C. K.", ["A"], links, de=False, missing=3)
    assert one.startswith("**1 document** in the library names C. K. as an author:")
    assert "3 documents" in one and "not checked" in one
    none = library_listing.author_listing("Jonas Weidner", [], {}, de=False, missing=0)
    assert none.startswith("No document in the library names **Jonas Weidner** as an author.")
    de = library_listing.author_listing("Alfred Kessler", ["A", "B"], links, de=True, missing=1)
    assert de.startswith("**2 Dokumente** in der Bibliothek nennen Alfred Kessler als Autor:") and "nicht geprüft" in de
    assert "—" not in en + one + none + de


def test_pipe_answers_a_persons_documents_from_the_front_matter():
    import inspect
    src = inspect.getsource(library_listing.person_reply)
    assert "author_listing(" in src and "doc_authors.classify(" in src and "author_topic(" in src


def test_store_front_reads_the_parsed_text_and_never_fails(tmp_path):
    engine = create_engine("sqlite://")
    _library_rows(engine, [("Doc", "h")])
    (tmp_path / "Doc" / "vlm").mkdir(parents=True)
    (tmp_path / "Doc" / "vlm" / "Doc.md").write_text(CHARACTERISTICS)
    assert da.store_front(engine, collection="c", source="Doc", out_dir=tmp_path, content_hash="h") is True
    assert da.documents_by(da.load(engine, "c"), "Christopher Brueckner") == ["Doc"]
    assert da.store_front(engine, collection="c", source="Nope", out_dir=tmp_path, content_hash=None) is False
    assert da.store_front(None, collection="c", source="Doc", out_dir=tmp_path, content_hash="h") is False   # no engine


def test_a_name_with_brackets_finds_its_parsed_text(tmp_path):
    # a source name with pattern characters ("[1]", "*", "?") is matched as text, not as a file pattern
    for source in ("DIssertation_NW-Endfassung[1]", "Diss*Draft?", "A [draft] (v2)"):
        (tmp_path / source / "vlm").mkdir(parents=True)
        (tmp_path / source / "vlm" / f"{source}.md").write_text(CHARACTERISTICS)
        assert da.parsed_text_path(tmp_path, source) == tmp_path / source / "vlm" / f"{source}.md"
        assert da.read_front(tmp_path, source)


def test_a_name_with_brackets_finds_its_first_page_title(tmp_path):
    # the same in the library list's title lookup (memento/library/documents.raw_titles)
    import json
    from memento.library import documents as ll
    source, title = "DIssertation_NW-Endfassung[1]", "Novel expanded and expandable polycarbonate particle foams"
    (tmp_path / source / "vlm").mkdir(parents=True)
    (tmp_path / source / "vlm" / f"{source}_content_list.json").write_text(json.dumps(
        [{"type": "text", "text": title, "text_level": 1, "page_idx": 0}]))
    assert ll.raw_titles(source, tmp_path / "in", tmp_path)[0] == title


def test_uploads_store_the_front_matter():
    import inspect
    import openwebui.functions.ingestion_pipe as ip
    assert "doc_authors.store_front(" in inspect.getsource(document.Ingester.register_document)
    src = open("memento/ingestion/run_pipeline.py").read()
    assert "doc_authors.store_front(" in src


def test_author_with_a_topic_searches_only_their_documents_with_the_list_scan():
    # a person's documents asked about a topic keep the list scan on
    import inspect
    assert "turn.author_docs = theirs" in inspect.getsource(library_listing.person_reply)
    assert "elif turn.author_docs:" in inspect.getsource(search_step.filters)    # the search filter
    # the list scan's filter: their documents (of the kind asked for, when some are)
    scan = inspect.getsource(lists.scan)
    assert "scope_docs = list(turn.author_docs or [])" in scan and "library_listing.author_scope(scope_docs, turn.in_progress)" in scan


def test_author_scope_filter():
    f = library_listing.author_scope(["A", "B"], set())
    assert f.must[0].key == "source" and list(f.must[0].match.any) == ["A", "B"]


def test_a_fresh_library_stores_the_front_matter_of_every_document(tmp_path):
    # the first upload into an empty library builds the list from the search index (documents.rebuild_documents)
    # instead of adding one row; the front matter of every listed document is stored there too
    from memento.library import documents as ldb
    engine = create_engine("sqlite://")
    ldb.ensure_tables(engine)
    for src, text in (("Hafnia", CHARACTERISTICS), ("BoTorch", BOTORCH)):
        (tmp_path / src / "vlm").mkdir(parents=True)
        (tmp_path / src / "vlm" / f"{src}.md").write_text(text)
    # fingerprints from the manifest: an author list is stored only for a known version
    manifest = {"documents": {s: {"content_hash": "h-" + s} for s in ("Hafnia", "BoTorch", "NoText")}}
    n = ldb.rebuild_documents(engine, collection="c", sources={"Hafnia", "BoTorch", "NoText"}, in_dir=tmp_path,
                              out_dir=tmp_path, section_hints={}, manifest=manifest, output_root=None)
    assert n == 3
    assert set(da.load(engine, "c")) == {"Hafnia", "BoTorch"}            # no parsed text: nothing, and no failure
    assert da.documents_by(da.load(engine, "c"), "Christopher Brueckner") == ["Hafnia"]


# ── a library fills its missing author lists itself ──
# Each library's worker fills the missing (or out-of-date) lists from the parsed text when it starts; no model call,
# never fails.

def _parsed(tmp_path, source, text):
    (tmp_path / source / "vlm").mkdir(parents=True)
    (tmp_path / source / "vlm" / f"{source}.md").write_text(text)


def test_missing_author_lists_are_filled_from_the_parsed_text(tmp_path):
    engine = create_engine("sqlite://")
    _library_rows(engine, [("Old", "h1"), ("New", "h2"), ("Unparsed", "h3")])
    _parsed(tmp_path, "Old", CHARACTERISTICS)
    _parsed(tmp_path, "New", BOTORCH)
    da.ensure_table(engine)
    da.put(engine, collection="c", source="New", content_hash="h2", front="kept as stored")
    stored, missing = da.fill_missing(engine, collection="c", out_dir=tmp_path)
    assert (stored, missing) == (1, ["Unparsed"])
    fronts = da.load(engine, "c")
    assert da.documents_by(fronts, "Christopher Brueckner") == ["Old"]
    assert fronts["New"] == "kept as stored"                          # a current list is left alone
    assert da.fill_missing(engine, collection="c", out_dir=tmp_path) == (0, ["Unparsed"])     # nothing twice


def test_an_out_of_date_author_list_is_read_again(tmp_path):
    engine = create_engine("sqlite://")
    _library_rows(engine, [("Doc", "h-new")])
    _parsed(tmp_path, "Doc", CHARACTERISTICS)
    da.ensure_table(engine)
    with engine.begin() as c:                                         # a list of the document's earlier version
        c.execute(da.FRONT.insert().values(collection="c", source="Doc", content_hash="h-old", front="old front"))
    assert da.fill_missing(engine, collection="c", out_dir=tmp_path) == (1, [])
    assert "Brueckner" in da.load(engine, "c")["Doc"]


def test_filling_never_fails(tmp_path):
    assert da.fill_missing(None, collection="c", out_dir=tmp_path) == (0, [])
    assert da.fill_missing(create_engine("sqlite://"), collection="c", out_dir=None) == (0, [])


def test_each_library_fills_its_author_lists_when_its_worker_starts():
    import inspect
    from pathlib import Path
    root = PROJECT_ROOT
    qp = load_function(root / "openwebui/functions/query_pipe.py", "memento_query")
    src = inspect.getsource(workers.warmup_cycle)
    ready = src.index('if not st.get("ready"):')
    assert ready < src.index("library_state.fill_front_matter(valves, engine)") < src.index('st["ready"] = True')
