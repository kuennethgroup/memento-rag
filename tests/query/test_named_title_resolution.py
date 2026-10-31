# Naming a document by its listed or full title.
from sqlalchemy import create_engine

import openwebui.functions.query_pipe as qp
from memento.library import documents as db
from memento.query.scope.paper_resolve import PaperResolver
from memento.query import library_state
from memento.query.scope import documents

BO = "Taking the Human Out of the Loop: A Review of Bayesian Optimization"
JANUS = "Fracture Mechanics of Nanostructured Polymer Blends with Janus Particles"
FR = "Bio-basierte Flammschutzsysteme aus phosphatierten Polyaromaten und Aminosäuren für Epoxide"
LOVERA = "Morphology-Charge Storage Relationships in Polymer Electrets"


def _library(monkeypatch, rows: dict) -> list:
    """A library {source: (first-page title, listed title)} as the pipe reads it; returns its sources."""
    e = create_engine("sqlite://")
    db.ensure_tables(e)
    for src, (parsed, listed) in rows.items():
        db.add_document(e, collection="lib", entry={"source": src, "title": listed, "parsed_title": parsed,
                                                    "pdf_title": None}, content_hash=src, uploaded_by="u",
                        output_root=None)
    monkeypatch.setattr(qp, "_db_engine", lambda: e)
    monkeypatch.setattr(documents, "_domain_graph", lambda domain: None)
    for cache in ("ALIASES", "TITLES", "LIB_TITLES"):
        monkeypatch.setattr(library_state, cache, {})
    return sorted(library_state.load_documents("lib", None, qp._db_engine))


STUDY = {"Diss_Lovera": (LOVERA, f"Dissertation: {LOVERA}"),
         "Vortrag_Lovera": (LOVERA, f"Talk (2019): {LOVERA}"),
         "Dissertation FR": (FR, f"Dissertation: {FR}"),
         "Dissertation_Mira Hartwig": (JANUS, f"Dissertation: {JANUS}")}


def test_a_file_name_matches_only_as_whole_words():
    r = PaperResolver([("Dissertation FR", FR), ("Dissertation_Mira Hartwig", JANUS)])
    assert r.resolve(f"Summarize Dissertation: {JANUS}").source == "Dissertation_Mira Hartwig"
    assert r.resolve("Summarize Dissertation FR").source == "Dissertation FR"


def test_the_listed_title_is_one_of_a_documents_names(monkeypatch):
    sources = _library(monkeypatch, STUDY)
    assert f"Talk (2019): {LOVERA}" in documents._doc_aliases(sources, None)["Vortrag_Lovera"]


def test_a_dissertation_and_its_talk_with_one_title_are_told_apart_by_the_listed_title(monkeypatch):
    sources = _library(monkeypatch, STUDY)
    assert documents.resolve_reference(f"Summarize Dissertation: {LOVERA}", sources, None) == "Diss_Lovera"
    assert documents.resolve_reference(f"Tell me about Talk (2019): {LOVERA}", sources, None) == "Vortrag_Lovera"
    # the shared title alone stays ambiguous
    assert documents.resolve_reference(f"Tell me about {LOVERA}", sources, None) is None


def test_a_listed_dissertation_title_is_not_taken_for_another_file_name(monkeypatch):
    sources = _library(monkeypatch, STUDY)
    assert documents.resolve_reference(f"Tell me about Dissertation: {JANUS}", sources, None) == "Dissertation_Mira Hartwig"


def test_a_full_title_containing_the_and_review_is_found(monkeypatch):
    sources = _library(monkeypatch, {"GpyOpt_review-Paper": (BO, BO), "Ax": ("Ax: adaptive experimentation", "Ax")})
    q = "Tell me about Taking the Human Out of the Loop: A review of Bayesian Optimization"
    assert documents.document_phrases(q) == ["the Loop: A review"]
    assert documents.resolve_reference(q, sources, None) == "GpyOpt_review-Paper"
