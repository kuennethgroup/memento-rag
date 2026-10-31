# Earlier answers as the answer model reads them: without their old citations, so a follow-up never copies
# "(file.pdf, p. 4, §2.6)" as its own citation style; the router keeps the document names.
from memento.query import presentation, texts
from memento.query.scope import conversation
from memento.retrieval.answer_pipeline import HISTORY_NOTE

HITS = [{"source": "Diss_Mueller", "page": 11}, {"source": "openalex_W123", "page": 0}, {"source": "Talk--X>y", "page": 4}]


def test_answer_copy_drops_old_citations_and_the_marker():
    answer = "Density was 3.9 g/cm3 [1], hardness 18 GPa [3]; see also [2]." + presentation.sources_marker(HITS, {0, 2})
    h = conversation.history_text(answer, cite=False)
    assert h == "Density was 3.9 g/cm3, hardness 18 GPa; see also."
    assert "Diss_Mueller" not in h and "details" not in h


def test_answer_copy_ignores_stored_names():
    names = {"1": "Example et al. - 2025 - Study.pdf, p. 4, §2.6"}
    assert conversation.history_text("Pixels were re-inserted [1].", names, cite=False) == "Pixels were re-inserted."


def test_router_copy_still_names_the_documents():
    names = {"1": "Doe 2026, p. 6, §2.4.2"}
    assert conversation.history_text("The MMD is a histogram [1].", names) == \
        "The MMD is a histogram (Doe 2026, p. 6, §2.4.2)."


def test_history_rule_says_how_to_cite_and_matches_the_rq3_builder():
    assert texts.HISTORY_RULE == HISTORY_NOTE
    assert texts.HISTORY_RULE.startswith("\nEarlier turns of this conversation are context only, not evidence")
    assert "only as [N]" in texts.HISTORY_RULE and "never by document name, page or section" in texts.HISTORY_RULE
