# The block under an answer: the count is what the search reached ("Searched the library" when capped), the wording
# follows separate facts (cited or not, weak scores or not, closest passages or none), a list counts only "cites
# nothing", and a low score is never written as a statement about meaning.
import types

import pytest

import openwebui.functions.query_pipe as qp
from memento.retrieval.search import related_heads
from memento.query import presentation, request, texts
from memento.query.flows import answer as answer_flow

H = lambda src, score, text="A passage about it.": {"source": src, "page": 1, "text": text, "rerank_score": score}


# ── the count ────────────────────────────────────────────────────────────────────────────────────────────────────────

class _Client:
    def __init__(self, n, blank=0):
        self.n, self.blank = n, blank

    def query_points_groups(self, **kw):
        groups = [types.SimpleNamespace(hits=[types.SimpleNamespace(
            payload={"source": f"D{i}", "page": 0, "type": "text", "text": "" if i < self.blank else "t"}, score=0.5,
            id=i)]) for i in range(min(self.n, kw["limit"]))]
        return types.SimpleNamespace(groups=groups)


def test_the_grouped_search_says_when_it_hit_its_cap():
    assert related_heads(_Client(35), "c", [0.0], None, 200).capped is False
    heads = related_heads(_Client(200, blank=1), "c", [0.0], None, 200)
    assert len(heads) == 199 and heads.capped is True             # capped although one group had no text


def test_the_count_is_taken_before_reranking(monkeypatch):
    import memento.retrieval.search as search
    monkeypatch.setattr(search, "related_heads", lambda *a, **k: search.RelatedHeads(
        [H(f"D{i}", 0.5) for i in range(35)], capped=False))
    monkeypatch.setattr(search, "rerank", lambda q, hits, *a, **k: hits[:30])
    out = answer_flow.related_heads(qp.Pipe().valves, None, [0.0], "q", None)
    assert len(out) == 30 and out.found == 35 and out.capped is False


# ── the wording ──────────────────────────────────────────────────────────────────────────────────────────────────────

def footer(**kw):
    base = dict(items=[], titles={}, searched=35, cited=True, weak=False, files_url="", list_answer=False)
    base.update(kw)
    return presentation.related_footer(**base)


def test_cited_and_strong_lists_other_documents():
    md = footer(items=[H("B", 0.4)], titles={"B": "Title B"})
    assert "Other documents with related passages (1), not used in this answer" in md


def test_cited_but_weak_says_so_and_never_claims_nothing():
    md = footer(cited=True, weak=True, items=[H("B", 0.05)], titles={"B": "Title B"})
    assert "The closest matches in the library were weak. Check the cited passages." in md
    assert "came close" not in md and "answers this" not in md


def test_a_list_answer_that_cites_is_never_weak():
    md = footer(cited=True, weak=True, list_answer=True, items=[H("B", 0.05)], titles={"B": "Title B"})
    assert "weak" not in md and "came close" not in md


def test_no_citations_with_strong_matches():
    md = footer(cited=False, weak=False, items=[H("B", 0.7)], titles={"B": "Title B"})
    assert "This answer cites no sources. The closest passages (1)" in md and "came close" not in md


def test_no_citations_weak_with_closest_passages():
    md = footer(cited=False, weak=True, items=[H("B", 0.05)], titles={"B": "Title B"})
    assert "Searched 35 documents: the closest passages (1)" in md


def test_no_citations_weak_nothing_close():
    md = footer(cited=False, weak=True)
    assert "Searched 35 documents: no passage came close to this question." in md


def test_capped_or_unknown_says_the_library():
    assert "Searched the library: no passage came close to this question." in footer(cited=False, weak=True, searched=None)
    assert "Searched the library: the closest passages (1)" in footer(cited=False, weak=True, searched=None,
                                                                       items=[H("B", 0.05)], titles={"B": "T"})


def test_nothing_to_say_is_empty():
    assert footer(cited=True, weak=False) == ""


@pytest.mark.parametrize("kw,de", [
    (dict(cited=False, weak=True), "35 Dokumente durchsucht: kein Abschnitt kam dieser Frage nah genug."),
    (dict(cited=False, weak=True, searched=None), "Bibliothek durchsucht: kein Abschnitt kam dieser Frage nah genug."),
    (dict(cited=False, weak=True, items=[H("B", 0.05)], titles={"B": "T"}), "35 Dokumente durchsucht: die nächsten "
                                                                             "Abschnitte (1)"),
    (dict(cited=False, weak=False, items=[H("B", 0.7)], titles={"B": "T"}), "Diese Antwort zitiert keine Quellen. Die "
                                                                             "nächsten Abschnitte (1)"),
    (dict(cited=True, weak=True, items=[H("B", 0.05)], titles={"B": "T"}), "Die nächsten Treffer in der Bibliothek waren "
                                                                            "schwach. Bitte prüfen Sie die zitierten Abschnitte."),
])
def test_german(kw, de):
    token = request.LANG.set("de")
    try:
        assert de in texts.tr(footer(**kw))
    finally:
        request.LANG.reset(token)


# ── nothing found ────────────────────────────────────────────────────────────────────────────────────────────────────

def test_nothing_found_texts_say_what_the_search_found():
    assert texts.NOTHING_IN_LIBRARY_MSG.startswith("I could not find anything in the library that answers this.")
    assert "The search found no passage that answers the user's question." in answer_flow.NOTHING_FOUND_PROMPT
    assert "library has no passage" not in answer_flow.NOTHING_FOUND_PROMPT
    assert "does not cover" not in answer_flow.NOTHING_FOUND_PROMPT
    token = request.LANG.set("de")
    try:
        assert texts.tr(texts.NOTHING_IN_LIBRARY_MSG).startswith("Ich konnte in der Bibliothek nichts finden, das das beantwortet.")
    finally:
        request.LANG.reset(token)
