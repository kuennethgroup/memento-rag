# Soft per-paper cap on the passages sent to the answer model.
from memento.retrieval.search import balance_papers


def _ranked(*rows):
    """(source, score) rows → hits in the reranker's order (best first)."""
    return [{"source": s, "rerank_score": sc, "text": f"{s} {i}"} for i, (s, sc) in enumerate(rows)]


def _count(hits, src):
    return sum(h["source"] == src for h in hits)


def test_a_competitive_other_paper_takes_the_dominant_papers_weakest_slots():
    ranked = _ranked(*[("A", 0.9 - i * 0.05) for i in range(8)], ("B", 0.40), ("B", 0.38))
    out = balance_papers(ranked, top_n=8, cap=6, ratio=0.5)
    assert len(out) == 8 and _count(out, "A") == 6 and _count(out, "B") == 2
    assert [h["rerank_score"] for h in out] == sorted((h["rerank_score"] for h in out), reverse=True)


def test_a_weak_other_passage_does_not_displace_a_strong_one():
    # the replaced passage scores 0.35; a newcomer needs at least 0.175
    ranked = _ranked(*[("A", 0.9 - i * 0.05) for i in range(12)], ("B", 0.10), ("C", 0.05))
    out = balance_papers(ranked, top_n=12, cap=6, ratio=0.5)
    assert _count(out, "A") == 12


def test_swapping_stops_at_the_first_newcomer_that_falls_short():
    # A has 9 in the top 9; B's 0.30 and 0.28 qualify against A's weakest (0.50, 0.55); C's 0.05 does not
    ranked = _ranked(*[("A", 0.9 - i * 0.05) for i in range(9)], ("B", 0.30), ("B", 0.28), ("C", 0.05))
    out = balance_papers(ranked, top_n=9, cap=6, ratio=0.5)
    assert _count(out, "A") == 7 and _count(out, "B") == 2 and _count(out, "C") == 0


def test_a_single_relevant_paper_keeps_every_slot():
    ranked = _ranked(*[("A", 0.5 - i * 0.02) for i in range(20)])
    assert balance_papers(ranked, top_n=20, cap=6, ratio=0.5) == ranked


def test_newcomers_respect_the_cap_too_and_two_dominant_papers_are_both_balanced():
    ranked = _ranked(*[("A", 0.9)] * 5, *[("B", 0.8)] * 5, *[("C", 0.7)] * 10)
    out = balance_papers(ranked, top_n=10, cap=4, ratio=0.5)
    assert len(out) == 10 and _count(out, "A") == 4 and _count(out, "B") == 4 and _count(out, "C") == 2


def test_off_short_lists_and_reranker_fallback_return_the_plain_top():
    ranked = _ranked(*[("A", 0.9 - i * 0.05) for i in range(8)], ("B", 0.40))
    assert balance_papers(ranked, top_n=5, cap=0, ratio=0.5) == ranked[:5]          # cap 0 = off
    assert balance_papers(ranked[:3], top_n=5, cap=1, ratio=0.5) == ranked[:3]      # fewer than top_n: nothing to choose
    fallback = [{**h, "rerank_fallback": True} for h in ranked]                   # no scores to compare
    assert balance_papers(fallback, top_n=5, cap=2, ratio=0.5) == fallback[:5]


def test_share_evenly_gives_every_file_its_share():
    # Document Chat, several files, none named: "the attached files" means all of them, so each file gets an equal share
    # of the passages, the rest goes to the best leftovers; ordered by score
    from memento.retrieval.search import share_evenly

    def _h(src, score):
        return {"source": src, "rerank_score": score, "text": f"{src} {score}"}
    ranked = [_h("A", 0.9 - i / 100) for i in range(10)] + [_h("B", 0.5 - i / 100) for i in range(10)] \
        + [_h("C", 0.2 - i / 100) for i in range(2)]
    out = share_evenly(ranked, top_n=8, sources=["A", "B", "C"])
    from collections import Counter
    c = Counter(h["source"] for h in out)
    assert len(out) == 8 and c["C"] == 2 and c["B"] >= 2 and c["A"] >= 2      # C has only 2: the rest is refilled
    assert [h["rerank_score"] for h in out] == sorted((h["rerank_score"] for h in out), reverse=True)
    assert share_evenly(ranked[:3], top_n=8, sources=["A", "B"]) == ranked[:3]   # fewer than top_n: all of them
