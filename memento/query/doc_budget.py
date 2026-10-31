# memento/query/doc_budget.py: which passages of one document go to the answer model
"""Research gaps and summaries of long documents send a budgeted selection of a document's passages, chosen by priority
and given to the model in reading order. One rule for every document: a normal paper fits completely, a very long one
gets its most important passages first. The counts go to the log, not to the reader."""
from __future__ import annotations

GAP_PRIORITY = ("limitation", "untested_condition", "future_work", "uncertainty")
CONCLUSION_LIKE = ("conclusion", "conclusions", "summary", "outlook", "concluding remarks", "summary and outlook",
                   "conclusion and outlook", "conclusions and outlook")


def _tokens(p: dict, chars_per_token: float, extra_chars: int = 0) -> float:
    return (len(p.get("text") or "") + extra_chars) / chars_per_token


def _fill(ordered: list, budget_tokens: float, chars_per_token: float, extra_chars: int = 0) -> tuple[list, float]:
    """Take whole passages in the given order while they fit (a passage too big for what is left is skipped, a
    smaller later one may still fit). extra_chars: what each passage costs besides its text (its [N] label)."""
    chosen, used = [], 0.0
    for p in ordered:
        t = _tokens(p, chars_per_token, extra_chars)
        if used + t <= budget_tokens:
            chosen.append(p)
            used += t
    return chosen, used


def _reading_order(chosen: list, all_passages: list) -> list:
    rank = {id(p): i for i, p in enumerate(all_passages)}
    return sorted(chosen, key=lambda p: (p.get("page", 0), rank.get(id(p), 0)))


def select_gap_passages(gap_chunks: list, section_chunks: list, budget_tokens: float = 20000,
                        chars_per_token: float = 3) -> tuple[list, dict]:
    """Gap-tagged passages by type (limitation, untested condition, future work, uncertainty, other), then the
    Discussion/Conclusion/Outlook passages that carry no tag, up to the budget; returned in reading order."""
    seen = {((g.get("page"), (g.get("text") or "").strip())) for g in gap_chunks}
    extra = [s for s in section_chunks if (s.get("page"), (s.get("text") or "").strip()) not in seen]
    rank = {t: i for i, t in enumerate(GAP_PRIORITY)}
    by_type = sorted(gap_chunks, key=lambda g: (rank.get(g.get("gap_type") or "", len(GAP_PRIORITY)), g.get("page", 0)))
    ordered = by_type + sorted(extra, key=lambda s: s.get("page", 0))
    chosen, used = _fill(ordered, budget_tokens, chars_per_token)
    everything = sorted(gap_chunks + extra, key=lambda p: p.get("page", 0))
    return _reading_order(chosen, everything), {"candidates": len(ordered), "chosen": len(chosen), "tokens": round(used)}


def select_summary_passages(passages: list, budget_tokens: float = 40000,
                            chars_per_token: float = 3, per_passage_chars: int = 0) -> tuple[list, dict]:
    """A long document's passages for a summary: abstract, conclusion-like sections, introduction, the first passage
    of every other section, then the rest by an even stride over the document; up to the budget, in reading order."""
    def name(p):
        return (p.get("section") or "").strip().lower()

    idx = list(range(len(passages)))
    abstract = [i for i in idx if name(passages[i]) == "abstract"]
    conclusion = [i for i in idx if name(passages[i]) in CONCLUSION_LIKE]
    intro = [i for i in idx if name(passages[i]) == "introduction"]
    firsts, seen = [], set()
    for i in idx:
        n = name(passages[i])
        if n and n not in seen:
            seen.add(n)
            firsts.append(i)
    taken = set(abstract + conclusion + intro + firsts)
    rest = [i for i in idx if i not in taken]
    # even spread: every k-th of the rest first, then the ones in between, so a partial fill still covers the document
    spread = []
    for step in (16, 8, 4, 2, 1):
        for i in rest[::step]:
            if i not in spread:
                spread.append(i)
    order, placed = [], set()
    for i in abstract + conclusion + intro + firsts + spread:
        if i not in placed:
            placed.add(i)
            order.append(i)
    chosen, used = _fill([passages[i] for i in order], budget_tokens, chars_per_token, per_passage_chars)
    return _reading_order(chosen, passages), {"total": len(passages), "chosen": len(chosen), "tokens": round(used)}
