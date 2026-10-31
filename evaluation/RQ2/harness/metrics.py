"""Per-question measures."""
from __future__ import annotations

from memento.graph.documents import work_id


def _pages(ctx):
    return {(h["source"], h.get("page", 0)) for h in ctx}


def _facts_of(gold):
    return {(g["source"], g["page"]): set((g.get("fact") or "").split("+")) - {""} for g in gold}


def fact_coverage(context, gold_pages):
    facts, seen = _facts_of(gold_pages), _pages(context)
    covered = set().union(*(facts[k] for k in seen if k in facts)) if seen & set(facts) else set()
    a, b = "a" in covered, "b" in covered
    return {"a": a, "b": b, "complete": a and b, "fact_recall": (a + b) / 2,
            "page_recall": len(seen & set(facts)) / len(facts) if facts else 0.0}


def reaches_b(context, gold_pages, origin):
    facts = _facts_of(gold_pages)
    return any("b" in facts.get((h["source"], h.get("page", 0)), set()) for h in context if h.get("origin") == origin)


def stream_reaches_b(stream_pages, gold_pages, k):
    """Did the arm's own ranked stream (before merging with G0) reach a page stating fact b
    within its top k? Spec §3 pilot check, independent of whether G0 already had B."""
    facts = _facts_of(gold_pages)
    return any("b" in facts.get((s, p), set()) for s, p in stream_pages[:k])


def doc_prf(pred_docs, gold_docs, unit, registry):
    f = (lambda d: work_id(d, registry.get(d))) if unit == "works" else (lambda d: d)
    pred, gold = {f(d) for d in pred_docs}, {f(d) for d in gold_docs}
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    return {"p": p, "r": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}


def control_hits(context, gold_pages):
    gold = {(g["source"], g["page"]) for g in gold_pages}
    got = _pages(context) & gold
    return {"hit": bool(got), "recall": len(got) / len(gold) if gold else 0.0}
