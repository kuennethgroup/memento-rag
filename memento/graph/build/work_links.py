"""G1 work-level links between documents (no LLM): SAME_WORK from the
document registry's duplicate/chapter/defense grouping plus near-identical
titles, and SAME_AUTHOR from the doctoral person folder."""
from __future__ import annotations

from itertools import combinations

from memento.graph.documents import is_person_folder
from memento.graph.store import GraphEdge

_KIND = (("duplicate_of", "duplicate"), ("parent", "chapter"), ("parent_diss", "defense"))
# a chapter sharing almost no text with its assigned dissertation was grouped by folder heuristics, not because it is
# part of that work
MIN_CHAPTER_OVERLAP = 0.1


def _edge(a: str, b: str, etype: str, props: dict) -> GraphEdge:
    s, e = sorted((a, b))
    return GraphEdge(s, "Paper", e, "Paper", etype, props)


def work_link_edges(registry: dict[str, dict], present: set[str]) -> list[GraphEdge]:
    edges: list[GraphEdge] = []
    same_work: set[tuple[str, str]] = set()
    for src, info in registry.items():
        if src not in present:
            continue
        for field, kind in _KIND:
            tgt = info.get(field)
            if field == "parent" and info.get("overlap_with_parent", 1.0) < MIN_CHAPTER_OVERLAP:
                continue
            if tgt and tgt in present and tgt != src:
                edges.append(_edge(src, tgt, "SAME_WORK", {"kind": kind}))
                same_work.add(tuple(sorted((src, tgt))))
                break
    by_student: dict[str, list[str]] = {}
    for src, info in registry.items():
        st = info.get("student")
        if src in present and is_person_folder(st):
            by_student.setdefault(st, []).append(src)
    for st, docs in by_student.items():
        for a, b in combinations(sorted(docs), 2):
            if (a, b) not in same_work:
                edges.append(_edge(a, b, "SAME_AUTHOR", {"student": st}))
    return edges


MIN_TITLE_LENGTH_RATIO = 0.8


def title_duplicate_edges(papers: list[dict], threshold: float = 0.95,
                          min_tokens: int = 5) -> list[GraphEdge]:
    from memento.graph.build.build_graph import _normalize_title, _token_set_ratio
    rows = [(p["id"], _normalize_title(p.get("title") or ""), p.get("year")) for p in papers]
    rows = [r for r in rows if len(r[1].split()) >= min_tokens]
    out = []
    for (ia, ta, ya), (ib, tb, yb) in combinations(rows, 2):
        if ya is not None and yb is not None and abs(ya - yb) > 1:
            continue
        na, nb = len(ta.split()), len(tb.split())
        if min(na, nb) / max(na, nb) < MIN_TITLE_LENGTH_RATIO:   # subset titles score 1.0 on token sets
            continue
        s = _token_set_ratio(ta, tb)
        if s >= threshold:
            out.append(_edge(ia, ib, "SAME_WORK", {"kind": "title_duplicate", "score": round(s, 3)}))
    return out
