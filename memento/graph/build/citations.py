"""Cross-paper citation resolution (pure; no graph access). A reference's raw
bibliography text is matched to in-corpus paper titles by token_set_ratio,
exactly as the original all-pairs resolver, but candidates come from an
inverted token index instead of scanning every paper for every reference."""
from __future__ import annotations

from collections import defaultdict

from memento.graph.build.build_graph import _normalize_title, _token_set_ratio


def _prep(papers, references, min_title_tokens, df_fraction, df_floor):
    refs = [r for r in references if r.get("raw")]
    ref_tokens = [frozenset(_normalize_title(r["raw"]).split()) for r in refs]
    df_cutoff = max(df_floor, int(df_fraction * len(ref_tokens)))
    return refs, ref_tokens, df_cutoff


def _match(ref, cands, threshold):
    raw_norm = _normalize_title(ref["raw"])
    best_id, best = None, 0.0
    for pid, title_norm, year in cands:
        if pid == ref["paper_id"]:
            continue
        if ref.get("year") is not None and year is not None and abs(ref["year"] - year) > 1:
            continue
        s = _token_set_ratio(title_norm, raw_norm)
        if s > best:
            best_id, best = pid, s
    if best_id is not None and best >= threshold:
        return {"ref_id": ref["id"], "citing": ref["paper_id"], "cited": best_id, "score": round(best, 3)}
    return None


def resolve_citations_all_pairs(papers, references, *, threshold=0.92, min_title_tokens=3,
                                df_fraction=0.005, df_floor=20):
    refs, ref_tokens, df_cutoff = _prep(papers, references, min_title_tokens, df_fraction, df_floor)
    cands, n_nd = [], 0
    for p in papers:
        if not p.get("title"):
            continue
        tn = _normalize_title(p["title"])
        toks = frozenset(tn.split())
        if len(toks) < min_title_tokens or sum(1 for rs in ref_tokens if toks <= rs) > df_cutoff:
            n_nd += 1
            continue
        cands.append((p["id"], tn, p.get("year")))
    matches = [m for r in refs if (m := _match(r, cands, threshold))]
    return {"matches": matches, "n_candidates": len(cands), "n_nondistinctive": n_nd, "df_cutoff": df_cutoff}


def resolve_citations(papers, references, *, threshold=0.92, min_title_tokens=3,
                      df_fraction=0.005, df_floor=20):
    refs, ref_tokens, df_cutoff = _prep(papers, references, min_title_tokens, df_fraction, df_floor)
    postings: dict[str, set[int]] = defaultdict(set)          # token -> reference indices
    for i, rs in enumerate(ref_tokens):
        for t in rs:
            postings[t].add(i)
    cands, n_nd = [], 0
    for p in papers:
        if not p.get("title"):
            continue
        tn = _normalize_title(p["title"])
        toks = frozenset(tn.split())
        if len(toks) < min_title_tokens:
            n_nd += 1
            continue
        containing = set.intersection(*(postings.get(t, set()) for t in toks))
        if len(containing) > df_cutoff:
            n_nd += 1
            continue
        cands.append((p["id"], tn, p.get("year"), toks, sum(len(t) for t in toks)))
    title_index: dict[str, list[int]] = defaultdict(list)     # token -> candidate indices
    for ci, c in enumerate(cands):
        for t in c[3]:
            title_index[t].append(ci)
    matches = []
    for r, rs in zip(refs, ref_tokens):
        shared_chars: dict[int, int] = defaultdict(int)       # candidate -> chars of shared tokens
        for t in rs:
            for ci in title_index.get(t, ()):
                shared_chars[ci] += len(t)
        short = [cands[ci][:3] for ci in sorted(ci for ci, h in shared_chars.items() if h * 2 >= cands[ci][4])]
        m = _match(r, short, threshold)
        if m:
            matches.append(m)
    return {"matches": matches, "n_candidates": len(cands), "n_nondistinctive": n_nd, "df_cutoff": df_cutoff}
