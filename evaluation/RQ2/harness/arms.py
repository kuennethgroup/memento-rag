"""RQ2 arms. Every arm takes a QuestionView and returns an ArmOut."""
from __future__ import annotations

import dataclasses
import re
import time
from collections import Counter
from dataclasses import dataclass, field

from evaluation.RQ2.harness.budget import fit_budget, hit_tokens, merge_streams
from evaluation.RQ2.harness.qterms import seed_groups_from_terms, seed_ids_from_terms
from evaluation.RQ2.harness.scope import best_chunk_per_doc, g0_retrieve, payload_hits, rank_stream
from evaluation.RQ2.questions.corpus_level import variant_scan
from memento.graph.documents import work_id
from memento.graph.query.graph_retrieval import graph_candidate_ids

RUNG_PARAMS = {
    "g1": dict(streams="structural"),
    "g2": dict(streams="all", traversal="bfs", max_hops=0, first_hop=True),
    "g3": dict(streams="all", traversal="ppr", first_hop=True),
}
FOLLOWUP_SYSTEM = ("You help a search engine find missing evidence. Given a question and the passages found so "
                   "far, write ONE follow-up search query for the information still needed to answer it. Output "
                   "only the query, in the question's language.")
_STOP = {"the", "of", "and", "or", "a", "an", "in", "on", "to", "with", "by", "for", "der", "die", "das", "und",
         "von", "mit", "zu", "den", "dem", "des", "ein", "eine"}


@dataclass
class ArmOut:
    arm: str
    design: str
    qid: str
    context: list
    block: str = ""
    docs: list = field(default_factory=list)
    info: dict = field(default_factory=dict)
    latency_s: float = 0.0

    @property
    def key(self):
        return f"{self.qid}|{self.arm}|{self.design}"


def _docs(hits):
    return list(dict.fromkeys(h["source"] for h in hits))


def without_docs(g0, docs):
    """G0's output with the given documents removed (tuning's bridge check: can the graph reach
    B's document from A's passages when G0 did not already hand it over?)."""
    ctx_ = [h for h in g0.context if h["source"] not in docs]
    return ArmOut(g0.arm, g0.design, g0.qid, ctx_, g0.block, _docs(ctx_), dict(g0.info), g0.latency_s)


def arm_g0(ctx, view):
    t = time.time()
    hits, qvec = g0_retrieve(ctx, view.question)
    return ArmOut("g0", "rag", view.qid, [dict(h, origin="g0") for h in hits], "", _docs(hits), {"qvec": qvec},
                  time.time() - t)


def _followup(ctx, question, hits):
    ctxt = "\n\n".join(f"[{i + 1}] {h['text'][:800]}" for i, h in enumerate(hits[:5]))
    r = ctx.llm.chat.completions.create(model=ctx.model, temperature=0, max_tokens=100,
                                        messages=[{"role": "system", "content": FOLLOWUP_SYSTEM},
                                                  {"role": "user", "content": f"Question: {question}\n\n{ctxt}"}],
                                        extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    lines = (r.choices[0].message.content or "").strip().splitlines()
    return lines[0].strip()[:300] if lines and lines[0].strip() else question


def arm_d2(ctx, view, g0):
    t = time.time()
    follow = _followup(ctx, view.question, g0.context)
    second, _ = g0_retrieve(ctx, follow)
    stream = rank_stream(ctx, view.question, second)
    merged, block = merge_streams(g0.context, stream, k_slots=ctx.k_slots, origin="d2")
    return ArmOut("d2", "multihop", view.qid, merged, block, _docs(merged), {"followup": follow},
                  g0.latency_s + time.time() - t)


def _linking_block(ctx, rung, g0_hits, added, seeds):
    if rung == "g1" or not added:
        return ""
    first = ctx.store.chunk_entities([h["chunk_id"] for h in g0_hits], max_df=ctx.max_df)
    new = ctx.store.chunk_entities([h["chunk_id"] for h in added], max_df=ctx.max_df)
    shared = sorted(set(first) & set(new), key=lambda e: -first[e])[:15]
    if rung == "g2":
        names = ctx.store.entities_by_ids(shared)
        return ("Linking entities: " + ", ".join(names[e] for e in shared if e in names)) if names else ""
    rels = ctx.store.relations_among(sorted(set(shared) | set(seeds)), limit=10)
    return ("Linking facts:\n" + "\n".join(f"({r['src']}) — {r['description']} — ({r['tgt']})" for r in rels)) if rels else ""


def lite_ctx(ctx, arm):
    """The arm context of a G2-lite arm: the same everything, its own graph store."""
    return dataclasses.replace(ctx, store=ctx.stores[arm])


def arm_graph(ctx, view, g0, rung, terms, name=None):
    """`rung` picks the retrieval (RUNG_PARAMS); `name` labels the arm (G2-lite arms run rung g2)."""
    name = name or rung
    t = time.time()
    seeds = seed_ids_from_terms(ctx.store, terms) if rung in ("g2", "g3") else {}
    ids = graph_candidate_ids(
        view.question, dense_chunk_ids=[h["chunk_id"] for h in g0.context], dense_sources=_docs(g0.context),
        store=ctx.store, enabled=True, query_vector=g0.info.get("qvec"), qdrant_client=ctx.qdrant,
        max_df=ctx.max_df, limit=ctx.stream_pool, registry=ctx.registry, chunk_source=ctx.chunk_source,
        extra_seeds=seeds, use_llm=False, seed_mode="dict", **RUNG_PARAMS[rung])
    stream = rank_stream(ctx, view.question, payload_hits(ctx, ids))
    merged, _ = merge_streams(g0.context, stream, k_slots=ctx.k_slots, origin=name)
    added = [h for h in merged if h["origin"] == name]
    block = _linking_block(ctx, rung, g0.context, added, seeds)
    merged, block = merge_streams(g0.context, stream, k_slots=ctx.k_slots, origin=name, block=block)
    return ArmOut(name, "multihop", view.qid, merged, block, _docs(merged),
                  {"n_stream": len(stream), "seeds": sorted(seeds),
                   "stream_pages": [[h["source"], h.get("page", 0)] for h in stream]},
                  g0.latency_s + time.time() - t)


def corpus_g1(ctx, view, g0):
    groups = {}
    for d in g0.docs:
        groups.setdefault(work_id(d, ctx.registry.get(d)), []).append(d)
    block = "Documents grouped by work:\n" + "\n".join(f"- {w}: {', '.join(ds)}" for w, ds in groups.items())
    hits, block = fit_budget(g0.context, block, hit_tokens(g0.context))
    return ArmOut("g1", "corpus", view.qid, hits, block, g0.docs, {}, g0.latency_s)


def _doc_arm(ctx, view, g0, arm, docs, label, info, t):
    hits = best_chunk_per_doc(ctx, docs[:20], g0.info.get("qvec"))
    block = f"{label} ({len(docs)}): " + "; ".join(docs)
    hits, block = fit_budget([dict(h, origin=arm) for h in hits], block, hit_tokens(g0.context))
    return ArmOut(arm, "corpus", view.qid, hits, block, docs, info, g0.latency_s + time.time() - t)


def _abstain(g0, arm, view, why):
    return ArmOut(arm, "corpus", view.qid, g0.context, "", g0.docs, {"abstained": True, "why": why}, g0.latency_s)


def corpus_lookup(ctx, view, g0, rung, terms, name=None):
    name = name or rung
    t = time.time()
    groups = seed_groups_from_terms(ctx.store, terms)
    if not groups:
        return _abstain(g0, name, view, "no seed entity")
    seeds = sorted({e for g in groups for e in g})
    info = {"seeds": seeds, "n_terms_resolved": len(groups), "abstained": False}
    hits, matched = Counter(), Counter()
    for g in groups:                          # a term matches a document if any of its variants is mentioned
        for r in ctx.store.papers_for_entities(g, need=1):
            if r["hits"] >= ctx.min_hits and (ctx.allowed is None or r["src"] in ctx.allowed):
                matched[r["src"]] += 1
                hits[r["src"]] += r["hits"]
    need = len(groups) if terms.get("combine") == "and" else 1
    while need > 1 and not any(n >= need for n in matched.values()):    # soft AND, same rule as LX
        need -= 1
        info["and_relaxed_to"] = need
    docs = sorted((d for d, n in matched.items() if n >= need), key=lambda d: (-matched[d], -hits[d], d))
    if rung == "g3" and terms.get("relation") and len(seeds) >= 2:
        words = {w for w in re.findall(r"\w+", terms["relation"].lower()) if w not in _STOP and len(w) > 2}
        rel_docs = {r["paper_id"] for r in ctx.store.relations_among(seeds, limit=1000)
                    if words & set(re.findall(r"\w+", (r["description"] or "").lower()))}
        kept = [d for d in docs if d in rel_docs]
        info["relation_filter"] = "applied" if kept else "empty→kept"
        docs = kept or docs
    if not docs:
        return _abstain(g0, name, view, "no document")
    return _doc_arm(ctx, view, g0, name, docs, "Documents found", info, t)


def corpus_lx(ctx, view, g0, terms, chunks):
    t = time.time()
    if not terms.get("terms"):
        return _abstain(g0, "lx", view, "no terms")
    scope = [c for c in chunks if ctx.allowed is None or c["source"] in ctx.allowed]
    per_term = [variant_scan(scope, t_["variants"]) for t_ in terms["terms"]]
    info = {"abstained": False}
    matched = Counter(d for p in per_term for d in p)
    need = len(per_term) if terms.get("combine") == "and" else 1
    while need > 1 and not any(n >= need for n in matched.values()):    # soft AND, same rule as the lookup
        need -= 1
        info["and_relaxed_to"] = need
    docs = {d for d, n in matched.items() if n >= need}
    count = {d: sum(len(p.get(d, [])) for p in per_term) for d in docs}
    ranked = sorted(docs, key=lambda d: (-matched[d], -count[d], d))     # same order as the graph lookup
    if not ranked:
        return _abstain(g0, "lx", view, "no match")
    return _doc_arm(ctx, view, g0, "lx", ranked, "Documents found by keyword scan", info, t)


def _sheet_matches(ctx, variants):
    from memento.graph import canonicalize
    from memento.graph.build.fact_sheets import match_counts
    return match_counts(ctx.qdrant, ctx.sheets, variants, canonicalize)


def corpus_fs(ctx, view, g0, terms):
    """Fact-sheets arm ("structure then query"): the same query terms as LX and the graph lookup, matched
    against the per-document fact sheets; same soft AND, same document list and context budget."""
    t = time.time()
    if not ctx.sheets or not ctx.qdrant.collection_exists(ctx.sheets):
        return _abstain(g0, "fs", view, "no fact sheets for this corpus")
    if not terms.get("terms"):
        return _abstain(g0, "fs", view, "no terms")
    per_term = [{d: n for d, n in _sheet_matches(ctx, t_["variants"]).items()
                 if ctx.allowed is None or d in ctx.allowed} for t_ in terms["terms"]]
    info = {"abstained": False}
    matched = Counter(d for p in per_term for d in p)
    need = len(per_term) if terms.get("combine") == "and" else 1
    while need > 1 and not any(n >= need for n in matched.values()):    # soft AND, same rule as LX
        need -= 1
        info["and_relaxed_to"] = need
    docs = {d for d, n in matched.items() if n >= need}
    count = {d: sum(p.get(d, 0) for p in per_term) for d in docs}
    ranked = sorted(docs, key=lambda d: (-matched[d], -count[d], d))
    if not ranked:
        return _abstain(g0, "fs", view, "no match")
    return _doc_arm(ctx, view, g0, "fs", ranked, "Documents found by fact sheets", info, t)
