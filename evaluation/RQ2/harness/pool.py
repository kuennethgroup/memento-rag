"""All pooling uses every arm; the judge sees page/document text only, never the graph."""
from __future__ import annotations

import dataclasses

from evaluation.RQ2.questions.corpus_level import screen
from evaluation.RQ2.questions.pooling import pool_and_judge
from memento.graph.documents import work_id


def extend_multihop_gold(llm, model, q, outs):
    facts = {"a": q.meta["fact_a"], "b": q.meta["fact_b"]}
    known = {"gold": [g for g in q.gold_pages if g.get("fact")], "unparsed": [], "judged": []}
    res = pool_and_judge(llm, model, facts, {o.key: o.context for o in outs}, known=known)
    have = {(g["source"], g["page"]) for g in q.gold_pages}
    new = [{"source": g["source"], "page": g["page"], "role": "pooled", "fact": g["fact"],
            "found_by": g.get("found_by", [])} for g in res["gold"] if (g["source"], g["page"]) not in have]
    meta = q.meta | {"pool_plan4": {"n_candidates": res["n_candidates"], "unparsed": res["unparsed"]}}
    return dataclasses.replace(q, gold_pages=q.gold_pages + new, meta=meta)


def extend_corpus_gold(llm, model, q, outs, registry=None, fetch=None):
    """`fetch(doc) -> passages` supplies text for documents an arm listed without showing
    any passage (a lookup lists more documents than fit in the context), so every
    arm's output is judged."""
    gold = set(q.gold_docs)
    passages = {}
    for o in outs:
        for h in o.context:
            passages.setdefault(h["source"], []).append(h)
        for d in o.docs:
            passages.setdefault(d, [])
    if fetch is not None:
        for d, ps in passages.items():
            if not ps and d not in gold:
                passages[d] = list(fetch(d))
    todo = {d: ps[:3] for d, ps in passages.items() if d not in gold and ps}
    res = screen(llm, model, q.meta["criterion"], todo) if todo else {"gold_docs": [], "unparsed_docs": [], "n_screened": 0}
    docs = q.gold_docs + [d for d in res["gold_docs"] if d not in gold]
    works = sorted({work_id(d, (registry or {}).get(d)) for d in docs}) if q.meta.get("unit") == "works" else q.gold_works
    meta = q.meta | {"screen_plan4": {"n_screened": res["n_screened"], "unparsed_docs": res["unparsed_docs"],
                                      "no_passage": sorted(d for d, ps in passages.items() if d not in gold and not ps)}}
    return dataclasses.replace(q, gold_docs=docs, gold_works=works, meta=meta)
