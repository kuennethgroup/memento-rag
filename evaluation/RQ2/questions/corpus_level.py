"""Corpus-level questions: seed terms by document frequency, composition, alias-aware
full scan + dense top-100 screening, works/documents counting unit."""
from __future__ import annotations

import random
import re

from evaluation.RQ2.questions.common import lang_of, language_instruction, llm_json
from evaluation.RQ2.questions.pooling import judge_facts
from evaluation.RQ2.questions.terms import term_key
from memento.graph.documents import work_id

UNIT = {"list": "documents", "count": "works", "compare": "works"}


def seed_terms(df_universe, *, lo, hi, seed, n):
    ok = sorted(t for t, docs in df_universe.items() if lo <= len(docs) <= hi)
    random.Random(seed).shuffle(ok)
    return ok[:n]


SEED_SYSTEM = (
    "You write corpus-level questions for a research archive. From the passages (which all mention TERM, in "
    "different documents) write ONE question of the requested TYPE whose answer requires finding ALL documents that "
    "satisfy one checkable criterion about TERM. TYPE list: 'Which documents report …?'. TYPE count: 'How many "
    "works/researchers …?' or 'Who worked on …?'. TYPE compare: a question comparing two sub-groups defined by the "
    "criterion (e.g. 'Are there more works on X with A or with B?'). Use the language the request names; never "
    "mention the passages. "
    "Output ONLY JSON: {\"question\": str, \"criterion\": str (a statement a single document either satisfies or "
    "not), \"variants\": [all surface forms of TERM: abbreviation, full name, German and English forms, formulas]}")


def compose_corpus_question(client, model, term, passages, qtype):
    user = f"TERM: {term}\nTYPE: {qtype}\n\n" + "\n\n".join(
        f"[{p['source']}] {p['text'][:1200]}" for p in passages[:6])
    lang = lang_of([p["text"] for p in passages[:6]])
    user += "\n\n" + language_instruction(lang)
    r = llm_json(client, model, SEED_SYSTEM, user, temperature=0.3)
    if not isinstance(r, dict) or not all(isinstance(r.get(k), str) and r[k].strip() for k in ("question", "criterion")):
        return None
    variants = [v for v in (r.get("variants") or []) if isinstance(v, str) and v.strip()]
    return {"question": r["question"].strip(), "criterion": r["criterion"].strip(),
            "variants": sorted(set(variants) | {term}), "unit": UNIT[qtype], "lang": lang}


def variant_scan(chunks, variants):
    keys = sorted({term_key(v) for v in variants if term_key(v)}, key=len, reverse=True)
    rx = re.compile(r"(?<![a-z0-9äöüß])(?:" + "|".join(re.escape(k) for k in keys) + r")")
    hits = {}
    for c in chunks:
        if rx.search(term_key(c.get("text", ""))):
            hits.setdefault(c["source"], []).append(c)
    return {s: sorted(cs, key=lambda c: -len(c["text"]))[:3] for s, cs in hits.items()}


def screen(client, model, criterion, docs):
    gold, unparsed = [], []
    for src, passages in sorted(docs.items()):
        r = judge_facts(client, model, {"c": criterion}, passages, batch=3)
        if r["labels"]:
            gold.append(src)
        elif len(r["unparsed"]) == len(passages):
            unparsed.append(src)
    return {"gold_docs": gold, "unparsed_docs": unparsed, "n_screened": len(docs)}


def count_answer(gold_docs, registry, unit):
    if unit == "works":
        works = sorted({work_id(d, registry.get(d)) for d in gold_docs})
        return len(works), works
    return len(gold_docs), []


def dense_top_docs(client, collection, question, allowed, k=100):
    from memento.config.settings import DOMAIN, EMBEDDER_URL
    QUERY_INSTRUCTION = DOMAIN["instruction"]   # same query instruction as G0 / RQ1
    from memento.retrieval.search import embed_query, point_to_hit, qdrant_query
    from evaluation.RQ2.questions.hardness import _filter
    qvec = embed_query(question, EMBEDDER_URL, QUERY_INSTRUCTION)
    pts = qdrant_query(client, collection, qvec, question, _filter(allowed), k, False, 0).points
    out = {}
    for pt in pts:
        h = point_to_hit(pt.payload or {}, pt.score, getattr(pt, "id", None))
        out.setdefault(h["source"], []).append(h)
    return {s: cs[:3] for s, cs in out.items()}
