"""Question terms for corpus-level lookup. One LLM call per question, shared by the
graph lookups (matched to Entity ids) and LX (matched in raw text), so H6 compares indexes, not query
understanding. Uses the question text only."""
from __future__ import annotations

from evaluation.RQ2.questions.common import llm_json

TERMS_SYSTEM = (
    "You extract the search terms of a research question. List only the 1 to 3 most specific terms a matching "
    "document must mention (the material, compound, technique, property or process the question is about); leave "
    "out broad topic words and descriptive phrases. Give each with its surface forms (at most 8): abbreviation, "
    "full name, German and English forms, formula. Say whether a document must match all terms ('and') or any ('or'). "
    "If the question asks about a specific relation between terms, quote it in a few words, else null. Output "
    "ONLY JSON {\"terms\": [{\"term\": str, \"variants\": [str]}], \"combine\": \"and\"|\"or\", "
    "\"relation\": str|null}.")
_EMPTY = {"terms": [], "combine": "or", "relation": None}
MAX_TERMS = 3          # smoke v2: broad context terms made the AND lookups miss most gold documents


def query_terms(llm, model, view, cache):
    if view.qid in cache:
        return cache[view.qid]
    r = None
    for _ in range(2):                        # one retry: a truncated/unparsable answer is not "no terms"
        r = llm_json(llm, model, TERMS_SYSTEM, f"Question: {view.question}")
        if isinstance(r, dict) and isinstance(r.get("terms"), list):
            break
    out = dict(_EMPTY)
    if isinstance(r, dict) and isinstance(r.get("terms"), list):
        terms = []
        for t in r["terms"]:
            if isinstance(t, dict) and isinstance(t.get("term"), str) and t["term"].strip():
                vs = [v for v in (t.get("variants") or []) if isinstance(v, str) and v.strip()]
                terms.append({"term": t["term"].strip(), "variants": list(dict.fromkeys(vs + [t["term"].strip()]))})
        out = {"terms": terms[:MAX_TERMS], "combine": "and" if r.get("combine") == "and" else "or",
               "relation": r["relation"] if isinstance(r.get("relation"), str) and r["relation"].strip() else None}
    cache[view.qid] = out
    return out


def _candidate_ids(form):
    from memento.graph import canonicalize          # the import memento/graph/build/extract_entities.py uses
    from memento.graph.build.extract_entities import _slugify
    known = canonicalize(form)
    return {known.canonical_id} if known is not None else {_slugify(form)} - {""}


def seed_ids_from_terms(store, terms):
    cands = set()
    for t in terms.get("terms", []):
        for form in [t["term"], *t.get("variants", [])]:
            cands |= _candidate_ids(form)
    return {eid: 1.0 for eid in sorted(store.existing_entity_ids(sorted(cands)))}


def seed_groups_from_terms(store, terms):
    """One seed group per term (its variants are alternatives): corpus lookups count matched
    terms, not matched seed ids — the same unit LX counts. Terms with no entity are dropped."""
    groups = []
    for t in terms.get("terms", []):
        cands = set()
        for form in [t["term"], *t.get("variants", [])]:
            cands |= _candidate_ids(form)
        found = store.existing_entity_ids(sorted(cands))
        if found:
            groups.append(sorted(found))
    return groups
