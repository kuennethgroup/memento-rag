"""Graph-native set retrieval: the modes top-k cannot express.

A set query ("all papers on X") asks for a whole set; ranked retrieval and reranking return the best passages and
miss most of it, while a one-hop MENTIONS lookup returns the complete set. So this module does not go through candidate
generation: it resolves the query to entities with the graph path's seeding logic and evaluates a set expression with an
unbounded structured lookup, returning whole papers.

Operators: union (papers mentioning any seed), intersection (every seed) and difference (minus papers mentioning an
excluded seed). No single chunk needs to contain X and Y for a paper to qualify, and dense retrieval cannot express
negation. Scope: set queries over entities the seeder can resolve (the domain registry in `dict` seed mode); a different
retrieval mode, not a better ranker."""
from __future__ import annotations

import re
from typing import Optional

from memento.config.settings import DOMAIN, GRAPH_QUERY_USE_LLM, GRAPH_SEED_MODE, PREDICATE_LLM_ENABLED
from memento.graph.store import GraphStore, get_store
from memento.graph.query import predicates as operands
from memento.graph.query.graph_retrieval import _build_seeds
from memento.graph.query.predicates import classify, validate
from memento.graph.query.predicates import Predicate, get_menu
from memento.graph.query.query_intent import Intent, looks_like_enumeration, parse_intent  # noqa: F401

_WORD_RE = re.compile(r"[a-zA-Z]+")


def _find_trigger(text: str, trigger_map: dict[str, Predicate]) -> Optional[Predicate]:
    """The first structural predicate whose trigger words appear in `text`, or None. Runs before entity seeding and,
    when it fires, replaces the mentions path: a citation or authorship question would otherwise seed unrelated entities
    from its words and be answered as a mentions union."""
    for tok in _WORD_RE.findall(text.lower()):
        pred = trigger_map.get(tok)
        if pred is not None:
            return pred
    return None


def _strip_triggers(text: str, triggers: frozenset[str]) -> str:
    """Remove trigger-word tokens before handing the remainder to an operand
    resolver -- "cite"/"cites" isn't part of any paper title or author name,
    and leaving it in only dilutes the resolver's token-overlap scoring."""
    kept = [w for w in text.split() if w.strip(".,;:!?").lower() not in triggers]
    return " ".join(kept)


_EDGE_PAPERS = (
    "MATCH (p:Paper)-[:`{edge}`]->(o:{end_label} {{id: $oid}}) "
    "RETURN DISTINCT p.id AS src, p.title AS title, p.year AS year "
    "ORDER BY src"
)


def _edge_papers(store: GraphStore, pred: Predicate, operand_id: str) -> list[dict]:
    """Papers on the subject side of one structural edge instance (the papers that CITES_PAPER a resolved paper, or
    AUTHORED_BY a resolved author). One template for every predicate predicates.py offers, keyed by
    `pred.edge`/`pred.end_label`: a new Paper-anchored edge type needs only a menu entry."""
    cypher = _EDGE_PAPERS.format(edge=pred.edge, end_label=pred.end_label)
    rows = [r.data() for r in store._run(cypher, oid=operand_id).records]
    # dummy hit/chunk fields, so this shares format_answer and the year post-filter with the mentions path; a structural
    # edge is binary, so the `min_hits` threshold does not apply (see set_query())
    for r in rows:
        r["hits"] = r["chunks"] = r["n_ents"] = 1
    return rows


def _operand_display(store: GraphStore, pred: Predicate, operand_id: str) -> str:
    label = pred.end_label
    prop = "title" if label == "Paper" else "name"
    rows = store._run(
        f"MATCH (o:{label} {{id: $oid}}) RETURN o.{prop} AS t", oid=operand_id
    ).records
    text = rows[0].data().get("t") if rows else None
    return text or operand_id

# Papers mentioning $eids, counted per entity: count(DISTINCT c), since a chunk mentioning two seeds gives two rows. The
# per-entity breakdown (min_h / max_h) makes a threshold mean "each entity mentioned ≥ n times", not one entity carrying
# a conjunction.
_PAPERS = (
    "MATCH (c:Chunk)-[:MENTIONS]->(e:Entity) WHERE e.id IN $eids "
    "WITH c.source AS src, e.id AS eid, count(DISTINCT c) AS h "
    "WHERE src IS NOT NULL "
    "WITH src, sum(h) AS chunks, count(eid) AS n_ents, "
    "     min(h) AS min_h, max(h) AS max_h "
    "WHERE n_ents >= $need "
    "OPTIONAL MATCH (p:Paper {source: src}) "
    "RETURN src, p.title AS title, p.year AS year, chunks, n_ents, min_h, max_h "
    "ORDER BY chunks DESC, src"
)


def _papers_for(store: GraphStore, eids: list[str], need: int = 1,
                combine: str = "or") -> list[dict]:
    """`hits` is the statistic a mention threshold applies to, and which one is
    correct depends on the operator: a conjunction is only as strong as its
    weakest entity (min), a union as strong as its best (max). With a single
    seed the two coincide, so union enumeration is unaffected."""
    if not eids:
        return []
    rows = [r.data() for r in store._run(_PAPERS, eids=eids, need=need).records]
    for r in rows:
        r["hits"] = r["min_h"] if combine == "and" else r["max_h"]
    return rows


def set_query(
    query: str,
    *,
    intent: Optional[Intent] = None,
    store: Optional[GraphStore] = None,
    seed_mode: str = GRAPH_SEED_MODE,
    use_llm: bool = GRAPH_QUERY_USE_LLM,
    seed_client=None,
    min_hits: int = 1,
    limit: int = 0,
) -> dict:
    """Evaluate the query's set expression over the graph. `limit` 0 means unbounded. `min_hits` filters papers by how
    many distinct chunks mention the seeds (substantive treatment vs a passing reference). Returns the seed entities
    with the results, so a bad resolution is visible. `abstained` is True when nothing resolved: callers fall back to
    ordinary retrieval, since "no papers match" and "could not read the question" differ."""
    it = intent or parse_intent(query)
    seed_kw = dict(seed_mode=seed_mode, use_llm=use_llm, seed_client=seed_client)

    owns = store is None
    store = store or get_store()
    try:
        # Structural predicates (cites_paper, authored_by, …) first (see _find_trigger). predicates.get_menu is
        # process-cached.
        menu = get_menu(store)
        pred = _find_trigger(it.include_text or query, menu.trigger_map())
        predicate_name: Optional[str] = None
        operand_display: Optional[str] = None
        if pred is not None:
            operand_text = _strip_triggers(it.include_text or query, pred.triggers)
            resolver = operands.RESOLVERS[pred.operand_type]
            operand_id = resolver(operand_text, store)
            if operand_id is None:
                return {"mode": it.mode, "combine": it.combine, "entities": [],
                        "excluded_entities": [], "papers": [], "n_papers": 0,
                        "predicate": pred.name, "abstained": True,
                        "reason": f"could not confidently resolve the "
                                  f"{pred.operand_type} named in a "
                                  f"'{pred.name}' query"}
            rows = _edge_papers(store, pred, operand_id)
            drop: set[str] = set()
            predicate_name = pred.name
            operand_display = _operand_display(store, pred, operand_id)
            inc: list[str] = []
            exc: list[str] = []
        else:
            inc = list(_build_seeds(it.include_text or query, **seed_kw))
            exc = list(_build_seeds(it.exclude_text, **seed_kw)) if it.exclude_text else []
            # An entity named on both sides is the seeder over-matching the
            # include span (the exclude clause usually repeats the topic
            # word). Exclusion wins -- it is the more specific signal.
            inc = [e for e in inc if e not in exc]
            if not inc:
                # Second-line fallback, off by default: neither the trigger words nor entity seeding found anything, so
                # the LLM is asked whether the query names a structural predicate in other words
                # (memento/graph/query/predicates.py). It only runs on queries that would abstain anyway.
                llm_pred = llm_operand_id = None
                if PREDICATE_LLM_ENABLED and menu.predicates:
                    raw = classify(query, menu)
                    validated = validate(raw, menu, query) if raw else None
                    if validated:
                        cand_pred, operand_text = validated
                        resolver = operands.RESOLVERS[cand_pred.operand_type]
                        cand_id = resolver(operand_text, store)
                        if cand_id is not None:
                            llm_pred, llm_operand_id = cand_pred, cand_id
                if llm_pred is None:
                    return {"mode": it.mode, "combine": it.combine, "entities": [],
                            "excluded_entities": exc, "papers": [], "n_papers": 0,
                            "abstained": True,
                            "reason": "no entity in the query resolved against "
                                      "the graph"}
                rows = _edge_papers(store, llm_pred, llm_operand_id)
                drop = set()
                predicate_name = llm_pred.name
                operand_display = _operand_display(store, llm_pred, llm_operand_id)
                inc, exc = [], []
            else:
                need = len(inc) if it.combine == "and" else 1
                rows = _papers_for(store, inc, need=need, combine=it.combine)
                # Exclusion is unconditional: a single mention is enough to
                # falsify "does not use Y", so `min_hits` deliberately does
                # not apply here.
                drop = {r["src"] for r in _papers_for(store, exc)} if exc else set()

        # Post-filter by Paper.year (resolve, then filter by year), the shape the temporal evaluation measured, rather
        # than a new Cypher shape. Shared by both branches above.
        year: dict[str, int] = {}
        if it.year_from is not None or it.year_to is not None:
            year = {r.data()["src"]: r.data()["y"] for r in store._run(
                "MATCH (p:Paper) WHERE p.source IN $srcs AND p.year IS NOT NULL "
                "RETURN p.source AS src, p.year AS y",
                srcs=[r["src"] for r in rows]).records}
    finally:
        if owns:
            store.close()

    # `min_hits` ("mentioned in ≥ N distinct chunks") has no analogue for a structural edge; applied there it would drop
    # every result whenever a caller passes min_hits > 1 (the mentions path does).
    effective_min_hits = 1 if predicate_name else min_hits
    rows = [r for r in rows if r["src"] not in drop and r["hits"] >= effective_min_hits]
    if it.year_from is not None:
        rows = [r for r in rows if year.get(r["src"], -1) >= it.year_from]
    if it.year_to is not None:
        rows = [r for r in rows if year.get(r["src"], 10**9) <= it.year_to]

    return {
        "mode": it.mode,
        "combine": it.combine,
        "entities": sorted(inc),
        "excluded_entities": sorted(exc),
        "predicate": predicate_name,
        "operand_display": operand_display,
        "year_from": it.year_from, "year_to": it.year_to,
        "n_papers": len(rows),
        "papers": rows[:limit] if limit else rows,
        "abstained": False,
        "reason": None,
    }


def aggregate_by_entity(query: str, **kw) -> dict:
    """Union-only form retained for the enumeration evaluation harness."""
    return set_query(query, **kw)


_DOC_NOUN = DOMAIN.get("document_noun", {"singular": "paper", "plural": "papers"})


def format_answer(result: dict, show: int = 15) -> str:
    """Render for a reader, leading with the count (for a set query the count is the answer). Uses
    config.DOMAIN["document_noun"] ("papers" by default), a display word only; the Neo4j schema is not config-driven."""
    if result.get("abstained"):
        return f"No matching entity found in the knowledge graph ({result['reason']})."
    n = result["n_papers"]
    noun = _DOC_NOUN.get("singular", "paper") if n == 1 else _DOC_NOUN.get("plural", "papers")
    if result.get("predicate"):
        # Names the resolved operand in the answer text -- so a mis-resolved
        # paper/author is visible to the reader rather than silent (a wrong
        # but confident-looking answer is harder to catch than an empty one).
        verb = {"cites_paper": "cite", "authored_by": "were authored by"}.get(
            result["predicate"], result["predicate"])
        what = f"{verb} {result.get('operand_display') or '(unresolved)'}"
        if result.get("year_from") is not None:
            what += f", published {result['year_from']} or later"
        if result.get("year_to") is not None:
            what += f", published {result['year_to']} or earlier"
        if result.get("mode") == "count":
            return f"{n} {noun} {what}."
        lines = [f"{n} {noun} {what}." + (f" First {min(show, n)}:" if n > show else "")]
        for r in result["papers"][:show]:
            title = (r.get("title") or "(title not extracted)")[:78]
            year = f" ({r['year']})" if r.get("year") else ""
            lines.append(f"  · {title}{year}")
        if n > show:
            lines.append(f"  … and {n - show} more.")
        return "\n".join(lines)
    joiner = " and " if result.get("combine") == "and" else ", "
    what = joiner.join(result["entities"])
    if result.get("excluded_entities"):
        what += ", excluding " + ", ".join(result["excluded_entities"])
    if result.get("year_from") is not None:
        what += f", published {result['year_from']} or later"
    if result.get("year_to") is not None:
        what += f", published {result['year_to']} or earlier"
    if result.get("mode") == "count":
        return f"{n} {noun} mention {what}."
    lines = [f"{n} {noun} mention {what}." + (f" First {min(show, n)}:" if n > show else "")]
    for r in result["papers"][:show]:
        title = (r.get("title") or "(title not extracted)")[:78]
        year = f" ({r['year']})" if r.get("year") else ""
        lines.append(f"  · {title}{year}  [{r['chunks']} chunks]")
    if n > show:
        lines.append(f"  … and {n - show} more.")
    return "\n".join(lines)
