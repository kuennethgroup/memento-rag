"""Structural set-query predicates: the menu, the operand resolvers and the LLM fallback that picks from the menu.

The structural set-query predicate menu, derived from the graph rather than written by hand.

memento/graph/query/query_intent.py and memento/graph/query/graph_aggregate.py answer "mentions"-shaped set queries over MENTIONS edges.
Another family is shaped around the graph's other paper-anchored edges ("which papers cite paper X", "which papers were
authored by Y"). This module lists those edges from the graph's live schema (GraphStore.relationship_types() /
.relationship_type_stats(), Neo4j's db.* procedures, no APOC) and offers an edge as a predicate when it is (1) not owned
by the mentions/temporal path (MENTIONS, RELATION, the intra-document edges, and CO_OCCURS/SIMILAR, which carry no paper
provenance); (2) anchored at Paper with an operand-typeable end label (Paper or Author, _CANDIDATE_ENDS); (3) its
derived trigger words do not collide with the router's own vocabulary (config.DOMAIN["set_query_vocabulary"]); this is
what leaves out HAS_REFERENCE/REFERS_TO, since "references" is a corpus noun.

A new Paper-anchored edge type becomes queryable with no edits here, unless its trigger words collide with the router's
vocabulary or its operand type is new (the operand resolvers below).

Operand resolvers (one per operand type the schema exposes: paper, author, …) share PaperResolver's asymmetry: a
wrong paper or author is worse than none, so ambiguous input returns None and graph_aggregate falls back to ranked
retrieval. The LLM classification is a second-line fallback for predicates the deterministic trigger cannot reach:
the model never sees or writes Cypher, only a predicate name copied from the menu and an operand span copied
verbatim from the query, and validate() rejects a name not in the menu before anything reaches a query.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from memento.config.settings import DOMAIN
from memento.graph.store import GraphStore

from typing import Optional
from memento.query.scope.paper_resolve import normalize, resolver_from_store
import json
import logging
import os
import re
from memento.config.settings import DOMAIN, REASONING_URL  # noqa: E402
from pydantic import BaseModel  # noqa: E402


# Edges the mentions/temporal path (query_intent.py + graph_aggregate.py) already owns, plus the two Entity-Entity edges
# without paper provenance (CO_OCCURS carries only `weight`, SIMILAR only a similarity score). SAME_WORK/SAME_AUTHOR are
# Paper-Paper retrieval links for the RQ2 ladder, not predicates; SAME_AUTHOR would otherwise claim "author" queries.
_OWNED_BY_MENTIONS_PATH = frozenset({
    "HAS_SECTION", "PARENT_SECTION", "IN_SECTION", "REFERENCES", "CITES",
    "MENTIONS", "RELATION", "CO_OCCURS", "SIMILAR", "SAME_WORK", "SAME_AUTHOR",
})

# End label -> the operand type a query names when it points at that label.
# Bounded by node label, not by relation -- see the operand resolvers below for why
# this stays a short, closed map even as edge types grow.
_CANDIDATE_ENDS = {"Paper": "paper", "Author": "author"}

# Parts of a predicate name that are no trigger of their own: connectives, and "paper"/"papers", the corpus noun every
# set query carries (it would make CITES_PAPER fire on any set query).
_STOP_PARTS = frozenset({"has", "by", "of", "to", "in", "for", "paper", "papers"})


def _inflect(word: str) -> set[str]:
    """Plausible surface forms of one predicate-name word part. Not a stemmer: covers the two verb forms the edge-type
    names use (CITES_PAPER "cites", AUTHORED_BY "authored") by de-inflecting to a rough base and regenerating from
    there."""
    if word.endswith("ing") and len(word) > 5:
        base = word[:-3]
    elif word.endswith("ed") and len(word) > 4:
        base = word[:-2]
    elif word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        base = word[:-1]
    else:
        base = word
    forms = {word, base}
    if base.endswith("e"):
        forms |= {base + "s", base + "d", base[:-1] + "ing"}
    else:
        forms |= {base + "s", base + "ed", base + "ing"}
    return forms


def _derive_triggers(predicate_name: str) -> frozenset[str]:
    parts = [p for p in predicate_name.lower().split("_") if p not in _STOP_PARTS]
    triggers: set[str] = set()
    for p in parts:
        triggers |= _inflect(p)
    return frozenset(triggers)


def _router_vocab() -> frozenset[str]:
    vocab = DOMAIN.get("set_query_vocabulary", {})
    words: set[str] = set()
    for key in ("corpus_nouns", "corpus_nouns_singular", "list_cues", "count_cues"):
        words.update(w.lower() for w in vocab.get(key, []))
    return frozenset(words)


@dataclass(frozen=True)
class Predicate:
    name: str              # "cites_paper" | "authored_by" -- lowercased edge type
    edge: str               # the literal Neo4j relationship type ("CITES_PAPER")
    operand_type: str       # "paper" | "author" -- key into memento.graph.query.operands.RESOLVERS
    end_label: str          # "Paper" | "Author"
    n_edges: int            # live count at menu-build time
    triggers: frozenset[str]
    examples: tuple[str, ...] = ()  # 1-2 REAL edge instances, for LLM grounding
                                     # (the LLM classification below) -- pulled
                                     # from the graph at menu-build time, never
                                     # authored, so they can't drift from reality


@dataclass(frozen=True)
class Menu:
    predicates: tuple[Predicate, ...]
    fingerprint: str        # sha256 of (name, operand_type) pairs -- pin this in a gate;
                             # menu introspection is DB-state-dependent (relationship
                             # types "in use"), so a change here should be a reviewed
                             # test failure, not a silent behaviour change.

    def by_name(self, name: str) -> "Predicate | None":
        return next((p for p in self.predicates if p.name == name), None)

    def trigger_map(self) -> dict[str, "Predicate"]:
        """token -> the predicate it would escalate to. Only tokens for
        predicates this menu actually offers -- callers must not invent
        trigger words that aren't backed by a live, offered predicate."""
        out: dict[str, Predicate] = {}
        for p in self.predicates:
            for t in p.triggers:
                out[t] = p
        return out


def _sample_examples(store: GraphStore, edge: str, end_label: str,
                     limit: int = 2) -> tuple[str, ...]:
    """Real `(subject id) -> (object display)` pairs for LLM grounding (the LLM classification's menu prompt, below),
    best effort and never hand-written, so an example cannot drift from what the graph contains."""
    try:
        rows = store._run(
            f"MATCH (p:Paper)-[:`{edge}`]->(o:{end_label}) "
            "RETURN p.id AS subj, coalesce(o.title, o.name, o.id) AS obj "
            "LIMIT $limit",
            limit=limit,
        ).records
        return tuple(f"{r['subj']} {edge.lower()} {r['obj']}" for r in rows)
    except Exception:
        return ()


def build_menu(store: GraphStore, *, min_edges: int = 1) -> Menu:
    predicates: list[Predicate] = []
    router_vocab = _router_vocab()
    try:
        rel_types = store.relationship_types()
    except NotImplementedError:
        rel_types = []

    for rel_type in rel_types:
        if rel_type in _OWNED_BY_MENTIONS_PATH:
            continue
        stats = store.relationship_type_stats(rel_type)
        if stats is None:
            continue
        start_label, end_label, n = stats
        if start_label != "Paper" or end_label not in _CANDIDATE_ENDS:
            continue
        if n < min_edges:
            continue
        name = rel_type.lower()
        triggers = _derive_triggers(name)
        if not triggers or (triggers & router_vocab):
            continue
        predicates.append(Predicate(
            name=name, edge=rel_type, operand_type=_CANDIDATE_ENDS[end_label],
            end_label=end_label, n_edges=n, triggers=triggers,
            examples=_sample_examples(store, rel_type, end_label),
        ))

    predicates.sort(key=lambda p: p.name)
    fingerprint = hashlib.sha256(
        "|".join(f"{p.name}:{p.operand_type}" for p in predicates).encode()
    ).hexdigest()[:16]
    return Menu(predicates=tuple(predicates), fingerprint=fingerprint)


_menu_cache: "Menu | None" = None


def get_menu(store: GraphStore, *, refresh: bool = False) -> Menu:
    """Process-cached -- the menu build costs a handful of Cypher calls
    (cheap, but not per-query cheap) and the schema changes only on
    ingestion, not between queries in a session."""
    global _menu_cache
    if _menu_cache is None or refresh:
        _menu_cache = build_menu(store)
    return _menu_cache


# ── Operand resolvers ─────────────────────────────────────────────────────────
_STOP_AUTHOR = frozenset({
    "who", "wrote", "written", "by", "the", "paper", "papers", "was", "is",
    "were", "are", "author", "authors", "authored", "which", "a", "an",
})

# A candidate author name must beat the runner-up by this margin, or two
# authors sharing a surname make the match ambiguous -- abstain rather than
# guess (same contract as PaperResolver.THRESHOLD/MARGIN).
_AUTHOR_SCORE_FLOOR = 0.5
_AUTHOR_MARGIN = 0.15


def resolve_paper(text: str, store: GraphStore) -> Optional[str]:
    """Paper-name operand → Paper.id via the title/id resolver. Confidence-gated (Match.confident, score ≥ 0.75),
    stricter than PaperResolver.resolve's 0.68, because a structural predicate answers with an unfiltered edge traversal
    and no reranker downstream."""
    match = resolver_from_store(store).resolve(text)
    return match.source if match and match.confident else None


def resolve_author(text: str, store: GraphStore) -> Optional[str]:
    """Surname/name-token match against `Author.name` (`GraphStore.
    all_authors`). Reuses `paper_resolve.normalize`'s LaTeX/punctuation
    folding so "Y. Smith" and "Smith, Y." both fold to comparable tokens."""
    tokens = [t for t in normalize(text).split()
              if t not in _STOP_AUTHOR and len(t) > 1]
    if not tokens:
        return None
    q_tok = set(tokens)

    scored: list[tuple[float, str]] = []
    for a in store.all_authors():
        name_tok = set(normalize(a.get("name") or "").split())
        if not name_tok:
            continue
        hit = len(name_tok & q_tok)
        if not hit:
            continue
        # coverage of the author's tokens, not the query's: a bare surname must not match every author sharing it at the
        # same score
        scored.append((hit / len(name_tok), a["id"]))
    if not scored:
        return None
    scored.sort(key=lambda r: -r[0])
    top, runner = scored[0], (scored[1][0] if len(scored) > 1 else 0.0)
    if top[0] < _AUTHOR_SCORE_FLOOR or (top[0] - runner) < _AUTHOR_MARGIN:
        return None
    return top[1]


RESOLVERS = {
    "paper": resolve_paper,
    "author": resolve_author,
}


# ── LLM classification (fallback) ─────────────────────────────────────────────
class PredicateTerm(BaseModel):
    predicate: str = ""
    operand_type: str = ""
    operand_text: str = ""


class PredicateResponse(BaseModel):
    terms: list[PredicateTerm] = []

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover — exercised only when openai isn't installed
    OpenAI = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Same env-var names as memento/graph/build/extract_entities.py, deliberately -- one
# reasoning endpoint, one provider switch, not a second copy of the config.
ENRICH_PROVIDER       = os.getenv("ENRICH_PROVIDER", "local")
CLOUD_BASE_URL        = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
CLOUD_MODEL           = os.getenv("CLOUD_REASONING_MODEL", "qwen3.5-397b-a17b")
REASONING_MODEL_NAME  = os.getenv("REASONING_MODEL_NAME", "/models/reasoning")

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def _get_client_and_model() -> tuple["OpenAI", str]:
    if OpenAI is None:
        raise RuntimeError("openai package not installed")
    if ENRICH_PROVIDER == "cloud":
        return (
            OpenAI(base_url=CLOUD_BASE_URL, api_key=os.getenv("OPENAI_API_KEY", "")),
            CLOUD_MODEL,
        )
    return OpenAI(base_url=f"{REASONING_URL}/v1",
                  api_key=os.getenv("REASONING_API_KEY", os.getenv("OPENAI_API_KEY", "none"))), REASONING_MODEL_NAME


def _parse_llm_json(content: str) -> Optional[PredicateResponse]:
    content = _THINK_RE.sub("", content or "").strip()
    m = _FENCE_RE.search(content)
    if m:
        content = m.group(1).strip()
    if not content:
        return None
    if not content.startswith("{"):
        start, end = content.find("{"), content.rfind("}")
        if start == -1 or end == -1 or end < start:
            return None
        content = content[start:end + 1]
    try:
        return PredicateResponse.model_validate(json.loads(content))
    except Exception:
        return None


def menu_prompt(menu: Menu) -> str:
    """The LLM's whole menu: predicate names, operand types, live edge counts and 1-2 real examples from the graph."""
    lines = []
    for p in menu.predicates:
        lines.append(f"  - {p.name}  (operand: {p.operand_type})  [{p.n_edges} edges]")
        for ex in p.examples:
            lines.append(f"      e.g. \"{ex}\"")
    predicates_block = "\n".join(lines)
    noun = DOMAIN.get("document_noun", {})
    plural, singular = noun.get("plural", "papers"), noun.get("singular", "paper")
    return (
        f"You translate a question about a corpus of {plural} into a set expression.\n"
        "Respond with a single JSON object and nothing else -- no prose, no "
        "markdown fences.\n\n"
        f"Available predicates (the subject is always a {singular}):\n"
        f"{predicates_block}\n\n"
        "Output schema:\n"
        '{"terms": [{"predicate": <one of the names above>, '
        '"operand_type": <the operand type shown for that predicate>, '
        '"operand_text": <verbatim span of the question naming the operand>}]}\n\n'
        "Rules:\n"
        "- Use only predicate names from the list above. Do not invent predicates.\n"
        "- `operand_text` must be an exact, verbatim span copied from the "
        "question -- do not normalise, translate, expand an abbreviation, or "
        "guess an identifier.\n"
        "- Emit at most one term.\n"
        '- If no predicate in the list expresses the question, return {"terms": []}.'
    )


def classify(query: str, menu: Menu, *, client=None, model_name: Optional[str] = None,
            temperature: float = 0.0, max_tokens: int = 200) -> Optional[dict]:
    """One best-effort LLM call: the parsed JSON dict, or None on any failure (no menu, network, timeout, malformed
    output). Callers treat None as an abstain and never retry with a guess."""
    if not menu.predicates:
        return None
    if client is None:
        try:
            client, model_name = _get_client_and_model()
        except Exception as e:
            logger.warning("Predicate classification unavailable: %s: %s",
                           type(e).__name__, e)
            return None
    system = menu_prompt(menu)
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": query},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body={
                "reasoning_effort": "low",
                "enable_thinking": False,
                # qwen3.6 (vLLM) deployments ignore top-level enable_thinking
                # and only honor chat_template_kwargs -- send both spellings
                # (same as memento/graph/build/extract_entities.py:_llm_pass).
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
    except Exception as e:
        logger.warning("Predicate classification call failed: %s: %s",
                       type(e).__name__, e)
        return None
    return _parse_llm_json(response.choices[0].message.content)


def validate(raw: Optional[PredicateResponse], menu: Menu, query: str) -> Optional[tuple[Predicate, str]]:
    """Checked, never trusted: `(predicate, operand_text)` or None. Rejects missing or malformed JSON; zero or more than
    one term (one predicate only, as the deterministic path); a predicate name outside the menu (the injection
    boundary); an `operand_type` that does not match the menu's; and an `operand_text` that is not a substring of the
    query (an invented identifier)."""
    if raw is None or len(raw.terms) != 1:
        return None
    term = raw.terms[0]
    pred = menu.by_name(term.predicate) if term.predicate else None
    if pred is None:
        return None
    if term.operand_type != pred.operand_type:
        return None
    if not term.operand_text.strip():
        return None
    if term.operand_text.strip().lower() not in query.lower():
        return None
    return pred, term.operand_text.strip()
