"""Query intent for set-shaped questions: the router in front of graph aggregation.

Ranked retrieval answers "which passages are best"; questions about the corpus as a set (which papers, how many, which
use X and Y, X but not Y) have no best passage, so top-k cannot express them. This module decides which family a query
belongs to and parses the set expression. A missed route degrades to ordinary retrieval, a false one answers a focused
question with hundreds of papers, so every rule is a conjunction and the classifier abstains when unsure. The
corpus-noun requirement matters most: aggregation words alone ("total", "number of") often describe physical quantities."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from memento.config.settings import DOMAIN

# Vocabulary comes from config, never from literals here. The cue words are
# language-bound rather than domain-bound, so a new domain reuses them unchanged
# while a new language replaces them — the same split section_hints already uses.
_V = DOMAIN.get("set_query_vocabulary", {})


def _alt(key: str, default: tuple = ()) -> str:
    """Regex alternation over a vocabulary list, longest first so that
    multi-word cues ("as well as") win over their own prefixes ("as")."""
    words = _V.get(key) or list(default)
    return "|".join(re.escape(w) for w in sorted(set(words), key=len, reverse=True))

# What the question counts or lists; without one of these the query is about a physical quantity, not the corpus. Plural
# only, plus collectives: a singular "the study" refers to one document.
_CORPUS = _alt("corpus_nouns")

# The one place a singular corpus noun is legitimate: under a universal
# quantifier, where the quantifier itself carries the enumeration
# ("list every paper that mentions X").
_UNIVERSAL = re.compile(
    r"\b(?:" + _alt("universal_quantifiers") + r")\s+(?:\w+\s+){0,2}?"
    r"(?:" + _alt("corpus_nouns_singular") + r")\b", re.I)

# except after a preposition, where the quantifier distributes over documents already in hand ("the value reported in
# every paper")
_DISTRIBUTIVE = re.compile(
    r"\b(?:" + _alt("distributive_preps") + r")\s+"
    r"(?:" + _alt("universal_quantifiers") + r")\b", re.I)

# Listing intent: "list/every/which/all ... papers". The corpus noun trails the
# cue closely enough that an unrelated later clause cannot supply it.
_ENUMERATE = re.compile(
    r"\b(?:" + _alt("list_cues") + r")\b[^.?!]{0,60}?\b(?:" + _CORPUS + r")\b", re.I)

# Counting intent -- same corpus requirement, answered with a scalar.
_COUNT = re.compile(
    r"\b(?:" + _alt("count_cues") + r")\b[^.?!]{0,40}?\b(?:" + _CORPUS + r")\b", re.I)

# A question that happens to mention papers but wants a passage, not a set:
# "what does the X paper say about Y". Aboutness verbs beat the
# listing cue.
_PASSAGE = re.compile(
    r"\b(?:" + _alt("passage_cues") + r"|what is the (?:value|mechanism|effect|"
    r"reason|role))\b", re.I)

# Set operators. Negation splits the query; conjunction changes how the
# remaining seeds combine.
_NEGATE = re.compile(
    r"\b(?:" + _alt("negation_cues") + r"|that (?:do|does|did) ?n[o\']t|"
    r"which (?:do|does|did) ?n[o\']t|do ?n[o\']t|does ?n[o\']t|no)\b", re.I)
_CONJOIN = re.compile(
    # "combining X with Y" is the same conjunction as "combined with", with the operands between the verb and the
    # preposition
    r"\b(?:" + _alt("conjunction_cues") + r"|combin\w*(?:\s+\S+){0,4}?\s+with)\b",
    re.I)

# Bare "X and Y" with an explicit joint-ness marker ("together", "in combination", "simultaneously", "jointly") is an
# intersection. Not every "and": "papers on hardness and toughness" is ambiguous, and an intersection default would
# empty results a union answers acceptably.
_CONJOIN_JOINT = re.compile(
    r"\band\b(?:\s+\S+){0,6}?\s+"
    r"(?:together|in\s+combination|simultaneously|at\s+the\s+same\s+time|"
    r"in\s+the\s+same\s+(?:paper|study|work|publication|article))\b",
    re.I)

# Temporal predicate: "papers since 2018", "from 2018 onwards", "after 2018", "2019 or later"; "before"/"until" are the
# symmetric case, included but not separately evaluated.
_YEAR = r"(?:19|20)\d{2}"
_SINCE = re.compile(
    r"\b(?:since|after|from)\s+(" + _YEAR + r")(?:\s+onwards?)?\b"
    r"|\b(" + _YEAR + r")\s+or\s+later\b", re.I)
_BEFORE = re.compile(
    r"\b(?:before|until|up to)\s+(" + _YEAR + r")\b"
    r"|\b(" + _YEAR + r")\s+or\s+earlier\b", re.I)


@dataclass
class Intent:
    """`mode` drives the answer shape; the texts go to the entity seeder. Spans, not entity names: entity resolution is
    shared with the retrieval path, and a second parser here would drift from it."""
    mode: str = "none"                       # none | enumerate | count
    include_text: str = ""
    exclude_text: str = ""
    combine: str = "or"                      # how to combine include seeds
    year_from: int | None = None             # inclusive: Paper.year >= this
    year_to: int | None = None               # inclusive: Paper.year <= this
    cues: list[str] = field(default_factory=list)

    @property
    def is_set_query(self) -> bool:
        return self.mode in ("enumerate", "count")


def _universal_hit(q: str) -> bool:
    """A universal quantifier over documents that is selecting, not distributing."""
    spans = [m.span() for m in _DISTRIBUTIVE.finditer(q)]
    return any(not any(s <= m.start() < e for s, e in spans)
               for m in _UNIVERSAL.finditer(q))


def parse_intent(query: str, force: bool = False) -> Intent:
    """Route and parse a query. `force=True` skips the routing gate and runs only the field extraction, for callers that
    already decided this is a set query (a learned detector, say). Default False."""
    q = (query or "").strip()
    if not q:
        return Intent()

    counting = bool(_COUNT.search(q))
    listing = bool(_ENUMERATE.search(q)) or _universal_hit(q)
    if not force:
        if not (counting or listing):
            return Intent()
        # "how many papers" is unambiguous even next to an aboutness verb; a
        # bare listing cue next to one is not.
        if _PASSAGE.search(q) and not counting:
            return Intent()

    cues = []
    include, exclude = q, ""
    neg = _NEGATE.search(q)
    if neg:
        include, exclude = q[: neg.start()], q[neg.end():]
        cues.append(f"negate:{neg.group(0).lower()}")

    conj = _CONJOIN.search(include) or _CONJOIN_JOINT.search(include)
    combine = "and" if conj else "or"
    if conj:
        cues.append(f"conjoin:{conj.group(0).lower()}")

    # the temporal phrase is stripped out of include_text, so a bare year never reaches the entity seeder (which and
    # when stay separate, as exclude_text separates negation)
    year_from = year_to = None
    since = _SINCE.search(include)
    if since:
        year_from = int(since.group(1) or since.group(2))
        include = include[: since.start()] + " " + include[since.end():]
        cues.append(f"since:{year_from}")
    before = _BEFORE.search(include)
    if before:
        year_to = int(before.group(1) or before.group(2))
        include = include[: before.start()] + " " + include[before.end():]
        cues.append(f"before:{year_to}")

    return Intent(
        # A query can carry both a counting and a listing cue ("how many papers list …"): counting is the more specific
        # request, so it wins.
        mode="count" if counting else "enumerate",
        include_text=include.strip(),
        exclude_text=exclude.strip(),
        combine=combine,
        year_from=year_from, year_to=year_to,
        cues=cues + (["count"] if counting else []) + (["list"] if listing else []),
    )


def looks_like_enumeration(query: str) -> bool:
    """Back-compatible boolean form."""
    return parse_intent(query).is_set_query
