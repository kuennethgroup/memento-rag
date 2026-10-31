"""Resolve a document named in a question (by title, file name or id) to its corpus `source` id.
Titles are matched as well as ids. Matching tolerates truncated titles and LaTeX notation, and abstains (None) on
ambiguous matches: a wrong document is worse than none, which falls back to the unfiltered search."""
from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Iterable, Optional

# Words that carry no identifying signal; kept small so domain terms survive.
_STOP = {
    "a", "an", "the", "of", "on", "in", "for", "and", "or", "to", "with", "by",
    "from", "at", "as", "its", "their", "this", "that", "these", "those",
    "what", "which", "does", "do", "did", "say", "says", "propose", "proposes",
    "paper", "papers", "study", "article", "about", "me", "tell", "is", "are",
    "was", "were", "it", "you", "please", "can", "could", "would", "give",
    "show", "list", "from", "there", "here", "above", "below", "previous",
    "last", "first", "one", "titled", "entitled", "called", "named",
}

_LATEX_CMD = re.compile(r"\\(?:mathrm|mathit|mathbf|text|textrm|rm|it|bf)\s*\{([^}]*)\}")
_LATEX_ANY = re.compile(r"\\[a-zA-Z]+")
_SUBSUP = re.compile(r"[_^]\s*\{?([0-9a-zA-Z]+)\}?")


def normalize(s: str) -> str:
    """Fold LaTeX, punctuation and case so parsed and typed titles compare equal."""
    s = s or ""
    s = _LATEX_CMD.sub(r"\1", s)
    s = _SUBSUP.sub(r" \1", s)
    s = _LATEX_ANY.sub(" ", s)
    s = s.replace("$", " ").replace("{", " ").replace("}", " ")
    s = re.sub(r"[-_/]+", " ", s)
    s = re.sub(r"[^0-9a-zA-Z ]+", " ", s)
    return re.sub(r"\s+", " ", s).lower().strip()


def _tokens(s: str) -> list[str]:
    return [t for t in normalize(s).split() if t not in _STOP and len(t) > 1]


@dataclass
class Match:
    source: str
    title: Optional[str]
    score: float
    method: str            # "source_id" | "title_exact" | "title_fuzzy"
    runner_up: float = 0.0

    @property
    def confident(self) -> bool:
        return self.score >= 0.75


class PaperResolver:
    """Build once per request from (source, title) pairs; query many times."""

    # A match must clear this AND beat the runner-up by MARGIN, or we abstain.
    THRESHOLD = 0.68
    MARGIN = 0.08
    MIN_TOKENS = 3

    # Source ids carry a provenance prefix nobody types (a citation shows `W123`, not `openalex_W123`). Prefixes come
    # from the corpus, never a fixed list: a component that names exactly one document is an identifier, one shared by
    # many is a namespace. It must also look opaque (long, letters mixed with digits), so words of file-name-style ids
    # do not count.
    MIN_BARE_ID = 8

    @staticmethod
    def _looks_like_id(tok: str) -> bool:
        if len(tok) < PaperResolver.MIN_BARE_ID:
            return False
        has_d = any(c.isdigit() for c in tok)
        has_a = any(c.isalpha() for c in tok)
        return (has_d and has_a) or (has_d and not has_a)

    def __init__(self, papers: Iterable[tuple[str, Optional[str]]]):
        rows = [(s, t) for s, t in papers if s]

        # Which components are unique across the whole corpus?
        seen: dict[str, set] = {}
        for src, _ in rows:
            for part in re.split(r"[^0-9a-zA-Z]+", src):
                p = part.lower()
                if self._looks_like_id(p):
                    seen.setdefault(p, set()).add(src)

        self._rows = []
        for src, title in rows:
            ids = {normalize(src)}
            for part in re.split(r"[^0-9a-zA-Z]+", src):
                p = part.lower()
                if self._looks_like_id(p) and len(seen.get(p, ())) == 1:
                    ids.add(p)
            self._rows.append((src, title, {i for i in ids if i},
                               normalize(title or ""), set(_tokens(title or ""))))

    def _score(self, q_norm: str, q_tok: set, t_norm: str, t_tok: set) -> float:
        if not t_tok or len(t_tok) < 2:
            return 0.0
        inter = len(t_tok & q_tok)
        if inter < min(self.MIN_TOKENS, len(t_tok)):
            return 0.0
        # F-measure weighted towards `cov_query`: a short generic title always covers itself, so the share of the
        # query's words the title explains decides. Truncated titles still score well.
        cov_title = inter / len(t_tok)
        cov_query = inter / max(len(q_tok), 1)
        b2 = 4.0                                   # F_2: cov_query weighted 2x
        denom = b2 * cov_title + cov_query
        base = ((1 + b2) * cov_title * cov_query / denom) if denom else 0.0
        # Contiguity check: token overlap alone would match a bag of common
        # domain words, so require the strings to actually align somewhere.
        if t_norm and q_norm:
            m = SequenceMatcher(None, t_norm, q_norm).find_longest_match(
                0, len(t_norm), 0, len(q_norm))
            run = m.size / max(len(t_norm), 1)
            base = 0.65 * base + 0.35 * min(run * 1.5, 1.0)
        return base

    def resolve(self, query: str) -> Optional[Match]:
        if not query:
            return None
        q_norm = normalize(query)
        q_tok = set(_tokens(query))
        # names match whole words only
        q_words = f" {q_norm} "

        # 1. Source id named literally, with or without its prefix.
        best_id, best_id_len = None, 0
        for src, title, ids, _, _ in self._rows:
            for s_norm in ids:
                if f" {s_norm} " in q_words and len(s_norm) > best_id_len:
                    best_id, best_id_len = (src, title), len(s_norm)
        if best_id:
            return Match(best_id[0], best_id[1], 1.0, "source_id")

        # 2. The full title appears verbatim. Trusted only when it covers most of the query's distinctive words: a short
        # generic title is contained in longer specific ones.
        exact = [(src, title, t_norm, t_tok)
                 for src, title, _, t_norm, t_tok in self._rows
                 if len(t_norm) >= 18 and f" {t_norm} " in q_words]
        strong = [e for e in exact
                  if not q_tok or len(e[3] & q_tok) / max(len(q_tok), 1) >= 0.5]
        if strong:
            src, title, t_norm, _ = max(strong, key=lambda r: len(r[2]))
            # Two sources sharing a title is a corpus duplicate, not a choice
            # the resolver should silently make.
            if len({e[0] for e in strong if e[2] == t_norm}) > 1:
                return None
            return Match(src, title, 1.0, "title_exact")

        # 3. Fuzzy — truncated titles, notation differences, paraphrase.
        scored = []
        for src, title, _, t_norm, t_tok in self._rows:
            s = self._score(q_norm, q_tok, t_norm, t_tok)
            if s > 0:
                scored.append((s, src, title))
        if not scored:
            return None
        # one entry per document: a document's other names (file name, parsed title, PDF title) must not count as the
        # runner-up
        best: dict = {}
        for row in scored:
            if row[0] > best.get(row[1], (0.0,))[0]:
                best[row[1]] = row
        scored = sorted(best.values(), key=lambda r: -r[0])
        top, runner = scored[0], (scored[1][0] if len(scored) > 1 else 0.0)
        if top[0] < self.THRESHOLD or (top[0] - runner) < self.MARGIN:
            return None
        return Match(top[1], top[2], top[0], "title_fuzzy", runner_up=runner)


# ── Short names: one distinctive word next to a document noun ──
# Guarded, as a wrong document is worse than none: the word occurs in exactly one document's names, is distinctive (5+
# letters, or an id or name shape) and stands next to a singular document noun. Two such words naming different
# documents give no match.
DOC_NOUNS = {
    "paper", "article", "study", "thesis", "dissertation", "document", "report", "publication", "manuscript",
    "review", "preprint", "talk", "presentation", "chapter",
    "arbeit", "artikel", "studie", "dokument", "bericht", "veroffentlichung", "vortrag", "prasentation",
    "abschlussarbeit", "masterarbeit", "bachelorarbeit", "doktorarbeit", "publikation", "kapitel",
}
_GENERIC = _STOP | DOC_NOUNS | {
    "using", "based", "towards", "toward", "approach", "analysis", "effect", "effects", "influence", "novel",
    "results", "method", "methods", "model", "models", "materials", "material", "properties", "property",
    "journal", "science", "version", "final", "draft", "english", "german", "einfluss", "untersuchung",
    "untersuchungen", "entwicklung", "zur", "und", "der", "die", "das", "von", "fur", "mit", "auf", "bei",
}


def _fold(s: str) -> str:
    import unicodedata
    return unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()


def _name_words(name: str) -> set[str]:
    """Distinctive words of a document name: 5+ letters, or an id/name shape (letters with digits or inner capitals)."""
    out = set()
    for raw in re.findall(r"[0-9A-Za-z]+", _fold(name)):
        w = raw.lower()
        if w in _GENERIC or len(w) < 3:
            continue
        shaped = any(c.isdigit() for c in raw) and any(c.isalpha() for c in raw) or any(c.isupper() for c in raw[1:])
        if len(w) >= 5 or shaped:
            out.add(w)
    return out


# kinds of document, as a query and a file name or title say them
_KINDS = [
    {"thesis", "dissertation", "diss", "doktorarbeit"},        # not "PhD"/"Promotion": they name defence talks too
    {"talk", "presentation", "slides", "vortrag", "prasentation", "folien", "kolloquium", "colloquium"},
    {"masterarbeit", "master", "bachelorarbeit", "bachelor", "abschlussarbeit"},
    {"review"},
    {"report", "bericht"},
]
_ARTICLES = {"the", "a", "an", "this", "that", "s", "der", "die", "das", "den", "dem", "des", "ein", "eine", "einer",
             "eines", "dieser", "diese", "dieses", "diesem", "diesen"}
# words that link a document noun to its name
_NAMING = {"on", "by", "of", "about", "titled", "entitled", "called", "named", "from",
           "von", "uber", "zu", "zum", "zur", "namens", "mit", "titel"}


def resolve_short_name(query: str, names: dict[str, list[str]]) -> Optional[str]:
    """The one document a query names by a distinctive word next to a singular document noun, else None. Next to: up to
    two words before the noun, or after it behind a naming word; anything else is a topic, not a name."""
    words = [w.lower() for w in re.findall(r"[0-9A-Za-z]+", _fold(query))]
    words = [w for w in words if w not in _ARTICLES]
    nouns = [i for i, w in enumerate(words) if w in DOC_NOUNS]
    if not nouns:
        return None
    owners: dict[str, set] = {}
    for src, aliases in names.items():
        for w in set().union(*(_name_words(a) for a in aliases if a)) if aliases else ():
            owners.setdefault(w, set()).add(src)
    near = set()
    for j in nouns:
        near |= {k for k in (j - 1, j - 2) if k >= 0}
        k = j + 1
        linked = 0
        while k < len(words) and words[k] in _NAMING and linked < 2:
            k, linked = k + 1, linked + 1
        if linked:
            near |= {k, k + 1}
    # the noun's kind (thesis, talk, ...) separates documents that share the name word
    kinds = set().union(*(k for k in _KINDS if any(words[j] in k for j in nouns)))
    found = set()
    for k in near:
        own = owners.get(words[k], set()) if k < len(words) else set()
        if 1 < len(own) <= 5 and kinds:          # kind words can sit inside compound words
            own = {src for src in own
                   if any(kind in w.lower() for a in names[src] if a for w in re.findall(r"[0-9A-Za-z]+", _fold(a))
                          for kind in kinds)}
        if len(own) == 1:
            found |= own
    return found.pop() if len(found) == 1 else None


def resolver_from_store(store) -> PaperResolver:
    rows = [r.data() for r in store._run(
        "MATCH (p:Paper) RETURN p.source AS src, p.title AS title").records]
    return PaperResolver((r["src"], r["title"]) for r in rows)
