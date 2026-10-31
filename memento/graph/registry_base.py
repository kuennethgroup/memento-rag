#!/usr/bin/env python3
"""Domain-agnostic entity-registry infrastructure. A domain registry (graph/entity_registry_<domain>.py) supplies only
data: REGISTRY (tuple[EntityDef, ...], the canonical vocabulary), RELATION_TYPES (dict, the relation vocabulary that
constrains memento/graph/build/extract_entities.py's LLM schema) and ENTITY_TYPES (usually DEFAULT_ENTITY_TYPES), and calls
build_registry(REGISTRY, ENTITY_TYPES) for its public API (canonicalize, find_entities_in_text, get_by_id, normalize).
Normalization, the alias index and dictionary matching (Greek-letter and subscript folding, longest alias wins,
duplicate detection) live here once.

A new domain: copy an existing registry to graph/entity_registry_<name>.py, replace REGISTRY and RELATION_TYPES (extend
DEFAULT_ENTITY_TYPES only if the domain needs a new category), and set `entity_registry: <name>` in the domain YAML;
memento/graph/__init__.py loads that module and re-exports its interface."""
from __future__ import annotations

import re
from dataclasses import dataclass

# Default Entity node `type` categories. Shared across domains because they
# describe *roles* in an experimental-science KG (what gets measured, what
# process produced it, what condition it was made under, what technique
# measured it) rather than ceramics-specific concepts. A domain may pass its
# own tuple to build_registry() if it genuinely needs different categories.
DEFAULT_ENTITY_TYPES = ("Material", "Process", "Property", "Condition", "Characterization")


@dataclass(frozen=True)
class EntityDef:
    canonical_id: str
    name: str
    type: str               # one of the registry's entity_types
    aliases: tuple[str, ...] = ()


# ── Normalization ─────────────────────────────────────────────────────────
_GREEK_MAP = {
    "α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "θ": "theta",
    "κ": "kappa", "λ": "lambda", "µ": "micro", "μ": "micro",
}
_SUBSCRIPT_TRANS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
_SUPERSCRIPT_TRANS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")


def normalize(text: str) -> str:
    """Lowercase + fold Greek letters to their Latin names + fold
    sub/superscript digits to plain digits + collapse whitespace.

    Used both to build a registry's alias index and to normalize text before
    dictionary matching, so e.g. "α-Al2O3", "alpha-Al2O3", and "Al₂O₃" all
    become comparable strings. Domain-agnostic — Greek letters and
    sub/superscript digits show up in formulas/units across materials-science
    domains, not just ceramics.
    """
    s = text.strip().lower()
    for greek, latin in _GREEK_MAP.items():
        s = s.replace(greek, latin)
    s = s.translate(_SUBSCRIPT_TRANS).translate(_SUPERSCRIPT_TRANS)
    s = re.sub(r"\s+", " ", s)
    return s.strip(" \t-_.,;:()[]")


def _normalize_with_map(text: str) -> tuple[str, list[int]]:
    """`normalize(text)` without the final strip, plus, for every character of
    the result, the index of the original character it came from. Lets
    matches on normalized text be placed in the original text."""
    chars: list[str] = []
    pos: list[int] = []
    for i, ch in enumerate(text):
        for c in ch.lower():
            for cc in _GREEK_MAP.get(c, c).translate(_SUBSCRIPT_TRANS).translate(_SUPERSCRIPT_TRANS):
                if cc.isspace():
                    if chars and chars[-1] == " ":
                        continue
                    cc = " "
                chars.append(cc)
                pos.append(i)
    return "".join(chars), pos


# ── LaTeX folding ─────────────────────────────────────────────────────────
# MinerU's VLM output writes formulas as LaTeX ("$Al_{2}O_{3}$",
# "$\mathrm{ZrB} _ {2}$", "$\delta$-Al..."); fold them to plain text before
# matching so "Al2O3"/"ZrB2"/"δ-Al2O3" aliases can match. Only digit
# subscripts/superscripts are folded ("Al_{1 - x}" is left alone).
_LATEX_CMD = re.compile(r"\\(?:mathrm|text|textrm|rm|mathbf|mathit|operatorname)\s*\{([^{}]*)\}")
_LATEX_SUB = re.compile(r"\s*[_^]\s*\{\s*([0-9][0-9.\s]*)\}|\s*[_^]\s*([0-9])")
_LATEX_GREEK = {"alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "theta": "θ",
                "kappa": "κ", "lambda": "λ", "mu": "µ"}
# a subscript in its own math span, "Al $_{2}$ O $_{3}$": join to the element
# before it, and to a following element symbol
_LATEX_SPLIT_SUB = re.compile(r"\s*\$\s*[_^]\s*\{\s*([0-9][0-9.\s]*)\}\s*\$(?:\s+(?=[A-Z]))?")
# "$\delta$ -x" -> "δ-x"
_GREEK_HYPHEN = re.compile(r"([αβγδθκλµ])\s*-\s*(?=[A-Za-z])")
_LATEX_GREEK_RE = re.compile(r"\\(" + "|".join(_LATEX_GREEK) + r")(?![A-Za-z])")


def fold_latex(text: str) -> str:
    """Plain-text form of inline LaTeX formulas; text without LaTeX is returned unchanged."""
    if "$" not in text and "\\" not in text:
        return text
    t = _LATEX_GREEK_RE.sub(lambda m: _LATEX_GREEK[m.group(1)], text)
    t = _LATEX_SPLIT_SUB.sub(lambda m: m.group(1).replace(" ", ""), t)
    prev = None
    while prev != t:          # nested \mathrm{\mathrm{..}}
        prev, t = t, _LATEX_CMD.sub(r"\1", t)
    t = _LATEX_SUB.sub(lambda m: (m.group(1) or m.group(2)).replace(" ", ""), t)
    return _GREEK_HYPHEN.sub(r"\1-", t.replace("$", ""))


def _boundary_pattern(normalized_alias: str) -> str:
    """Wrap an already-normalized alias in lookaround word-boundary
    assertions, but only on edges that are themselves alphanumeric — aliases
    like "AlO(OH)" (normalized "alo(oh)") naturally end on ")" and don't need
    (or work well with) a trailing boundary assertion."""
    pre = r"(?<![A-Za-z0-9])" if normalized_alias[0].isalnum() else ""
    post = r"(?![A-Za-z0-9])" if normalized_alias[-1].isalnum() else ""
    return pre + re.escape(normalized_alias) + post


SHORT_ALIAS_MAX = 3  # aliases this short match case-sensitively on the original text


def _is_exact_case(term: str) -> bool:
    """Short aliases and all-caps acronyms ("FAST", "STEM", "3Y-TZP") match
    with exact case only: case-folded they collide with ordinary words
    ("fast", "stem cells", "ps", "hp")."""
    t = term.strip()
    return len(t) <= SHORT_ALIAS_MAX or (any(ch.isalpha() for ch in t) and t == t.upper())


def _cs_boundary_pattern(alias: str) -> str:
    """Boundary pattern for a short or all-caps exact-case alias matched on the original
    (un-normalized) text, so "PS" matches "(PS)" but not "ps", "ups" or "PSU"."""
    pre = r"(?<![A-Za-z0-9])" if alias[0].isalnum() else ""
    post = r"(?![A-Za-z0-9])" if alias[-1].isalnum() else ""
    return pre + re.escape(alias) + post


# ── Registry ──────────────────────────────────────────────────────────────
class Registry:
    """Built from a domain's `REGISTRY` tuple. See `build_registry()`."""

    def __init__(self, entities: tuple[EntityDef, ...],
                 entity_types: tuple[str, ...] = DEFAULT_ENTITY_TYPES):
        self.entities = entities
        self.entity_types = entity_types
        self._by_id: dict[str, EntityDef] = {}
        self._alias_index: dict[str, EntityDef] = {}   # normalized, case-insensitive
        self._cs_index: dict[str, EntityDef] = {}      # short/all-caps aliases, exact case

        for ent in entities:
            if ent.canonical_id in self._by_id:
                raise ValueError(f"Duplicate canonical_id in registry: {ent.canonical_id!r}")
            if ent.type not in entity_types:
                raise ValueError(f"Unknown entity type {ent.type!r} for {ent.canonical_id!r}")
            self._by_id[ent.canonical_id] = ent
            for term in (ent.name,) + ent.aliases:
                t = term.strip()
                if _is_exact_case(t):
                    index, key = self._cs_index, t
                else:
                    index, key = self._alias_index, normalize(term)
                existing = index.get(key)
                if existing is not None and existing.canonical_id != ent.canonical_id:
                    raise ValueError(
                        f"Alias {term!r} (key {key!r}) claimed by both "
                        f"{existing.canonical_id!r} and {ent.canonical_id!r}")
                index[key] = ent

        # Longest-first: a longer alias wins over a shorter one it contains when
        # both would match at the same starting position.
        self._combined_re = re.compile(
            "|".join(_boundary_pattern(k)
                     for k in sorted(self._alias_index, key=len, reverse=True))
        ) if self._alias_index else None
        self._cs_re = re.compile(
            "|".join(_cs_boundary_pattern(k)
                     for k in sorted(self._cs_index, key=len, reverse=True))
        ) if self._cs_index else None

    def get_by_id(self, canonical_id: str) -> EntityDef | None:
        """Look up a registry entry by its canonical_id."""
        return self._by_id.get(canonical_id)

    def canonicalize(self, text: str) -> EntityDef | None:
        """Resolve free text to a registry entry via exact
        (post-normalization) name/alias match; short and all-caps terms
        (see _is_exact_case) must match with exact case. Returns None if `text` isn't
        a known term — it may still be a novel entity for the
        LLM-extraction path (memento/graph/build/extract_entities.py) to surface."""
        text = fold_latex(text or "")
        t = text.strip()
        if _is_exact_case(t) and t in self._cs_index:
            return self._cs_index[t]
        # exact-case aliases are never in _alias_index, so a short word that is also an
        # exact-case alias stays None, while an upper-case spelling of a normal alias still resolves
        return self._alias_index.get(normalize(text))

    def find_entities_in_text(self, text: str) -> list[EntityDef]:
        """Scan `text` for any registry name/alias (dictionary match — the
        high-precision half of the hybrid extraction strategy). Mixed-case
        aliases match case-insensitively on normalized text; short and
        all-caps ones (see _is_exact_case) with exact case on the original
        text; an exact-case match inside a longer match is dropped. Returns
        the distinct EntityDefs found, in first-occurrence order."""
        text = fold_latex(text or "")
        hits: list[tuple[int, EntityDef]] = []
        spans: list[tuple[int, int]] = []
        if self._combined_re is not None:
            norm, pos = _normalize_with_map(text)
            for m in self._combined_re.finditer(norm):
                ent = self._alias_index.get(m.group(0))
                if ent is not None:
                    start, end = pos[m.start()], pos[m.end() - 1] + 1
                    spans.append((start, end))
                    hits.append((start, ent))
        if self._cs_re is not None:
            # sub/superscript digits fold 1:1, so match offsets stay valid ("K₁c" -> "K1c")
            plain = text.translate(_SUBSCRIPT_TRANS).translate(_SUPERSCRIPT_TRANS)
            for m in self._cs_re.finditer(plain):
                # "BN" inside "c-BN": the longer match wins
                if any(m.start() < e and s0 < m.end() for s0, e in spans):
                    continue
                ent = self._cs_index.get(m.group(0))
                if ent is not None:
                    hits.append((m.start(), ent))
        found: dict[str, EntityDef] = {}
        for _, ent in sorted(hits, key=lambda h: h[0]):
            found.setdefault(ent.canonical_id, ent)
        return list(found.values())


def build_registry(entities: tuple[EntityDef, ...],
                    entity_types: tuple[str, ...] = DEFAULT_ENTITY_TYPES) -> Registry:
    return Registry(entities, entity_types)
