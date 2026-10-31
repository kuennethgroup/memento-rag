#!/usr/bin/env python3
"""Bibliographic and author metadata (the Reference/Author node types of the knowledge graph; see
memento/graph/build/build_graph.py, memento/graph/store.py). Best effort and regex-only; Reference.raw and .year are always kept.

extract_paper_metadata(items) -> {"title", "authors", "year", "doi"}: the title is the first heading that is not
publisher furniture, within a bounded number of headings; authors are the first plain-text item after the title, split
on commas with LaTeX affiliation/ORCID markers stripped; year and DOI scraped from the front matter.

parse_bibliography(items) -> [{"marker", "raw", "authors", "title", "year", "doi"}]: one entry per numbered citation
("[12]", "12.", "12)", bare "12 Author"), from MinerU's list items or ref_text items; authors and title stay None.

link_intext_citations(chunks, references) -> [(chunk_id, marker)]: in-text markers ("[n]", "[n, m]", "[n-m]") that
resolve to a parsed reference, the (Chunk)-[:CITES]->(Reference) edges."""
from __future__ import annotations

import re

from memento.config.settings import DOMAIN

# How many items after the title to scan for the author line — authors
# normally come immediately after the title, sometimes with a graphical
# abstract image in between.
_AUTHOR_SEARCH_WINDOW = 3

# How many leading items to scan for year/DOI front-matter scraping.
_FRONT_MATTER_SCAN_ITEMS = 20

# LaTeX affiliation/ORCID superscript markers on author names, e.g.
# "$^{a,b}$", "$^{ID,*}$", "$^{c}$" — stripped before splitting on commas.
_AFFIL_MARKER_RE = re.compile(r'\$\^\{[^}]*\}\$')

# A 4-digit 19xx/20xx year in parentheses, e.g. "... (2023) 4017-4029." — the
# *last* such match in a reference/front-matter string is almost always the
# publication year (issue numbers like "(3)" appear earlier and aren't 4
# digits).
_YEAR_RE = re.compile(r'\((\d{4})\)')

_DOI_RE = re.compile(r'10\.\d{4,9}/\S+', re.IGNORECASE)

# Publisher-layout profile — see config.DOMAIN["parsing"] for what each value
# means and how to re-fit it for a corpus from a different publisher mix.
_PARSING = DOMAIN["parsing"]
_BANNER_TOKENS = _PARSING["title_banner_tokens"]
_TITLE_SEARCH_HEADINGS = _PARSING["title_search_headings"]
_MIN_TITLE_TOKENS = _PARSING["title_min_tokens"]
_REF_SECTION_RE = re.compile(_PARSING["reference_section_pattern"], re.I)

_WORD_RE = re.compile(r'[a-z0-9]+')

# Leading heading numbering, e.g. "2. Results and Discussion", "3.1 Methods".
_HEADING_NUM_RE = re.compile(r'^\s*\d+(?:\.\d+)*\s*[.)]?\s*')


def _norm_heading(text: str) -> str:
    return re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9\s]', ' ', text.lower())).strip()


# A heading that *is* a section name is a section heading, not a title. Reuses
# the same vocabulary the section classifier is driven by, so a new domain
# inherits this by editing one place.
_SECTION_HEADINGS = frozenset(
    [_norm_heading(k) for k in DOMAIN["section_hints"]]
    + [_norm_heading(a) for al in DOMAIN["section_hints"].values() for a in al]
    + [_norm_heading("Front Matter")]
)

# Canonical section names that hold the bibliography — the in-text citation
# scanner (link_intext_citations) skips chunks in these sections because
# bibliography entries also start with "[n]" but aren't citations.
_REF_SECTION_NAMES = frozenset(
    k for k in DOMAIN["section_hints"]
    if _REF_SECTION_RE.match(k)
)

def _is_banner(heading: str) -> bool:
    """True when every word of a level-1 heading is publisher furniture."""
    words = _WORD_RE.findall(heading.lower())
    return bool(words) and all(w in _BANNER_TOKENS for w in words)


def _is_section_heading(heading: str) -> bool:
    return _norm_heading(_HEADING_NUM_RE.sub('', heading)) in _SECTION_HEADINGS

# Bibliography numbering varies by publisher: "[12] Author", "12. Author", "12) Author", bare "12 Author"; the number is
# what in-text "[n]" citations resolve against. The 1-3 digit bound keeps a wrapped line starting with a year from
# reading as a new entry.
_BIB_ENTRY_RE = re.compile(_PARSING["bib_entry_pattern"], re.DOTALL)

# In-text citation markers ("[6]", "[1, 2]", "[3-5]", "[3–5]"), configured per domain in
# config.DOMAIN["parsing"]["citation_marker_pattern"] (author-year styles need another pattern).
_CITATION_RE = re.compile(_PARSING["citation_marker_pattern"])


def _clean_author_text(text: str) -> list[str]:
    cleaned = _AFFIL_MARKER_RE.sub('', text)
    authors = []
    for part in cleaned.split(','):
        name = re.sub(r'\s+', ' ', part).strip(' \t*')
        if name:
            authors.append(name)
    return authors


def _extract_year(text: str) -> int | None:
    years = [int(m) for m in _YEAR_RE.findall(text) if 1900 <= int(m) <= 2099]
    return years[-1] if years else None


def _extract_doi(text: str) -> str | None:
    m = _DOI_RE.search(text)
    return m.group(0).rstrip('.,;)') if m else None


def extract_paper_metadata(items: list[dict]) -> dict:
    """Best-effort front-matter metadata: title, authors, year, doi."""
    title: str | None = None
    authors: list[str] = []

    title_idx: int | None = None
    seen_headings = 0
    fallback: tuple[str, int] | None = None
    for idx, item in enumerate(items):
        if item.get('text_level') != 1:
            continue
        text = (item.get('text') or '').strip()
        if not text:
            continue
        seen_headings += 1
        if fallback is None:
            fallback = (text, idx)
        if (not _is_banner(text) and not _is_section_heading(text)
                and len(text.split()) >= _MIN_TITLE_TOKENS):
            title, title_idx = text, idx
            break
        if seen_headings >= _TITLE_SEARCH_HEADINGS:
            break
    if title is None and fallback is not None:
        text, idx = fallback
        # Furniture is worse than nothing here: cross-paper resolution matches
        # reference strings against Paper.title, so a junk title turns the
        # paper into a citation attractor, while an unset one merely excludes
        # it as a citation target.
        if not _is_banner(text) and not _is_section_heading(text):
            title, title_idx = text, idx

    if title_idx is not None:
        for item in items[title_idx + 1: title_idx + 1 + _AUTHOR_SEARCH_WINDOW]:
            if item.get('text_level'):
                break  # hit the next heading before finding an author line
            if item.get('type') == 'text':
                text = (item.get('text') or '').strip()
                if text:
                    authors = _clean_author_text(text)
                    break

    front_matter_text = ' '.join(
        (item.get('text') or '')
        for item in items[:_FRONT_MATTER_SCAN_ITEMS]
        if item.get('type') in ('text', 'list')
    )
    return {
        "title":   title,
        "authors": authors,
        "year":    _extract_year(front_matter_text),
        "doi":     _extract_doi(front_matter_text),
    }


def parse_bibliography(items: list[dict]) -> list[dict]:
    """Split the References section into one dict per "[n] ..." entry.

    MinerU emits bibliography entries in two formats depending on layout:
      - type=="list", section=="References": structured list with list_items
      - type=="ref_text", section=="References": one chunk per entry in .text

    Both are handled. seen_markers deduplicates in case a paper has both
    (e.g. MinerU emits a partial list chunk then individual ref_text items).
    """
    references: list[dict] = []
    seen_markers: set[int] = set()

    def _add(raw_entry: str) -> None:
        m = _BIB_ENTRY_RE.match(raw_entry.strip())
        if not m:
            return
        marker = int(m.group(1))
        if marker in seen_markers:
            return
        seen_markers.add(marker)
        raw = m.group(2).strip()
        references.append({
            "marker":  marker,
            "raw":     raw,
            "authors": None,
            "title":   None,
            "year":    _extract_year(raw),
            "doi":     _extract_doi(raw),
        })

    for item in items:
        if not _REF_SECTION_RE.match(item.get('section') or ''):
            continue
        if item.get('type') == 'list':
            for raw_entry in item.get('list_items') or []:
                _add(raw_entry)
        elif item.get('type') == 'ref_text':
            _add(item.get('text') or '')

    return references


def _expand_citation_group(group: str) -> list[int]:
    """"1, 2" -> [1, 2]; "3-5" / "3–5" -> [3, 4, 5]."""
    markers: list[int] = []
    for part in group.split(','):
        part = part.strip()
        sep = '-' if '-' in part else ('–' if '–' in part else None)
        if sep:
            lo_s, _, hi_s = part.partition(sep)
            try:
                lo, hi = int(lo_s.strip()), int(hi_s.strip())
            except ValueError:
                continue
            if 0 <= hi - lo <= 50:  # sanity bound against garbage ranges
                markers.extend(range(lo, hi + 1))
        else:
            try:
                markers.append(int(part))
            except ValueError:
                continue
    return markers


def link_intext_citations(chunks: list[dict], references: list[dict]) -> list[tuple[str, int]]:
    """Find (chunk_id, marker) pairs for in-text "[n]" citations that resolve
    to a parsed reference's marker. The References list itself is skipped
    (its entries also start with "[n]" but aren't citations)."""
    valid_markers = {r["marker"] for r in references}
    if not valid_markers:
        return []

    pairs: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    for c in chunks:
        if c.get("section") in _REF_SECTION_NAMES:
            continue
        if c["type"] not in {"text", "list"}:
            continue
        text = (c["item"].get("text") or "")
        for m in _CITATION_RE.finditer(text):
            for marker in _expand_citation_group(m.group(1)):
                if marker in valid_markers:
                    key = (c["chunk_id"], marker)
                    if key not in seen:
                        seen.add(key)
                        pairs.append(key)
    return pairs
