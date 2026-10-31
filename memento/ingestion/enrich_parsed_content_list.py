#!/usr/bin/env python3
"""Post-parsing enrichment of MinerU's content_list.json.

1. Figure context: for each image/chart item, the author's text that references the figure by label ("Fig. 3"); falls
back to positional proximity (±1 page) when there is no label or no reference.
2. Section assignment: every item gets `section` (a canonical bucket such as Abstract, Methods, Results, Conclusion,
used for section-filtered retrieval) and `section_raw` (the nearest heading verbatim). The VLM parse backend gives every
heading the same text_level, so title, section and subsection are told apart by position (first heading = title) and
text (numbering depth, known section names); see assign_sections(). Both ingestion paths read these fields.

Writes content_list_enriched.json next to the original, which is never modified.

python memento/ingestion/enrich_parsed_content_list.py <path/to/content_list.json>
python memento/ingestion/enrich_parsed_content_list.py --paper-dir <path/to/paper_dir>"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

from memento.config.settings import DOMAIN  # noqa: E402
from memento.common.pdf import PYMUPDF  # noqa: E402
from memento.ingestion.text_layer import correct_items  # noqa: E402

SURROUNDING_TEXT_TYPES = {'text', 'list'}
SURROUNDING_MAX_CHARS  = 300
SURROUNDING_WINDOW     = 25

# Caption-recovery search params
SIBLING_SEARCH_WINDOW   = 4    # how many items forward/back to look for a sibling sub-panel
BBOX_OVERLAP_THRESHOLD  = 0.5  # min fraction of overlap along one axis to count as "siblings"
# Overlap alone is not enough on the cross axis: stacked but unrelated figures in one column share almost all their
# x-extent. Sibling panels of one figure are packed tightly, so the gap on the other axis must not exceed the panels'
# own extent.
SIBLING_MAX_GAP_RATIO   = 1.0
ADJACENT_CAPTION_WINDOW = 2    # how many items forward to look for an orphaned caption text

IMAGE_TYPES = {'image', 'chart'}

# Figure/table/equation reference words are publisher and language conventions, configured in
# config.DOMAIN["structural_labels"]; each word also accepts a trailing ".".
_STRUCTURAL_LABELS = DOMAIN.get("structural_labels", {})


def _label_alt(key: str) -> str:
    words = _STRUCTURAL_LABELS.get(key) or []
    # longest first, so a longer entry is not shadowed by its own prefix (as memento/graph/query/query_intent.py's _alt())
    return "|".join(re.escape(w) for w in sorted(set(words), key=len, reverse=True))


# A label number is "5", "5a" or, in dissertations numbered by chapter, "4.2" or "4.5.5". "Fig. 5. Caption" stays "Fig.
# 5": the dot must be followed by a digit.
_LABEL_NUMBER = r'\d+(?:\.\d+)*[a-zA-Z]?'
# A caption's own number may also be chapter-numbered with a hyphen ("Abbildung 4-1", "Figure 10-25"); only caption
# labels read it so, and in running text "Figs. 3-5" stays a range (_LIST_TAIL uses _LABEL_NUMBER). The parser sometimes
# writes "Figure 1. 2 Global …" for Figure 1.2: a bare number after ". " is read as part of it ("Fig. 5. 3D map" stays
# "Fig. 5").
_CAPTION_NUMBER = r'\d+(?:(?:[.-]|\.\s(?=\d+\b))\d+)*[a-zA-Z]?'
_FIG_LABEL_RE = re.compile(
    r'\b((?:' + _label_alt("figure") + r')\.?\s*' + _CAPTION_NUMBER + r')(?![\d])',
    re.IGNORECASE,
)

# Table labels, e.g. "Table 1", "Tab. 2" — used both for caption-driven table
# enrichment and for charts whose caption describes a table ("chart-as-table"
# items from the VLM parse backend).
_TABLE_LABEL_RE = re.compile(
    r'\b((?:' + _label_alt("table") + r')\.?\s*' + _CAPTION_NUMBER + r')(?![\d])',
    re.IGNORECASE,
)


def _label_families(key: str) -> list[set[str]]:
    """The words of one structural_labels list grouped into families: two words are one family when one is a prefix of
    the other ("fig" < "figs", "figure" < "figures"; "abb" < "abbildung"; "tab" < "table", "tabelle"). A citation with
    any word of the caption label's family counts ("Fig. 5" in the caption, "Figure 5 shows" in the text). Words outside
    the family never match ("Scheme 1" is not "Fig. 1")."""
    words = sorted({w.lower() for w in (_STRUCTURAL_LABELS.get(key) or [])})
    families: list[set[str]] = []
    for w in words:
        joined = [f for f in families if any(w.startswith(v) or v.startswith(w) for v in f)]
        merged = {w}.union(*joined) if joined else {w}
        families = [f for f in families if f not in joined] + [merged]
    return families


_LABEL_FAMILIES = _label_families("figure") + _label_families("table")
_LABEL_PARTS_RE = re.compile(r'^\s*([^\W\d_]+)\.?\s*(\d+(?:[.-]\d+)*)([a-zA-Z]?)\s*$')
# "Supplementary Fig. 2", "Extended Data Fig. 2", "SI Fig. 2" is not the paper's Figure 2, neither as a citation nor as
# a caption.
_SUPPLEMENT_BEFORE_RE = re.compile(
    r'(?:\b(?:supplementary|supplemental|supporting|extended)|(?-i:\bE?SI))'
    r'(?:\s+(?:information|info|materials?|data|online))?\.?\s*$', re.IGNORECASE)
# Lists and ranges of labels: "Figs. 3 and 4", "Figures 3, 4 and 5", "Figures 3.31 to 3.35", "Abb. 2–4". A comma alone
# never joins two numbers ("In Fig. 3, 25 samples were tested"), so a comma list needs a conjunction or range word.
_LIST_JOIN = (r'\s*(?:,(?=\s*' + _LABEL_NUMBER + r'\s*(?:,|and|und|&|or|oder)\s)|,?\s*(?:and|und|&|or|oder))\s*')
_RANGE_JOIN = r'\s*(?:to|bis|through|–|—|-)\s*'
_LIST_TAIL = r'(?:(?:' + _RANGE_JOIN + r'|' + _LIST_JOIN + r')' + _LABEL_NUMBER + r'(?![\d]|\.\d))+'
_MAX_RANGE = 12  # a range like "Figs. 1–5" spans at most this many numbers; "Figure 10-25" (hyphen numbering) is not a
                 # range

# MinerU/LaTeX equation numbering, e.g. "...\tag {3}\n$$" or "...\tag{10}$$".
# This is a parser-output convention, not a language/publisher one, so it
# stays a literal (unlike the reference vocabulary below it).
_EQ_TAG_RE = re.compile(r'\\tag\s*\{\s*([0-9]+[a-zA-Z]?)\s*\}')

# Author references to a numbered equation: "Eq. (3)", "Equation 3",
# "eqn. (3)" — built per-equation from its \tag number.
_EQ_REF_RE_TEMPLATE = r'\b(?:(?:' + _label_alt("equation") + r')s?\.?)\s*\(?\s*{tag}\)?\b'

# A "composite" caption describes multiple sub-panels: has a figure label AND
# at least two panel markers like "(a)", "(b)", "(c）" (the latter is a common
# OCR artifact — full-width closing paren).
_PANEL_MARKER_RE = re.compile(r'\(\s*[a-z]\s*[）)]')

# A "panel fragment" caption is just a stray label with no real description,
# e.g. ['(a)'], ['(b)', '(c）'] — what's left over when MinerU misattributes a
# shared caption to one sibling and drops fragments onto the others.
_PANEL_LABEL_ONLY_RE = re.compile(r'^\(?\s*[a-z]\s*[）)]?$', re.IGNORECASE)

# A caption line that starts a standalone text chunk ("Fig. 3: …", "Table 2. …"): recovers captions MinerU emitted as
# separate text items, often when a figure sits near a page break.
_CAPTION_LINE_RE = re.compile(
    r'^\s*((?:' + _label_alt("figure") + '|' + _label_alt("table") +
    r')\.?\s*' + _CAPTION_NUMBER + r')(?:[:.](?!\d))\s*(.+)',   # "Abbildung 4.2 zeigt" is prose, not "Abbildung 4" + caption
    re.IGNORECASE,
)


# ── Split-paragraph merging ──
# MinerU emits each layout block as a separate item, so a paragraph split across a page break or around an inline
# equation, figure or table becomes several items, the later ones starting mid-sentence. The continuation's text is
# folded into the paragraph-start item before any other enrichment runs.
_SENTENCE_END_RE = re.compile(r'[.!?:;)\]]\s*$')


def _looks_like_continuation(prev: dict, item: dict) -> bool:
    """True if `item` continues the paragraph in `prev`: `prev` does not end a sentence and `item` starts with a
    lowercase letter, on the same page or across a page break."""
    prev_text = (prev.get('text') or '').strip()
    next_text = (item.get('text') or '').strip()
    if not prev_text or not next_text:
        return False
    if _SENTENCE_END_RE.search(prev_text):
        return False
    first = next_text[0]
    return first.isalpha() and first.islower()


def merge_page_boundary_paragraphs(items: list) -> int:
    """Fold continuation text items into the preceding paragraph-start item. Continuations are marked `low_info`
    (skipped at embedding) and `merged_continuation` (so context search does not count their text twice). The head item
    gets `page_end` when the merge crosses a page boundary, so citations can show the page span. Returns the number of
    items merged."""
    head = None
    n = 0
    for item in items:
        if item.get('type') == 'text' and item.get('text_level'):
            head = None  # heading — starts a new section, breaks any pending paragraph
            continue
        if item.get('type') != 'text':
            continue  # figures/tables/equations between text blocks don't break the paragraph
        if head is not None and _looks_like_continuation(head, item):
            head['text'] = (head.get('text') or '').rstrip() + ' ' + (item.get('text') or '').lstrip()
            if item.get('page_idx') != head.get('page_idx'):
                head['page_end'] = item.get('page_idx')
            item['low_info'] = True
            item['merged_continuation'] = True
            n += 1
            continue
        head = item
    return n


# ── Section assignment ────────────────────────────────────────────────────────

# Canonical section buckets, matched in order (first hit wins) against the normalized heading. Every bucket has English
# and German triggers, checked as is and with spaces removed (letter-spaced headings such as "A B S T R A C T"). Order:
# "Front Matter" before "Abstract" ("graphical abstract" is front matter); combined phrases ("Results and Discussion",
# "Zusammenfassung und Ausblick") before their parts; "Background" before "References" ("Literaturübersicht" is a
# review, not references). German "Zusammenfassung"/"Kurzfassung" map to Abstract here; assign_sections() reassigns them
# to Conclusion after the body sections.
SECTION_BUCKETS = [
    ("Front Matter", ["article info", "article history", "graphical abstract",
                      "highlights",
                      "artikelinformationen", "graphische zusammenfassung"]),
    ("Results and Discussion", ["results and discussion",
                                "ergebnisse und diskussion"]),
    # checked before the Methods bucket's "experimental": an "Experimental Results" heading is Results, while
    # "Experimental" alone is Methods
    ("Results",      ["experimental results", "experimentelle ergebnisse"]),
    ("Conclusion",   ["zusammenfassung und ausblick", "summary and outlook"]),
    ("Abstract",     ["abstract",
                      "zusammenfassung", "kurzfassung"]),
    ("Keywords",     ["keywords", "key words", "index terms",
                      "schlüsselwörter", "schlagwörter", "stichwörter",
                      "stichworte"]),
    ("Introduction", ["introduction",
                      "einleitung", "einführung"]),
    ("Background",   ["background", "related work", "literature review",
                      "hintergrund", "stand der technik", "stand der forschung",
                      "verwandte arbeiten", "literaturübersicht"]),
    ("Methods",      ["method", "experimental", "materials and methods",
                      "material and methods",
                      "methode", "methodik", "experimentell",
                      "material und methoden", "materialien und methoden",
                      "versuchsdurchführung", "durchführung"]),
    ("Results",      ["results",
                      "ergebnisse"]),
    ("Discussion",   ["discussion",
                      "diskussion"]),
    ("Conclusion",   ["conclusion", "concluding", "summary", "outlook",
                      "future work", "future scope", "future direction",
                      "perspectives",
                      "fazit", "schlussfolgerung", "schlussbemerkung",
                      "ausblick", "resümee"]),
    ("Nomenclature", ["nomenclature", "list of symbols", "list of abbreviations",
                      "nomenklatur", "formelzeichen", "symbolverzeichnis",
                      "abkürzungsverzeichnis"]),
    ("Declarations", ["acknowledg", "funding", "conflict of interest",
                      "conflicts of interest", "competing interest",
                      "declaration", "author contribution", "credit authorship",
                      "data availability", "code availability", "ethics",
                      "consent", "orcid",
                      "danksagung", "interessenkonflikt", "finanzierung",
                      "förderung", "autorenbeiträge", "datenverfügbarkeit"]),
    ("References",   ["references", "bibliography",
                      "literaturverzeichnis", "literatur", "quellenverzeichnis",
                      "referenzen", "bibliographie"]),
    ("Appendix",     ["appendix", "supplementary", "supporting information",
                      "anhang", "zusatzmaterial", "ergänzende informationen"]),
]

# Sections whose presence marks "we're past the paper's front part" — used by
# the positional Zusammenfassung→Conclusion rule in assign_sections().
_BODY_SECTIONS = {"Introduction", "Background", "Methods", "Results",
                  "Discussion", "Results and Discussion"}

# Leading section numbering: "1. Introduction", "3.1. Density evolution", "2 | Materials and Methods", "3.5.Shear
# Viscosity", "2. 2 Synthesis" (a stray space around the dot still reads as depth 2). The (?![a-zA-Z]) guard keeps a
# heading that starts with "3D …" from reading as a numbered section.
_NUMBERING_RE = re.compile(r'^\s*(\d+(?:\s*\.\s*\d+)*)(?![a-zA-Z])[\s.):|]*(.*)$')

# A "Keywords:" / "Index Terms:" / "Schlüsselwörter:" line has no heading of its own and would inherit the section
# before it (often the abstract): tagged separately so it does not pollute abstract retrieval.
_KEYWORDS_LINE_RE = re.compile(
    r'^\s*(keywords|key\s*words|index\s+terms'
    r'|schlüsselwörter|schlagwörter|stichwörter|stichworte)\s*[:.]',
    re.IGNORECASE,
)

# An abstract emitted as a body paragraph with a run-in heading ("Abstract. …", "Zusammenfassung: …")
# instead of a standalone heading. Only applied to front-matter items (before the first real section heading).
_INLINE_ABSTRACT_RE = re.compile(
    r'^\s*(abstract|zusammenfassung|kurzfassung)\s*[:.–—-]\s+\S',
    re.IGNORECASE,
)


def _bucket_match(heading: str) -> str | None:
    """Return the canonical bucket for a heading text, or None."""
    norm = re.sub(r'\s+', ' ', heading.replace('\\', '')).strip().lower()
    nospace = norm.replace(' ', '')
    for canonical, triggers in SECTION_BUCKETS:
        for trig in triggers:
            if trig in norm or trig.replace(' ', '') in nospace:
                return canonical
    return None


# Real section headings are short. A long unnumbered heading is more likely a second title, a running header or masthead
# line tagged as a heading, or a body sentence (a masthead line such as "REVIEW" can push the real title into the second
# heading slot). Numbered headings do not need this guard.
_MAX_UNNUMBERED_HEADING_WORDS = 8


# Journal furniture MinerU tags as a heading that is never a real section; only consulted on the raw-text fallback path
# (a heading matching a SECTION_BUCKETS entry is never suppressed). Kept short: repeated running headers are caught by
# the repeated-heading rule in assign_sections(); this covers masthead lines that appear once, mid-document.
_MASTHEAD_HEADING_RE = re.compile(
    r'^\s*(open\s*access\b.*|.*\bopen\s*access\s*article\s*$'
    r'|publisher(\'s)?\s*note\b.*|publisher\s*$'
    r'|online\s+content\s*$|table\s+of\s+contents\s*$'
    r'|\*?\s*corresponding\s+author\b.*|author\s+manuscript\s*$)$',
    re.IGNORECASE,
)


def _classify_heading(heading: str, allow_raw_fallback: bool = False) -> str | None:
    """Classify a heading (any text_level, see assign_sections()). Returns the canonical section it starts (possibly the
    cleaned heading itself, for a top-level section without a canonical match), or None if it should not change the
    current section: sub-numbered headings ("3.1. …"), long unnumbered headings (titles, masthead noise;
    _MAX_UNNUMBERED_HEADING_WORDS) and "Label:"-style run-in headings ("Edited by:", "Citation:").

    `allow_raw_fallback` (set once a canonical section has been seen, past the front matter) lets a short unnumbered
    heading that matches no bucket become its own section; before that, masthead labels would become spurious one-chunk
    sections."""
    text = heading.replace('\\', '').strip()
    m = _NUMBERING_RE.match(text)
    if m and m.group(1):
        depth = m.group(1).count('.') + 1
        rest = m.group(2).strip()
        if depth > 1 or not rest:
            return None  # numbered subsection ("3.1. ...") — keep current section
        # bare-integer numbering = a real top-level section even if unrecognized
        return _bucket_match(rest) or rest
    if len(text.split()) > _MAX_UNNUMBERED_HEADING_WORDS:
        return None
    bucket = _bucket_match(text)
    if bucket:
        return bucket
    if not allow_raw_fallback:
        return None
    # A trailing colon marks a run-in label whose content is a separate chunk
    # ("Correspondence:", "Citation:", "Define models:"), not a section title.
    if text.endswith(':'):
        return None
    if _MASTHEAD_HEADING_RE.match(text):
        return None
    return text


# ── Low-information fragment filtering ──
# A standalone text chunk without a heading level and with at most 2 words is sometimes a real keyword, but often
# furniture MinerU typed as body text: page numbers, all-caps running-header tags, or table-of-contents and chart labels
# repeated across the document. They can score just high enough on the reranker to fill a result slot without content,
# so they are flagged and both ingestion paths skip them, like SKIP_TYPES.
_PURE_NUMERIC_RE = re.compile(r'^[\d.,%/()\-–—]+$')


_LOW_INFO_CHAR_THRESHOLD = 50

def _is_low_information_fragment(text: str, occurrences: int, section: str | None) -> bool:
    if section == 'Keywords':
        return False
    if len(text) < _LOW_INFO_CHAR_THRESHOLD:
        return True
    words = text.split()
    if len(words) > 2:
        return False
    if _PURE_NUMERIC_RE.match(text):
        return True
    if occurrences >= 2:
        return True
    return text.isupper() and len(text) <= 15


def mark_low_information(items: list) -> int:
    """
    Set `low_info=True` on items matching _is_low_information_fragment.
    Requires a document-wide pass to count occurrences of each short text
    string first. Returns the number of items flagged.
    """
    counts: dict[str, int] = {}
    candidates = []
    for item in items:
        if item.get('type') != 'text' or item.get('text_level'):
            continue
        text = (item.get('text') or '').strip()
        if not text:
            continue
        # Old gate: only checked ≤2 words. Now also consider short fragments
        # below _LOW_INFO_CHAR_THRESHOLD regardless of word count.
        if len(text.split()) > 2 and len(text) >= _LOW_INFO_CHAR_THRESHOLD:
            continue
        counts[text] = counts.get(text, 0) + 1
        candidates.append(item)

    n = 0
    for item in candidates:
        text = item['text'].strip()
        if _is_low_information_fragment(text, counts[text], item.get('section')):
            item['low_info'] = True
            n += 1
    return n


# ── Lists ──
# MinerU puts a list's content in `list_items` and gives it no `text`, and both ingestion paths skip items without text,
# so lists (numbered conclusions, for instance) would never be embedded. Content lists get `text` (items joined by
# newlines) and are embedded like paragraphs; reference lists, tables of contents and near-empty lists are flagged
# low_info. A table of contents is recognized by dot leaders before a page number or by its heading, not by items that
# merely end in a number.
_TOC_LEADER_RE = re.compile(r'(\.{3,}|…)\s*[\dIVXLCivxlc]+\s*$')
_TOC_HEADING_RE = re.compile(r'contents|inhalt|verzeichnis|gliederung|list of (figures|tables|schemes|abbreviations)'
                             r'|\bindex\b', re.I)


# ── "…is given by:" + the equation or list that follows ──
# A paragraph that introduces an equation or a list and that equation or list are separate chunks, so retrieval could
# return the introduction without its content. The directly following equation or content list is appended to such a
# paragraph's text; the equation or list keeps its own chunk as well.
_BETWEEN_TYPES = {'page_number', 'header', 'footer', 'aside_text', 'page_footnote'}   # page furniture in between
_CONTINUATION_MAX_CHARS = 1500


def attach_continuations(items: list) -> int:
    """Append the next equation or content list to text items ending with ':'. Run after prepare_lists (lists need
    their `text`). Returns the number of paragraphs completed."""
    n = 0
    for i, item in enumerate(items):
        text = (item.get('text') or '').rstrip()
        if item.get('type') != 'text' or item.get('text_level') or item.get('low_info') or not text.endswith(':'):
            continue
        j = i + 1
        while j < len(items) and items[j].get('type') in _BETWEEN_TYPES:
            j += 1
        nxt = items[j] if j < len(items) else {}
        cont = (nxt.get('text') or '').strip()
        if nxt.get('type') not in ('equation', 'list') or nxt.get('low_info') or not cont:
            continue
        if len(cont) > _CONTINUATION_MAX_CHARS:
            cont = cont[:_CONTINUATION_MAX_CHARS].rstrip() + ' …'
        item['text'] = f"{text}\n{cont}"
        item['continued_by'] = nxt.get('type')
        n += 1
    return n


def prepare_lists(items: list) -> dict:
    """Give content lists their `text`; flag references, tables of contents and near-empty lists low_info.
    Needs `section`/`section_raw` (run after assign_sections). Returns counts per kind, for logging."""
    stats = {"content": 0, "references": 0, "toc": 0, "empty": 0}
    for item in items:
        if item.get('type') != 'list':
            continue
        entries = [str(x).strip() for x in (item.get('list_items') or []) if str(x).strip()]
        text = '\n'.join(entries)
        if not text:
            stats["empty"] += 1
            continue
        item['text'] = text
        if item.get('sub_type') == 'ref_text' or item.get('section') == 'References':
            kind = "references"
        elif (sum(bool(_TOC_LEADER_RE.search(e)) for e in entries) / len(entries) > 0.5
              or _TOC_HEADING_RE.search(item.get('section_raw') or '')):
            kind = "toc"
        elif len(text) < _LOW_INFO_CHAR_THRESHOLD:
            kind = "empty"
        else:
            kind = "content"
        if kind != "content":
            item['low_info'] = True
        stats[kind] += 1
    return stats


def assign_sections(items: list) -> dict:
    """Annotate every item (in place) with `section` (canonical bucket) and `section_raw` (nearest heading verbatim).
    Returns {canonical: count} over non-heading items, for logging.

    The paper title is the first heading by position, not by text_level: this VLM parse backend gives every heading
    text_level 1, so a level-based rule would make every section heading a second title."""
    section, raw = 'Front Matter', None
    seen_body = False
    seen_first_heading = False
    # True once a heading matched a real SECTION_BUCKETS entry — i.e. we're
    # past the front matter/masthead, so an unrecognized short heading is far
    # more likely a real (thematically named, unnumbered) section than journal
    # furniture. Gates the raw-text fallback in _classify_heading().
    seen_canonical = False
    _CANONICAL = {c for c, _ in SECTION_BUCKETS}
    # Headings that repeat verbatim in this document are running headers, not section starts (a real section heading
    # appears once). Only used to veto the raw-text fallback; a canonical match is never suppressed.
    _heading_counts: dict[str, int] = {}
    for _it in items:
        if _it.get('text_level'):
            _t = (_it.get('text') or '').replace('\\', '').strip()
            _heading_counts[_t] = _heading_counts.get(_t, 0) + 1
    _repeated = {t for t, c in _heading_counts.items() if c >= 2}
    # the section the 'Keywords' run started from: restored if a non-canonical heading fails to move off 'Keywords'
    # before body content resumes, so 'Keywords' does not stick to the rest of the document (and exclude it from
    # retrieval)
    pre_keywords_section = 'Front Matter'
    KEYWORDS_MAX_WORDS = 15  # individual keyword terms/phrases are short; a
                              # longer item is body content, not a keyword.
    stats: dict[str, int] = {}
    for item in items:
        text = (item.get('text') or '').strip()
        level = item.get('text_level')
        if level and not seen_first_heading:
            seen_first_heading = True
            # The paper title — following chunks (authors, affiliations,
            # date, an unheaded abstract) are front matter, not "the <title>
            # section".
            section, raw = 'Front Matter', text
        elif level:
            new_section = _classify_heading(text, allow_raw_fallback=seen_canonical)
            if (new_section is not None and new_section not in _CANONICAL
                    and new_section.strip() in _repeated):
                new_section = None  # running header, not a section start
            if new_section:
                seen_canonical = seen_canonical or new_section in _CANONICAL
                if new_section == 'Abstract' and seen_body:
                    # German papers: a trailing "Zusammenfassung" (after the
                    # body sections) is the conclusion, not the abstract
                    new_section = 'Conclusion'
                section = new_section
                seen_body = seen_body or new_section in _BODY_SECTIONS
            raw = text
        item_section = section
        if not level and item.get('type') in SURROUNDING_TEXT_TYPES:
            if _KEYWORDS_LINE_RE.match(text):
                # keep 'Keywords' as the current section too: the single keyword terms that follow (no heading of their
                # own) would otherwise inherit the section before and miss the Keywords handling in
                # _is_low_information_fragment() and retrieval filtering
                if section != 'Keywords':
                    pre_keywords_section = section
                section = item_section = 'Keywords'
            elif section == 'Keywords' and len(text.split()) > KEYWORDS_MAX_WORDS:
                # a non-canonical heading between the keyword list and here failed to classify, so the section stayed
                # 'Keywords', but this item is body content: stop the sticky assignment rather than exclude the rest of
                # the section from retrieval
                section = pre_keywords_section
                item_section = section
            elif section == 'Front Matter' and _INLINE_ABSTRACT_RE.match(text):
                item_section = 'Abstract'
        item['section'] = item_section
        item['section_raw'] = raw
        if not level:
            stats[item_section] = stats.get(item_section, 0) + 1
    return stats


def _main_label(pattern: re.Pattern, caption: str) -> str | None:
    # a supplementary / extended-data caption gets no label: it would otherwise be matched as the main figure
    caption = caption.strip()
    m = pattern.search(caption)
    if not m or _SUPPLEMENT_BEFORE_RE.search(caption[max(0, m.start() - 40):m.start()]):
        return None
    return re.sub(r'(?<=\d)\.\s+(?=\d)', '.', m.group(1))


def _extract_figure_label(caption: str) -> str | None:
    return _main_label(_FIG_LABEL_RE, caption)


def _extract_table_label(caption: str) -> str | None:
    return _main_label(_TABLE_LABEL_RE, caption)


def _caption_list(item: dict) -> list[str]:
    return (item.get('image_caption') or []) if item.get('type') == 'image' else (item.get('chart_caption') or [])


def _is_composite_caption(caption_list: list[str]) -> bool:
    """True if caption describes multiple sub-panels of one figure, e.g.
    'Figure 5: (a) … (b) … (c) …'"""
    full = ' '.join(caption_list)
    return bool(_FIG_LABEL_RE.search(full)) and len(_PANEL_MARKER_RE.findall(full)) >= 2


def _is_panel_fragment(caption_list: list[str]) -> bool:
    """True if caption is empty or just stray panel-label fragments like ['(a)', '(b)']
    — i.e. not a real description, and a candidate for recovery."""
    if not caption_list:
        return True
    return all(_PANEL_LABEL_ONLY_RE.match(c.strip()) for c in caption_list if c.strip())


def _bbox_overlap_fraction(b1: list, b2: list, axis: int) -> float:
    """Fraction of overlap between two bboxes [x0,y0,x1,y1] along one axis
    (axis=1 -> y-overlap, for horizontally arranged panels sharing a row;
     axis=0 -> x-overlap, for vertically stacked panels sharing a column),
    relative to the smaller box's extent on that axis."""
    lo, hi = max(b1[axis], b2[axis]), min(b1[axis + 2], b2[axis + 2])
    overlap = max(0, hi - lo)
    span = min(b1[axis + 2] - b1[axis], b2[axis + 2] - b2[axis])
    return overlap / span if span else 0.0


def _adjacent_on_cross_axis(b1: list, b2: list, axis: int) -> bool:
    """True if b1/b2 are close together on the axis PERPENDICULAR to the one
    `_bbox_overlap_fraction` checked (cross_axis = 1-axis) — i.e. the actual
    stacking direction for that layout (column stacking checked via x-overlap
    at axis=0 means the cross/stacking axis is y=1, and vice versa). Gap is
    measured relative to the panels' own extent on the cross axis, so this
    doesn't fire on genuinely separate figures that just happen to share a
    column/row on the same page."""
    cross = 1 - axis
    gap = max(b1[cross] - b2[cross + 2], b2[cross] - b1[cross + 2])
    if gap <= 0:
        return True  # already overlapping/touching on the cross axis too
    extent = min(b1[cross + 2] - b1[cross], b2[cross + 2] - b2[cross])
    return extent > 0 and gap <= extent * SIBLING_MAX_GAP_RATIO


def _find_sibling_group_caption(all_items: list, idx: int) -> str | None:
    """Multi-panel figures split into separate image items: when an image's own caption is empty or a stray panel label,
    look for a nearby image/chart on the same page whose box overlaps this one along an axis (sub-panels of one figure)
    and carries the full caption, and return that caption for this panel too."""
    item = all_items[idx]
    page, bbox = item.get('page_idx'), item.get('bbox')
    if not bbox:
        return None

    for offset in range(1, SIBLING_SEARCH_WINDOW + 1):
        for j in (idx - offset, idx + offset):
            if not (0 <= j < len(all_items)):
                continue
            sib = all_items[j]
            if sib.get('type') not in IMAGE_TYPES or sib.get('page_idx') != page:
                continue
            sib_bbox = sib.get('bbox')
            if not sib_bbox:
                continue
            for axis in (1, 0):  # try row layout (y-overlap), then column layout (x-overlap)
                if (_bbox_overlap_fraction(bbox, sib_bbox, axis) >= BBOX_OVERLAP_THRESHOLD
                        and _adjacent_on_cross_axis(bbox, sib_bbox, axis)):
                    cap = _caption_list(sib)
                    if _is_composite_caption(cap):
                        return ' '.join(cap)
    return None


def _find_adjacent_caption(all_items: list, idx: int) -> str | None:
    """A caption emitted as an orphaned text item: when an image's own caption is empty, check the next 1-2 text/list
    items for a line starting with a figure label ("Fig. 3: …"), also across a page break (reading order, page_idx
    ignored)."""
    for j in range(idx + 1, min(len(all_items), idx + 1 + ADJACENT_CAPTION_WINDOW)):
        item = all_items[j]
        if item.get('type') not in SURROUNDING_TEXT_TYPES:
            continue
        text = item.get('text', '').strip()
        if _CAPTION_LINE_RE.match(text):
            return text
    return None


def _excerpt_around(text: str, pos: int, max_chars: int = SURROUNDING_MAX_CHARS) -> str:
    """
    Return a ~max_chars excerpt of `text` centered on index `pos`, so the
    excerpt actually contains whatever is at `pos` (e.g. a "Figure N" mention)
    instead of just the start of the paragraph.
    """
    half  = max_chars // 2
    start = max(0, pos - half)
    end   = min(len(text), start + max_chars)
    start = max(0, end - max_chars)  # re-anchor if we hit the end of text
    return text[start:end]


def _iter_label_matches(all_items: list, label: str, exclude_idx: int | None = None) -> list[tuple[int, str, int]]:
    """
    Scan the whole document for text/list items whose text mentions `label`
    (e.g. "Fig. 3", "Table 1"). The label's word counts in any spelling of its
    family (_label_families: "Fig. 3" also finds "Figure 3", "Figs. 3",
    "Fig 3"; "Abb. 3" finds "Abbildung 3"; "Tabelle 2" finds "Tab. 2"); the
    number must match exactly ("4.2" is not "4", "1" is not "10"), and a
    mention right after "Supplementary" is skipped. Labels whose word is not
    in the vocabulary keep the literal search.

    Returns one (idx, text, match_start) tuple per matching item, in document
    order. Retries with the base label (trailing sub-figure letter stripped,
    e.g. "Fig. 3a" -> "Fig. 3") if the exact label has no matches, and then
    with lists and ranges of its family ("Figs. 3 and 4", "Figures 3.31 to
    3.35", "Abb. 2–4") if nothing cites the label on its own.
    """
    def _pattern(search_label: str) -> re.Pattern:
        parts = _LABEL_PARTS_RE.match(search_label)
        family = next((f for f in _LABEL_FAMILIES if parts and parts.group(1).lower() in f), None)
        if family:
            # any word of the label's family, then the exact number; "4.2" must not continue as "4.21", and "4" must
            # not continue as "4.2" (a chapter-numbered figure is another figure), while "3" still matches "3a"
            words = "|".join(re.escape(w) for w in sorted(family, key=len, reverse=True))
            number = re.escape(parts.group(2) + parts.group(3))
            # "Table 2-2" (hyphen numbering) is not Table 2; "Figs. 3-5" is found by the list tier below
            return re.compile(r'\b(?:' + words + r')\.?\s*' + number + r'(?!\d|\.\d|-\d)', re.IGNORECASE)
        escaped = re.escape(search_label)
        escaped = escaped.replace(r'\.', r'\.?').replace(r'\ ', r'\s*')
        return re.compile(r'\b' + escaped + r'(?!\d)', re.IGNORECASE)

    def _search(search_label: str) -> list[tuple[int, str, int]]:
        # the trailing-digit guard keeps "Fig. 1" from matching inside "Fig. 10" to "Fig. 19", while "3a"/"3b" still
        # match "Fig. 3" (a plain word boundary would reject those)
        pattern = _pattern(search_label)
        hits = []
        for idx, item in enumerate(all_items):
            if idx == exclude_idx:
                continue
            if item.get('type') not in SURROUNDING_TEXT_TYPES:
                continue
            if item.get('merged_continuation'):
                continue
            text = item.get('text', '').strip()
            if len(text) < 20:
                continue
            m = next((m for m in pattern.finditer(text)
                      if not _SUPPLEMENT_BEFORE_RE.search(text[max(0, m.start() - 40):m.start()])), None)
            if m:
                hits.append((idx, text, m.start()))
        return hits

    def _search_lists(search_label: str) -> list[tuple[int, str, int]]:
        # the label's number inside a list or range of its family ("Figs. 3 and 4", "Figures 3.31 to 3.35"); a range
        # stays within one chapter and spans at most _MAX_RANGE numbers
        parts = _LABEL_PARTS_RE.match(search_label)
        family = next((f for f in _LABEL_FAMILIES if parts and parts.group(1).lower() in f), None)
        if not family:
            return []
        target = parts.group(2)
        words = "|".join(re.escape(w) for w in sorted(family, key=len, reverse=True))
        pattern = re.compile(r'\b(?:' + words + r')\.?\s*(' + _LABEL_NUMBER + _LIST_TAIL + r')', re.IGNORECASE)
        # a book that numbers its figures "Figure 3-8" (chapter 3, figure 8) uses the hyphen for numbering, not ranges
        hyphen_numbered = any(re.match(r'\s*[^\W\d_]+\.?\s*\d+-\d', c)
                              for it in all_items if it.get('type') in ('image', 'chart', 'table')
                              for c in (it.get('image_caption') or it.get('chart_caption') or it.get('table_caption')
                                        or [])[:1] if isinstance(c, str))
        hits = []
        for idx, item in enumerate(all_items):
            if idx == exclude_idx or item.get('type') not in SURROUNDING_TEXT_TYPES or item.get('merged_continuation'):
                continue
            text = item.get('text', '').strip()
            if len(text) < 20:
                continue
            for m in pattern.finditer(text):
                if _SUPPLEMENT_BEFORE_RE.search(text[max(0, m.start() - 40):m.start()]):
                    continue
                pieces = re.split(r'(' + _RANGE_JOIN + r'|' + _LIST_JOIN + r')', m.group(1), flags=re.IGNORECASE)
                numbers = [re.sub(r'[a-zA-Z]$', '', n) for n in pieces[0::2]]
                if len({n.count('.') for n in numbers}) != 1:
                    continue  # "Fig. 3 and 4.2 wt%" is no list of figures
                cited = set(numbers)
                for (a, join), b in zip(zip(numbers, pieces[1::2]), numbers[1:]):
                    head_a, _, last_a = a.rpartition('.')
                    head_b, _, last_b = b.rpartition('.')
                    if hyphen_numbered and join == '-':
                        cited = set()  # "Figure 3-8" is one figure of chapter 3, not Figures 3 to 8
                        break
                    if re.fullmatch(_RANGE_JOIN, join, re.IGNORECASE) and head_a == head_b:
                        if not 0 < int(last_b) - int(last_a) <= _MAX_RANGE:
                            cited = set()  # "Table 2-2", "Figure 10-25": numbering, not a range
                            break
                        cited.update((head_a + '.' if head_a else '') + str(k)
                                     for k in range(int(last_a), int(last_b) + 1))
                if target in cited:
                    hits.append((idx, text, m.start()))
                    break
        return hits

    hits = _search(label)
    base_label = re.sub(r'[a-zA-Z]$', '', label).strip()
    if not hits and base_label != label:
        hits = _search(base_label)
    if not hits:
        # only when nothing cites the label on its own: the shared passage of a list or range
        hits = _search_lists(base_label)
    return hits


def find_all_label_reference_indices(all_items: list, label: str, exclude_idx: int | None = None) -> list[int]:
    """Exhaustive version of `_find_referencing_chunks` for build_graph.py: every item index whose text mentions
    `label`, with no cap and no proximity bound (one REFERENCES edge per citing chunk). Same tiers as the enrichment:
    list and range citations count only when no passage cites the label on its own."""
    return [idx for idx, _, _ in _iter_label_matches(all_items, label, exclude_idx)]


def _find_referencing_chunks(all_items: list, label: str, image_idx: int) -> list[str]:
    """
    Search entire document for text chunks explicitly mentioning label.
    Returns up to 3 matches sorted by proximity to image_idx, each as an
    excerpt centered on the label mention itself.
    Retries with base label (strips trailing sub-figure letter) if no matches.
    """
    hits = _iter_label_matches(all_items, label, exclude_idx=image_idx)
    hits = sorted(hits, key=lambda x: abs(x[0] - image_idx))
    return [_excerpt_around(text, pos) for _, text, pos in hits[:3]]


def _extract_equation_tag(eq_text: str) -> str | None:
    """Extract the equation number from a LaTeX `\\tag {N}` (e.g. "3", "10a"),
    or None if the equation is unnumbered."""
    m = _EQ_TAG_RE.search(eq_text)
    return m.group(1) if m else None


def _iter_equation_matches(all_items: list, tag: str, exclude_idx: int | None = None) -> list[tuple[int, str, int]]:
    """
    Scan the whole document for text/list items that cite equation `tag` by
    number (e.g. "Eq. (3)", "Equation 3"). Returns one (idx, text, match_start)
    tuple per matching item, in document order.
    """
    pattern = re.compile(_EQ_REF_RE_TEMPLATE.format(tag=re.escape(tag)), re.IGNORECASE)
    hits = []
    for idx, item in enumerate(all_items):
        if idx == exclude_idx:
            continue
        if item.get('type') not in SURROUNDING_TEXT_TYPES:
            continue
        if item.get('merged_continuation'):
            continue
        text = item.get('text', '').strip()
        if len(text) < 20:
            continue
        m = pattern.search(text)
        if m:
            hits.append((idx, text, m.start()))
    return hits


def find_all_equation_reference_indices(all_items: list, tag: str, exclude_idx: int | None = None) -> list[int]:
    """
    Exhaustive version of `_find_equation_referencing_chunks` for
    build_graph.py: every item index in `all_items` that cites equation `tag`,
    with no cap and no proximity bound.
    """
    return [idx for idx, _, _ in _iter_equation_matches(all_items, tag, exclude_idx)]


def _find_equation_referencing_chunks(all_items: list, tag: str, eq_idx: int) -> list[str]:
    """
    Search the document for text chunks that cite this equation by number
    (e.g. "Eq. (3)", "Equation 3"). Returns up to 3 matches sorted by
    proximity to eq_idx, each as an excerpt centered on the mention.
    """
    hits = _iter_equation_matches(all_items, tag, exclude_idx=eq_idx)
    hits = sorted(hits, key=lambda x: abs(x[0] - eq_idx))
    return [_excerpt_around(text, pos) for _, text, pos in hits[:3]]


def _positional_surrounding(all_items: list, idx: int, page: int) -> list[str]:
    """
    Fallback: nearest text chunk before and after idx within ±1 page.
    """
    results = []
    for direction in [-1, 1]:
        r = range(idx + direction,
                  max(-1, idx - SURROUNDING_WINDOW - 1) if direction == -1
                  else min(len(all_items), idx + SURROUNDING_WINDOW + 1),
                  direction)
        for j in r:
            c = all_items[j]
            if c.get('type') in SURROUNDING_TEXT_TYPES and not c.get('merged_continuation'):
                if page >= 0 and c.get('page_idx', -1) >= 0 and abs(c.get('page_idx') - page) <= 1:
                    t = c.get('text', '').strip()
                    if len(t) > 20:
                        # direction=-1 (the preceding chunk): the relevant
                        # sentence is at the END of that chunk (e.g. "...is
                        # given by:" introducing an equation, or the sentence
                        # right before a figure) — take the tail, not the
                        # head. direction=1 (the following chunk): the
                        # relevant sentence is at the start.
                        excerpt = t[-SURROUNDING_MAX_CHARS:] if direction == -1 else t[:SURROUNDING_MAX_CHARS]
                        results.append(excerpt)
                        break
    return results


# ── Multi-panel figures: one item per figure ──
# MinerU often saves each panel of a figure as its own image item, with the real caption on one panel only and a label
# or nothing on the others; one passage per panel would make a citation show one panel. merge_figure_panels() folds such
# panels into one item.
PANEL_SKIP_TYPES   = {'header', 'footer', 'page_number', 'aside_text', 'page_footnote'}   # may sit between panels
PANEL_BLOCK_TYPES  = {'text', 'list', 'table', 'equation'}      # the union box must not cover these
PANEL_TEXT_OVERLAP = 0.10       # max fraction of a text block's area the union box may cover
PANEL_CROP_ZOOM    = 2          # crop resolution: 2 = 144 dpi
PANEL_CROP_MARGIN  = (10, 20)   # crop margin (x, y) in 0-1000 page units: panel labels ("(a) …") sit just outside
                                # the panel boxes (MinerU reads them as captions); stops at a neighbouring text block


def mark_qr_codes(items: list, base_dir: Path | None) -> int:
    """Flag images that are QR codes (a journal's link to the article): `qr_code` and `low_info`, so they are not
    embedded and never join a multi-panel figure. Returns how many."""
    if base_dir is None:
        return 0
    try:
        from memento.common.qr_code import is_qr_code
    except Exception:                                  # never let this check stop an upload
        return 0
    n = 0
    for b in items:
        if b.get('type') in IMAGE_TYPES and b.get('img_path') and is_qr_code(Path(base_dir) / b['img_path']):
            b['qr_code'] = b['low_info'] = True
            n += 1
    return n


def _panel_groups(items: list) -> list[list[int]]:
    """Indices of the image/chart items that form one figure: runs of consecutive image/chart items on one page
    (headers, footers, page numbers and side text may sit between them), split at the items whose caption names a
    figure ("Figure 1", "Fig. 4", …, anywhere in the caption). Each captionless item joins the next captioned one;
    trailing ones join the one before. Only groups of two or more with exactly one captioned item."""
    runs, cur = [], []
    for i, b in enumerate(items):
        kind = b.get('type')
        if b.get('qr_code'):                          # a QR code sits beside figures but is none of their panels
            continue
        if kind in IMAGE_TYPES:
            if cur and items[cur[-1]].get('page_idx') != b.get('page_idx'):
                runs.append(cur)
                cur = []
            cur.append(i)
        elif kind in PANEL_SKIP_TYPES:
            continue
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    groups = []
    for run in runs:
        anchors = [i for i in run if _FIG_LABEL_RE.search(' '.join(_caption_list(items[i])))]
        if not anchors:
            continue
        members = {a: [] for a in anchors}
        for i in run:
            if i in members:
                members[i].append(i)
                continue
            later = [a for a in anchors if a > i]
            members[later[0] if later else anchors[-1]].append(i)
        groups += [sorted(m) for m in members.values() if len(m) > 1]
    return groups


def _union_bbox(boxes: list) -> list:
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]


def _covers_text(items: list, page, box: list) -> bool:
    """True if `box` covers more than PANEL_TEXT_OVERLAP of any text, list, table or equation block on `page`."""
    for b in items:
        bb = b.get('bbox')
        if b.get('page_idx') != page or b.get('type') not in PANEL_BLOCK_TYPES or not bb:
            continue
        w = max(0, min(box[2], bb[2]) - max(box[0], bb[0]))
        h = max(0, min(box[3], bb[3]) - max(box[1], bb[1]))
        area = max(1, (bb[2] - bb[0]) * (bb[3] - bb[1]))
        if w * h / area > PANEL_TEXT_OVERLAP:
            return True
    return False


def _crop_box(items: list, page, box: list) -> list:
    """`box` widened by PANEL_CROP_MARGIN on each side, but never past the edge of a text, list, table or equation
    block of the page lying on that side, nor past the page."""
    mx, my = PANEL_CROP_MARGIN
    x0, y0, x1, y1 = box[0] - mx, box[1] - my, box[2] + mx, box[3] + my
    for b in items:
        bb = b.get('bbox')
        if b.get('page_idx') != page or b.get('type') not in PANEL_BLOCK_TYPES or not bb:
            continue
        if bb[0] < box[2] and bb[2] > box[0]:          # overlaps the figure horizontally: limits top / bottom
            if bb[3] <= box[1]:
                y0 = max(y0, bb[3])
            if bb[1] >= box[3]:
                y1 = min(y1, bb[1])
        if bb[1] < box[3] and bb[3] > box[1]:          # overlaps vertically: limits left / right
            if bb[2] <= box[0]:
                x0 = max(x0, bb[2])
            if bb[0] >= box[2]:
                x1 = min(x1, bb[0])
    return [max(0, x0), max(0, y0), min(1000, x1), min(1000, y1)]


def _figure_image(members: list, box: list, out: Path, base_dir: Path | None, source_pdf: Path | None,
                  crop_box: list | None = None) -> str:
    """Write the whole figure to `out`: a crop of the PDF page at `box` (MinerU's 0-1000 page-normalized
    coordinates, as services/file_server.py reads them), else the panels' own images placed on a white canvas at their
    relative positions. The page crop uses `crop_box` when given (the box plus a margin for panel labels).
    Without a usable PDF an image already written for these panels is kept ("reused"): the
    graph builder and some scripts re-run the enrichment without the PDF and must not replace a page crop.
    Returns "crop", "canvas" or "reused"; raises when no image can be made."""
    import fitz  # PyMuPDF, used across ingestion and the file server
    page_idx = members[0].get('page_idx', 0)
    has_pdf = bool(source_pdf) and Path(source_pdf).is_file()
    if not has_pdf and out.is_file():
        return 'reused'
    if has_pdf:
        try:
            with fitz.open(str(source_pdf)) as doc:
                page = doc[page_idx]
                r, c = page.rect, crop_box or box
                clip = fitz.Rect(r.x0 + c[0] * r.width / 1000, r.y0 + c[1] * r.height / 1000,
                                 r.x0 + c[2] * r.width / 1000, r.y0 + c[3] * r.height / 1000)
                page.get_pixmap(matrix=fitz.Matrix(PANEL_CROP_ZOOM, PANEL_CROP_ZOOM), clip=clip).save(str(out))
            return 'crop'
        except Exception as e:
            print(f"  [figure merge] crop failed for page {page_idx + 1}, placing the panels instead: {e!r}")
    paths = [(base_dir / m['img_path']) if base_dir and m.get('img_path') else None for m in members]
    if not all(p and p.is_file() for p in paths):
        raise FileNotFoundError('panel images missing')
    doc = fitz.open()
    try:
        page = doc.new_page(width=box[2] - box[0], height=box[3] - box[1])
        for m, p in zip(members, paths):
            b = m['bbox']
            page.insert_image(fitz.Rect(b[0] - box[0], b[1] - box[1], b[2] - box[0], b[3] - box[1]), filename=str(p))
        page.get_pixmap(matrix=fitz.Matrix(PANEL_CROP_ZOOM * 2, PANEL_CROP_ZOOM * 2)).save(str(out))
    finally:
        doc.close()
    return 'canvas'


def merge_figure_panels(items: list, images_dir: Path | None, source_pdf: Path | None = None,
                        dry_run: bool = False) -> dict:
    """Fold the panels of one figure into a single item, in place: union box, the members' captions in order, one
    image of the whole figure (images/figure_<hash>.jpg under images_dir), and `merged_panels` recording the members.
    Refused (left as they are) when the union box would cover a text block of the page, or when no image can be made.
    dry_run: count only, write nothing. Returns {"groups", "panels", "refused", "crop", "canvas", "reused"}."""
    import hashlib
    stats = {'groups': 0, 'panels': 0, 'refused': 0, 'crop': 0, 'canvas': 0, 'reused': 0}
    base_dir = Path(images_dir).parent if images_dir else None
    replace: dict[int, dict] = {}
    drop: set[int] = set()
    for group in _panel_groups(items):
        members = [items[i] for i in group]
        if not all(m.get('bbox') for m in members):
            continue
        box = _union_bbox([m['bbox'] for m in members])
        page = members[0].get('page_idx')
        if _covers_text(items, page, box):
            stats['refused'] += 1
            continue
        anchor = next(i for i in group if _FIG_LABEL_RE.search(' '.join(_caption_list(items[i]))))
        name = 'figure_' + hashlib.sha1('|'.join(str(m.get('img_path')) for m in members).encode()).hexdigest()[:16]
        rel = f'images/{name}.jpg'
        if not dry_run:
            try:
                Path(images_dir).mkdir(parents=True, exist_ok=True)
                with PYMUPDF:
                    how = _figure_image(members, box, Path(images_dir) / f'{name}.jpg', base_dir, source_pdf,
                                        crop_box=_crop_box(items, page, box))
            except Exception as e:
                print(f"  [figure merge] page {page + 1 if isinstance(page, int) else page}: left as panels ({e!r})")
                stats['refused'] += 1
                continue
            stats[how] += 1
        kind = 'chart' if all(m.get('type') == 'chart' for m in members) else 'image'
        merged = {k: v for k, v in items[anchor].items()
                  if k not in ('image_caption', 'chart_caption', 'image_footnote', 'chart_footnote', 'img_path')}
        merged.update(type=kind, bbox=box, img_path=rel,
                      merged_panels=[{'bbox': m['bbox'], 'img_path': m.get('img_path'),
                                      'caption': ' '.join(_caption_list(m))} for m in members])
        merged[f'{kind}_caption'] = [c for m in members for c in _caption_list(m)]
        merged[f'{kind}_footnote'] = [f for m in members for f in (m.get('image_footnote') or m.get('chart_footnote') or [])]
        replace[anchor] = merged
        drop.update(i for i in group if i != anchor)
        stats['groups'] += 1
        stats['panels'] += len(group)
    if replace:
        items[:] = [replace.get(i, b) for i, b in enumerate(items) if i not in drop]
    return stats


def enrich_content_list(content_list_path: Path, source_pdf: Path | None = None) -> Path:
    """
    Read content_list.json, enrich image/chart items with author reference text
    and annotate every item with section/section_raw, write
    content_list_enriched.json. Returns path to the enriched file.
    source_pdf: the parsed PDF, for cropping merged multi-panel figures from
    the page (without it the panels' own images are placed together).
    """
    content_list_path = Path(content_list_path)
    with open(content_list_path) as f:
        items = json.load(f)

    # first, while each element still sits on its own page
    corrected = correct_items(items, source_pdf)['changes'] if source_pdf else []
    changes_file = content_list_path.parent / 'text_layer_changes.json'
    if corrected:
        with open(changes_file, 'w') as f:
            json.dump(corrected, f, ensure_ascii=False, indent=1)
    else:                       # an earlier run's list would describe text that is no longer there
        changes_file.unlink(missing_ok=True)

    n_merged = merge_page_boundary_paragraphs(items)
    n_qr = mark_qr_codes(items, content_list_path.parent)
    fig_stats = merge_figure_panels(items, content_list_path.parent / 'images', source_pdf)

    enriched = []
    stats = {'label_match': 0, 'positional_fallback': 0, 'none': 0}
    recovery_stats = {'sibling_group': 0, 'adjacent_text': 0}
    eq_stats = {'positional': 0, 'positional+label': 0, 'label_match': 0, 'none': 0}
    table_stats = {'label_match': 0, 'positional_fallback': 0, 'none': 0}

    for idx, item in enumerate(items):
        item = dict(item)  # shallow copy — don't mutate original

        if item.get('type') in IMAGE_TYPES:
            caption_list = _caption_list(item)
            caption = ' '.join(caption_list)
            page = item.get('page_idx', -1)

            # Caption recovery: when MinerU's own caption is empty or a stray panel label, try to recover the real
            # caption before falling back to the surrounding-text search.
            recovered_caption: str | None = None
            recovery_method: str | None = None
            if not caption.strip() or _is_panel_fragment(caption_list):
                recovered_caption = _find_sibling_group_caption(items, idx)
                if recovered_caption:
                    recovery_method = 'sibling_group'
                else:
                    recovered_caption = _find_adjacent_caption(items, idx)
                    if recovered_caption:
                        recovery_method = 'adjacent_text'

            effective_caption = recovered_caption or caption

            label = _extract_figure_label(effective_caption) if effective_caption else None
            table_label = None
            if not label and effective_caption:
                # "chart-as-table": a chart item whose caption is "Table N: …" (the VLM backend sometimes emits tabular
                # content as a chart with a markdown `content` field). Only the caption's own leading label counts: a
                # later "compare with Table 1" is a cross-reference.
                table_label = _extract_table_label(effective_caption)
                if table_label and effective_caption.strip().find(table_label) > 4:
                    table_label = None
            ref_label = label or table_label

            ref_chunks: list[str] = []
            method = 'none'

            if ref_label:
                ref_chunks = _find_referencing_chunks(items, ref_label, idx)
                if ref_chunks:
                    method = 'label_match'

            if not ref_chunks:
                ref_chunks = _positional_surrounding(items, idx, page)
                method = 'positional_fallback' if ref_chunks else 'none'

            # The recovered caption IS the author's description of this figure —
            # put it first in enriched_context, ahead of generic reference chunks.
            context_parts = ([recovered_caption] if recovered_caption else []) + ref_chunks

            item['enriched_context'] = '\n'.join(context_parts)
            item['enrichment_method'] = method
            if label:
                item['figure_label'] = label
            if table_label:
                item['table_label'] = table_label
            if recovery_method:
                item['recovered_caption'] = recovered_caption
                item['caption_recovery_method'] = recovery_method
                recovery_stats[recovery_method] += 1
            stats[method] += 1

        elif item.get('type') == 'table':
            caption_list = item.get('table_caption') or []
            caption = ' '.join(caption_list)
            page = item.get('page_idx', -1)

            table_label = _extract_table_label(caption) if caption else None
            ref_chunks = []
            method = 'none'

            if table_label:
                ref_chunks = _find_referencing_chunks(items, table_label, idx)
                if ref_chunks:
                    method = 'label_match'

            if not ref_chunks:
                ref_chunks = _positional_surrounding(items, idx, page)
                method = 'positional_fallback' if ref_chunks else 'none'

            item['enriched_context'] = '\n'.join(ref_chunks)
            item['enrichment_method'] = method
            if table_label:
                item['table_label'] = table_label
            table_stats[method] += 1

        elif item.get('type') == 'equation':
            # Authors introduce an equation with the sentence right before it and often define its symbols in the
            # sentence after, so positional context is the primary signal here (unlike figures, where it is a fallback);
            # label citations ("Eq. (3)") are added when present.
            page = item.get('page_idx', -1)
            surrounding = _positional_surrounding(items, idx, page)
            tag = _extract_equation_tag(item.get('text', ''))
            label_refs = _find_equation_referencing_chunks(items, tag, idx) if tag else []
            label_refs = [r for r in label_refs if r not in surrounding]

            context_parts = surrounding + label_refs
            if surrounding and label_refs:
                eq_method = 'positional+label'
            elif surrounding:
                eq_method = 'positional'
            elif label_refs:
                eq_method = 'label_match'
            else:
                eq_method = 'none'

            item['enriched_context'] = '\n'.join(context_parts)
            item['enrichment_method'] = eq_method
            if tag:
                item['equation_tag'] = tag
            eq_stats[eq_method] += 1

        enriched.append(item)

    section_stats = assign_sections(enriched)
    list_stats = prepare_lists(enriched)
    n_continued = attach_continuations(enriched)
    n_low_info = mark_low_information(enriched)

    out_path = content_list_path.parent / (
        content_list_path.name.replace('_content_list', '_content_list_enriched')
    )
    assert out_path != content_list_path, (
        f"Output path equals input path — filename must contain '_content_list': {content_list_path}"
    )
    # Atomic replace: parallel ingestion runs may enrich the same file; readers never see a partial one.
    tmp_path = out_path.with_name(f"{out_path.name}.{os.getpid()}.tmp")
    with open(tmp_path, 'w') as f:
        json.dump(enriched, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, out_path)

    if n_merged:
        print(f"  Merged {n_merged} page-boundary paragraph continuation(s)")
    if fig_stats['groups'] or fig_stats['refused']:
        print(f"  Merged {fig_stats['groups']} multi-panel figure(s) ({fig_stats['panels']} panels; "
              f"{fig_stats['crop']} cropped from the page, {fig_stats['canvas']} placed); "
              f"{fig_stats['refused']} left as panels")
    img_total = stats['label_match'] + stats['positional_fallback'] + stats['none']
    print(f"  Enriched {img_total} image/chart items: "
          f"{stats['label_match']} label match, "
          f"{stats['positional_fallback']} positional fallback, "
          f"{stats['none']} no context found")
    if recovery_stats['sibling_group'] or recovery_stats['adjacent_text']:
        print(f"  Recovered {recovery_stats['sibling_group'] + recovery_stats['adjacent_text']} "
              f"missing/fragment captions: "
              f"{recovery_stats['sibling_group']} via sibling-group, "
              f"{recovery_stats['adjacent_text']} via adjacent-text")
    eq_total = sum(eq_stats.values())
    if eq_total:
        print(f"  Enriched {eq_total} equation items: "
              f"{eq_stats['positional+label']} positional+label, "
              f"{eq_stats['positional']} positional only, "
              f"{eq_stats['label_match']} label match only, "
              f"{eq_stats['none']} no context found")
    table_total = sum(table_stats.values())
    if table_total:
        print(f"  Enriched {table_total} table items: "
              f"{table_stats['label_match']} label match, "
              f"{table_stats['positional_fallback']} positional fallback, "
              f"{table_stats['none']} no context found")
    print("  Sections: " + ", ".join(f"{s} ×{n}" for s, n in section_stats.items()))
    if n_low_info:
        print(f"  Flagged {n_low_info} low-information fragment(s) (TOC/header noise) — excluded from embedding")
    if n_qr:
        print(f"  Flagged {n_qr} QR code image(s) — excluded from embedding")
    if any(list_stats.values()):
        print(f"  Lists: {list_stats['content']} embedded as text; excluded {list_stats['references']} reference, "
              f"{list_stats['toc']} table-of-contents, {list_stats['empty']} empty/near-empty")
    if n_continued:
        print(f"  Completed {n_continued} paragraph(s) ending in ':' with the equation/list that follows")
    print(f"  Written: {out_path}")
    return out_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('content_list', nargs='?', help='Path to content_list.json')
    group.add_argument('--paper-dir', help='Path to parsed paper directory (finds content_list automatically)')
    args = parser.parse_args()

    if args.content_list:
        path = Path(args.content_list)
    else:
        paper_dir = Path(args.paper_dir)
        matches = [p for p in paper_dir.glob('*_content_list.json') if 'enriched' not in p.name and 'v2' not in p.name]
        if not matches:
            matches = [p for p in paper_dir.glob('*_content_list_v2.json') if 'enriched' not in p.name]
        if not matches:
            print(f"No content_list.json found in {paper_dir}")
            sys.exit(1)
        path = matches[0]

    print(f"Enriching: {path}")
    enrich_content_list(path)
