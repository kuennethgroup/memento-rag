"""Look-alike characters in parsed text, corrected against the PDF's own text layer.

The vision model reads prose well but can confuse characters in strings without surrounding language, such as O and 0
in chemical notation or l and 1 in a code. The parsed text stays the text: a word changes only when the layer holds the
same word at the same place, the two differ only in look-alike characters, and the layer does not use the parsed word
anywhere else (then it is a real word of this document, and the layer's version a slip of its own). Orbital labels stay:
in C 1s the layer can be the wrong one. Scanned pages are skipped, because an OCR layer makes the same mistakes, and so is
a document with so many changes that its layer must be broken (a wrong font map). TEXT_LAYER_CHECK=0 switches the check
off. This is a heuristic: the raw parse is kept and each change is listed beside it.
"""
from __future__ import annotations

import logging
import os
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

from memento.common.pdf import PYMUPDF

log = logging.getLogger(__name__)

CONFUSABLE = {frozenset(p) for p in ("O0", "o0", "l1", "I1", "Il", "S5", "B8", "Z2", "G6")}
# fields that hold text, per element type; equations are LaTeX, headers and page numbers carry nothing to correct
FIELDS = {"text": ("text",), "ref_text": ("text",), "page_footnote": ("text",), "aside_text": ("text",),
          "list": ("list_items",), "table": ("table_body", "table_caption", "table_footnote"),
          "image": ("image_caption", "image_footnote"), "chart": ("chart_caption", "chart_footnote")}
WORD = re.compile(r"[^\W_]+")
KEEP = re.compile(r"\$[^$]*\$|\\\(.*?\\\)|\\\[.*?\\\]|<[^>]+>|&#?\w+;", re.S)     # LaTeX and HTML markup
ORBITAL = re.compile(r"\d[spdf]\d?")     # C 1s, Ti 2p3: notation, whatever the layer spells
MIN_WORDS = 5           # shorter strings have no context to align
MIN_MATCH = 0.6         # share of an element's words that must match the layer exactly
MIN_PAGE_WORDS = 20
MAX_IMAGE_SHARE = 0.7   # a page image at least this large: a scan
MAX_CHANGED = 0.02      # share of a document's checked words: more means a broken layer


def lookalike(parsed: str, layer: str) -> bool:
    """Same length, different only in look-alike characters. A swap between two letters (I, l) counts only in a code
    such as AlN, never in an ordinary word."""
    if len(parsed) != len(layer) or parsed == layer or ORBITAL.fullmatch(parsed):
        return False
    swaps = [frozenset(p) for p in zip(parsed, layer) if p[0] != p[1]]
    if not all(s in CONFUSABLE for s in swaps):
        return False
    if any(c.isdigit() for c in parsed + layer):
        return True
    return len(parsed) > 1 and sum(c.isupper() for c in layer) >= 2


def _words(text: str) -> list[re.Match]:
    keep = [m.span() for m in KEEP.finditer(text)]
    return [m for m in WORD.finditer(text) if not any(a <= m.start() < b for a, b in keep)]


def _edits(text: str, layer: list[str], known: set[str] = frozenset()) -> list[tuple[int, int, str]]:
    """(start, end, layer word) for each word of `text` to replace; `known`: every word of the document's layer."""
    words = _words(text)
    if len(words) < MIN_WORDS:
        return []
    parsed = [m.group() for m in words]
    matcher = SequenceMatcher(None, parsed, layer, autojunk=False)
    if sum(b.size for b in matcher.get_matching_blocks()) < MIN_MATCH * len(parsed):
        return []
    edits = []
    for op, a, b, c, d in matcher.get_opcodes():
        if op == "replace" and b - a == d - c:
            edits += [(words[i].start(), words[i].end(), layer[j])
                      for i, j in zip(range(a, b), range(c, d))
                      if lookalike(parsed[i], layer[j]) and parsed[i] not in known]
    return edits


def page_words(page) -> list[str] | None:
    """The page's text layer as words; None for a page without a trustworthy layer (scan, OCR overlay, no text)."""
    text = page.get_text()
    words = WORD.findall(unicodedata.normalize("NFKC", text))
    if len(words) < MIN_PAGE_WORDS:
        return None
    area = abs(page.rect)
    if any(abs(page.rect & info["bbox"]) >= MAX_IMAGE_SHARE * area for info in page.get_image_info()):
        return None
    spans = page.get_texttrace()
    drawn = sum(len(s["chars"]) for s in spans if s["type"] != 3)
    return words if drawn >= sum(len(s["chars"]) for s in spans) / 2 else None


def _strings(item: dict) -> list[tuple[str, int | None, str]]:
    """(field, position in a list field or None, text) of an element."""
    out = []
    for name in FIELDS.get(item.get("type"), ()):
        value = item.get(name)
        if isinstance(value, str):
            out.append((name, None, value))
        elif isinstance(value, list):
            out += [(name, k, s) for k, s in enumerate(value) if isinstance(s, str)]
    return out


def correct_items(items: list[dict], pdf_path) -> dict:
    """Correct `items` (MinerU content list) in place. Returns the pages checked and skipped, the words checked, the
    changes made, and `rejected`: changes not made because the document's layer looks broken. Never raises: on any
    failure the items stay as parsed."""
    stats = {"pages": 0, "skipped": 0, "words": 0, "changes": [], "rejected": 0}
    if os.getenv("TEXT_LAYER_CHECK", "1") == "0" or Path(pdf_path).suffix.lower() != ".pdf":
        return stats
    try:
        import fitz
        with PYMUPDF, fitz.open(pdf_path) as doc:
            layers = [page_words(page) for page in doc]
    except Exception as e:                  # a damaged file is the parser's problem, not a reason to stop here
        log.warning("text layer not read for %s: %s", Path(pdf_path).name, e)
        return stats
    stats["pages"] = sum(w is not None for w in layers)
    stats["skipped"] = len(layers) - stats["pages"]
    known = {w for words in layers if words for w in words}
    planned = []
    try:
        for item in items:
            page = item.get("page_idx")
            if not isinstance(page, int) or not 0 <= page < len(layers) or layers[page] is None:
                continue
            for name, pos, text in _strings(item):
                stats["words"] += len(_words(text))
                edits = _edits(text, layers[page], known)
                if edits:
                    planned.append((item, name, pos, text, edits))
    except Exception as e:                  # a malformed element: nothing has been changed yet
        log.warning("text layer check left out for %s: %s", Path(pdf_path).name, e)
        return stats
    n = sum(len(p[4]) for p in planned)
    if n > max(3, MAX_CHANGED * stats["words"]):
        stats["rejected"] = n
        log.warning("text layer of %s looks broken: %d of %d words differ", Path(pdf_path).name, n, stats["words"])
        return stats
    for item, name, pos, text, edits in planned:
        fixed = text
        for start, end, word in reversed(edits):
            stats["changes"].append({"page": item["page_idx"], "field": name, "old": fixed[start:end], "new": word})
            fixed = fixed[:start] + word + fixed[end:]
        if pos is None:
            item[name] = fixed
        else:
            item[name][pos] = fixed
    return stats
