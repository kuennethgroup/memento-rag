"""Figures in an answer: a request to see figures (or a table or an equation), which cited figures are shown under the
answer and when, their captions, and icons and reference-list images left out.
"""
import re

from memento.query import presentation, request


# ── Image/figure intent ──
# A request for images retrieves image and chart chunks (the figures render inline), not text that mentions them.
# Conservative: needs a figure word and request framing, and never fires on analysis phrasings ("compare the SEM images
# of A and B"), which keep full-context retrieval.
_IMAGE_TRIGGER_RE = re.compile(
    r"\b(image|figure|fig\.|diagram|micrograph|photo|picture|chart"
    r"|abbildung|bild|grafik|diagramm|schaubild|mikroskopaufnahme)\w*",
    re.IGNORECASE,
)


_IMAGE_REQUEST_RE = re.compile(
    r"\b(show|give|display|list|find|get|see|any|which|what"
    r"|zeig|gib|finde|liste|welche|gibt)\w*",
    re.IGNORECASE,
)


# "figure(s) of merit" is a materials-science term, not a figure request
_FIGURE_OF_MERIT_RE = re.compile(r"\bfigures?\s+of\s+merit\b", re.IGNORECASE)


def detect_image_intent(query: str) -> bool:
    nq = query.lower()
    if _FIGURE_OF_MERIT_RE.search(nq):
        return False
    if any(w in nq for w in request.ANALYSIS_WORDS):
        return False
    return bool(_IMAGE_TRIGGER_RE.search(nq) and _IMAGE_REQUEST_RE.search(nq))


def figure_request(intent: str, element: str) -> bool:
    """A request to see figures (their cited figures are shown in the answer): the router's image_request, unless it
    names a table or an equation."""
    return intent == "image_request" and element not in ("table", "equation")


def no_kind_note(kind: str, where: str | None, de: bool) -> str:
    """A figure (or table / equation) request whose own search found nothing of that kind in the requested documents:
    said plainly, then the answer from their other passages (the search never widens to the whole library)."""
    if de:
        what = {"figure": "keine indizierte Abbildung", "table": "keine indizierte Tabelle",
                "equation": "keine indizierte Gleichung"}.get(kind, "keine indizierte Abbildung")
        return f"_Ich habe dazu {what} in {where or 'der Bibliothek'} gefunden._\n\n"
    what = {"figure": "indexed figure", "table": "indexed table", "equation": "indexed equation"}.get(kind, "indexed figure")
    return f"_I found no {what} for this in {where or 'the library'}._\n\n"


def figure_markdown(i: int, h: dict, url: str) -> str:
    """A figure under the answer with a caption naming its source: "(5)" as in the sources list, since "[5]" in the text
    becomes a citation chip; markdown characters in names escaped."""
    label = re.sub(r"([\\`*_{}\[\]<>#|])", r"\\\1", presentation.citation_label(presentation.source_name(h, full=True)))
    return f"\n\n![{presentation.shown(h['source'])} p.{h['page'] + 1}]({url})\n\n*Figure ({i + 1}): {label}*"


# a line that names a figure ("Fig. 5.4 [6]", "Figure 2", "Abbildung 3"): its citations are figures the answer discusses
_FIGURE_WORD_RE = re.compile(r"\b(?:fig(?:ure)?s?|abb(?:ildung(?:en)?)?)\b", re.I)


def _figures_named(answer: str) -> set:
    """The passages (0-based) cited on a line of the answer that names a figure."""
    return {int(n) - 1 for line in (answer or "").splitlines() if _FIGURE_WORD_RE.search(line)
            for n in re.findall(r"\[(\d+)\]", line)}


def figures_to_show(hits: list, cited: set, answer: str, seen: set, cap: int | None, figure_asked: bool = False,
                     skip=()) -> tuple[list, int]:
    """(the cited figures to show, in citation order; how many more there are). At most `cap` (None: all): first those
    the answer names (_figures_named), then those the model saw as images (`seen`), then citation order. Only figures
    the display would show (shown_figure) count."""
    cand = [i for i in sorted(cited) if 0 <= i < len(hits) and i not in skip and shown_figure(hits[i], figure_asked)]
    if cap is None:
        return cand, 0
    named = _figures_named(answer)
    first = sorted(cand, key=lambda i: (i not in named, i not in seen, i))[:max(cap, 0)]
    return sorted(first), len(cand) - len(first)


def more_figures_note(n: int, de: bool) -> str:
    """The line under the shown figures when the cap left some out (the side panel opens every cited source)."""
    if n <= 0:
        return ""
    if de:
        return (f"\n\n_{n} weitere Abbildung ist in den zitierten Quellen._" if n == 1 else
                f"\n\n_{n} weitere Abbildungen sind in den zitierten Quellen._")
    return (f"\n\n_{n} more figure is in the cited sources._" if n == 1 else
            f"\n\n_{n} more figures are in the cited sources._")


def first_paragraph_end(text: str) -> int | None:
    """Where the answer's first paragraph ends (the index after its blank line), or None while it is still being
    written. Leading blank lines do not count."""
    start = len(text) - len(text.lstrip())
    end = text.find("\n\n", start)
    return None if end < 0 else end + 2


def first_paragraph_figures(text: str) -> set | None:
    """A figure request: once the answer's first paragraph is complete, the passages it cites (0-based), shown right
    below it; None while the paragraph is still being written."""
    end = first_paragraph_end(text)
    return None if end is None else {int(n) - 1 for n in re.findall(r"\[(\d+)\]", text[:end])}


def show_figures(hits: list, cited: set, low: float, figure_asked: bool = False) -> bool:
    """Figures under the answer at all? Not when the search found nothing close (_nothing_found): such an answer says
    "not covered" and cites passages only to show what the library does cover. A question that asks for a figure shows
    the figures it cites whatever their score (figures score low against a text question)."""
    if figure_asked:
        return bool(cited)
    top = max((h.get("rerank_score") or 0.0 for h in hits), default=0.0)
    return not presentation.nothing_found(cited=cited, top_score=top, low=low)


def shown_figure(h: dict, figure_asked: bool = False) -> bool:
    """Show this cited image/chart under the answer? Not a journal's QR code, never a tiny one (an icon or logo), and
    from page 1 (journal header, badges, graphical abstract) only when the question asks for a figure."""
    if h.get("type") not in ("image", "chart") or not h.get("img_path") or presentation.reference_image(h):
        return False
    bb = h.get("bbox")
    if bb and len(bb) == 4 and (bb[2] - bb[0] < presentation.MIN_FIGURE_SIZE or bb[3] - bb[1] < presentation.MIN_FIGURE_SIZE):
        return False
    if not figure_asked and (h.get("page") or 0) == 0:
        return False
    try:
        from memento.common.qr_code import is_qr_code
        return not is_qr_code(h["img_path"])
    except Exception:
        return True


