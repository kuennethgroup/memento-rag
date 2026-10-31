"""Query-time research-gap retrieval and intent detection.

Gap passages are tagged at ingestion (memento/research_gaps/tagging.py) with payload fields gap_relevant (bool),
gap_signal (one line on the gap) and gap_type (limitation | future_work | untested_condition | uncertainty). Retrieval
is a metadata filter, not a semantic search (no embedder, no reranker); the reasoning model drops false positives and
structures the answer."""

import re
from dataclasses import dataclass


from memento.config.settings import DOMAIN
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

# ── Gap intent detection ──
# Conservative: needs a gap keyword and a paper/document word; "What are the limitations?" alone may be about a
# technique, not a paper's stated gaps.

_GAP_KEYWORDS = re.compile(
    r"\b("
    r"research\s+gap|research\s+gaps"
    r"|limitation|limitations"
    r"|future\s+work|future\s+research|future\s+direction|future\s+directions"
    r"|open\s+question|open\s+questions|open\s+problem|open\s+problems"
    r"|untested|unexplored|uninvestigated|understudied"
    r"|what(?:'s| is| are)\s+missing"
    r"|gaps?\s+in\s+(?:the\s+)?(?:research|literature|knowledge|understanding)"
    r"|shortcoming|shortcomings"
    r"|Forschungslücke|Forschungslücken"
    r"|Einschränkung|Einschränkungen|Limitation|Limitationen"
    r"|zukünftige\s+(?:Arbeit|Forschung)"
    r"|offene\s+Frage|offene\s+Fragen"
    r")\b",
    re.IGNORECASE,
)

_PAPER_CONTEXT = re.compile(
    r"\b("
    r"paper|papers|study|studies|thesis|dissertation|article|publication"
    r"|document|report|work"
    r"|Arbeit|Studie|Dissertation|Veröffentlichung|Dokument|Bericht"
    r")\b",
    re.IGNORECASE,
)


def detect_gap_intent(query: str) -> bool:
    """Does the query ask for research gaps in a paper? True only with a gap keyword and a paper/document word. A missed
    gap intent falls back to ranked retrieval; a false positive would answer a focused question with a gap analysis."""
    has_gap_keyword = bool(_GAP_KEYWORDS.search(query))
    has_paper_context = bool(_PAPER_CONTEXT.search(query))
    return has_gap_keyword and has_paper_context


def detect_gap_intent_with_paper(query: str, paper: str | None) -> bool:
    """Like detect_gap_intent, but also returns True when a paper is
    already resolved from the conversation context and a gap keyword
    is present — handles "what are the limitations?" in a paper-scoped
    conversation."""
    if not bool(_GAP_KEYWORDS.search(query)):
        return False
    return bool(_PAPER_CONTEXT.search(query)) or paper is not None


def fetch_gap_chunks(
    client: QdrantClient,
    collection: str,
    paper_id: str,
) -> list[dict]:
    """Scroll all gap-tagged chunks of a paper, sorted by page: dicts with text, source, page, section, gap_signal,
    gap_type, type."""
    chunks = []
    offset = None
    while True:
        pts, offset = client.scroll(
            collection_name=collection,
            scroll_filter=Filter(must=[
                FieldCondition(key="source", match=MatchValue(value=paper_id)),
                FieldCondition(key="gap_relevant", match=MatchValue(value=True)),
            ]),
            with_payload=True,
            with_vectors=False,
            limit=500,
            offset=offset,
        )
        for p in pts:
            pl = p.payload or {}
            text = (pl.get("text") or "").strip()
            if not text:
                continue
            chunks.append({
                "text": text,
                "source": pl.get("source", ""),
                "page": pl.get("page", 0),
                "section": pl.get("section", ""),
                "section_raw": pl.get("section_raw", ""),
                "gap_signal": pl.get("gap_signal", ""),
                "gap_type": pl.get("gap_type", ""),
                "type": pl.get("type", "text"),
                "bbox": pl.get("bbox"),
                "page_end": pl.get("page_end"),
                "img_path": pl.get("img_path"),
            })
        if offset is None:
            break

    chunks.sort(key=lambda c: c["page"])
    return chunks


# Sections where gaps typically concentrate: the fallback when a paper has no gap tags. Canonical section names from the
# domain config plus their section_hints aliases, so a German "Diskussion" heading also matches.
_GAP_ROLES = DOMAIN.get("section_roles", {}).get("gap_fallback", [])
_FALLBACK_SECTIONS = set(_GAP_ROLES)


def fetch_fallback_sections(
    client: QdrantClient,
    collection: str,
    paper_id: str,
) -> list[dict]:
    """Fetch Discussion/Conclusion/Outlook chunks as a fallback when no
    gap-tagged chunks exist.  Returns the same dict shape as
    fetch_gap_chunks, with gap_signal set to 'section-fallback'."""
    from qdrant_client.models import MatchAny
    chunks = []
    offset = None
    while True:
        pts, offset = client.scroll(
            collection_name=collection,
            scroll_filter=Filter(must=[
                FieldCondition(key="source", match=MatchValue(value=paper_id)),
                FieldCondition(key="section", match=MatchAny(
                    any=list(_FALLBACK_SECTIONS))),
            ]),
            with_payload=True,
            with_vectors=False,
            limit=500,
            offset=offset,
        )
        for p in pts:
            pl = p.payload or {}
            text = (pl.get("text") or "").strip()
            # text and lists only: limitations and future work are sentences, not figures or tables
            if not text or pl.get("type", "text") not in ("text", "list"):
                continue
            chunks.append({
                "text": text,
                "source": pl.get("source", ""),
                "page": pl.get("page", 0),
                "section": pl.get("section", ""),
                "section_raw": pl.get("section_raw", ""),
                "gap_signal": "section-fallback",
                "gap_type": "",
                "type": pl.get("type", "text"),
                "bbox": pl.get("bbox"),
                "page_end": pl.get("page_end"),
                "img_path": pl.get("img_path"),
            })
        if offset is None:
            break
    chunks.sort(key=lambda c: c["page"])
    return chunks


GAP_SYSTEM_PROMPT = """\
You are a scientific assistant analyzing passages from one paper in the \
corpus for the research gaps it names.

Most passages were pre-tagged as potentially containing research gaps, \
limitations, future work directions, or untested conditions.  Passages \
marked "Not gap-tagged" come from the paper's Discussion, Conclusion or \
Outlook sections and may state gaps the tagging missed.  Some passages \
may be false positives — only include gaps the authors genuinely state \
or clearly imply.

Do NOT infer gaps from your own knowledge of the field.
Do NOT include a passage just because it was tagged — use your judgment.
If a tagged passage does not actually contain a research gap, skip it.

Write each real gap as one plain sentence that ends with the passage(s) \
it comes from: [N], several as [2][5].

Layout, exactly this (2026-09-30: runs used to vary, e.g. "Type: \
future_work" labels and restarting numbers):
- Group the gaps under these headings, in this order, each as a level-3 \
heading: "### Limitations", "### Uncertainties", "### Untested \
conditions", "### Future work", "### Gaps in earlier work it addresses", \
"### Open problems in the field".  Leave out a heading that has no gaps.
- "Limitations", "Uncertainties" and "Untested conditions" are those of \
THIS study: its own data, methods, assumptions and results.  A statement \
that earlier research on a topic is missing, limited or unexplored, or a \
weakness of other authors' methods (usually in the Introduction, \
Background, Literature review, Related work or State of the art section), \
is a gap this study addresses: put it under "### Gaps in earlier work it \
addresses", never under Limitations.  Use each passage's section to decide.
- A textbook, book, review or survey reports no study of its own: do not \
use "### Gaps in earlier work it addresses".  Put the open problems, \
limitations and missing knowledge of the field it describes under "### Open \
problems in the field", and keep "Limitations" for what the authors say about \
their own text (for example what it does not cover).
- Under each heading a numbered list; the numbering continues across \
headings (1, 2, 3 under the first, 4, 5 under the next, and so on).
- No bold text, no type labels, no introduction and no closing summary.
- Write in the language of the question.  For a German question the \
headings are "### Einschränkungen", "### Unsicherheiten", "### Ungetestete \
Bedingungen", "### Zukünftige Arbeiten", "### Lücken in früheren Arbeiten, die \
sie schließt", "### Offene Probleme des Fachgebiets".
If no genuine gaps are found, say so plainly in one sentence."""

# a talk's own aims are not gaps: only a talk (its library title starts with "Talk") gets this rule, which in a blind
# judgement helped talks and hurt theses. Every other document keeps GAP_SYSTEM_PROMPT, and its stored answers their
# keys
TALK_AIMS_RULE = ("The document's own aims, objectives or research questions are not gaps: do not list them, also not "
                  "as future work (the slides of a talk often state them).\n")
_SKIP_LINE = "If a tagged passage does not actually contain a research gap, skip it.\n"
GAP_SYSTEM_PROMPT_TALK = GAP_SYSTEM_PROMPT.replace(_SKIP_LINE, _SKIP_LINE + TALK_AIMS_RULE, 1)


def gap_system_prompt(title: str | None) -> str:
    """The research-gap prompt for a document: a talk's (library title "Talk…") with TALK_AIMS_RULE, else
    GAP_SYSTEM_PROMPT."""
    return GAP_SYSTEM_PROMPT_TALK if re.match(r"\s*Talk\b", title or "") else GAP_SYSTEM_PROMPT


# the last one: a book's or review's open problems of its field, which the book itself does not address
GAP_HEADINGS_DE = {"Einschränkungen", "Unsicherheiten", "Ungetestete Bedingungen", "Zukünftige Arbeiten",
                   "Lücken in früheren Arbeiten, die sie schließt", "Offene Probleme des Fachgebiets"}
# the gaps in earlier work a study addresses, kept apart from the study's own limitations
EARLIER_WORK_HEADINGS = {"Gaps in earlier work it addresses", "Lücken in früheren Arbeiten, die sie schließt"}
_GAP_BLOCK = "memento_gaps"
# the Thorough check's notes the pipe puts under an answer, and the "Verification unavailable" note of a failed check
# (memento/agentic_validation/strategies.py): shown after the blocks
from memento.agentic_validation.strategies import NOTE_RE as _CHECK_NOTE
_TRAILING_NOTES_RE = re.compile(rf"(?:\n[ \t]*)+(?:{_CHECK_NOTE}[ \t]*(?:\n[ \t]*)*)+\Z")
# (the pipe's own "_Answered by …_" lines may follow the last block)
_NOTE_IN_LAST_BLOCK_RE = re.compile(rf"\n+({_CHECK_NOTE})[ \t]*\n+</details>[ \t]*\n*"
                                    r"(?=(?:[ \t]*_[^\n]+_[ \t]*\n*)*\Z)")
_HEADING_ORDER = {h: k for k, pair in enumerate(
    [("Limitations", "Einschränkungen"), ("Uncertainties", "Unsicherheiten"),
     ("Untested conditions", "Ungetestete Bedingungen"), ("Future work", "Zukünftige Arbeiten"),
     ("Gaps in earlier work it addresses", "Lücken in früheren Arbeiten, die sie schließt"),
     ("Open problems in the field", "Offene Probleme des Fachgebiets")]) for h in pair}


def _sections(text: str) -> tuple[str, list[tuple[str, str]]]:
    """(text before the first "### " heading, [(heading, body)])."""
    parts = re.split(r"(?m)^###[ \t]+(.+?)[ \t]*$", text or "")
    return parts[0], [(parts[i].strip(), parts[i + 1].strip("\n")) for i in range(1, len(parts) - 1, 2)]


# one kind of point asked for ("What future work does X propose?"): (English heading, German heading, English cue,
# German cue)
_GAP_KINDS = [("Future work", "Zukünftige Arbeiten", r"future\s+work|next\s+steps?|further\s+(?:work|research)",
               r"zukünftig\w*\s+arbeit\w*|ausblick"),
              ("Limitations", "Einschränkungen", r"limitation", r"einschränkung"),
              ("Uncertainties", "Unsicherheiten", r"uncertaint", r"unsicherheit"),
              ("Untested conditions", "Ungetestete Bedingungen", r"untested", r"ungetestet")]
_GAP_GENERIC_RE = re.compile(r"research\s+gaps?|open\s+questions?|forschungslücke|offene\w*\s+fragen", re.I)


def gap_focus(question: str) -> str | None:
    """The heading of the one kind of point a question asks for ("What future work does X propose?" → "Future work";
    German questions → the German heading), else None (research gaps in general, or several kinds)."""
    q = question or ""
    if _GAP_GENERIC_RE.search(q):
        return None
    hits = [(en, de, bool(re.search(cue_de, q, re.I))) for en, de, cue_en, cue_de in _GAP_KINDS
            if re.search(cue_en, q, re.I) or re.search(cue_de, q, re.I)]
    if len(hits) != 1:
        return None
    en, de, german = hits[0]
    return de if german else en


def strip_leaked(text: str) -> str:
    """The answer without leaked reasoning: after the first "###" heading only the known headings, numbered points (with
    their continuation lines) and the pipe's "_…_" lines are kept (thinking stays on for gap answers and can run on into
    the visible text)."""
    start = (text or "").find("###")
    if start < 0:
        return text
    kept, in_point, skipping = [], False, False
    for line in text[start:].splitlines():
        s = line.strip()
        if s.startswith("###"):
            # only the known headings; an unknown "###" line is leaked reasoning, and its points with it
            skipping = s.lstrip("#").strip() not in _HEADING_ORDER
            in_point = False
            if not skipping:
                kept.append(line)
            continue
        if skipping:
            continue
        if not s:
            in_point = False
            kept.append(line)
            continue
        if s.startswith("_"):
            in_point = False
            kept.append(line)
            continue
        if re.match(r"\d+\.\s", s):
            in_point = True
            kept.append(line)
            continue
        if in_point:
            kept.append(line)
    out = text[:start] + re.sub(r"\n{3,}", "\n\n", "\n".join(kept))
    return out + "\n" if text.endswith("\n") and not out.endswith("\n") else out


GAP_POINTS_SHOWN = 10          # points shown per block, the rest folded after it


def _split_points(body: str, n: int) -> tuple:
    """(the first `n` numbered points with their continuation lines, the rest, how many in the rest)."""
    starts = [m.start() for m in re.finditer(r"(?m)^\s*\d+\.\s", body)]
    if len(starts) <= n:
        return body, "", 0
    return body[:starts[n]].rstrip(), body[starts[n]:].strip(), len(starts) - n


def _more_block(heading: str, rest: str, k: int, german: bool) -> list:
    """The folded block with a heading's points beyond GAP_POINTS_SHOWN."""
    label = f"{heading}: {k} weitere" if german else f"{heading}: {k} more"
    return ["", f'<details type="{_GAP_BLOCK}">', f"<summary>{label}</summary>", "", rest, "", "</details>", ""]


_BLOCK_RE = re.compile(rf'<details type="{_GAP_BLOCK}">\n<summary>(.+?) \((\d+)\)</summary>\n\n(.*?)\n\n</details>',
                       re.S)


def _cap_blocks(text: str) -> str:
    """A stored answer in the block layout with its long blocks capped like a new one (N11); capped ones unchanged."""
    german = any(f"<summary>{h} (" in text for h in GAP_HEADINGS_DE)

    def cap(m):
        first, rest, k = _split_points(m.group(3).strip(), GAP_POINTS_SHOWN)
        if not rest:
            return m.group(0)
        block = f'<details type="{_GAP_BLOCK}">\n<summary>{m.group(1)} ({m.group(2)})</summary>\n\n{first}\n\n</details>'
        return block + "\n" + "\n".join(_more_block(m.group(1), rest, k, german))
    return _BLOCK_RE.sub(cap, text)


def calm_gap_answer(text: str, focus: str | None = None) -> str:
    """One document's research-gap answer in the calm layout: a count table of what the authors state, then one
    collapsed block per heading, its numbered points unchanged. `focus` (gap_focus): that heading's points shown open
    first. An answer without headings (the "no gaps" sentence) and one already in this layout are returned as they are;
    the headings stay in the answer's language. The pipe's closing notes (Thorough check) go after the blocks, never
    inside the last one."""
    if not text:
        return text
    if f'<details type="{_GAP_BLOCK}">' in text:
        # an answer stored with the note inside the last block: moved out on replay; long blocks capped
        text = _NOTE_IN_LAST_BLOCK_RE.sub(lambda m: f"\n\n</details>\n\n{m.group(1)}\n\n", text)
        return _cap_blocks(text)
    m = _TRAILING_NOTES_RE.search(text)
    notes = m.group(0).strip() if m else ""
    if notes:
        text = text[:m.start()]
    before, secs = _sections(strip_leaked(text))
    if not secs:
        if re.search(r"(?m)^###", text):
            # headings, none of them known: leaked reasoning, never shown
            return f"{notes}\n" if notes else ""
        return f"{text.rstrip()}\n\n{notes}\n" if notes else text
    german = any(h in GAP_HEADINGS_DE for h, _ in secs)
    # the fixed order of the known headings; others keep theirs
    secs = sorted(secs, key=lambda hb: _HEADING_ORDER.get(hb[0], len(_HEADING_ORDER)))
    count = lambda body: len(re.findall(r"(?m)^\s*\d+\.\s", body))
    head = "| Was die Autoren nennen | Punkte |" if german else "| What the authors state | Points |"
    out = [before.strip(), ""] if before.strip() else []
    if focus and any(h == focus for h, _ in secs):
        b = next(b for h, b in secs if h == focus)
        first, rest, k = _split_points(b.strip(), GAP_POINTS_SHOWN)
        out += [f"**{focus} ({count(b)})**", "", first, ""]
        if rest:
            out += _more_block(focus, rest, k, german)
    out += [head, "|---|---|", *[f"| {h} | {count(b)} |" for h, b in secs], ""]
    for h, b in secs:
        if focus and h == focus:
            continue
        first, rest, k = _split_points(b.strip(), GAP_POINTS_SHOWN)
        out += ["", f'<details type="{_GAP_BLOCK}">', f"<summary>{h} ({count(b)})</summary>", "", first, "",
                "</details>", ""]
        if rest:
            out += _more_block(h, rest, k, german)
    if notes:
        out += ["", notes]
    return "\n".join(out).strip() + "\n"


# ── one parser for a research-gap answer ──
# Display, storing, the extensions and stored answers all ask it. A heuristic: it reduces leaked reasoning but cannot
# remove every semantic case.
@dataclass
class GapParse:
    kind: str           # "sections" (known headings with points), "no_gaps" (the one-sentence answer), "rejected"
    text: str           # what may be shown: the known sections (points with their continuation lines), or the sentence
    rejected: bool      # something was dropped (an unknown heading, deliberation, text outside the layout), or nothing
                        # usable at all: such an answer is shown in its accepted parts and never stored


# a line of the model's own deliberation: conservative on purpose (real points starting "Could not …", "Waiting …" stay)
_DELIBERATION_RE = re.compile(r"^(?:let'?s\s+think|let\s+us\s+think|wait\b[,:]?|hmm+\b(?!-)|could\s+\[\d+\]\s+be)", re.I)
_WORDS_RE = re.compile(r"[A-Za-zÄÖÜäöüß]{3,}")
from memento.agentic_validation.strategies import PIPE_LINE_RE as _PIPE_NOTE_RE   # only the pipe's own notes
_POINT_RE = re.compile(r"\d+\.\s+(.*)")
NO_GAP_MAX_CHARS = 400            # the one-sentence "no gaps" answer (as _HeadingGate.SHORT in the pipe)
# a no-gap statement: one complete sentence saying that the passages name no gaps, in one of a few fixed shapes (EN /
# DE), so a hedging sentence that merely mentions limitations does not pass
_KIND_EN = (r"(?:(?:genuine|explicit|real|specific|clear)\s+)?(?:research\s+)?(?:gaps?|limitations?|open\s+(?:questions|"
            r"problems)|future\s+work|uncertaint(?:y|ies)|next\s+steps)")
_KINDS_EN = rf"{_KIND_EN}(?:\s*(?:,|or|and|,\s*or|,\s*and)\s+{_KIND_EN})*"
_THING_EN = r"(?:passages?|sources?|excerpts?|text|document|paper|study|thesis|work|authors?|article)"
_SUBJ_EN = rf"(?:the|this|these)\s+(?:provided\s+|given\s+|available\s+)?{_THING_EN}(?:\s+provided)?"
_PLACE_EN = rf"(?:\s+(?:in|within|from)\s+(?:the|this|these)\s+(?:provided\s+|given\s+)?{_THING_EN})?"
_VERB_EN = r"(?:name|state|mention|report|identify|describe|discuss|contain|give|propose|list|acknowledge)"
_DONE_EN = r"(?:named|stated|mentioned|reported|identified|described|discussed|given|proposed|listed|found|acknowledged)"
_KIND_DE = (r"(?:(?:echten?|expliziten?|konkreten?|eindeutigen?)\s+)?(?:forschungs)?(?:lücken|einschränkungen|"
            r"limitationen|offenen?\s+(?:fragen|probleme)|zukünftigen?\s+arbeiten|unsicherheiten|ausblicke?)")
_KINDS_DE = rf"{_KIND_DE}(?:\s*(?:,|oder|und)\s+{_KIND_DE})*"
_THING_DE = r"(?:passagen|quellen|auszüge|auszügen|textstellen|dokument|arbeit|studie|text|autoren|paper)"
_PLACE_DE = rf"(?:\s+in\s+den\s+(?:bereitgestellten\s+|vorliegenden\s+)?{_THING_DE})?"
_DONE_DE = r"(?:genannt|erwähnt|beschrieben|angegeben|berichtet|identifiziert|vorgeschlagen)"
_NO_GAP_SENTENCE_RE = re.compile(
    r"^(?:"
    rf"{_SUBJ_EN}\s+(?:does|do|did)\s+not\s+{_VERB_EN}\s+(?:any\s+)?{_KINDS_EN}{_PLACE_EN}"
    rf"|{_SUBJ_EN}\s+{_VERB_EN}s?\s+no\s+{_KINDS_EN}{_PLACE_EN}"
    rf"|(?:there\s+are\s+)?no\s+{_KINDS_EN}\s+(?:are|is|were|was)\s+(?:explicitly\s+|clearly\s+)?{_DONE_EN}{_PLACE_EN}"
    rf"|there\s+are\s+no\s+{_KINDS_EN}{_PLACE_EN}"
    rf"|(?:die|das|der)\s+(?:bereitgestellten\s+|vorliegenden\s+)?{_THING_DE}\s+(?:nennt|nennen|erwähnt|erwähnen|"
    rf"enthält|enthalten|beschreibt|beschreiben|berichtet|berichten)\s+keine\s+{_KINDS_DE}"
    rf"|(?:in\s+den\s+(?:bereitgestellten\s+|vorliegenden\s+)?{_THING_DE}\s+|es\s+)?(?:werden|wurden)\s+keine\s+"
    rf"{_KINDS_DE}{_PLACE_DE}\s+{_DONE_DE}"
    rf"|es\s+gibt\s+keine\s+{_KINDS_DE}{_PLACE_DE}"
    r")\s*[.!]?$", re.I)


def _plain(text: str) -> str:
    """The words without presentation emphasis (*, _): what the deliberation and no-gap checks judge (I4, R5)."""
    return re.sub(r"[*_]", "", text or "").strip()


def _no_gap_sentence(sentence: str) -> bool:
    return bool(_NO_GAP_SENTENCE_RE.match(re.sub(r"[*_]", "", sentence).strip()))


def parse_gap_answer(text: str) -> GapParse:
    """The research-gap answer as the layout allows it (GAP_SYSTEM_PROMPT): known "###" headings with numbered points
    (a point may wrap over several lines), or, without any heading, the no-gap statement, every sentence of it.
    Everything else is dropped and flagged; nothing usable → "rejected"."""
    t = (text or "").strip()
    before, secs = _sections(t)
    if not secs:
        sentences = [x for x in re.split(r"(?<=[.!?])\s+", t) if x.strip()]
        if t and len(t) <= NO_GAP_MAX_CHARS and not t.lstrip().startswith("#") and all(map(_no_gap_sentence, sentences)):
            return GapParse("no_gaps", t, False)
        return GapParse("rejected", "", True)
    rejected = bool(before.strip())
    kept = []
    for heading, body in secs:
        if heading not in _HEADING_ORDER:
            rejected = True
            continue
        lines, in_point = [], False
        for line in body.splitlines():
            s = line.strip()
            if not s:
                in_point = False
                lines.append("")
                continue
            if _PIPE_NOTE_RE.match(s):         # the pipe's own notes ("_Answered by …_"): not the model's, not kept
                in_point = False
                continue
            plain = re.sub(r"[*_]", "", s).strip()           # emphasis is presentation: judge the words (I4)
            point = _POINT_RE.match(s)
            if point:
                ok = bool(_WORDS_RE.search(re.sub(r"\[\d+\]", "", point.group(1)))) \
                    and not _DELIBERATION_RE.match(_plain(point.group(1)))       # emphasis too (R5)
                in_point = ok
                if ok:
                    lines.append(line)
                else:
                    rejected = True
                continue
            if in_point and not _DELIBERATION_RE.match(plain):
                lines.append(line)
                continue
            rejected, in_point = True, False
        body_ok = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
        if re.search(r"(?m)^\s*\d+\.\s", body_ok):
            kept.append(f"### {heading}\n{body_ok}")
    if not kept:
        return GapParse("rejected", "", True)
    return GapParse("sections", "\n\n".join(kept) + "\n", rejected)


def split_check_notes(text: str) -> tuple[str, str]:
    """(the answer, the Thorough check's notes at its end)."""
    m = _TRAILING_NOTES_RE.search(text or "")
    return ((text[:m.start()], m.group(0).strip()) if m else (text or "", ""))


def gap_answer_clean(text: str) -> bool:
    """A research-gap answer exactly in its layout, nothing dropped (parse_gap_answer), so no deliberation is stored."""
    if not (text or "").strip():
        return True                     # nothing written, nothing leaked
    p = parse_gap_answer(text)
    return p.kind != "rejected" and not p.rejected


_SUMMARY_LINE_RE = re.compile(r"^<summary>(.+?)(?: \(\d+\)|: \d+ (?:more|weitere))</summary>$")
_FOCUS_LINE_RE = re.compile(r"^\*\*(.+?) \(\d+\)\*\*$")


def stored_gap_ok(text: str) -> bool:
    """May a stored research-gap answer be replayed? One in the calm layout (calm_gap_answer) carries only that layout:
    the count table, blocks and "more" blocks of known headings, the focus heading, numbered points with their
    continuation lines and the pipe's notes; a raw one must pass parse_gap_answer cleanly without the pipe's notes
    (added at storing, after the check). An answer with nothing but the pipe's notes, or nothing at all, is never
    replayed."""
    if f'<details type="{_GAP_BLOCK}">' not in (text or ""):
        body = "\n".join(line for line in re.sub(_CHECK_NOTE, "", text or "").splitlines()
                         if not _PIPE_NOTE_RE.match(line.strip()))
        return bool(body.strip()) and gap_answer_clean(body)
    in_point = False
    for line in re.sub(_CHECK_NOTE, "", text).splitlines():
        s = line.strip()
        if not s:
            in_point = False
            continue
        if s.startswith("|") or s == "</details>" or s.startswith(f'<details type="{_GAP_BLOCK}">'):
            in_point = False
            continue
        m = _SUMMARY_LINE_RE.match(s) or _FOCUS_LINE_RE.match(s)
        if m:
            if m.group(1) not in _HEADING_ORDER:
                return False
            in_point = False
            continue
        if _PIPE_NOTE_RE.match(s):                          # the pipe's own notes only
            continue
        point = _POINT_RE.match(s)
        if point:
            words = re.sub(r"\[\d+\]", "", point.group(1))
            if not _WORDS_RE.search(words) or _DELIBERATION_RE.match(_plain(point.group(1))):     # emphasis too (R5)
                return False
            in_point = True
            continue
        if in_point and not _DELIBERATION_RE.match(re.sub(r"[*_]", "", s).strip()):
            continue
        return False
    return True


_ANY_BLOCK_RE = re.compile(rf'\n*<details type="{_GAP_BLOCK}">\n<summary>(.+?)(?: \(\d+\)|: \d+ (?:more|weitere))</summary>'
                           r'.*?</details>\n*', re.S)


def study_points(text: str) -> str:
    """A research-gap answer without its "Gaps in earlier work it addresses" section: what the study itself leaves open,
    the basis for "Possible extensions". Also for a stored answer in the calm layout: its earlier-work blocks and table
    row are left out."""
    if f'<details type="{_GAP_BLOCK}">' in (text or ""):
        out = _ANY_BLOCK_RE.sub(lambda m: "\n\n" if m.group(1) in EARLIER_WORK_HEADINGS else m.group(0), text)
        out = re.sub(r"(?m)^\| (?:" + "|".join(map(re.escape, EARLIER_WORK_HEADINGS)) + r") \| \d+ \|\n", "", out)
        return re.sub(r"\n{3,}", "\n\n", out)
    before, secs = _sections(text)
    if not any(h in EARLIER_WORK_HEADINGS for h, _ in secs):
        return text
    kept = [f"### {h}\n{b.strip()}" for h, b in secs if h not in EARLIER_WORK_HEADINGS]
    return "\n\n".join(([before.strip()] if before.strip() else []) + kept) + "\n"


def _gap_label(c: dict) -> str:
    """The tag line of one passage: its gap tag, or that it comes untagged from a closing section."""
    if c.get("gap_type") and c.get("gap_signal") != "section-fallback":
        return f"\nPre-tagged as: {c['gap_type']} — {c['gap_signal']}"
    return f"\nNot gap-tagged: from the {c.get('section') or 'closing'} section"


def build_gap_prompt(chunks: list[dict], original_query: str) -> str:
    """Build the model's message for gap reasoning."""
    context = "\n\n---\n\n".join(
        f"[{i + 1}] {c['source']}, page {c['page'] + 1}"
        + (f", Section: {c['section']}" if c.get("section") else "")
        + _gap_label(c)
        + f"\n{c['text']}"
        for i, c in enumerate(chunks)
    )
    # the answer's language, stated last: "the language of the question" alone let a German cover page pull an English
    # question's answer into German
    from memento.common.text_language import question_language
    lang = "German" if question_language(original_query) == "de" else "English"
    return f"Passages:\n{context}\n\nQuestion: {original_query}\n\nWrite the answer in {lang}."


# ── "Possible extensions": the model's own suggestions, when the question asks for them ──
SUGGEST_SYSTEM_PROMPT = """\
You suggest how a study could be extended. You get the limitations, uncertainties, untested conditions and future \
work its authors state (each point cites its passage as [N]) and the passages themselves.
Write up to five possible extensions of the study. Each is one sentence that builds on ONE point the authors state \
and cites that point's passage exactly as the point does ([N]). Add no new facts about the study, no numbers that \
are not in the passages and no other citations. Prefer concrete, feasible next steps (what to vary, measure, \
simulate or compare). Do not repeat a point the authors already state as future work.
Output ONLY a numbered list (1., 2., …), no heading, no introduction, no closing remark, in the language asked for."""
_IDEAS_BLOCK = "memento_ideas"


def suggest_messages(authors_answer: str, chunks: list[dict], question: str) -> list[dict]:
    """The request for the suggestions: the authors' points (the gap answer) and their passages, numbered as there."""
    from memento.common.text_language import question_language
    passages = "\n\n".join(f"[{i + 1}] {c['source']}, page {int(c.get('page') or 0) + 1}\n{c.get('text') or ''}"
                           for i, c in enumerate(chunks))
    lang = "German" if question_language(question) == "de" else "English"
    return [{"role": "system", "content": SUGGEST_SYSTEM_PROMPT},
            {"role": "user", "content": f"What the authors state:\n{authors_answer}\n\nPassages:\n{passages}\n\n"
                                        f"Question: {question}\n\nWrite the suggestions in {lang}."}]


def parse_suggestions(raw: str, allowed: set, limit: int = 5) -> list[str]:
    """The numbered suggestions that cite only passages the authors' points cite (and at least one); at most `limit`."""
    out = []
    for m in re.finditer(r"(?m)^\s*\d+[.)]\s+(.+?)\s*$", raw or ""):
        item = m.group(1).strip()
        cites = {int(n) for n in re.findall(r"\[(\d+)\]", item)}
        if cites and cites <= set(allowed):
            out.append(item)
    return out[:limit]


def unfold_suggestions(block: str) -> str:
    """The "Possible extensions" block as an open section (when asked how a study could be extended); anything else
    unchanged."""
    m = re.search(rf'<details type="{_IDEAS_BLOCK}">\s*<summary>(.*?)</summary>\s*(.*?)\s*</details>', block or "",
                  re.S)
    if not m:
        return block
    return f"{block[:m.start()]}### {m.group(1).strip()}\n\n{m.group(2).strip()}\n\n{block[m.end():].lstrip()}"


def suggestions_block(items: list[str], german: bool) -> str:
    """The suggestions set apart from what the authors state, with their label."""
    if not items:
        return ""
    title = "Mögliche Erweiterungen (von der KI vorgeschlagen)" if german else "Possible extensions (suggested by the AI)"
    label = ("_Von der KI vorgeschlagen, nicht von den Autoren. Bitte prüfen._" if german
             else "_Suggested by the AI, not stated by the authors. Please verify._")
    body = "\n".join(f"{k}. {t}" for k, t in enumerate(items, start=1))
    return f'\n\n<details type="{_IDEAS_BLOCK}">\n<summary>{title}</summary>\n\n{label}\n\n{body}\n\n</details>\n\n'
