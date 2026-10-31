"""What a reply shows around the answer: status events, the sources and names markers, citation labels, file links
and names, the related-documents footer, a folded long list, a held-back heading, and the typed-out text.
"""
import asyncio
import re
from urllib.parse import quote

from memento.query import request, texts


# Each answer ends with a collapsed "Sources in this answer" block listing what its [N] markers cite. Open WebUI replays
# only chat text to the pipe (citation events are UI-only), so a later turn needs it to know what "[2]" meant. A
# <details> block survives the frontend's history processing; unknown HTML would print as text.
_SOURCES_TYPE = "memento_sources"


SOURCES_MARK_RE = re.compile(r'\s*<details type="' + _SOURCES_TYPE + r'">.*?</details>\s*', re.S)


SOURCES_LINE_RE = re.compile(r"^\s*-\s*[\[(](\d+)[\])]\s*(.+?)\s*$", re.M)   # "- (3) …"; older answers "- [3] …"


def status_event(msg: str, done: bool) -> dict:
    """A status event. The final "Done" is sent hidden, so no grey "Done" line stays above the answer; other final notes
    stay visible."""
    data = {"description": texts.tr(msg), "done": done}           # German for a German question
    if done and msg == "Done":
        data["hidden"] = True
    return {"type": "status", "data": data}


# Documents the answer did not read but with a related passage, and for an answer that cites nothing the closest
# passages of the whole library: a collapsed block under the answer, from a per-document search that runs while the
# answer is written. Kept out of the next turn's context like the sources block.
_RELATED_TYPE = "memento_related"


RELATED_MARK_RE = re.compile(r'\s*<details type="' + _RELATED_TYPE + r'">.*?</details>\s*', re.S)


def image_label(n: int) -> str:
    """The text part before a source's image, on its own line: the chat template joins text parts without a separator
    (same as retrieval/answer_pipeline.image_label)."""
    return f"\n\nImage for Source [{n}]:"


def nothing_found(*, cited: set, top_score: float, low: float) -> bool:
    """The search found nothing that answers the question: the answer cites nothing, or its best passage scores below
    `low`. Independent of the answer's wording and language."""
    return not cited or (top_score or 0.0) < low


def _found_nothing(*, cited: set, top_score: float, low: float, figure_asked: bool) -> bool:
    """_nothing_found for the block under an answer; a figure request found something when it cites a figure, whatever
    the figures' scores (figures score low against a text question)."""
    return not cited if figure_asked else nothing_found(cited=cited, top_score=top_score, low=low)


def related_documents(heads: list, *, used: set, nothing_found: bool, top_score: float, ratio: float,
                       min_score: float, limit: int, near: int) -> list:
    """From the best passage of each document (reranked, best first): when nothing was found, the `near` closest of any
    document; otherwise up to `limit` documents the answer did not read whose passage scores at least `ratio` × the
    answer's best passage and at least `min_score`."""
    if nothing_found:            # the closest passages that are at least somewhat related
        return [h for h in heads if (h.get("rerank_score") or 0.0) >= min_score][:near]
    floor = max(ratio * (top_score or 0.0), min_score)
    return [h for h in heads if h["source"] not in used and (h.get("rerank_score") or 0.0) >= floor][:limit]


def plain(text: str) -> str:
    """Text for a markdown list line: no [N] (OpenWebUI turns it into a citation pill), no markup characters."""
    t = re.sub(r"\[([^\]]*)\]", r"(\1)", " ".join(str(text or "").split()))
    return re.sub(r"([\\`*_{}<>#|])", r"\\\1", t)


def _related_lines(items: list, titles: dict, files_url: str, quote: bool) -> list:
    lines = []
    for h in items:
        label = plain(f"{titles.get(h['source']) or h['source']}, {page_label(h['page'], h.get('page_end'))}")
        bbox = h.get("bbox")
        url = source_file_url(files_url, h["source"], h["page"], [bbox] if isinstance(bbox, list) else None,
                               text=h.get("text"), section=h.get("section_raw"))
        line = f"- {label}" + (f" · [open]({url})" if url else "")
        if quote:
            q = " ".join(str(h.get("text") or "").split())
            q = q if len(q) <= 160 else q[:157].rsplit(" ", 1)[0] + "…"
            line += f"\n  \"{plain(q)}\""
        lines.append(line)
    return lines


def related_footer(*, items: list, titles: dict, searched: int | None, cited: bool, weak: bool, files_url: str,
                    list_answer: bool = False) -> str:
    """The block under an answer, worded from separate facts: whether the answer cites its sources, whether the best
    scores were weak (a list answer is about documents, so only "cites nothing" counts) and which documents' passages
    are shown. `searched`: the documents the search reached, None when it hit its cap or is unknown. A low score is
    reported as a score, never as "nothing answers this"."""
    weak = weak and not list_answer
    where = (f"Searched {searched} document{'s' if searched != 1 else ''}" if searched else "Searched the library")
    if cited and not weak:
        if not items:
            return ""
        summary, quote = f"Other documents with related passages ({len(items)}), not used in this answer", False
    elif cited:
        summary, quote = texts.RELATED_WEAK, True
    elif not weak:
        if not items:
            return ""
        summary, quote = f"This answer cites no sources. The closest passages ({len(items)})", True
    elif items:
        summary, quote = f"{where}: the closest passages ({len(items)})", True
    else:
        return (f'\n\n<details type="{_RELATED_TYPE}">\n<summary>{where}: no passage came close to this question.'
                f'</summary>\n\n{texts.RELATED_NONE_BODY}\n</details>\n')
    body = "\n".join(_related_lines(items, titles, files_url, quote)) or texts.RELATED_NONE_BODY
    return f'\n\n<details type="{_RELATED_TYPE}">\n<summary>{summary}</summary>\n\n{body}\n</details>\n'


def related_applies(enabled: bool, *, paper, named_docs: list, image_intent: bool, whole_doc: bool,
                     query_intent: str) -> bool:
    """Only for questions across the library: not for a named document, figures, summaries or sections."""
    return (enabled and not paper and not named_docs and not image_intent and not whole_doc
            and query_intent in ("ranked_retrieval", "multi_hop", "set_query", "gap_query"))


def sources_marker(hits: list, cited: set) -> str:
    """The collapsed list of the sources an answer cites: "(3) document name, p. 6, §2.4.2". Round brackets because Open
    WebUI turns every "[3]" into a citation chip."""
    return names_marker([source_name(h) for h in hits], cited, docs=[h.get("source") for h in hits])


_NAME_LOCATION_RE = re.compile(r"^(.*\S)\s*\(([^()]*)\)\s*$")     # "Doc (p. 6, §2.4.2)" → Doc, p. 6, §2.4.2


ALL_FROM_RE = re.compile(r"^(?:All from|Alle aus) \*\*(.+?)\*\*\.\s*$", re.M)


def _plain_label(text: str) -> str:
    return " ".join(re.sub(r"[<>\[\]\n]", " ", text or "").split())


def _one_document_marker(names: list, cited: list, docs: list | None) -> str:
    """The list for an answer whose cited sources all come from one document: its name once, then page and section per
    line. One number per line, as _history_text reads them; "" when the sources come from several documents or a name
    has no location. `docs`: each source's document identity (None when unknown); only identities decide "one document",
    as two files can share a name."""
    parts = [_NAME_LOCATION_RE.match(names[i] or "") for i in cited]
    ids = {docs[i] if docs and i < len(docs) else None for i in cited}
    if not all(parts) or len({p.group(1) for p in parts}) != 1 or len(ids) != 1 or None in ids:
        return ""
    k = len(cited)
    lines = [f"- ({i + 1}) {_plain_label(p.group(2))}" for i, p in zip(cited, parts)]
    return (f'\n\n<details type="{_SOURCES_TYPE}">\n<summary>Cited in this answer: {k} passage{"s" if k != 1 else ""} '
            f'of this document</summary>\n\nAll from **{_plain_label(parts[0].group(1))}**.\n\n'
            f'_The numbers match the links in the text._\n\n' + "\n".join(lines) + "\n</details>\n")


def stored_doc_ids(sources: list) -> list:
    """The document identity of each stored source (OpenWebUI's message sources, or citation events' data), by
    position: its metadata "X#N" / "X#gapN" without the passage suffix. None when unsure: no or several metadata
    entries (OpenWebUI merges sources with the same name), a "[k]" name that is not this position, a literature paper."""
    out = []
    for k, s in enumerate(sources or []):
        metas = s.get("metadata") if isinstance(s.get("metadata"), list) else [s.get("metadata") or {}]
        meta = metas[0] if len(metas) == 1 and isinstance(metas[0], dict) else {}
        ident = str(meta.get("source") or "")
        num = re.match(r"\[(\d+)\]", str(meta.get("name") or ""))
        ok = bool(ident) and bool(num) and int(num.group(1)) == k + 1 and not LITERATURE_ID_RE.match(ident)
        out.append(PASSAGE_SUFFIX_RE.sub("", ident) if ok else None)
    return out


def replay_sources(answer: str, citations: list) -> str:
    """A stored answer's sources list in the current layout, rebuilt from its stored citation events (presentation
    only). Unchanged when the sources come from several documents, an identity is unknown or a listed number has no
    stored citation."""
    m = SOURCES_MARK_RE.search(answer or "")
    if not m:
        return answer
    srcs = [(ev or {}).get("data") or {} for ev in citations or []]
    cited = sorted({int(n) - 1 for n, _ in SOURCES_LINE_RE.findall(m.group(0))})
    if not cited or any(not 0 <= i < len(srcs) for i in cited):
        return answer
    new = _one_document_marker([(x.get("source") or {}).get("name", "") for x in srcs], cited, stored_doc_ids(srcs))
    core = re.search(r"<details.*</details>", m.group(0), re.S)
    if not new or not core:
        return answer
    return answer[:m.start() + core.start()] + texts.tr(new).strip() + answer[m.start() + core.end():]


def names_marker(names: list, cited: set, docs: list | None = None) -> str:
    """The same list from the sources' stored names ("Doc (p. 6, §2.4.2)"), e.g. for a reworked answer. `docs`: each
    source's document, when known; sources all from one document get the shorter list (_one_document_marker)."""
    one = _one_document_marker(names, [i for i in sorted(cited) if 0 <= i < len(names)], docs) \
        if any(0 <= i < len(names) for i in cited) else ""
    if one:
        return one
    lines = []
    for i in sorted(cited):
        if 0 <= i < len(names):
            lines.append(f"- ({i + 1}) {citation_label(names[i])}")
    if not lines:
        return ""
    # Open WebUI's own "N Sources" button lists every passage the model received; this list only the cited ones, and its
    # heading says how the two relate.
    n, k = len(names), len(lines)
    read = ("the 1 source" if n == 1 else f"all {n} sources" if k == n else f"{k} of the {n} sources")
    return (f'\n\n<details type="{_SOURCES_TYPE}">\n<summary>Cited in this answer: {read} Memento read</summary>\n\n'
            f'_The numbers match the links in the text._\n\n' + "\n".join(lines) + "\n</details>\n")


PASSAGE_SUFFIX_RE = re.compile(r"#(?:gap)?\d+$")


LITERATURE_ID_RE = re.compile(r"^paper#\d+$")         # a published paper of a research-gap report, not a library document


def shown(source: str) -> str:
    names = request.SHOWN_NAMES.get()
    return names.get(source, source) if names else source


def context_name(source: str, names: dict) -> str:
    """A document's name as the answer model reads it: its title with the file name when they differ, else the shown
    name; citations keep the file names."""
    return names.get(source) or shown(source)


def source_name(h: dict, full: bool = False) -> str:
    tag = request.section_tag(h.get("section_raw"), full)
    return f"{shown(h['source'])} ({page_label(h['page'], h.get('page_end'))}{', ' + tag if tag else ''})"


def citation_label(name: str) -> str:
    """A stored citation name "Doc (p. 6, §2.4.2)" as history text: "Doc, p. 6, §2.4.2"."""
    label = re.sub(r"\s*\(([^()]*)\)\s*$", r", \1", name or "")
    return " ".join(re.sub(r"[<>\[\]\n]", " ", label).split())


def _names_per_answer(stored_messages: list) -> list[dict]:
    """For each assistant message of a stored chat branch, in order: {"N": label} from the citations OpenWebUI saved
    with it. They are in emission order, which is the order of our [N] (one citation event per numbered source)."""
    out = []
    for m in stored_messages:
        if m.get("role") != "assistant":
            continue
        out.append({str(i + 1): citation_label((s.get("source") or {}).get("name", ""))
                    for i, s in enumerate(m.get("sources") or [])})
    return out


async def stored_citation_names(metadata: dict | None) -> list[dict]:
    """[N] labels of the earlier answers in this chat (current branch, the answer being written excluded)."""
    chat_id = (metadata or {}).get("chat_id")
    if not chat_id or str(chat_id).startswith("local:"):
        return []
    try:
        from open_webui.models.chats import Chats
        from open_webui.utils.misc import get_message_list
        chat = await Chats.get_chat_by_id(chat_id)
        hist = ((chat.chat if chat else {}) or {}).get("history") or {}
        chain = get_message_list(hist.get("messages") or {}, (metadata or {}).get("message_id") or hist.get("currentId"))
        return _names_per_answer(chain[:-1] if chain and chain[-1].get("role") == "assistant" else chain)
    except Exception:
        return []


def page_label(page: int, page_end=None) -> str:
    """Citation label for a chunk's page(s): a paragraph merged across a page boundary
    (enrich_parsed_content_list.merge_page_boundary_paragraphs) has `page_end` > `page`, so the label shows the span;
    the highlight shows the first page only."""
    if isinstance(page_end, int) and page_end > page:
        return f"pp. {page + 1}–{page_end + 1}"
    return f"p. {page + 1}"


def source_file_url(files_url: str, stem: str, page: int,
                     bboxes: list | None = None, text: str | None = None,
                     section: str | None = None) -> str | None:
    """URL for a citation's embed_url, so clicking it opens the cited place in the side panel.

    One link format for every file type: the file server's /view/<stem>/<page_idx> decides how to show it
    (PDF or image → rendered page with the bbox highlight; Office → reading view of the slide/section/table with
    `highlight` matched against the parsed text). `bboxes`: [x0, y0, x1, y1] in MinerU's 0-1000 page-normalized
    coordinates."""
    if not files_url or not stem:
        return None
    params = [f"bbox={b[0]},{b[1]},{b[2]},{b[3]}" for b in (bboxes or []) if isinstance(b, list) and len(b) == 4]
    if text:
        params.append(f"highlight={quote(text[:200])}")
    if section:
        params.append(f"section={quote(str(section)[:120])}")
    return f"{files_url.rstrip('/')}/view/{quote(stem)}/{int(page or 0)}" + ("?" + "&".join(params) if params else "")


# the listing shows the first LIST_FIRST names; the rest sit in collapsible blocks (chat answers are markdown, so
# <details> is the only clickable element): a chain of "Show LIST_STEP more", each nested in the one before, plus one
# "Show all" block when more than one step remains
LIST_FIRST, LIST_STEP = 20, 50


LIST_ALL_TYPE = "memento_list_all"


_LIST_ENTRY = re.compile(r"(?:[*+-]|\d+[.)])\s+\S")      # a top-level list entry: at the start of the line, no indent


# the count sentence LIST_ADDENDUM asks for, after the list: its number is set to the number of entries shown
_COUNT_STARTS = ("I found ", "These are the ", "Ich habe ", "Das sind die ")


_COUNT_LINE = re.compile(r"^(\s*(?:%s))(\d+)\b" % "|".join(re.escape(s) for s in _COUNT_STARTS))


def _fold_entries(entries: list, de: bool) -> str:
    """Entries LIST_FIRST+1 … of a long list (each entry its lines) in the library listing's folds: "Show 5 more
    (21–25 of 25)", in steps of LIST_STEP, and "Show all N" when there is more than one step."""
    n = len(entries)

    def items(a: int, b: int) -> str:
        return "\n".join("".join(e).rstrip("\n") for e in entries[a:b])

    chain = ""
    for a in reversed(range(LIST_FIRST, n, LIST_STEP)):
        b = min(a + LIST_STEP, n)
        more = f"{b - a} weitere anzeigen ({a + 1}–{b} von {n})" if de else f"Show {b - a} more ({a + 1}–{b} of {n})"
        chain = f"<details>\n<summary>{more}</summary>\n\n{items(a, b)}\n\n{chain}</details>\n\n"
    if n - LIST_FIRST > LIST_STEP:
        every = f"Alle {n} anzeigen" if de else f"Show all {n}"
        chain += (f'<details type="{LIST_ALL_TYPE}">\n<summary>{every}</summary>\n\n'
                  f'{items(LIST_FIRST, n)}\n\n</details>\n\n')
    return chain


class ListFold:
    """A list answer as it streams: text passes through at once, line by line, while a list has at most LIST_FIRST
    entries; later entries are held and come as one fold when the list ends, with the listing's labels. feed() returns
    what may be shown now, close() the rest."""

    def __init__(self, de: bool | None = None):
        self.de = (request.LANG.get() == "de") if de is None else de
        self.buf, self.sent = "", 0          # the line being written, and how much of it is shown
        self.count = 0                       # entries of the current list (0: not in a list)
        self.blanks: list = []               # blank lines after a list line: a fold drops them, else they are shown
        self.entries: list = []              # every entry of the current list (each its lines), for the fold
        self.folding = False
        self.total = 0                       # top-level entries of every list so far: the count sentence's number

    def _counted(self, line: str) -> str:
        """A count sentence ("I found 40 …") with the number of entries shown."""
        return _COUNT_LINE.sub(lambda m: f"{m.group(1)}{self.total}", line, count=1) if self.total else line

    def _maybe_count(self, partial: str) -> bool:
        """Could the line being written be a count sentence (so it is held until it is complete)?"""
        s = partial.lstrip()
        return bool(self.total) and any(c.startswith(s) or s.startswith(c) for c in _COUNT_STARTS)

    def _line(self, line: str) -> str:
        sent, self.sent = self.sent, 0
        entry = bool(_LIST_ENTRY.match(line))
        if self.folding:
            if not line.strip():
                self.blanks.append(line)
                return ""
            if entry:
                self.blanks = []
                self.entries.append([line])
                self.total += 1
                return ""
            if line[:1].isspace():           # a sub-entry or a wrapped line of the last entry
                self.entries[-1] += self.blanks + [line]
                self.blanks = []
                return ""
            out = "\n" + _fold_entries(self.entries, self.de)       # the list ended: the fold, then this line
            self.folding, self.count, self.entries, self.blanks = False, 0, [], []
            return out + self._counted(line)
        if entry and self.count >= LIST_FIRST:                      # entry 21: the rest of the list is held
            self.folding, self.blanks = True, []
            self.entries.append([line])
            self.count += 1
            self.total += 1
            return ""
        if not line.strip() and self.count:
            self.blanks.append(line)
            return ""
        out, self.blanks = "".join(self.blanks), []
        if entry:
            self.entries = self.entries if self.count else []
            self.count += 1
            self.total += 1
            self.entries.append([line])
        elif self.count and line[:1].isspace():
            self.entries[-1].append(line)
        else:
            if self.count:                                          # a plain line: the list ended
                self.count, self.entries = 0, []
            if not sent and not line[:1].isspace():
                line = self._counted(line)
        return out + line[sent:]

    def feed(self, piece: str) -> str:
        self.buf += piece or ""
        out = []
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            out.append(self._line(line + "\n"))
        # the line being written shows at once unless it could be entry LIST_FIRST + 1 or the count sentence
        if self.buf.strip() and not self.folding and self.count < LIST_FIRST and not self._maybe_count(self.buf):
            out.append("".join(self.blanks) + self.buf[self.sent:])
            self.blanks, self.sent = [], len(self.buf)
        return "".join(out)

    def close(self) -> str:
        out = self._line(self.buf) if self.buf else ""
        self.buf = ""
        if self.folding:
            out += "\n" + _fold_entries(self.entries, self.de)
            self.folding = False
        out += "".join(self.blanks)
        self.blanks = []
        return out


def fold_list(text: str, de: bool | None = None) -> str:
    """`text` with each list of more than LIST_FIRST entries folded after its first LIST_FIRST (_ListFold)."""
    f = ListFold(de)
    return f.feed(text) + f.close()


class HeadingGate:
    """The visible research-gap answer from its first heading on. Its layout starts with a "###" heading; with thinking
    capped, the model may keep planning in the visible text, which must not be shown or stored. Text before the first
    heading is held back and dropped; an answer without any heading (the one-sentence "no gaps" answer) is shown whole
    at the end, and stored only if it is short."""
    SHORT = 400

    def __init__(self):
        self.buf, self.opened, self.dropped, self.storable = "", False, 0, True

    def feed(self, text: str) -> str:
        if self.opened:
            return text
        self.buf += text
        m = re.search(r"(?m)^#{1,6} ", self.buf)
        if not m:
            return ""
        self.opened, self.dropped = True, len(self.buf[:m.start()].strip())
        out, self.buf = self.buf[m.start():], ""
        return out

    def finish(self) -> str:
        """The held text of an answer that never reached a heading (else "")."""
        if self.opened:
            return ""
        out, self.buf = self.buf, ""
        self.storable = len(out.strip()) <= self.SHORT
        return out


# A finished answer (stored, or held for a check) is typed out like a generated one, about 200 characters/s in pieces of
# ~3 characters. Collapsible blocks, figures and reasoning are sent whole: a half-sent <details> shows raw tags.
_TYPE_ATOMIC_RE = re.compile(r"<details[\s\S]*?</details>|!\[[^\]]*\]\([^)]*\)|<think>[\s\S]*?</think>")


TYPE_PIECE = 4


def _type_pieces(text: str, size: int = TYPE_PIECE) -> list[str]:
    """The text in small pieces for typing it out; collapsible blocks, images and reasoning stay whole."""
    out, pos = [], 0
    for m in _TYPE_ATOMIC_RE.finditer(text or ""):
        plain = text[pos:m.start()]
        out += [plain[i:i + size] for i in range(0, len(plain), size)]
        out.append(m.group(0))
        pos = m.end()
    plain = (text or "")[pos:]
    out += [plain[i:i + size] for i in range(0, len(plain), size)]
    return [p for p in out if p]


async def type_out(text: str, rate: float, max_s: float = 0, typed_chars: int = 0, sleep=None):
    """`text` typed out at `rate` characters per second (blocks and figures cost no time), in at most `max_s` seconds
    when that is set; rate 0: all at once. `typed_chars`: only the first that many characters typed, the rest sent at
    once; 0 = the whole text."""
    if not text:
        return
    text = texts.relink_for_copy(texts.tr(text))         # our notes in the question's language and links to this copy's models, before the
                                      # text is cut into pieces (a link cut in two would not be rewritten)
    if not rate or rate <= 0:
        yield text
        return
    sleep = sleep or asyncio.sleep
    pieces = _type_pieces(text)
    typed, rest = pieces, []
    if typed_chars and typed_chars > 0:
        done = 0
        for k, p in enumerate(pieces):
            done += 0 if _TYPE_ATOMIC_RE.fullmatch(p) else len(p)
            if done >= typed_chars:
                typed, rest = pieces[:k + 1], pieces[k + 1:]
                break
    plain = sum(len(p) for p in typed if not _TYPE_ATOMIC_RE.fullmatch(p))
    total = plain / rate if not max_s or max_s <= 0 else min(plain / rate, max_s)
    per_char = total / plain if plain else 0
    for p in typed:
        yield p
        if not _TYPE_ATOMIC_RE.fullmatch(p):
            await sleep(len(p) * per_char)
    if rest:
        yield "".join(rest)


LIST_ALL_RE = re.compile(r'\s*<details type="' + LIST_ALL_TYPE + r'">.*?</details>\s*', re.S)


MIN_FIGURE_SIZE = 80    # on the page's 1000 grid: smaller boxes are icons, logos, journal badges


def icon_image(h: dict) -> bool:
    """A clear icon, not a figure: an image/chart passage whose box is under MIN_FIGURE_SIZE on both sides (publisher
    badges, licence marks, journal icons) or a QR code; such badges can carry a chapter's opening text and rank high.
    Lenient on purpose: a borderline image (one small side, no box) stays; the model and figures.shown_figure still
    judge it."""
    if h.get("type") not in ("image", "chart") or not h.get("img_path"):
        return False
    bb = h.get("bbox")
    if bb and len(bb) == 4 and bb[2] - bb[0] < MIN_FIGURE_SIZE and bb[3] - bb[1] < MIN_FIGURE_SIZE:
        return True
    try:
        from memento.common.qr_code import is_qr_code
        return bool(is_qr_code(h["img_path"]))
    except Exception:
        return False


# sections whose images are never figures: author photos and logos; exact names only (a heading like "Preferred
# Orientation" is a result)
_REFERENCE_SECTION_RE = re.compile(r"^(references|bibliography|literatur(verzeichnis)?|about the authors?|"
                                   r"authors?'? biograph\w*|biograph\w*)$", re.I)


def reference_image(h: dict) -> bool:
    """An image/chart passage from a References or author-biography section."""
    return (h.get("type") in ("image", "chart")
            and bool(_REFERENCE_SECTION_RE.match((h.get("section") or "").strip())))


def without_icons(hits: list) -> list:
    """The search results without icons (icon_image) and reference-section images (reference_image), in their
    order; all of them when only those were found."""
    kept = [h for h in hits if not icon_image(h) and not reference_image(h)]
    return kept or hits


_APP_LINKS = {"Library Upload": "/?model=memento-ingestion-prod", "Document Chat": "/?model=memento-document-chat"}


def link_app_names(text: str) -> str:
    """The first mention of each way to use your own documents as its link (the model may drop the prompt's links); one
    already linked stays."""
    for name, url in _APP_LINKS.items():
        if f"[{name}](" in text:
            continue
        text = re.sub(rf"(?<![\w\[]){re.escape(name)}(?![\w\]])", f"[{name}]({url})", text, count=1)
    return text


def gap_citation(i: int, h: dict, files_url: str) -> dict:
    """The citation event of research-gap passage [i+1], named "[N] document (p. 3)" like every other answer's: Open
    WebUI maps [N] to sources through their names and drops repeated names, so two passages from one page would shift
    later numbers; the "[" also makes the pill green."""
    source_obj = {"name": source_name(h)}
    file_url = source_file_url(files_url, h["source"], h["page"],
                                [h["bbox"]] if isinstance(h.get("bbox"), list) else None,
                                text=h.get("text"), section=h.get("section_raw"))
    if file_url:
        source_obj["embed_url"] = file_url
        source_obj["url"] = file_url
    return {"type": "citation",
            "data": {"document": [h["text"]],
                     "metadata": [{"source": f"{h['source']}#gap{i}",
                                   "name": f"[{i + 1}] {shown(h['source'])} ({page_label(h['page'], h.get('page_end'))})",
                                   "page": h["page"], "section": h.get("section")}],
                     "source": source_obj}}


def typing_speed(valves, metadata, at_once: bool = False) -> tuple:
    """(characters per second, cap) for typing out a finished answer; at once for the background warm-up and for an
    answer Thorough check has checked (the reader already waited for the check)."""
    if at_once or (metadata or {}).get("warmup"):
        return 0, 0, 0
    return valves.REPLAY_CHARS_PER_S, valves.REPLAY_MAX_S, valves.REPLAY_TYPED_CHARS


def citation_event(i: int, h: dict, files_url: str) -> dict:
    """The citation event of answer source [i+1]: its own card in Open WebUI's "Sources" footer, named "[N] document
    (page)", which opens the passage's page."""
    # Each chunk gets its own card in Open WebUI's "Sources" footer; the page number keeps cards of one document
    # apart.
    source_obj = {"name": source_name(h)}
    bbox = h.get("bbox")
    file_url = source_file_url(
        files_url, h["source"], h["page"],
        [bbox] if isinstance(bbox, list) else None,
        text=h.get("text"), section=h.get("section_raw"),
    )
    if file_url:
        # embed_url makes Open WebUI open the source in its viewer on click, instead of the excerpt modal
        source_obj["embed_url"] = file_url
        # The footer's "Sources" card opens another modal (CitationsModal) that ignores embed_url and links
        # source.url when it contains "http"; setting url too gives it a working link to this chunk's page.
        source_obj["url"] = file_url
    return {
        "type": "citation",
        "data": {
            "document": [h["text"]],
            "metadata": [
                {
                    "source": f"{h['source']}#{i}",
                    "name": f"[{i + 1}] {shown(h['source'])} ({page_label(h['page'], h.get('page_end'))})",
                    "page": h["page"],
                    "section": h.get("section"),
                }
            ],
            "source": source_obj,
        },
    }


def chunk_label(h: dict, whole_doc: bool, ctx_names: dict) -> str:
    """A source's label in the answer model's context: its document (the title the model reads), page and section."""
    sec, raw = h.get("section"), h.get("section_raw")
    # a summary's sources are all one document, named once above them
    label = (f"page {h['page'] + 1}" if whole_doc
             else f"{context_name(h['source'], ctx_names)}, page {h['page'] + 1}")
    if sec:
        if raw and raw.lower() != sec.lower():
            label += f", Section: {sec} › {raw}"
        else:
            label += f", Section: {sec}"
    return label
