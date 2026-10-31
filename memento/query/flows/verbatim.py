"""A section shown word for word: which section the question means (its number, its heading, the document's own
sections), where it starts and ends, and how much of it is shown.
"""
import asyncio
import re

from memento.query import library_state, presentation, prompts, texts
from memento.query.flows import figures


# "Give me the abstract of X" wants the section's actual text, not a summary. Retrieval runs as usual; afterwards, for a
# show-me request, the named document (else the documents of the best passages) is read in reading order, the section
# the query names is found by its heading (section_span) and its passages are reproduced verbatim. No matching heading:
# the normal cited answer, so a loose match changes only the format, never what was retrieved. Section words come from
# the domain profile's section_hints; without them, heading words that appear in the query.

# The section is found by its heading text in reading order, not by the canonical `section` label (it sticks across
# later headings) or by position on the page (two-column layouts); long sections are not cut at a fixed number of
# passages.
VERBATIM_VISIBLE, VERBATIM_MAX = 20, 200


_TOC_LINE_RE = re.compile(r"(\.{3,}|…)\s*\d{1,4}\s*$|^\s*\d+(?:\.\d+)*\.?\s+\S.*\s\d{1,4}\s*$")


def _toc_like(texts: list) -> bool:
    """Passages that are table-of-contents lines: most lines end in a page number after dot leaders or a section
    number ("4.4 Controlling the micro- and nanostructure ..... 183")."""
    lines = [l for t in texts for l in (t or "").splitlines() if l.strip()]
    return bool(lines) and sum(bool(_TOC_LINE_RE.search(l.strip())) for l in lines) * 2 > len(lines)


def _heading_number(raw: str) -> str | None:
    m = re.match(r"\s*(\d+(?:\.\d+)*)\.?\s", raw or "")
    return m.group(1) if m else None


# a conclusion chapter is often split into summary-like sibling headings ("7.1 Zusammenfassung", "7.2 Ausblick")
_SIBLING_VOCAB = {"Conclusion": ("Conclusion", "Abstract")}


def section_span(passages: list, query: str, section_hints: dict, ranked: list) -> list:
    """The passages of the section the query asks for, from one document in reading order ([] = no such heading).

    The document is a sequence of heading runs (consecutive passages under one heading). The heading is chosen by the
    domain vocabulary ("conclusion" → conclusion, fazit, summary, outlook, …; a single heading that is exactly the
    query's word wins), by a section number in the query (that number and its subsections), else by 2+ of its words (or
    its only word) in the query. Words of the document's name are ignored. Of several qualifying headings: the one the
    reranker ranked highest, else the last for Conclusion and the first otherwise. The span runs to the next heading,
    numbered subsections included; a numbered conclusion also takes adjacent summary-like siblings."""
    # the document's own name is not a section request
    title = set(re.findall(r"[a-zäöüß0-9]+", (passages[0]["source"] if passages else "").lower()))
    nq = " ".join(w for w in re.findall(r"[a-zäöüß0-9]+(?:\.[0-9]+)*", query.lower()) if w not in title)
    hints = section_hints or {}
    name = next((n for n, trig in hints.items() if any(t in nq for t in trig)), None)
    triggers = [t for t in hints.get(name) or [] if t in nq] or list(hints.get(name) or [])
    m_num = re.search(r"\b(\d+(?:\.\d+)*)\b", nq) if not name else None
    wanted = m_num.group(1) if m_num else None

    runs = []                                         # [heading, first index, end index]
    for k, h in enumerate(passages):
        raw = h.get("section_raw") or ""
        if runs and raw in (runs[-1][0], ""):
            runs[-1][2] = k + 1
        else:
            runs.append([raw, k, k + 1])

    def num(r):
        return _heading_number(r[0]) or ""

    # a table of contents parsed as headings
    toc = {k for k, r in enumerate(runs) if _toc_like([passages[i]["text"] for i in range(r[1], r[2])])}

    def under(r, prefix):                             # the heading numbered `prefix` or one of its subsections
        return bool(prefix) and (num(r) == prefix or num(r).startswith(prefix + "."))

    def by_words(words):
        return [k for k, r in enumerate(runs) if r[0] and any(w in r[0].lower() for w in words)]

    if name:
        cands = by_words(hints[name])
        bare = lambda r: re.sub(r"^\s*\d+(?:\.\d+)*\.?\s*", "", r[0]).strip(" *").lower()
        exact = [k for k in cands if bare(runs[k]) in [t for t in hints[name] if t in nq]]
        if len(exact) == 1:                           # the one heading literally named ("Kurzfassung")
            cands = exact
    elif wanted:
        cands = [k for k, r in enumerate(runs) if under(r, wanted)]
    else:
        def named(r):                                 # 2+ of the heading's words, or its only word, in the query
            words = set(re.findall(r"[a-zäöüß]{4,}", r[0].lower()))
            hit = sum(w in nq.split() for w in words)
            return bool(words) and hit >= min(2, len(words))
        cands = [k for k, r in enumerate(runs) if r[0] and named(r)]
    in_toc = [k for k in cands if k in toc]
    cands = [k for k in cands if k not in toc]
    if not cands and name:
        # the section is named only in the table of contents (its chapter heading has no text of its own): take the
        # chapter by its number
        wanted = next((num(runs[k]) for k in in_toc if num(runs[k])), None)
        cands = [k for k, r in enumerate(runs) if wanted and under(r, wanted) and k not in toc]
    if not cands:
        return []
    run_of = {}                                       # passage index → run index
    for k, r in enumerate(runs):
        run_of.update(dict.fromkeys(range(r[1], r[2]), k))
    where = {(h["text"], h.get("section_raw") or ""): i for i, h in enumerate(passages)}
    # the reranker decides only with a strong signal (one of its top 5 passages); deeper ranks shift between runs
    anchor = next((run_of[where[key]] for key in ((r["text"], r.get("section_raw") or "") for r in ranked[:5])
                   if key in where and run_of[where[key]] in cands), None)
    if anchor is None:
        if name == "Conclusion":                      # the last one: the final chapter, not a chapter's "Fazit"
            anchor = cands[-1]
        else:                                         # least nested first
            top = min(num(runs[k]).count(".") if num(runs[k]) else -1 for k in cands)
            anchor = next(k for k in cands if (num(runs[k]).count(".") if num(runs[k]) else -1) == top)
    if wanted:                                        # "section 5.2": from its first run, all of 5.2.*
        anchor = cands[0]
    prefix = wanted or num(runs[anchor])
    first = last = anchor
    def continues(k):                                 # an unnumbered heading ("Fazit") inside the numbered section
        if under(runs[k], prefix):
            return True
        if num(runs[k]):
            return False
        nxt = next((j for j in range(k + 1, len(runs)) if num(runs[j])), None)
        return nxt is not None and under(runs[nxt], prefix)

    while prefix and last + 1 < len(runs) and continues(last + 1):
        last += 1
    if name and not wanted and "." in num(runs[anchor]):
        parent, depth = num(runs[anchor]).rsplit(".", 1)[0], num(runs[anchor]).count(".")
        family = [w for n in _SIBLING_VOCAB.get(name, (name,)) for w in hints.get(n) or []]

        def sibling(r):
            return (num(r).count(".") == depth and num(r).startswith(parent + ".")
                    and any(w in r[0].lower() for w in family))

        k = first - 1
        while k >= 0:                                 # step over an earlier sibling's subsections to the sibling
            while k >= 0 and num(runs[k]).startswith(parent + ".") and num(runs[k]).count(".") > depth:
                k -= 1
            if k >= 0 and sibling(runs[k]):
                first, k = k, k - 1
            else:
                break
        while last + 1 < len(runs) and sibling(runs[last + 1]):
            last += 1
            head = num(runs[last])
            while last + 1 < len(runs) and under(runs[last + 1], head):
                last += 1
    return passages[runs[first][1]:runs[last][2]]


async def find_section(turn) -> None:
    """A section asked for as it stands: the section's passages in reading order, from the named document or the best
    passages' documents."""
    for doc in ([turn.paper] if turn.paper else list(dict.fromkeys(h["source"] for h in turn.hits))[:3]):
        try:
            passages = await asyncio.to_thread(library_state.ordered_passages, turn.client, turn.collection, doc)
        except Exception:
            continue
        span = section_span(passages, turn.query, turn.dom.get("section_hints", {}),
                             [h for h in turn.hits if h["source"] == doc])
        if span:
            turn.verbatim_total = len(span)
            turn.hits, turn.verbatim = span[:VERBATIM_MAX], True
            await turn.status(f"Reproducing '{span[0].get('section_raw')}' section of {doc}...")
            break


async def reply(turn):
    """The section reproduced: its passages as they stand with their numbers, the rest folded, its figures."""
    yield f"## {turn.hits[0].get('section_raw') or 'Section'} ({presentation.shown(turn.hits[0]['source'])})\n\n"
    for i, h in enumerate(turn.hits[:VERBATIM_VISIBLE]):
        yield f"{h['text'].strip()} [{i + 1}]\n\n"
    rest = turn.hits[VERBATIM_VISIBLE:]
    if rest:       # the rest of a long section behind one collapsible block: nothing is cut silently
        body = "".join(f"{h['text'].strip()} [{i + 1}]\n\n" for i, h in enumerate(rest, VERBATIM_VISIBLE))
        yield (f"<details>\n<summary>Show the rest of the section ({len(rest)} more passages)</summary>\n\n"
               f"{body}</details>\n\n")
    if turn.verbatim_total > len(turn.hits):
        yield (f"_The section continues in the document ({turn.verbatim_total - len(turn.hits)} more passages): open a "
               f"citation to read on._\n\n")
    for i, h in enumerate(turn.hits[:VERBATIM_VISIBLE]):
        if figures.shown_figure(h):
            url = prompts.verified_image_url(h["img_path"])
            if url:   # missing or unreadable image → text only
                yield figures.figure_markdown(i, h, url)
    await turn.status("Done", done=True)
    turn.done = True
