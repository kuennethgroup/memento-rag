"""The open-research-questions report built from the gap ledger. Every fact comes from the ledger and is rendered here
by code: the questions, who raises them, their status in the library and the literature, the sources [N], how and when
they were checked. The model writes only the overview, "why it matters" per question and suggested directions; their
citations and question numbers are checked mechanically, and the overview and "why it matters" against the sent sources
(the Thorough-check machinery, without new searches) before anything is shown."""
from __future__ import annotations

import json
import re

from memento.research_gaps import fulltext as ft

STATUS_ORDER = {"contested": 0, "open": 1, "partly": 2}
REL_ORDER = {"addresses": 0, "contradicts": 1, "partly": 2}
MAX_LIBRARY, MAX_PAPERS, MAX_STATING = 3, 3, 2
SERVICES = {"semantic_scholar": "Semantic Scholar", "openalex": "OpenAlex", "alexandria": "Offline index"}
MARK = "#### §"                         # section markers in the text sent to the check


PER_DOCUMENT = 2                        # questions of one document shown in full, so one document cannot fill the
                                        # report


def select(gaps: list[dict], *, relevance: dict | None = None, full: int = 10, related_counts: dict | None = None,
           per_document: int = PER_DOCUMENT) -> dict:
    """{"full", "more", "addressed"}: the questions to show in full, the ones only listed, and the library-addressed
    ones. relevance {gap id: score}: a topic's relevant questions (others left out), ordered by score first. Then
    status (contested, open, partly), the number of raising documents, the number of related questions (a question
    linked to many is central to the library), id. At most `per_document` questions of one document in full."""
    rel, links = relevance or {}, related_counts or {}
    pool = [g for g in gaps if g.get("library_status") and (relevance is None or g["id"] in rel)]

    def key(g):
        return (-rel.get(g["id"], 0.0), STATUS_ORDER.get(g["library_status"], 9), -len(g.get("documents") or []),
                -links.get(g["id"], 0), g["id"])
    ordered = sorted((g for g in pool if g["library_status"] in STATUS_ORDER), key=key)
    shown, per = [], {}
    for g in ordered:                            # first pass: at most `per_document` of one document
        first = (g.get("documents") or [""])[0]
        if len(shown) < full and per.get(first, 0) < per_document:
            shown.append(g)
            per[first] = per.get(first, 0) + 1
    for g in ordered:                            # then free slots in order: the cap never leaves one empty
        if len(shown) < full and g not in shown:
            shown.append(g)
    shown = [g for g in ordered if g in shown]
    rest = [g for g in ordered if g not in shown]
    return {"full": shown, "more": rest,
            "addressed": sorted((g for g in pool if g["library_status"] == "addressed"), key=key)}


TABLE_ROWS = 20                         # rows of a collapsed table in a report


def page_select(gaps: list[dict], *, page: int = 1, per_page: int = 10, relevance: dict | None = None,
                related_counts: dict | None = None, per_document: int = PER_DOCUMENT) -> dict:
    """select() for report page `page` ("more research gaps"): the earlier pages taken in turn, each the next `per_page`
    in select's order and per-document cap, so page 2 continues page 1 exactly. Adds "start" (the first question's
    number) and "total" (the open questions of the whole scope)."""
    rest, sel = list(gaps), None
    for _ in range(max(1, page)):
        if sel is not None:
            taken = {g["id"] for g in sel["full"]}
            rest = [g for g in rest if g["id"] not in taken]
        sel = select(rest, relevance=relevance, full=per_page, related_counts=related_counts,
                     per_document=per_document)
    first = select(gaps, relevance=relevance, full=per_page, related_counts=related_counts, per_document=per_document)
    total = len(first["full"]) + len(first["more"])
    return {**sel, "addressed": first["addressed"], "start": (max(1, page) - 1) * per_page + 1, "total": total}


def by_document(gaps: list[dict], names: dict) -> list[dict]:
    """[{"source", "name", "count", "top"}]: each document with the open questions it raises (a question raised by
    two documents counts for both), most first; "top" is its most open one in the report's order."""
    per: dict = {}
    for g in gaps:
        if g.get("library_status") in STATUS_ORDER:
            for d in dict.fromkeys(g.get("documents") or []):
                per.setdefault(d, []).append(g)
    rows = []
    for d, gs in per.items():
        top = sorted(gs, key=lambda g: (STATUS_ORDER[g["library_status"]], -len(g.get("documents") or []), g["id"]))[0]
        rows.append({"source": d, "name": names.get(d, d), "count": len(gs), "top": top["statement"]})
    return sorted(rows, key=lambda r: (-r["count"], r["name"].lower()))


_SCOPE_HEAD = re.compile(r'^## Open research questions (?:in "(?P<doc>.+)"|on (?P<topic>.+?)|in the library)'
                         r'(?: \(questions \d+–\d+\))?\s*$', re.M)


def report_scope(report_md: str) -> dict | None:
    """What an earlier report covered, for its next page: {"kind": library | topic | document, "value", "last": the
    highest question number shown}; None when it is no report."""
    m = _SCOPE_HEAD.search(report_md or "")
    if not m:
        return None
    nums = [int(n) for n in re.findall(r"(?m)^### (\d+)\. ", report_md)]
    kind = "document" if m.group("doc") else "topic" if m.group("topic") else "library"
    return {"kind": kind, "value": m.group("doc") or m.group("topic"), "last": max(nums, default=0)}


def relevant_only(scores: dict, *, ratio: float = 0.4, floor: float = 0.05) -> dict:
    """A topic's questions: reranker scores of statements against a topic are low in absolute terms, so a question is
    kept when it scores at least `ratio` × the best, and at least `floor`."""
    if not scores:
        return {}
    cut = max(floor, ratio * max(scores.values()))
    return {k: v for k, v in scores.items() if v >= cut}


_DOI_PREFIX = re.compile(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", re.I)


def doi_key(doi: str | None) -> str:
    """A DOI as one paper's identity: the services write it as "10.x/…", "doi:10.x/…" or a doi.org link, in any
    case (DOIs are case-insensitive); "" when there is none."""
    d = _DOI_PREFIX.sub("", (doi or "").strip()).strip().rstrip(".,;").lower()
    return d if d.startswith("10.") else ""


def _paper_key(e: dict) -> tuple:
    """One outside paper: its DOI when it has one (the same paper may be found by several services), else the service's
    own id; never its title (two papers can share one, and the services spell titles differently)."""
    d = doi_key(e.get("doi"))
    return ("doi", d) if d else ("paper", e.get("source") or e.get("title"))


def _merge_paper(src: dict, e: dict) -> None:
    """A paper already numbered, found again (another question, another service): keep the best details. The year and
    DOI link from whichever has them; the title of Semantic Scholar or OpenAlex over the offline index's (which keeps
    the PDF's spacing); the longer abstract; every service that found it."""
    origin = e.get("origin") or ""
    if src["services"] == ["alexandria"] and origin != "alexandria" and e.get("title"):
        src["title"] = src["source"] = e["title"]
    src["year"] = src.get("year") or e.get("year")
    if not str(src.get("url") or "").startswith("https://doi.org/") and str(e.get("url") or "").startswith(
            "https://doi.org/"):
        src["url"] = e["url"]
    src["doi"] = src.get("doi") or e.get("doi") or ""
    if len(e.get("text") or "") > len(src.get("text") or ""):
        src["text"] = e["text"]
    if origin in SERVICES and e.get("source"):
        src.setdefault("ids", {}).setdefault(origin, e["source"])
    if origin and origin not in src["services"]:
        src["services"] = [k for k in SERVICES if k in {*src["services"], origin}]


# ── paper cards: an outside paper shown in the side panel like a library page ──
_ABBREV = re.compile(r"(?:\b(?:e\.g|i\.e|et al|Fig|Figs|Eq|Eqs|Ref|Refs|vs|ca|approx|No|cf|resp)\.|\b[A-Z]\.)$")


def paper_sentences(text: str) -> list[str]:
    """An abstract as sentences (for marking the ones related to a question); joined with spaces they give the
    abstract back (whitespace normalised). Not split after common abbreviations ("e.g.", "Fig.", "et al.")."""
    t = " ".join((text or "").split())
    if not t:
        return []
    out: list[str] = []
    for part in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\[\"'“])", t):
        if out and _ABBREV.search(out[-1]):
            out[-1] = f"{out[-1]} {part}"
        else:
            out.append(part)
    return out


def pick_marks(scores: list[float], *, floor: float = 0.05, ratio: float = 0.5) -> list[int]:
    """The sentences to mark for one question (reranker scores, one per sentence): the best if it reaches `floor`,
    a second if it scores at least `ratio` × the best; none when nothing stands out."""
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    if not order or scores[order[0]] < floor:
        return []
    keep = [order[0]] + [i for i in order[1:2] if scores[i] >= max(floor, ratio * scores[order[0]])]
    return sorted(keep)


_S2_ID = re.compile(r"[0-9a-f]{40}")
_OA_ID = re.compile(r"(?:https://openalex\.org/)?(W\d+)")


def service_links(ids: dict) -> list[dict]:
    """[{"name", "url"}] per service that found a paper, in SERVICES order; the url only from a well-formed id (Semantic
    Scholar paper id, OpenAlex work id); the offline literature index has no page."""
    out = []
    for k in SERVICES:
        if k not in (ids or {}):
            continue
        v, url = str(ids[k] or ""), ""
        if k == "semantic_scholar" and _S2_ID.fullmatch(v):
            url = f"https://www.semanticscholar.org/paper/{v}"
        elif k == "openalex" and _OA_ID.fullmatch(v):
            url = f"https://openalex.org/{_OA_ID.fullmatch(v).group(1)}"
        out.append({"name": SERVICES[k], "url": url})
    return out


def gather_sources(gaps: list[dict], evidence: dict, passages: dict,
                   answered: list[dict] = ()) -> tuple[list[dict], dict]:
    """(numbered sources, refs per gap). Sources: the stating passages (the stored passage, located through its point
    when it still exists), the library evidence that still resolves to a passage (else dropped), the outside papers with
    their stored abstract (one without stays unnumbered: it cannot be checked). Each source once. `answered`: questions
    the library may answer; only their answering passages, numbered after all the others, so the report's own numbers
    and the writer's sources stay the same."""
    sources: list[dict] = []
    index: dict = {}

    def add(key, src) -> int:
        if key not in index:
            sources.append(src)
            index[key] = len(sources)
        return index[key]
    refs = {}
    for g in gaps:
        r = {"stating": [], "library": [], "papers": [], "unnumbered": [], "reasons": {}, "unnumbered_reasons": []}
        seen_docs = []
        for m in g.get("mentions") or []:
            if m["source"] in seen_docs or len(seen_docs) >= MAX_STATING:
                continue
            seen_docs.append(m["source"])
            p = passages.get(str(m.get("point_id")))
            src = {**p, "kind": "stating"} if p else {
                "kind": "stating", "source": m["source"], "page": m.get("page") or 0, "text": m.get("passage") or "",
                "type": "text", "section": None, "section_raw": None, "bbox": None, "page_end": None, "img_path": None}
            if not p:
                src["text"] = m.get("passage") or ""
            r["stating"].append(add(("pass", str(m.get("point_id")), m["source"]), src))
        ev = sorted(evidence.get(g["id"]) or [], key=lambda e: REL_ORDER.get(e["relation"], 9))
        lib = [e for e in ev if e["origin"] == "library" and e["relation"] in REL_ORDER]
        for e in lib:
            if len(r["library"]) >= MAX_LIBRARY:
                break
            p = passages.get(str(e.get("point_id")))
            if p:
                n = add(("pass", str(e["point_id"]), p["source"]), {**p, "kind": "library"})
                r["library"].append((n, e["relation"]))
                r["reasons"][n] = e.get("reason") or ""
        for e in (e for e in ev if e["origin"] != "library" and e["relation"] in REL_ORDER):
            if len(r["papers"]) + len(r["unnumbered"]) >= MAX_PAPERS:
                break
            key = _paper_key(e)
            n = index.get(key)
            if n is None and not e.get("text"):        # no abstract here or elsewhere: listed, cannot be checked
                r["unnumbered"].append((e.get("title") or "", e.get("year"), e.get("url") or "", e["relation"]))
                r["unnumbered_reasons"].append(e.get("reason") or "")
                continue
            if n is None:
                n = add(key, {"kind": "paper", "source": e.get("title") or "", "title": e.get("title") or "",
                              "year": e.get("year"), "url": e.get("url") or "", "doi": e.get("doi") or "", "page": 0,
                              "text": e["text"], "type": "text", "section": None, "section_raw": None, "bbox": None,
                              "page_end": None, "img_path": None,
                              "services": [e["origin"]] if e.get("origin") in SERVICES else [],
                              "ids": {e["origin"]: e.get("source")} if e.get("origin") in SERVICES else {}})
            else:
                _merge_paper(sources[n - 1], e)
            if ft.read_in_full(e.get("text")):      # per question: a merged source holds another question's excerpts
                r.setdefault("read", set()).add(n)
            if any(k == n for k, _ in r["papers"]):  # the same paper twice for one question: its first (strongest) verdict
                continue
            r["papers"].append((n, e["relation"]))
            r["reasons"][n] = e.get("reason") or ""
        refs[g["id"]] = r
    for g in answered:
        r = {"stating": [], "library": [], "papers": [], "unnumbered": [], "reasons": {}, "unnumbered_reasons": []}
        for e in evidence.get(g["id"]) or []:
            if e.get("origin") != "library" or e.get("relation") != "addresses" or len(r["library"]) >= MAX_LIBRARY:
                continue
            p = passages.get(str(e.get("point_id")))
            if p:
                n = add(("pass", str(e["point_id"]), p["source"]), {**p, "kind": "library"})
                if all(k != n for k, _ in r["library"]):
                    r["library"].append((n, "addresses"))
                    r["reasons"][n] = e.get("reason") or ""
        refs[g["id"]] = r
    return sources, refs


def own_source_count(refs: dict, gaps: list[dict]) -> int:
    """How many of the numbered sources belong to these questions (the report's own, before any answering passages):
    the writing model and its check see only these."""
    return max([n for g in gaps for r in [refs.get(g["id"]) or {}] for n in
                list(r.get("stating") or []) + [k for k, _ in (r.get("library") or []) + (r.get("papers") or [])]],
               default=0)


# ── facts, rendered by code ───────────────────────────────────────────────────

def _date(d, tz: str | None = None) -> str:
    """The day a check ran, in the server's time zone (or `tz`): stored times are UTC."""
    if not d:
        return "an earlier date"
    try:
        from zoneinfo import ZoneInfo
        from datetime import timezone as _tz
        d = (d if d.tzinfo else d.replace(tzinfo=_tz.utc)).astimezone(ZoneInfo(tz) if tz else None)
    except Exception:
        pass
    return f"{d.day} {d.strftime('%b %Y')}"


def _one_line(t) -> str:
    return " ".join(str(t or "").split())


def _cites(ns) -> str:
    return "".join(f"[{n}]" for n in ns)


def _services(csv: str | None) -> str:
    names = [SERVICES[s] for s in ("semantic_scholar", "openalex", "alexandria") if s in (csv or "").split(",")]
    return (", ".join(names[:-1]) + " and " + names[-1]) if len(names) > 1 else (names[0] if names else "no service")


def cap_docs_checked(gaps: list[dict], library_size: int | None) -> list[dict]:
    """Copies of the questions whose "checked against the other N documents" is never more than the library has: N at
    most the library's size minus the documents that raise the question. Size unknown: unchanged."""
    if not library_size:
        return list(gaps)
    out = []
    for g in gaps:
        n, most = g.get("library_docs_checked"), max(library_size - len(g.get("documents") or []), 0)
        out.append({**g, "library_docs_checked": min(n, most)} if n is not None else dict(g))
    return out


def library_line(g: dict, refs: dict) -> str:
    st, n = g.get("library_status"), g.get("library_docs_checked")
    by = {rel: [k for k, r in refs["library"] if r == rel] for rel in REL_ORDER}
    others = f"the other {n} documents" if n else "the other documents"
    if st == "open":
        return f"Nothing in the library answers it (checked against {others}, {_date(g.get('library_checked_at'))})."
    if st == "contested":
        return f"Documents of the library disagree {_cites(by['contradicts'] + by['addresses'] + by['partly'])}."
    if st == "addressed":
        return f"The library may already answer this {_cites(by['addresses'] + by['partly'])}: check the evidence."
    return f"Related findings in the library {_cites(by['partly'] + by['addresses'])}, not a full answer."


def _read_from(refs: dict) -> str:
    """What this question's verdicts on its papers rest on."""
    return "abstracts and full-text excerpts" if refs.get("read") else "abstracts"


def reads_full_text(evidence: dict) -> bool:
    """Whether any stored verdict of these questions read a paper's full text, near misses included."""
    return any(ft.read_in_full(e.get("text")) for rows in evidence.values() for e in rows)


def literature_line(g: dict, refs: dict, titles: dict) -> str:
    st = g.get("web_status")
    if not st:
        return "Not checked."
    papers = [f"{_one_line(titles.get(k, ('', None))[0])} ({titles.get(k, ('', None))[1] or 'n.d.'}) [{k}]"
              for k, _rel in refs["papers"]]
    papers += [f"{_one_line(t)} ({y or 'n.d.'})" for t, y, _u, _r in refs["unnumbered"]]
    where = f" ({_services(g.get('web_sources'))}, {_date(g.get('web_checked_at'))})"
    if st == "no_results":
        return f"No papers found for the search{where}."
    if st == "nothing_found":
        return f"Nothing found among the closest papers read{where}."
    read = f"(judged from {_read_from(refs)})"
    label = {"possibly_addressed": f"Possibly addressed {read}", "partly": f"Related findings {read}",
             "contested": f"Papers report conflicting findings {read}"}.get(st, st)
    return f"{label}: {'; '.join(papers)}{where}."


LIB_REL = {"addresses": "Answers it", "contradicts": "Disagrees", "partly": "Partly"}
WEB_REL = {"addresses": "Possibly answers it", "contradicts": "Disagrees", "partly": "Partly"}
LIB_HEAD = {"addressed": "may already be answered", "partly": "partly answered", "contested": "documents disagree"}
WEB_HEAD = {"possibly_addressed": "possibly answered", "partly": "related findings", "contested": "papers disagree"}


def _reason(text: str) -> str:
    """The check's reason, whole (one sentence, never cut mid-sentence)."""
    return _one_line(text)


def library_block(g: dict, refs: dict, sources: list, names: dict, pages: bool = True) -> list[str]:
    """The library status, then every piece of evidence on its own line: which document, page, [N], how, and why (the
    check's own reason): which document answers a question, and which only partly. `pages=False` in chat, where the
    citation pill shows the page; the PDF and the writing model keep it."""
    st, n = g.get("library_status"), g.get("library_docs_checked")
    when = f"checked against the other {n} documents" if n else "checked against the other documents"
    when += f", {_date(g.get('library_checked_at'))}"
    if st == "open":
        return [f"- **In the library:** nothing answers it ({when})."]
    out = [f"- **In the library:** {LIB_HEAD.get(st, st)} ({when}):"]
    for k, rel in sorted(refs["library"], key=lambda x: REL_ORDER.get(x[1], 9)):
        s = sources[k - 1]
        why = refs.get("reasons", {}).get(k)
        page = f", p. {int(s.get('page') or 0) + 1}" if pages else ""
        out.append(f"  - {LIB_REL.get(rel, rel)}: {names.get(s['source'], s['source'])}{page} [{k}]"
                   + (f": {_reason(why)}" if why else ""))
    if not refs["library"]:
        out.append("  - (the passages found then are no longer in the library)")
    return out


def literature_block(g: dict, refs: dict, sources: list) -> list[str]:
    st = g.get("web_status")
    if not st:
        return ["- **In the literature:** not checked yet."]
    where = f"{_services(g.get('web_sources'))}, {_date(g.get('web_checked_at'))}"
    if st == "no_results":
        return [f"- **In the literature:** no papers found for the search ({where})."]
    if st == "nothing_found":
        return [f"- **In the literature:** nothing found among the closest papers read ({where})."]
    out = [f"- **In the literature:** {WEB_HEAD.get(st, st)} ({where}; judged from {_read_from(refs)}):"]
    for k, rel in sorted(refs["papers"], key=lambda x: REL_ORDER.get(x[1], 9)):
        s = sources[k - 1]
        why = refs.get("reasons", {}).get(k)
        out.append(f"  - {WEB_REL.get(rel, rel)}: {_one_line(s.get('title') or s['source'])} "
                   f"({s.get('year') or 'n.d.'}) [{k}]" + (f": {_reason(why)}" if why else ""))
    for (t, y, url, rel), why in zip(refs["unnumbered"], refs.get("unnumbered_reasons") or [""] * 99):
        out.append(f"  - {WEB_REL.get(rel, rel)}: {_one_line(t)} ({y or 'n.d.'})" + (f", {url}" if url else "")
                   + (f": {_reason(why)}" if why else ""))
    return out


SHORT_WORDS, SHORT_MAX = 8, 12
TABLE_LIB = {"open": "nothing answers it", "partly": "partly answered", "addressed": "may answer it",
             "contested": "documents disagree"}
TABLE_WEB = {"nothing_found": "nothing found", "partly": "related findings", "possibly_addressed": "possibly answered",
             "contested": "papers disagree", "no_results": "no results"}


def short_title(statement: str, given: str | None) -> str:
    """The question's title in the report's table: the model's short title, at most 12 words, else the question's first
    8 words."""
    words = (given or "").split()
    if words:
        return " ".join(words[:SHORT_MAX]) + ("…" if len(words) > SHORT_MAX else "")
    words = _one_line(statement).split()
    return " ".join(words[:SHORT_WORDS]) + ("…" if len(words) > SHORT_WORDS else "")


def _cell(t: str) -> str:
    return str(t).replace("|", "/")


def render_question(k: int, g: dict, refs: dict, *, why: str, names: dict, sources: list) -> str:
    raised = ", ".join(f"{names.get(d, d)} {_cites([n])}" for d, n in
                       zip(dict.fromkeys(m["source"] for m in g.get("mentions") or []), refs["stating"]))
    out = [f"### {k}. {g['statement']}", "",
           f"- **Raised by:** {raised}",
           *library_block(g, refs, sources, names, pages=False),
           *literature_block(g, refs, sources)]
    if why.strip():
        out.append(f"- **Why it matters:** {why.strip()}")
    return "\n".join(out)


# ── the model's prose ─────────────────────────────────────────────────────────

PROSE_SYSTEM = """\
You write parts of a report on OPEN RESEARCH QUESTIONS found in the documents of a research library. You get numbered \
sources (passages of the library's documents, and abstracts of published papers) and numbered questions, each with \
the facts already established about it. Those facts are shown to the reader exactly as given: do not repeat, change \
or judge them, and never say whether a question is open or answered.
Write ONLY:
- "overview": two or three sentences on what the questions are about across the documents; cite sources as [N].
- "why": for each question number, one or two sentences on why it matters, in the raising documents' own terms, \
citing the question's sources as [N]. Use only what the sources say.
- "suggestions": up to three suggested research directions, each building on named question numbers \
("builds_on"). These are labelled as your suggestions; add no new facts and no citations.
- "short": for each question number, a short title of at most eight words (no citations), for a table.
Cite each source in its own brackets, as [1][4], never [1, 4]. Write about the science, never about "the provided \
text", "the sources" or "the passage".
Output ONLY JSON: {"overview": "...", "why": {"1": "...", "2": "..."}, "suggestions": [{"text": "...", \
"builds_on": [1, 3]}], "short": {"1": "...", "2": "..."}}"""


def prose_messages(questions: list[dict], sources: list[dict], topic: str | None = None,
                   max_chars: int = 1200) -> list[dict]:
    """questions: [{"n", "statement", "facts", "sources": [N…]}]."""
    def block(k, s):
        text = " ".join(str(s.get("text") or "").split())[:max_chars]
        if s["kind"] == "paper":
            read = "abstract and excerpts" if ft.read_in_full(text) else "abstract"
            return f"[{k}] {s['source']} ({s.get('year') or 'n.d.'}), {read}\n{text}"
        return f"[{k}] {s['source']}, page {int(s.get('page') or 0) + 1}\n{text}"
    src = "\n\n".join(block(k + 1, s) for k, s in enumerate(sources))
    qs = "\n\n".join(f"{q['n']}. {q['statement']}\nFacts: {q['facts']}\nIts sources: {_cites(q['sources'])}"
                     for q in questions)
    about = f"Topic: {topic}\n\n" if topic else "Topic: the whole library\n\n"
    return [{"role": "system", "content": PROSE_SYSTEM},
            {"role": "user", "content": f"{about}Sources:\n{src}\n\nQuestions:\n{qs}"}]


def _split_cites(m) -> str:
    out = []
    for part in re.split(r"\s*[,;]\s*", m.group(1)):
        a, _, b = part.partition("-") if "-" in part else part.partition("–")
        nums = range(int(a), int(b) + 1) if b and a.isdigit() and b.isdigit() and int(b) - int(a) < 20 else \
            [int(part)] if part.isdigit() else []
        out += [f"[{n}]" for n in nums]
    return "".join(dict.fromkeys(out))


def _clean_cites(text: str, n_sources: int) -> str:
    text = re.sub(r"\[(\d+(?:\s*[,;\-–]\s*\d+)+)\]", _split_cites, text or "")
    text = re.sub(r"\[(\d+)\]", lambda m: m.group(0) if 1 <= int(m.group(1)) <= n_sources else "", text or "")
    return re.sub(r"\s+([.,;:])", r"\1", re.sub(r"[ \t]{2,}", " ", text)).strip()


def parse_prose(raw: str, *, n_sources: int, n_questions: int) -> dict:
    """The model's JSON, checked: citations to sources never sent removed; "why" only for existing question numbers;
    suggestions without citations, kept only when they build on existing questions. Raises ValueError if unreadable."""
    from memento.agentic_validation.jsonfix import loads_lenient
    try:
        a, b = raw.index("{"), raw.rindex("}") + 1
        data = loads_lenient(raw[a:b])
    except Exception as e:
        raise ValueError(f"unreadable report prose: {e}") from e
    if not isinstance(data, dict):
        raise ValueError("report prose is not an object")
    why = {}
    for k, v in (data.get("why") or {}).items():
        try:
            n = int(k)
        except (TypeError, ValueError):
            continue
        if 1 <= n <= n_questions and str(v or "").strip():
            why[n] = _clean_cites(str(v), n_sources)
    sugs = []
    for s in data.get("suggestions") or []:
        if not isinstance(s, dict):
            continue
        text = re.sub(r"\s*\[\d+\]", "", str(s.get("text") or "")).strip()
        try:
            on = sorted({int(x) for x in s.get("builds_on") or []})
        except (TypeError, ValueError):
            continue
        if text and on and all(1 <= x <= n_questions for x in on):
            sugs.append({"text": text, "builds_on": on})
    short = {}
    for k, v in (data.get("short") or {}).items():
        try:
            n = int(k)
        except (TypeError, ValueError):
            continue
        t = _one_line(re.sub(r"\s*\[\d+\]", "", str(v or ""))).strip(" .")
        if 1 <= n <= n_questions and t:
            short[n] = t
    return {"overview": _clean_cites(str(data.get("overview") or ""), n_sources), "why": why,
            "suggestions": sugs[:3], "short": short}


def checkable_text(prose: dict) -> str:
    """The overview and every "why it matters", each under a marker heading, as one text for the check."""
    parts = [f"{MARK}OV\n\n{prose['overview']}"] if prose.get("overview") else []
    parts += [f"{MARK}W{k}\n\n{v}" for k, v in sorted(prose.get("why", {}).items()) if v]
    return "\n\n".join(parts)


def split_checked(text: str) -> dict:
    """The checked text back into its sections (a section the check emptied has lost its heading: missing)."""
    out = {"overview": "", "why": {}}
    for m in re.finditer(rf"(?m)^{re.escape(MARK)}(OV|W\d+)\s*\n(.*?)(?=^{re.escape(MARK)}|\Z)", text or "", re.S):
        body = m.group(2).strip()
        if not body:
            continue
        if m.group(1) == "OV":
            out["overview"] = body
        else:
            out["why"][int(m.group(1)[1:])] = body
    return out


# a sentence that only makes sense after an earlier one (the check may remove the sentence before it)
_REFERS_BACK = re.compile(r"^(?:They|Their|Them|It|Its|Also|Moreover|Furthermore|However|Additionally|In addition|"
                          r"Similarly|Likewise|Besides|The latter|The former)\b")


def standalone_start(text: str) -> str:
    """`text` from its first sentence that stands on its own: leading sentences that refer back are dropped."""
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", (text or "").strip())
    while sentences and _REFERS_BACK.match(sentences[0]):
        sentences.pop(0)
    return " ".join(sentences).strip()


# the question framing around a report's topic, so the heading shows the topic, not the whole question
_TOPIC_FRAME_RE = re.compile(
    r"^(?:please\s+)?(?:can you\s+)?(?:find|list|show(?: me)?|give me|what(?: are)?|which(?: are)?|are there)?\s*"
    r"(?:the\s+|some\s+|any\s+)?(?:(?:potential|main|key|open|current|important|remaining)\s+)*"
    r"(?:research\s+)?(?:gaps|questions|problems|issues)(?:\s+are there)?"
    r"(?:\s+(?:in|on|about|for|across|regarding|of)\b)?\s*", re.I)
_TOPIC_FRAME_DE_RE = re.compile(r"^welche\s+(?:offenen\s+)?(?:forschungs)?(?:fragen|lücken|probleme)\s+"
                                r"(?:gibt es\s+)?(?:zu|in|bei|über|zum|zur)\s+", re.I)
_LIBRARY_RE = re.compile(r"^(?:across\s+|in\s+)?(?:the\s+)?(?:whole\s+)?(?:library|corpus|documents?)$", re.I)


def display_topic(topic: str | None) -> str | None:
    """The topic as the report's heading shows it: the question framing removed ("What are the open research
    questions in X?" → "X"); None for the whole library; a question that cannot
    be reduced is shown in quotes. The report's search uses the full question."""
    t = (topic or "").strip().rstrip("?.! ").strip()
    if not t:
        return None
    # also the frame itself without a question word (the router's rewrite "open research questions in polymer
    # informatics")
    if re.match(r"^(?:find|list|show|give|what|which|are|can|please|welche)\b", t, re.I) or re.match(
            r"^(?:(?:potential|main|key|open|current|important|remaining)\s+)*(?:research\s+)?"
            r"(?:gaps|questions|problems|issues)\b", t, re.I):
        reduced = _TOPIC_FRAME_DE_RE.sub("", _TOPIC_FRAME_RE.sub("", t)).strip()
        reduced = re.sub(r"\s+(?:across|in)\s+(?:the\s+)?(?:whole\s+)?library$", "", reduced, flags=re.I).strip()
        if not reduced or _LIBRARY_RE.match(reduced) or re.fullmatch(r"(?:potential\s+)?(?:research\s+)?gaps", reduced,
                                                                     re.I):
            return None
        if reduced != t:
            return reduced
    if _LIBRARY_RE.match(t):
        return None
    if len(t.split()) > 6 or (topic or "").strip().endswith("?"):
        return f'"{(topic or "").strip()}"'
    return t


def opening_line(full: list[dict], *, more: int, topic: str | None, document: str | None = None,
                 start: int = 1, total: int | None = None) -> str:
    """The report's first line, written by the code from the ledger's facts, so every report opens the same way whatever
    survives the overview check; the checked overview follows it. `document`: the report of one document's questions
    (its name as the picker shows it)."""
    if not full:
        return ""
    n = len(full)
    count = f"**{n} open research question{'s' if n != 1 else ''}**"
    paged = total is not None and (start > 1 or total > n)     # a page of a longer list
    if paged:
        count = (f"**Questions {start}–{start + n - 1} of {total}** open research questions" if start > 1 else
                 f"**{n} of {total} open research questions**")
    if document:
        shared = sum(1 for g in full if len(g.get("documents") or []) > 1)
        also = ("" if not shared else ", also by other documents" if n == 1
                else ", both also by other documents" if shared == n == 2
                else ", all also by other documents" if shared == n else f", {shared} of them also by other documents")
        head = f'{count} raised by "{document}"{also}.'
    else:
        docs = {d for g in full for d in (g.get("documents") or [])}
        where = f"on {topic}" if topic else "in the library"
        head = f"{count} {where}, raised by {len(docs)} document{'s' if len(docs) != 1 else ''}."
    line = (f"{head} Each was checked against the other documents in the library"
            + (" and against published papers." if any(g.get("web_status") for g in full) else "."))
    return line + (" Further ones are listed after them." if more and not paged else "")


def _and(ns) -> str:
    ns = [str(n) for n in ns]
    return ns[0] if len(ns) == 1 else ", ".join(ns[:-1]) + " and " + ns[-1]


def _collapsed(kind: str, title: str, lines: list[str]) -> list[str]:
    """A collapsed block (OpenWebUI's Collapsible; blank lines around it, or renderers merge neighbours)."""
    return ["", f'<details type="{kind}">', f"<summary>{title}</summary>", "", *lines, "", "</details>", ""]


def papers_how_line(full_text: bool) -> str:
    """The report's method line on published papers."""
    read = ("titles, abstracts and, where the offline index holds a paper's full text, its best-matching passages"
            if full_text else "titles and abstracts only")
    return f"- **Published papers:** searched in Semantic Scholar, OpenAlex and an offline index, and judged from {read}."


def assemble(sel: dict, sources: list[dict], refs: dict, prose: dict, *, names: dict, topic: str | None,
             n_documents: int, evidence: dict, checked: bool, pending: list | None = None,
             document: str | None = None, by_document: list | None = None) -> str:
    """The report in Markdown. `prose`: the checked overview and "why it matters" texts and the suggestions.
    `document`: one document's questions (its name as the picker shows it); the heading still starts "## Open research
    questions", which the proposal route and the proposal buttons look for."""
    titles = {k + 1: (s.get("title") or s["source"], s.get("year")) for k, s in enumerate(sources)
              if s["kind"] == "paper"}
    # a page of the report: sel["start"] is its first question's number; prose stays numbered 1… per page
    start, total = sel.get("start") or 1, sel.get("total")
    end = start + len(sel["full"]) - 1
    topic = display_topic(topic)                # the heading and first line show the topic, not the question
    head = (f'## Open research questions in "{document}"' if document else
            f"## Open research questions on {topic}" if topic else "## Open research questions in the library")
    if start > 1:
        head += f" (questions {start}–{end})"
    out = [head, ""]
    if pending:                                 # uploaded, not yet through the updater
        shown = ", ".join(f'"{names.get(p, p)}"' for p in pending[:3]) + (" and more" if len(pending) > 3 else "")
        n = len(pending)
        out += [f"_{'Papers' if n > 1 else 'Paper'} still processing, try again later: {shown}._", ""]
    if not checked:                             # unchecked prose is never shown (trust chain §11.6)
        prose = {**prose, "overview": "", "why": {}}
    opening = opening_line(sel["full"], more=len(sel.get("more") or []), topic=topic, document=document,
                           start=start, total=total)
    if document and not sel["full"] and sel.get("addressed"):
        opening = (f'The open questions "{document}" names may already be answered by other documents in the library: '
                   f"see below.")
    if opening:
        out += [opening, ""]
    overview = standalone_start(prose.get("overview") or "")
    if overview:
        out += [overview, ""]
    if sel["full"]:                             # the table, then one block per question
        out += ["| # | Question | Library | Literature |", "|---|---|---|---|"]
        for k, g in enumerate(sel["full"], start=1):
            out.append(f"| {start + k - 1} | {_cell(short_title(g['statement'], (prose.get('short') or {}).get(k)))} | "
                       f"{TABLE_LIB.get(g.get('library_status'), 'not checked yet')} | "
                       f"{TABLE_WEB.get(g.get('web_status'), 'not checked yet')} |")
        out.append("")
    for k, g in enumerate(sel["full"], start=1):
        d = start + k - 1
        out += ["", '<details type="memento_evidence">', f"<summary>Evidence for question {d}</summary>", "",
                render_question(d, g, refs[g["id"]], why=prose.get("why", {}).get(k, ""), names=names,
                                sources=sources), "", "</details>", ""]
    if sel["full"]:
        two = f" or **questions {start} and {start + 1}**" if len(sel["full"]) > 1 else ""
        out += ["_To draft a research proposal, click **Draft a proposal on this question** under a question, or reply "
                f"**draft a proposal on question {start}**{two}._", ""]
    if total and end < total:                   # more after this page
        nxt = min(len(sel["full"]), total - end)
        out += [f"_For the next {nxt}, reply **more research gaps**. To narrow it down, use **Draft a proposal** on the "
                "start screen with a topic or a document._", ""]
    # the rest as tables in collapsed blocks (each can be long), counts in the titles, none when empty; at most
    # TABLE_ROWS rows; the whole library lists documents instead of every further question
    def count(shown, of):
        return f"{shown} of {of}" if of > shown else f"{of}"
    if by_document is not None:
        if by_document:
            rows = by_document[:TABLE_ROWS]
            out += _collapsed("memento_bydoc", f"Open questions by document ({count(len(rows), len(by_document))})",
                              ["_Click **Show its questions** to see one document's open questions and draft a "
                               "proposal on one._", "", "| Document | Open questions | Most open one |", "|---|---|---|",
                               *[f"| {_cell(r['name'])} | {r['count']} | {_cell(short_title(r['top'], None))} |"
                                 for r in rows]])
    elif sel.get("more"):
        more = sel["more"][:TABLE_ROWS]
        out += _collapsed("memento_more", f"Further open questions ({count(len(more), len(sel['more']))})",
                          ["| Question | Library | Literature |", "|---|---|---|",
                           *[f"| {_cell(g['statement'])} | {TABLE_LIB.get(g.get('library_status'), 'not checked yet')} | "
                             f"{TABLE_WEB.get(g.get('web_status'), 'not checked yet')} |" for g in more]])
    if sel.get("addressed") and start == 1:
        rows = []
        for g in sel["addressed"][:TABLE_ROWS]:
            cited = [(names.get(sources[n - 1]["source"], sources[n - 1]["source"]), n)
                     for n, _ in (refs.get(g["id"]) or {}).get("library") or []]
            if cited:                           # the answering pages, cited, grouped by document
                per: dict = {}
                for t, n in cited:
                    per.setdefault(t, []).append(n)
                by = "; ".join(f"{_one_line(t)} " + " ".join(f"[{n}]" for n in ns) for t, ns in per.items())
            else:                               # its passage gone or not loaded: the document named
                by = "; ".join(dict.fromkeys(_one_line(names.get(e["source"], e["source"]))
                                             for e in evidence.get(g["id"]) or []
                                             if e.get("origin") == "library" and e.get("relation") == "addresses"))
            rows.append(f"| {_cell(g['statement'])} | {_cell(by) if by else 'not named'} |")
        out += _collapsed("memento_answered",
                          f"Possibly already answered in the library ({count(len(rows), len(sel['addressed']))})",
                          ["| Question | Answered by |", "|---|---|", *rows])
    if prose.get("suggestions"):
        out += _collapsed("memento_ideas", f"Directions suggested by the AI ({len(prose['suggestions'])})",
                          ["_Suggested by the AI, not from the documents. Please verify._", "",
                           "| Direction | Builds on |", "|---|---|",
                           *[f"| {_cell(s['text'])} | question{'s' if len(s['builds_on']) > 1 else ''} "
                             f"{_and([start + n - 1 for n in s['builds_on']])} |" for s in prose["suggestions"]]])
    # checked: True (every statement judged), "partial" (some could not be judged: kept, and said), "no_claims" (no
    # statement was found to check: kept, and said), False (not checked: the prose was left out above)
    partly = ", and some statements could not be checked this time" if checked == "partial" else ""
    written = ('**Written by the AI:** the overview, "why it matters" and the suggested directions. The overview and '
               '"why it matters" were not checked against the sources: no statements to check were found in them. '
               "Statuses and sources come from the checks, not from the AI." if checked == "no_claims" else
               '**Written by the AI:** the overview, "why it matters" and the suggested directions. The overview and '
               f'"why it matters" were checked against the sources; unsupported statements were removed{partly}. '
               "Statuses and sources come from the checks, not from the AI." if checked else
               '**Written by the AI:** the overview, "why it matters" and the suggested directions. The overview and '
               '"why it matters" were left out this time because they could not be checked against the sources. '
               "Statuses and sources come from the checks, not from the AI.")
    out += _collapsed("memento_how", "How this report was made", [
        "- **Questions:** what the documents themselves name as limitations or future work, each checked against the "
        "passage it comes from.",
        "- **In the library:** each question was searched across all passages of all other documents in the library; "
        "the closest passages were read to judge whether they answer it.",
        papers_how_line(reads_full_text(evidence)),
        f"- {written}",
        "- Documents added after the checks are not included yet. A domain expert should judge the questions before "
        "relying on them."])
    return "\n".join(out).strip() + "\n"


# ── loading ──

def topic_relevance(topic: str, gaps: list[dict], *, nearest, rerank_fn, k: int = 50, floor: float = 0.0) -> dict:
    """{gap id: reranker score} of the questions that fit a topic: the k nearest in the gap collection
    (`nearest(topic, k)` → gap ids), then the reranker reads the statements; scores below `floor` are left out."""
    by_id = {g["id"]: g for g in gaps}
    cand = [by_id[i] for i in nearest(topic, k) if i in by_id]
    if not cand:
        return {}
    hits = [{"text": g["statement"], "type": "text", "gap_id": g["id"]} for g in cand]
    ranked = rerank_fn(topic, hits, len(hits))
    return {h["gap_id"]: float(h.get("rerank_score") or 0) for h in ranked if (h.get("rerank_score") or 0) >= floor}


def needed_points(gaps: list[dict], evidence: dict, answered: list[dict] = ()) -> list[str]:
    """The Qdrant point ids a report's sources come from (stating passages, library evidence; for `answered`
    questions only the passages that answer them)."""
    ids = [str(m.get("point_id")) for g in gaps for m in g.get("mentions") or []]
    ids += [str(e.get("point_id")) for g in gaps for e in evidence.get(g["id"]) or [] if e.get("origin") == "library"]
    ids += [str(e.get("point_id")) for g in answered for e in evidence.get(g["id"]) or []
            if e.get("origin") == "library" and e.get("relation") == "addresses"]
    return list(dict.fromkeys(i for i in ids if i and i != "None"))


def load_passages(client, collection: str, point_ids: list[str]) -> dict:
    """{point id: hit} for the ids that still exist (a deleted document's passages are simply missing)."""
    from memento.retrieval.search import point_to_hit
    ids = [int(i) if i.isdigit() else i for i in point_ids]
    out = {}
    for k in range(0, len(ids), 256):
        for p in client.retrieve(collection, ids=ids[k:k + 256], with_payload=True):
            out[str(p.id)] = point_to_hit(p.payload or {}, 0.0, p.id)
    return out
