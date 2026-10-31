"""A draft research proposal from questions of the open-research-questions report: after a report, "draft a proposal on
question 2" (or "questions 2 and 5") builds on those ledger questions and all their stored evidence, as a text reply and
a PDF. Code writes the facts (the questions word for word, what the library and the literature have, the references);
the chosen model writes the title, motivation and background (cited, then checked against the sources; unsupported
statements removed, unchecked ones never shown) and the proposed work (objectives, approach, expected outcomes, risks),
which is labelled as the model's suggestion and cites questions, not sources."""
from __future__ import annotations

import re
from pathlib import Path

from memento.research_gaps import fulltext as ft
from memento.research_gaps.report import _clean_cites, _one_line

MARK = "#### §"
_REQ = re.compile(r"\b(?:draft|write|make|create)?\b.*?\bproposal\b.*?\b(?:on|for|about|from)\s+(?:the\s+)?"
                  r"(?:questions?\s+)?(\d+(?:\s*(?:,|and|&|\+)\s*\d+)*)\b", re.I | re.S)


def proposal_request(text: str) -> list[int] | None:
    """The question numbers a follow-up asks a proposal for ("draft a proposal on question 2", "… questions 1 and
    3"), sorted; None when it is no such request."""
    m = _REQ.search(text or "")
    if not m:
        return None
    nums = sorted({int(x) for x in re.findall(r"\d+", m.group(1))})
    return nums or None


def report_questions(report_md: str) -> dict:
    """{number: question} of a report's questions shown in full ("### 2. How …?")."""
    return {int(n): q.strip() for n, q in re.findall(r"(?m)^### (\d+)\. (.+)$", report_md or "")}


PROPOSAL_SYSTEM = """\
You draft a RESEARCH PROPOSAL that builds on open research questions found in the documents of a research library. \
You get numbered sources (passages of the library's documents, and abstracts of published papers), the questions \
(Q1, Q2, …) and the facts established about each. Write in English, for researchers of the field.
- "title": a short working title.
- "summary": two or three plain sentences for the reader: the question, and how the proposed work would answer it \
(no citations).
- "motivation": why the questions matter, in the sources' own terms (3–5 sentences), every statement cited as [N].
- "background": what is already known and what the library and the literature already offer for these questions \
(4–8 sentences), every statement cited as [N]. Use only what the sources say.
- "objectives": 2–4 objectives of the proposed work.
- "approach": how the work could be done (methods, materials, measurements or simulations), referring to the \
questions as Q1, Q2.
- "outcomes": the expected results and their use.
- "risks": the main risks and how to handle them.
Objectives, approach, outcomes and risks are your suggestions: no source citations there, no invented results. \
Cite each source in its own brackets, as [1][4], never [1, 4]. Write about the science, never about "the provided \
text" or "the sources".
Output ONLY JSON: {"title": "...", "summary": "...", "motivation": "...", "background": "...", "objectives": ["..."], \
"approach": "...", "outcomes": "...", "risks": "..."}"""


def proposal_messages(questions: list[str], facts: list[str], sources: list[dict], max_chars: int = 1200) -> list:
    def block(k, s):
        text = " ".join(str(s.get("text") or "").split())[:max_chars]
        if s["kind"] == "paper":
            read = "abstract and excerpts" if ft.read_in_full(text) else "abstract"
            return f"[{k}] {s['source']} ({s.get('year') or 'n.d.'}), {read}\n{text}"
        return f"[{k}] {s['source']}, page {int(s.get('page') or 0) + 1}\n{text}"
    src = "\n\n".join(block(k + 1, s) for k, s in enumerate(sources))
    qs = "\n\n".join(f"Q{k}. {q}\nFacts:\n{f}" for k, (q, f) in enumerate(zip(questions, facts), start=1))
    return [{"role": "system", "content": PROPOSAL_SYSTEM},
            {"role": "user", "content": f"Sources:\n{src}\n\nQuestions:\n{qs}"}]


def _no_cites(text) -> str:
    t = re.sub(r"\s*\[\d+(?:\s*[,;\-–]\s*\d+)*\]", "", str(text or ""))
    return re.sub(r"\s+([.,;:])", r"\1", re.sub(r"[ \t]{2,}", " ", t)).strip()


def parse_proposal(raw: str, *, n_sources: int) -> dict:
    """The model's JSON: citations to sources never sent removed from motivation and background; no source citations
    in the title and the proposed work. Raises ValueError when unreadable."""
    from memento.agentic_validation.jsonfix import loads_lenient
    try:
        data = loads_lenient(raw[raw.index("{"):raw.rindex("}") + 1])
    except Exception as e:
        raise ValueError(f"unreadable proposal: {e}") from e
    if not isinstance(data, dict):
        raise ValueError("proposal is not an object")
    objectives = data.get("objectives") or []
    if isinstance(objectives, str):
        objectives = [objectives]
    sentences = re.findall(r"[^.!?]+[.!?]+", _no_cites(data.get("summary")))
    return {"title": _no_cites(data.get("title")) or "Untitled",
            "summary": " ".join(x.strip() for x in sentences[:3]),
            "motivation": _clean_cites(str(data.get("motivation") or ""), n_sources),
            "background": _clean_cites(str(data.get("background") or ""), n_sources),
            "objectives": [o for o in (_no_cites(x) for x in objectives) if o][:5],
            "approach": _no_cites(data.get("approach")), "outcomes": _no_cites(data.get("outcomes")),
            "risks": _no_cites(data.get("risks"))}


def checkable_text(p: dict) -> str:
    return "\n\n".join(f"{MARK}{k.upper()}\n\n{p[k]}" for k in ("motivation", "background") if p.get(k))


def split_checked(text: str) -> dict:
    out = {"motivation": "", "background": ""}
    for m in re.finditer(rf"(?m)^{re.escape(MARK)}(MOTIVATION|BACKGROUND)\s*\n(.*?)(?=^{re.escape(MARK)}|\Z)",
                         text or "", re.S):
        out[m.group(1).lower()] = m.group(2).strip()
    return out


def references(sources: list[dict], names: dict) -> list[str]:
    out = []
    for k, s in enumerate(sources, start=1):
        if s["kind"] == "paper":
            out.append(f"[{k}] {_one_line(s.get('title') or s['source'])} ({s.get('year') or 'n.d.'})."
                       + (f" {s['url']}" if s.get("url") else ""))
        else:
            out.append(f"[{k}] {names.get(s['source'], s['source'])}, p. {int(s.get('page') or 0) + 1}")
    return out


PROPOSED_LABEL = "_Suggested by the AI, not from the documents. Please verify._"


def _block(kind: str, title: str, lines: list[str]) -> list[str]:
    return ["", f'<details type="{kind}">', f"<summary>{title}</summary>", "", *lines, "", "</details>", ""]


def _collapsed(prose: dict, *, questions: list[str], facts: list[str], sources: list[dict], names: dict,
               checked: bool, link: str) -> str:
    """The chat reply: title, summary and the PDF link visible; every section folded; the proposed work set apart as the
    AI's suggestion."""
    out = [f"## Draft research proposal: {prose.get('title') or 'Untitled'}", ""]
    if prose.get("summary"):
        out += [prose["summary"], ""]
    if link.strip():
        out += [link.strip(), ""]
    rq = []
    for k, (q, f) in enumerate(zip(questions, facts + [""] * len(questions)), start=1):
        rq += [f"**Q{k}. {q}**", ""] + ([f, ""] if f else [])
    out += _block("memento_proposal", "Research question" + ("s" if len(questions) > 1 else "") + " and evidence", rq)
    if checked and prose.get("motivation"):
        out += _block("memento_proposal", "Motivation", [prose["motivation"]])
    if checked and prose.get("background"):
        out += _block("memento_proposal", "Background", [prose["background"]])
    work = [PROPOSED_LABEL, ""]
    if prose.get("objectives"):
        work += ["**Objectives**", "", *[f"{k}. {o}" for k, o in enumerate(prose["objectives"], start=1)], ""]
    for key, head in (("approach", "Approach"), ("outcomes", "Expected outcomes"), ("risks", "Risks")):
        if prose.get(key):
            work += [f"**{head}.** {prose[key]}", ""]
    out += _block("memento_proposed", "Proposed work (suggested by the AI)", work)
    refs = references(sources, names)
    out += _block("memento_proposal", f"References ({len(refs)})", [f"- {r}" for r in refs])
    out += ["---", "", _how_made(checked, sources)]
    return "\n".join(out).strip() + "\n"


def _how_made(checked, sources=()) -> str:
    """`checked`: True, "partial" (some statements could not be judged: kept and said), "no_claims" (no statement was
    found to check: kept and said) or False (left out)."""
    partly = ", and some statements could not be checked this time" if checked == "partial" else ""
    check = ("Motivation and background were not checked against the sources: no statements to check were found in them."
             if checked == "no_claims" else
             f"Motivation and background were checked against the sources; unsupported statements were removed{partly}."
             if checked else "Motivation and background were left out because they could not be checked against "
             "the sources this time.")
    read = ("their abstracts and, where the offline index holds them, passages of their full text"
            if any(ft.read_in_full(s.get("text")) for s in sources) else "their abstracts")
    return (f"_How this draft was made: the research questions, what the library and the literature "
            f"already have, and the references come from Memento's checks of the documents and of published papers "
            f"(papers judged from {read}). {check} The proposed work is the model's suggestion. A domain "
            f"expert should judge the draft before it is used._")


def assemble(prose: dict, *, questions: list[str], facts: list[str], sources: list[dict], names: dict,
             checked: bool, collapsed: bool = False, link: str = "") -> str:
    """The proposal in Markdown: `collapsed` for the chat reply, plain for the PDF."""
    if collapsed:
        return _collapsed(prose, questions=questions, facts=facts, sources=sources, names=names, checked=checked,
                          link=link)
    out = [f"## Draft research proposal: {prose.get('title') or 'Untitled'}", ""]
    if prose.get("summary"):
        out += [prose["summary"], ""]
    out += ["### Research question" + ("s" if len(questions) > 1 else ""), ""]
    for k, (q, f) in enumerate(zip(questions, facts + [""] * len(questions)), start=1):
        out += [f"**Q{k}. {q}**", ""] + ([f, ""] if f else [])
    if checked:
        if prose.get("motivation"):
            out += ["### Motivation", "", prose["motivation"], ""]
        if prose.get("background"):
            out += ["### Background", "", prose["background"], ""]
    out += ["### Proposed work", "", PROPOSED_LABEL, ""]
    if prose.get("objectives"):
        out += ["**Objectives**", "", *[f"{k}. {o}" for k, o in enumerate(prose["objectives"], start=1)], ""]
    for key, head in (("approach", "Approach"), ("outcomes", "Expected outcomes"), ("risks", "Risks")):
        if prose.get(key):
            out += [f"**{head}.** {prose[key]}", ""]
    out += ["### References", "", *[f"- {r}" for r in references(sources, names)], ""]
    out += ["---", "", _how_made(checked, sources)]
    return "\n".join(out).strip() + "\n"


_CSS = ("body { font-family: sans-serif; font-size: 10.5pt; line-height: 1.4; } h2 { font-size: 16pt; } "
        "h3 { font-size: 12.5pt; margin-top: 14pt; } li { margin-bottom: 3pt; } em { color: #555; }")


def to_html(markdown_text: str) -> str:
    """Markdown → HTML for the PDF; the chat's 2-space sub-items made 4-space, as Python-Markdown nests them."""
    import markdown
    text = re.sub(r"(?m)^  - ", "    - ", markdown_text)
    return markdown.markdown(text, extensions=["sane_lists"])


def write_pdf(markdown_text: str, path) -> Path:
    """The proposal as an A4 PDF (PyMuPDF's HTML layout; no other dependency)."""
    import fitz
    from memento.common.pdf import PYMUPDF
    html = f"<html><body>{to_html(markdown_text)}</body></html>"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with PYMUPDF:
        story = fitz.Story(html=html, user_css=_CSS)
        writer = fitz.DocumentWriter(str(path))
        page, where = fitz.paper_rect("a4"), fitz.paper_rect("a4") + (56, 56, -56, -56)
        more = True
        while more:
            dev = writer.begin_page(page)
            more, _ = story.place(where)
            story.draw(dev)
            writer.end_page()
        writer.close()
    return path
