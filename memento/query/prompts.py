"""The answer model's instructions shared by the flows: the answer prompt (the domain's, with the plain-formula rule),
the element, suggestion and extension addenda, the query rewrite, and images as data URLs.
"""
import base64
import re

from memento.query import request, texts


def element_addendum(element: str) -> str:
    """A table asked for: the table itself, then a short explanation."""
    if element == "table":
        return ("\nThe question asks for a table: show the table, or its relevant rows and columns, as a Markdown table "
                "taken from the sources, with its citation, then explain it briefly.")
    return ""


def figures_for(*, element: str, image_intent: bool, query_intent: str = "") -> tuple[bool, bool]:
    """(figures under the answer at all, treated as a figure request). Not under an equation or table question, nor
    under a list or count answer; "which figures show …" (element figure without a figure search) shows the figures it
    cites like a figure request."""
    if element in ("table", "equation") and not image_intent:
        return False, False
    if query_intent == "set_query" and not image_intent and element != "figure":
        return False, False
    return True, image_intent or element == "figure"


SUGGEST_ADDENDUM = """

The user also asks for ideas beyond what the sources state (to think further, speculate or extrapolate). First answer \
from the sources as usual, with [N] citations. Then add a section headed "### Possible extensions (suggested by the \
AI)" (in a German answer "### Mögliche Erweiterungen (von der KI vorgeschlagen)") with your own ideas: no [N] citations \
there, and never present them as stated in the documents. Keep any length or format the user asked for."""


def suggest_addendum(suggest: bool, query_intent: str) -> str:
    """SUGGEST_ADDENDUM when the router says the question asks for ideas beyond the documents; a list stays a list, and
    a research-gap answer has its own "Possible extensions" block."""
    return SUGGEST_ADDENDUM if suggest and query_intent not in ("set_query", "gap_query", "summary") else ""


_EXTENSIONS_HEAD_RE = re.compile(r"(?m)^###\s*(?:Possible extensions \(suggested by the AI\)|Mögliche Erweiterungen "
                                 r"\(von der KI vorgeschlagen\))")


_HEADING_LINE_RE = re.compile(r"(?m)^(#{1,6})[ \t]+\S.*$")


def split_extensions(text: str) -> tuple:
    """(the text the Thorough check judges, [(the heading line the section stood before, or None at the end, the
    section)]). Each "Possible extensions" section is the AI's own ideas by design, so it is not judged; it ends at the
    next heading of level ≤ 3, so a later section is judged. Several sections stay several."""
    text = text or ""
    heads = list(_EXTENSIONS_HEAD_RE.finditer(text))
    if not heads:
        return text, []
    parts, exts, pos = [], [], 0
    for m in heads:
        if m.start() < pos:
            continue
        parts.append(text[pos:m.start()])
        nxt = next((h for h in _HEADING_LINE_RE.finditer(text, m.end()) if len(h.group(1)) <= 3), None)
        end = nxt.start() if nxt else len(text)
        anchor = nxt.group(0).strip() if nxt and not _EXTENSIONS_HEAD_RE.match(nxt.group(0)) else None
        exts.append((anchor, text[m.start():end].strip()))
        pos = end
    parts.append(text[pos:])
    return "\n\n".join(p.strip() for p in parts if p.strip()) + "\n", exts


def with_extensions(final: str, extensions: list) -> str:
    """The checked answer with its extensions sections back: one that stood before a heading goes back before it, the
    others after the check's note; each says it was not checked."""
    if not extensions:
        return final
    out, tail = final.rstrip(), []
    for anchor, section in extensions:
        head, _, rest = section.partition("\n")
        labelled = f"{head}\n{texts.EXTENSIONS_UNCHECKED_NOTE}" + (f"\n\n{rest.strip()}" if rest.strip() else "")
        m = re.search(rf"(?m)^{re.escape(anchor)}[ \t]*$", out) if anchor else None
        if m:
            out = f"{out[:m.start()].rstrip()}\n\n{labelled}\n\n{out[m.start():]}"
        else:
            tail.append(labelled)
    return out + "".join(f"\n\n{t}" for t in tail)


# The answer instruction lives in config/domains/<domain>.yaml (one copy only); a domain without one gets generic.yaml's
# (_answer_prompt). This line is the last resort if no config file can be read.
SYSTEM_PROMPT = ("Answer using ONLY the numbered sources below and cite every factual statement with its number [N]. "
                 "If the sources do not answer the question, say so; do not guess.")


# Open WebUI renders $…$ as maths only with a space before the opening $ and none after the closing one, so paper-style
# LaTeX pairs the wrong $ signs and shows raw code: formulas are asked for as plain text, for every domain.
PLAIN_FORMULAS = ("Write chemical formulas, units and symbols as plain text with Unicode characters (e.g. HfO₂, "
                  "μC cm⁻², 200 °C, Tg), never in LaTeX or with $ signs.")


def answer_prompt(dom: dict | None) -> str:
    """The domain's answer instruction, else config/domains/generic.yaml's, else the one-line last resort; plus the
    plain-text formula rule."""
    if (dom or {}).get("system_prompt"):
        base = dom["system_prompt"]
    else:
        try:
            base = request.domain("generic").get("system_prompt") or SYSTEM_PROMPT
        except Exception:
            base = SYSTEM_PROMPT
    return base.rstrip() + " " + PLAIN_FORMULAS


REWRITE_SYSTEM_PROMPT = """\
You rewrite a user's latest chat message into a single, self-contained search \
query for a document-retrieval system. Use the prior conversation turns ONLY \
to resolve references in the latest message — pronouns ("it", "that"), \
ellipsis, or an implicit subject/paper.
Rules:
- Output ONLY the rewritten query — no preamble, no quotes, no explanation.
- If the latest message is already self-contained, return it unchanged.
- Never answer the question; only rewrite it as a search query.
- Keep the rewritten query in the same language as the latest message."""


async def rewrite_query(llm, model: str, query: str, history: list) -> str:
    """Resolve pronouns and ellipsis in `query` using `history` ({"role", "content"} dicts) with one cheap,
    non-streaming LLM call. Returns `query` unchanged (no call) without history or on any failure: the rewrite only
    helps retrieval."""
    if not history:
        return query
    convo = "\n".join(f"{h['role']}: {h['content']}" for h in history)
    try:
        resp = await llm.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
                {"role": "user", "content": f"Conversation so far:\n{convo}\n\nLatest message: {query}"},
            ],
            temperature=0.0,
            max_tokens=128,
            # see Task-handler comment in pipe() re: chat_template_kwargs nesting
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        rewritten = (resp.choices[0].message.content or "").strip()
        return rewritten or query
    except Exception:
        return query


def verified_image_url(path: str) -> str | None:
    """A figure as a data URL the model and the browser can both read, or None (then the caption alone is used). SVG is
    rasterized and the image type set correctly by memento/common/model_servers.image_data_url; a mislabelled image
    fails the whole request."""
    try:
        import io
        from PIL import Image
        from memento.common.model_servers import image_data_url
        url = image_data_url(path)
        with Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))) as im:
            im.verify()
        return url
    except Exception:
        return None


# the length by kind of document
SUMMARY_ADDENDUM = """

The numbered sources below are the COMPLETE document, passage by passage in reading order ([N] = one passage). \
Write a summary of the whole document: its aim, the methods or approach, the main results with the key numbers, the \
conclusions, and limitations the authors state. Unless the user asks for another length or format, use short \
headed sections and fit the length to the document: about 300–500 words for a paper, a talk or a short report, and \
about 700–1,000 words for a thesis or a dissertation, with the results of each of its main parts. Cite the passages \
with [N] as usual."""


# a long document's summary from a budgeted selection of its main parts
SUMMARY_PARTIAL_ADDENDUM = """

The numbered sources below are the main parts of a long document, passage by passage in reading order ([N] = one \
passage): its abstract, conclusions, introduction, the start of each section and passages spread across it. Write a \
summary of the whole document from them: its aim, the methods or approach, the main results with the key numbers, \
the conclusions, and limitations the authors state. Do not mention that parts were selected. Unless the user asks for \
another length or format, use short headed sections and fit the length to the document: about 300–500 words for a \
paper, a talk or a short report, and about 700–1,000 words for a thesis or a dissertation, with the results of each \
of its main parts. Cite the passages with [N] as usual."""


# what one summary source costs besides its text: its label "[12] page 3, Section: Results › 3.2 …" and separator
SUMMARY_LABEL_CHARS = 60
