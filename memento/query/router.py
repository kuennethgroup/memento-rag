"""LLM-based query intent classification and rewrite.

One LLM call rewrites the query (resolving pronouns and ellipsis from the conversation) and classifies it into a routing
intent:
- gap_query: research gaps, limitations, future work
- set_query: a complete set or list of items from the corpus
- corpus_listing: which documents are in the corpus
- image_request: show a specific figure, image or chart
- verbatim_section: a section verbatim (abstract, …)
- proposal: draft something from earlier content
- multi_hop: a factual question needing two linked passages (answered like ranked_retrieval unless a graph arm is on)
- ranked_retrieval: the default, a factual question answered by retrieval
- chitchat: greeting, thanks, "what can you do?"
- general: a message no document could answer (date or time, arithmetic with all numbers given, how to use the app,
rewording pasted text)

Structured JSON output {"intent", "query", "reasoning": "deep" | "light"}; "reasoning" lets the pipe switch thinking off
for simple questions (THINKING_AUTO). Anything uncertain or unparsable falls back to ranked_retrieval with "deep". The
first turn is classified too."""

import json
import re

from pydantic import BaseModel, field_validator

_VALID_INTENTS = {
    "gap_query", "set_query", "corpus_listing",
    "image_request", "verbatim_section", "proposal",
    "multi_hop", "ranked_retrieval", "chitchat", "general", "summary", "rework",
}


def pipe_intent(intent: str) -> str:
    """The intent the production query pipe acts on. `multi_hop` exists for RQ2 graph routing;
    production answers it exactly like ranked_retrieval (incl. RQ3 validation, which is gated
    on intent == "ranked_retrieval")."""
    return "ranked_retrieval" if intent == "multi_hop" else intent


# Intents that always get the model's full reasoning, whatever the router says about the question.
_DEEP_INTENTS = {"gap_query", "proposal", "multi_hop", "set_query"}
# "doc_kind": the router's word → one of these kinds, the most specific first ("PhD thesis" is a dissertation)
_DOC_KINDS = (("master's thesis", ("master",)), ("bachelor's thesis", ("bachelor",)),
              ("dissertation", ("dissertation", "phd", "doctoral", "doktorarbeit")),
              ("thesis", ("thes", "abschlussarbeit")),
              ("talk", ("talk", "vortr", "presentation", "präsentation", "slides", "folien")),
              ("chapter", ("chapter", "kapitel")))


class RouterResponse(BaseModel):
    intent: str = "ranked_retrieval"
    query: str = ""
    reasoning: str = "deep"
    # the question wants ideas beyond what the documents state (suggest, extend, what next)
    suggest: bool = False
    # the part of a paper asked for, so a table request searches tables: figure | table | equation | none; additive to
    # the intents
    element: str = "none"
    # what the search covers ("which other papers use X?" is about the library even when X is a document's short name):
    # document | documents | library | none; additive to the intents
    scope: str = "none"
    # the person whose documents are asked for, as named, else ""; additive to the intents
    author: str = ""
    # list and count questions: the topic the documents must cover, without the kind of document and the question's
    # other parts (the topic alone ranks the right documents far better), else ""; and the kind of document asked for
    # (dissertation | master's thesis | bachelor's thesis | thesis | talk | chapter, else ""); additive to the intents
    topic: str = ""
    doc_kind: str = ""
    ok: bool = True     # False = the router call failed and these are the defaults (callers may fall back to regex)

    @field_validator("suggest", mode="before")
    @classmethod
    def _coerce_suggest(cls, v) -> bool:
        return v is True

    @field_validator("element", mode="before")
    @classmethod
    def _coerce_element(cls, v) -> str:
        v = str(v).strip().lower() if isinstance(v, str) else ""
        return v if v in ("figure", "table", "equation") else "none"

    @field_validator("scope", mode="before")
    @classmethod
    def _coerce_scope(cls, v) -> str:
        v = str(v).strip().lower() if isinstance(v, str) else ""
        return v if v in ("document", "documents", "library") else "none"

    @field_validator("author", mode="before")
    @classmethod
    def _coerce_author(cls, v) -> str:
        v = " ".join(v.split())[:100] if isinstance(v, str) else ""
        return "" if v.lower() in ("none", "null", "n/a") else v

    @field_validator("topic", mode="before")
    @classmethod
    def _coerce_topic(cls, v) -> str:
        v = " ".join(v.split())[:200] if isinstance(v, str) else ""
        return "" if v.lower() in ("none", "null", "n/a") else v

    @field_validator("doc_kind", mode="before")
    @classmethod
    def _coerce_doc_kind(cls, v) -> str:
        v = " ".join(str(v).lower().replace("’", "'").split()) if isinstance(v, str) else ""
        for kind, words in _DOC_KINDS:
            if any(w in v for w in words):
                return kind
        return ""

    @field_validator("intent")
    @classmethod
    def _coerce_intent(cls, v: str) -> str:
        return v if v in _VALID_INTENTS else "ranked_retrieval"

    @field_validator("reasoning")
    @classmethod
    def _coerce_reasoning(cls, v: str) -> str:
        return "light" if str(v).strip().lower() == "light" else "deep"

    @property
    def needs_thinking(self) -> bool:
        return self.reasoning != "light" or self.intent in _DEEP_INTENTS

ROUTER_SYSTEM_PROMPT = """\
You route user queries in a scientific document retrieval system. \
Given the user's latest message and any prior conversation, output JSON with:
1. "intent": classify into exactly one of these categories:
   - "gap_query": asking about the research gaps, limitations, uncertainties, untested conditions, open questions, \
future work or next steps of a document or of a topic, whether the user asks what they ARE or asks to suggest, \
propose or extend them (e.g. "what future work does the thesis propose?", "suggest future work for this study", \
"how could this work be extended?", "what are the next steps after this study?", "Welche offenen Fragen lässt die \
Arbeit?", "Was könnte man darauf aufbauend untersuchen?"). \
Asking which technologies, applications or uses could follow from a document's findings, or to think its content \
further (e.g. "welche Technologien können daraus entstehen?", "what could this be used for?") is NOT gap_query → \
"ranked_retrieval" with "suggest": true. \
Asking HOW to do, solve, approach or improve a technical task or problem (e.g. "how would you solve the \
fingerprinting of a copolymer?", "how can I predict the glass transition temperature?", "wie würdest du das \
lösen?") is NOT gap_query, also right after a research-gap answer → "ranked_retrieval" (with "suggest": true when \
it asks for ideas). \
A question ABOUT the previous research-gap answer (whether an item belongs under its heading, what a point means or \
where it comes from, e.g. "those in the last section are not limitations, are they?", "why is this listed as future \
work?") is NOT a new gap_query → "ranked_retrieval", and its "query" restates the question with the heading or \
points it refers to, as named in the previous answer
   - "set_query": asking for a complete list/set of items from the corpus \
(e.g. "list all papers about alumina", "which studies use SEM")
   - "corpus_listing": asking to list or count ALL documents in the corpus, with no topic \
(e.g. "which documents are in the library?", "how many papers do you have?"). Asking whether documents \
ON A TOPIC exist ("is there a paper about X?", "do you have anything on Y?") is NOT corpus_listing → "set_query"; \
asking for documents of a kind ("are there any theses?", "which reviews are in the library?", "welche Bücher gibt \
es?") is NOT corpus_listing either → "set_query"
   - "image_request": asking to see/show/display a specific figure, image, \
chart, SEM micrograph, or diagram
   - "verbatim_section": asking ONLY for the text of one named section of a document, to read it as written \
(e.g. "give me the abstract", "show the conclusion of X", "zeig mir die Kurzfassung", "section 3.2 of X"). \
Asking what a section SAYS ABOUT something, or to explain, summarize or evaluate it ("what does the conclusion say \
about fatigue?", "explain the abstract", "summarize the conclusion") is NOT verbatim → "ranked_retrieval". \
If unsure → "ranked_retrieval"
   - "proposal": asking to DRAFT or WRITE a research proposal, or to write something based on information \
already discussed in this conversation (a summary of the answers so far, a comparison). Asking to suggest or \
list ideas for further research is "gap_query", not "proposal"
   - "summary": asking for a summary or overview of ONE whole document \
(e.g. "summarize the polyBERT paper", "what is this thesis about overall?", "fasse das Paper zusammen"). \
A summary of a topic across documents, of one section, or of one aspect of a document ("summarize the conclusion \
of X", "summarize what X says about fatigue", "what are the main results of X?", "welche Methoden nutzt X?") is NOT \
summary → "ranked_retrieval". If unsure → "ranked_retrieval"
   - "multi_hop": a factual question that describes one thing through a fact and asks \
about another fact of it, so the answer needs two passages, often from different documents \
(e.g. "what hardness did the carbide reach that was sintered at the lowest temperature?")
   - "rework": the message ONLY asks to reshape the previous answer, with no new information: shorten or \
summarize it, reformat it (as a table, as bullet points, in three sentences), translate it, simplify or reword it \
(e.g. "summarize that in three bullet points", "make it shorter", "as a table please", "translate it into German", \
"fasse das kürzer zusammen", "als Tabelle"). Asking for MORE or NEW information about it ("tell me more about the \
second one", "explain why", "more detail on the method", "what about X?", "can you expand on it?", "elaborate on \
that", "go into more detail", "can you explain that further?", "tell me more", "erzähl mir mehr", "kannst du das \
ausführlicher erklären?", "genauer bitte", "kannst du das vertiefen?") is NOT rework → "ranked_retrieval", also \
when the previous answer was a summary: its "query" names the topic and asks for detail (e.g. "PolyBERT in more \
detail: methods, results and limitations"), not for a summary. \
Only possible when there is a previous answer; if unsure → "ranked_retrieval". \
An objection or correction to the previous answer's content ("these are their conclusions, how are they future \
work?", "that is not what I asked", "that is wrong, the paper says …") is NOT rework: give it the intent of the \
question it objects to, and as "query" that question restated with the objection's point
   - "ranked_retrieval": any other factual question about content in the papers
   - "chitchat": ONLY a greeting, thanks, small talk or a question about the assistant itself \
("hello", "thanks!", "what can you do?") with no question about document content. \
"Hi, what is the melting point of PTFE?" is a content question → not chitchat
   - "general": a message that NO research document could answer: the current date or time, arithmetic or \
a unit conversion with every number given in the message, how to use this app, or rewording/translating text \
the user pasted in. Scientific or technical concepts, materials, methods or results ("what is DFT?", \
"how does sintering work?") are NOT general → "ranked_retrieval": the library may cover them. If unsure → \
"ranked_retrieval"
2. "query": the latest message rewritten as a standalone search query. \
Resolve pronouns ("it", "that paper", "this"), ellipsis, and implicit \
references using the conversation context. Keep the same language. \
Keep every part of the message: its topic and what it asks about it (e.g. "AI in rubber compounding", not just \
"rubber compounding"). If already self-contained, return unchanged.
3. "reasoning": how much reasoning the answer needs:
   - "light": the answer is ONE fact, value, name or definition stated directly in a source \
(e.g. "what is the melting point of PTFE?"), OR the message only asks about or reworks the \
previous answer: which document/page it came from, translate it, shorten it, reformat it, repeat it
   - "deep": everything else — explanations, comparisons, cause and effect, combining several \
sources, calculations or unit conversions, conflicting findings, evaluations, drafting
   - If in doubt, "deep"
4. "suggest": true when the user wants ideas beyond what the documents state (asks to suggest, propose or extend \
work, what could be done next, ideas for further research), in any wording or language; otherwise false
5. "element": the kind of part of a document the user asks to see, find or check, else "none". The library's \
documents are stored as text passages, tables, equations, figures and images (micrographs, photos, diagrams), charts \
(plots, graphs) and lists. "figure" = a figure, image, micrograph, diagram, chart or plot; "table" = a table (e.g. "is \
there a table of the results?", "show the results table"); "equation" = an equation or formula; "none" = anything \
else, including questions about results or values without asking for a table, figure or equation. This field does \
not change "intent".
6. "scope": what the answer must search. "document" = the user asks about the content of ONE document they name or \
refer to (its title or short name, "the X paper", "this thesis"), e.g. "summarize the polyBERT paper", "what does \
the polyBERT paper report about fingerprints?"; "documents" = about two or more named documents ("compare A with \
B"); "library" = about a method, tool, material or topic across the library: which or how many documents use, \
apply, cite or mention it, whether it has been used in other research or other papers, in any wording or language \
(e.g. "which other papers use polyBERT?", "has polyBERT been used to discover new polymers?", "welche Arbeiten \
nutzen polyBERT?"), also when the method's name is also a document's name; "none" = unsure or no document is named. \
This field does not change "intent".
7. "author": the person whose documents the user asks for (documents, papers or work by, from or of a person, \
written or co-authored by them, "von", or theses and documents they supervised or reviewed: "betreut", \
"begutachtet"), exactly as the user names them (e.g. "are there any documents from Christopher Brueckner?" → \
"Christopher Brueckner", "Gibt es Paper von Alfred Kessler?" → "Alfred Kessler", "Which theses did Holger Rückert \
supervise?" → "Holger Rückert", "Welche Arbeiten hat Christian Böhmer betreut?" → \
"Christian Böhmer"), else "". A \
method, model, equation or effect named after a person (Landau-Khalatnikov, Preisach, Kohn-Sham, Schulz-Zimm, \
Fox-Flory, Gasteiger, Bayesian) is not an author. This field does not change "intent".
8. "topic": when the message asks which documents, or how many, cover something (a list or count): that topic \
alone, the material, method, property or subject the documents must be about, without the kind of document and \
without the question's other parts, in the user's language (e.g. "Which dissertations in the library worked on \
phosphorus-based flame retardants, and which compounds did they use?" → "phosphorus-based flame retardants", "list \
all papers about PEEK" → "PEEK", "Welche Arbeiten beschäftigen sich mit Hafniumoxid?" → "Hafniumoxid"); else "". \
This field does not change "intent".
9. "doc_kind": the kind of document such a message asks for: "dissertation" (also PhD or doctoral thesis), \
"master's thesis" (Masterarbeit), "bachelor's thesis" (Bachelorarbeit), "thesis" (theses of any level, \
Abschlussarbeiten), "talk" (talks, presentations, slides, Vorträge), "chapter" (book chapters); "" for papers, \
documents, works, or when no kind is named. This field does not change "intent".

Rules:
- Output ONLY valid JSON: {"intent": "...", "query": "...", "reasoning": "...", "suggest": false, "element": "none", \
"scope": "none", "author": "", "topic": "", "doc_kind": ""}
- When the user asks to EXPLAIN, DISCUSS, or ANALYZE a technical limitation \
or property (e.g. "explain the limitations of DICY"), that is a factual \
question about content → "ranked_retrieval", NOT "gap_query". \
"gap_query" is reserved for meta-questions about the PAPER's stated gaps.
- Default to "ranked_retrieval" if uncertain
- Never answer the question — only classify and rewrite"""


def _router_user_msg(query: str, history: list[dict]) -> str:
    if history:
        convo = "\n".join(f"{h['role']}: {h['content']}" for h in history)
        return f"Conversation so far:\n{convo}\n\nLatest message: {query}"
    return f"Latest message: {query}"


def _parse(raw: str, query: str) -> RouterResponse:
    raw = (raw or "").strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    parsed = RouterResponse.model_validate(json.loads(raw))
    if not parsed.query:
        parsed.query = query
    return parsed


async def route(llm, model: str, query: str, history: list[dict]) -> RouterResponse:
    """One no-thinking LLM call: intent + standalone query + reasoning depth.
    On any failure: RouterResponse(intent="ranked_retrieval", query=query, reasoning="deep")."""
    try:
        resp = await llm.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": _router_user_msg(query, history)},
            ],
            temperature=0.0,
            max_tokens=256,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return _parse(resp.choices[0].message.content, query)
    except Exception:
        return RouterResponse(query=query, ok=False)


async def classify_and_rewrite(
    llm,
    model: str,
    query: str,
    history: list[dict],
) -> tuple[str, str]:
    """Classify intent and rewrite query in a single LLM call (see route()).
    Returns (intent, rewritten_query); on any failure ("ranked_retrieval", original_query)."""
    r = await route(llm, model, query, history)
    return r.intent, r.query


def classify_and_rewrite_sync(
    client,
    model: str,
    query: str,
    history: list[dict],
) -> tuple[str, str]:
    """Synchronous version for test scripts."""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": _router_user_msg(query, history)},
            ],
            temperature=0.0,
            max_tokens=256,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        r = _parse(resp.choices[0].message.content, query)
        return r.intent, r.query
    except Exception:
        return "ranked_retrieval", query
