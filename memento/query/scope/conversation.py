"""What a message refers to in its conversation: "this paper" and "these" from the previous turn, the documents an
earlier answer cited, the question behind a gap follow-up, and the history as text.
"""
import asyncio
import re

from memento.query import presentation, request, texts
from memento.query.scope import documents


_IDEAS_RE = re.compile(r'\n*<details type="memento_ideas">.*?</details>\n*', re.S)


_IDEAS_OPEN_RE = re.compile(r"\n*###\s*(?:Possible extensions \(suggested by the AI\)|Mögliche Erweiterungen \(von der "
                            r"KI vorgeschlagen\)).*?(?=\n_Answered by|\Z)", re.S)


def without_suggestions(text: str) -> str:
    """An answer as it is stored: without the model's own suggestions (stored separately, action gaps_ext) and without
    the note that they could not be written."""
    out = _IDEAS_RE.sub("\n\n", text or "") if "memento_ideas" in (text or "") else (text or "")
    out = _IDEAS_OPEN_RE.sub("\n\n", out)        # shown unfolded when asked to extend; never stored either
    for note in (texts.SUGGEST_FAILED_NOTE, texts.DE_FIXED[texts.SUGGEST_FAILED_NOTE]):          # English or German
        out = out.replace(note, "\n") if note in out else out
    return out


GAP_QUESTION_EN = "Which limitations, uncertainties, untested conditions and future work do the authors state?"


GAP_QUESTION_DE = ("Welche Einschränkungen, Unsicherheiten, ungetesteten Bedingungen und zukünftigen Arbeiten nennen die "
                   "Autoren?")


def gap_question(question: str) -> str:
    """The question the authors' part of a research-gap answer is written for: the start screen's own question as it is,
    else a neutral one in the question's language. The reader's wording goes to the suggestions only; in the gap prompt
    it made the model list its own ideas."""
    if documents.PICKED.match(question or ""):
        return question
    from memento.common.text_language import question_language
    return GAP_QUESTION_DE if question_language(question) == "de" else GAP_QUESTION_EN


def chunks_from_citations(citations: list) -> list[dict]:
    """The passages behind a stored answer's citation events, in their [N] order (for its suggestions)."""
    out = []
    for ev in citations or []:
        d = (ev or {}).get("data") or {}
        meta = (d.get("metadata") or [{}])[0] if isinstance(d.get("metadata"), list) else (d.get("metadata") or {})
        out.append({"source": re.sub(r"#\d+$", "", str((meta or {}).get("source") or "")), "page": (meta or {}).get("page") or 0,
                    "text": " ".join(d.get("document") or [])})
    return out


def _conversation_document(stored: list | None, library) -> str | None:
    """The document the conversation is about: the one holding at least 60 % of the previous answer's citations, with at
    least two of them; None when no document clearly leads or the citations are unknown. `stored`: the sources Open
    WebUI kept with that answer (stored_sources)."""
    counts = {}
    for s in stored or []:
        meta = (s.get("metadata") or [{}])[0] if isinstance(s.get("metadata"), list) else (s.get("metadata") or {})
        src = re.sub(r"#\d+$", "", str((meta or {}).get("source") or ""))
        if src in library:
            counts[src] = counts.get(src, 0) + 1
    total = sum(counts.values())
    if not total:
        return None
    best, n = max(counts.items(), key=lambda kv: kv[1])
    return best if n >= 2 and n >= 0.6 * total else None


# ── "this paper" after an earlier turn ──
# The document is what the previous answer cited, and the previous request must agree; passage counts never decide.
_CITE_RE = re.compile(r"\[(\d+)\]")


_NOT_ANSWER_RE = re.compile(r"<think>.*?</think>|<details type=\"reasoning\".*?</details>|```.*?```|"
                            r'<details type="memento_sources">.*?</details>|<details type="memento_related">.*?</details>',
                            re.S)


def cited_documents(content: str, stored: list | None, library) -> tuple[set, bool]:
    """(the library documents the answer's [N] markers cite, ambiguous). Each N is mapped through the numbered name
    stored with the citations ("[N] …" in metadata[0].name; source.name carries no number), not by position;
    "X#N" / "X#gapN" passage ids give the document. Markers in reasoning, code and the sources / related blocks do not
    count; a literature paper of a report never counts. A cited number without a mapping, with two documents, or with
    a document no longer in the library makes the result ambiguous."""
    numbers = {int(n) for n in _CITE_RE.findall(_NOT_ANSWER_RE.sub(" ", content or ""))}
    mapping: dict = {}
    for s in stored or []:
        metas = s.get("metadata") if isinstance(s.get("metadata"), list) else [s.get("metadata") or {}]
        for meta in metas or []:
            m = re.match(r"\[(\d+)\]", str((meta or {}).get("name") or ""))
            ident = str((meta or {}).get("source") or "")
            if not m or not ident:
                continue
            doc = None if presentation.LITERATURE_ID_RE.match(ident) else presentation.PASSAGE_SUFFIX_RE.sub("", ident)
            mapping.setdefault(int(m.group(1)), set()).add(doc)
    cited, ambiguous = set(), False
    for n in numbers:
        docs = mapping.get(n)
        if not docs or len(docs) > 1:
            ambiguous = True
            continue
        doc = next(iter(docs))
        if doc is not None and doc in library:
            cited.add(doc)
        elif doc is not None:
            # cited but no longer in the library (removed or being replaced): the answer was about it too, so the
            # remaining one is not "the" document
            ambiguous = True
    return cited, ambiguous


def request_scope(question: str, sources, dom=None, collection: str | None = None) -> tuple[str, str | None]:
    """The previous request's scope: "time" (my latest upload, the most recent: not re-run, the library has changed
    since), "several" (two or more documents named), "one" (one picked, named or described document, with it),
    "unresolved" (it described a document that matches none now, or the lookup failed), "none" (no document)."""
    try:
        if documents.mentions_my_uploads(question) or documents.wants_recent(question):
            return "time", None
        picked = documents.picked_document(question, set(sources), collection)
        if picked:
            return "one", picked[0]
        if len(documents.named_documents(question, list(sources), dom) or []) >= 2:
            return "several", None
        doc = documents.resolve_reference(question, list(sources), dom)
        if doc:
            return "one", doc
        return ("unresolved", None) if documents.document_phrases(question) else ("none", None)
    except Exception:
        return "unresolved", None


def this_document(cited: set, ambiguous: bool, scope: str, previous_doc: str | None) -> str | None:
    """The document "this paper" means, or None (ask): exactly one cited document, and a previous request that named
    that one, or named none, or was time-relative; never after a comparison, an unresolved request or an ambiguous
    citation."""
    if ambiguous or len(cited) != 1:
        return None
    doc = next(iter(cited))
    if scope == "one":
        return doc if previous_doc == doc else None
    return doc if scope in ("time", "none") else None


def these_documents(cited: set, ambiguous: bool) -> list | None:
    """The documents "them" / "these" mean: two or more library documents the previous answer cited, every citation
    mapped; None otherwise (one document: this_document; else ask)."""
    if ambiguous or len(cited) < 2:
        return None
    return sorted(cited)


# a plural reference back to the previous answer's documents ("which of them", "welche davon")
PLURAL_BACKREF_RE = re.compile(r"\b(?:them|these|those|they)\b|\b(?:davon|von ihnen|unter ihnen|denen)\b"
                                r"|\bdiese[rn]?\s+(?:arbeiten|dissertationen|dokumente|masterarbeiten|bachelorarbeiten|"
                                r"vortr[äa]ge|papers?|studien)\b", re.I)


def them_scope(question: str, previous_docs) -> list | None:
    """The documents of "Which of them …?" after an answer about several documents: those documents, when the question's
    own words name none (the router may copy the previous titles into its query, which the name matcher resolves only
    partly); else None."""
    if not previous_docs or not PLURAL_BACKREF_RE.search(question or "") or documents.document_phrases(question):
        return None
    return list(previous_docs)


async def stored_previous_turn(metadata: dict | None):
    """(the previous question, its answer's text, the citations stored with that answer) of this chat's current branch,
    or None (no earlier answer, an API call without a chat, an error)."""
    chat_id = (metadata or {}).get("chat_id")
    if not chat_id or str(chat_id).startswith("local:"):
        return None
    try:
        from open_webui.models.chats import Chats
        from open_webui.utils.misc import get_message_list
        chat = await Chats.get_chat_by_id(chat_id)
        hist = ((chat.chat if chat else {}) or {}).get("history") or {}
        chain = get_message_list(hist.get("messages") or {}, (metadata or {}).get("message_id") or hist.get("currentId"))
        chain = chain[:-1] if chain and chain[-1].get("role") == "assistant" else chain
        for i in range(len(chain) - 1, -1, -1):
            if chain[i].get("role") == "assistant":
                question = next((request.message_text(m.get("content")) for m in reversed(chain[:i]) if m.get("role") == "user"),
                                "")
                return question, request.message_text(chain[i].get("content")), list(chain[i].get("sources") or [])
        return None
    except Exception:
        return None


_ATTACH_WORDS = set("""summarize summarise summary summarise zusammenfassen fasse zusammen zusammenfassung explain explanation
erkläre erklär erklären what what's whats is it this that these the a an of about attached file files paper papers document
documents pdf please bitte can could would you me give tell kannst könntest du mir gib das die der dies diese dieses
worum geht es was ist um den dem tldr tl dr in short brief briefly kurz kurze angehängte angehängten datei dateien""".split())


def attachment_only(question: str) -> bool:
    """A message about an attached file alone ("summarise this", "what is this paper about?"), not a question for the
    library: Library Chat cannot read the file, so the hand-over note is the whole reply."""
    words = re.findall(r"[a-zäöüß']+", (question or "").lower())
    return bool(words) and all(w in _ATTACH_WORDS for w in words)


async def stored_sources(metadata: dict | None) -> list | None:
    """The sources OpenWebUI stored with the last earlier answer in this chat ([] if it had none), or None when the
    chat cannot be read (API calls without a chat, errors)."""
    chat_id = (metadata or {}).get("chat_id")
    if not chat_id or str(chat_id).startswith("local:"):
        return None
    try:
        from open_webui.models.chats import Chats
        from open_webui.utils.misc import get_message_list
        chat = await Chats.get_chat_by_id(chat_id)
        hist = ((chat.chat if chat else {}) or {}).get("history") or {}
        chain = get_message_list(hist.get("messages") or {}, (metadata or {}).get("message_id") or hist.get("currentId"))
        chain = chain[:-1] if chain and chain[-1].get("role") == "assistant" else chain
        for m in reversed(chain):
            if m.get("role") == "assistant":
                return list(m.get("sources") or [])
        return []
    except Exception:
        return None


def history_text(text: str, names: dict | None = None, cite: bool = True) -> str:
    """An earlier assistant answer as context for the next turn: stale [N] markers become the document names they
    pointed to (this turn's [N] are different sources). `names` = {"N": label} from the citations Open WebUI stored with
    that answer; an older answer's visible sources block is read and dropped. `cite=False`: the markers are dropped
    instead, for the model that writes a cited answer (it would copy "(file.pdf, p. 4, §2.6)" as its own citations)."""
    names = dict(names or {})
    for m in presentation.SOURCES_MARK_RE.finditer(text or ""):
        # a list for one document names it once ("All from **Doc**."); its labels are rebuilt as "Doc, p. 6, §2.4.2", so
        # the next turn's context is the same
        one = presentation.ALL_FROM_RE.search(m.group(0))
        names.update({n: (f"{one.group(1)}, {label}" if one else label)
                      for n, label in presentation.SOURCES_LINE_RE.findall(m.group(0))})
    text = presentation.SOURCES_MARK_RE.sub("\n", texts.MODEL_NOTE_RE.sub("\n", text or ""))
    text = presentation.RELATED_MARK_RE.sub("\n", text)     # documents not used / closest passages: not context
    text = presentation.LIST_ALL_RE.sub("\n", text)          # a corpus listing's "Show all" repeats the names above it
    if not cite:
        return re.sub(r"[ \t]*\[\d+\]", "", text).strip()
    return re.sub(r"\[(\d+)\]", lambda mm: f"({names[mm.group(1)]})" if mm.group(1) in names else "", text).strip()


async def previous_document(sources, metadata) -> str | None:
    """The one library document the previous answer cited, or None."""
    prev = await stored_previous_turn(metadata)
    if not prev or not sources:
        return None
    cited, ambiguous = cited_documents(prev[1], prev[2], set(sources))
    return next(iter(cited)) if len(cited) == 1 and not ambiguous else None


async def conversation_target(valves, sources, dom, metadata) -> str | None:
    """The document an unnamed reference means ("this paper", "this study"): the one the previous answer cited, when
    its question agrees (this_document); None: ask, or show the topic's open questions."""
    prev = await stored_previous_turn(metadata)
    if not prev or not sources:
        return None
    question, content, stored = prev
    cited, ambiguous = cited_documents(content, stored, set(sources))
    scope, doc = await asyncio.to_thread(request_scope, question, sorted(sources), dom, valves.COLLECTION)
    import logging
    logging.getLogger("memento.query").info("conversation document: cited %s%s; previous request %s %s",
                                            sorted(cited), " (ambiguous)" if ambiguous else "", scope, doc or "")
    return this_document(cited, ambiguous, scope, doc)


async def conversation_documents(sources, metadata) -> list | None:
    """The documents a plural reference means ("which of them"): the previous answer's cited documents when there
    are two or more (these_documents); None → the single-document rule, or ask."""
    prev = await stored_previous_turn(metadata)
    if not prev or not sources:
        return None
    _, content, stored = prev
    cited, ambiguous = cited_documents(content, stored, set(sources))
    return these_documents(cited, ambiguous)


# a question that refers back to the previous answer's document without naming one ("what are its limitations?")
BACKREF_RE = re.compile(r"\b(?:it|its|it's|they|their|them|this|that|these|those|same|there(?:in)?|"
                         r"es|sein\w*|ihr\w*|dessen|deren|dies\w*|darin|dort|daraus)\b", re.I)


THIS_DOC_RE = re.compile(r"\b(?:this|that|the attached)\s+(?:paper|document|article|study|thesis|file|pdf|work|"
                          r"manuscript|review)\b|\bdies(?:es|e|en|er)\s+(?:paper|dokument|arbeit|artikel|studie|datei|"
                          r"pdf|thesis|abschlussarbeit)\b", re.I)


def unnamed_document(question: str, *, dc: bool, paper, named_docs, history) -> bool:
    """"Summarize this paper" in Library Chat with nothing behind "this": no document named or picked, not Document
    Chat, and no earlier turn it could refer to. Memento then asks which document instead of answering from the whole
    library."""
    return (not dc and not paper and not named_docs and not history
            and bool(THIS_DOC_RE.search(question or "")))


def unnamed_document_msg(de: bool) -> str:
    """The hand-over for unnamed_document: name it, pick it on the start screen, or attach your own in Document Chat."""
    if de:
        return ("Können Sie mir sagen, welches Dokument Sie meinen? Sie können den Titel eingeben oder es auf dem "
                "Startbildschirm unter **Summarize a document** auswählen. Wenn es Ihre eigene Datei ist und nicht in der "
                "Bibliothek liegt, können Sie sie im [Document Chat](/?model=memento-document-chat) anhängen.")
    # polite and short
    return ("Could you tell me which document you mean? You can type its title or choose it under **Summarize a "
            "document** on the start screen. If it is your own file and not in the library, you can attach it in "
            "[Document Chat](/?model=memento-document-chat).")
