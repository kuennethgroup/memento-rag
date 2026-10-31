"""One message of Library Chat: the domain profile, the message text, the history the router and the answer model
get, and the message's context (its language, the copy's link ids, Document Chat's shown names), set and reset at
the request boundary.
"""
import contextlib
import contextvars
import os
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any


# The domain profile (config/domains/<DOMAIN_NAME>.yaml); if it cannot be loaded, the fallback below keeps the pipe
# working.
FALLBACK_INSTRUCTION = "Retrieve content relevant to the query, including supporting figures, charts, and tables."


_DOMAINS: dict = {}


def default_collection() -> str:
    """The configured domain's primary collection, `<collection_prefix>-rag` (COLLECTION_PREFIX overrides the profile's
    prefix). A saved COLLECTION Valve still wins."""
    from memento.config.domain_loader import default_collection as configured
    return configured()


def domain(name: str) -> dict:
    if name not in _DOMAINS:
        try:
            import importlib
            _DOMAINS[name] = importlib.import_module("memento.config.domain_loader").load_domain(name)
        except Exception:
            return {}
    return _DOMAINS[name]


def llm_key(valves) -> str:
    """This copy's model key (Library Chat's answers, Library Upload's captions and tags): the environment variable
    named by its LLM_API_KEY_ENV Valve; empty when unset, never another copy's key."""
    name = (getattr(valves, "LLM_API_KEY_ENV", "") or "").strip()
    return os.getenv(name, "") if name else ""


def rerank_instruction(dom: dict | None) -> str | None:
    """The reranker instruction of this copy's domain: EMBED_INSTRUCTION when set, else the domain's own; None keeps the
    process domain's."""
    return os.getenv("EMBED_INSTRUCTION") or (dom or {}).get("rerank_instruction") or None


def domain_claim_rules(dom: dict | None) -> tuple:
    """The model-free claim checks of this copy's library (domain profile `claim_rules`); a profile without
    the key keeps all three, as evaluated in RQ3. An unknown rule name raises."""
    from memento.agentic_validation.scientific_checks import CLAIM_RULES, claim_rules
    return claim_rules((dom or {}).get("claim_rules", CLAIM_RULES))


def continue_request(body: dict) -> bool:
    """Open WebUI's "Continue Response": the request ends with the model's own answer and whatever the pipe writes is
    appended to it. Memento's answers are complete when sent: nothing to add."""
    msgs = (body or {}).get("messages") or []
    return bool(msgs) and (msgs[-1] or {}).get("role") == "assistant"


def normalize(s: str) -> str:
    return re.sub(r"[-_]+", " ", s).lower().strip()


def message_text(content) -> str:
    """Extract plain text from a chat message's `content`, which is either a
    plain string or OpenWebUI's multimodal list form
    (`[{"type": "text", "text": ...}, {"type": "image_url", ...}, ...]`)."""
    if isinstance(content, list):
        text = " ".join(p.get("text", "") for p in content if p.get("type") == "text")
    else:
        text = content or ""
    # Open WebUI prepends an <attached_files> block to user messages with files (for its own tools); it is not part of
    # the question or the history.
    return _ATTACHED_FILES_RE.sub("", text)


_ATTACHED_FILES_RE = re.compile(r"<attached_files>.*?</attached_files>\s*", re.S)


# Rendered artifacts that accumulate in assistant turns but shouldn't be
# replayed back to the model as conversation history: inline base64 figures
# (see step 10's render loop) and <think>...</think> reasoning blocks.
BASE64_IMAGE_RE = re.compile(r"!\[[^\]]*\]\(data:image/[^)]*\)")


THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.S)


# ── Conversation history budget ──
# Earlier turns are replayed within a size budget, so one long answer (a verbatim chapter, a long listing) cannot fill
# the answer model's prompt. The router rewrites a follow-up into a standalone question; each earlier message is cut to
# HISTORY_MESSAGE_TOKENS, all together to HISTORY_TOTAL_TOKENS (newest kept first); the router gets the gist of each
# turn (ROUTER_TURN_CHARS, enough to resolve "the sixth one" of a list).
HISTORY_MESSAGE_TOKENS, HISTORY_TOTAL_TOKENS, ROUTER_TURN_CHARS = 1500, 6000, 2500


SHORTENED_NOTE = "\n[…the rest of this earlier message is left out]"


# Memento Ingestion's upload reports (a chat can switch models) are not conversation — never history for Query
INGEST_REPORT_RE = re.compile(r"^\s*(Checking|Processing|Uploading|Upload of) \*\*\d+\*\* file|✅ Done · \d+ (stored|parsed)",
                               re.M)


def _shorten(text: str, max_chars: int) -> str:
    """The beginning of `text`, cut at a paragraph, line or sentence end, marked as shortened."""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    for sep in ("\n\n", "\n", ". "):
        i = cut.rfind(sep)
        if i > max_chars // 2:
            cut = cut[:i + (1 if sep == ". " else 0)]
            break
    return cut.rstrip() + SHORTENED_NOTE


def budget_history(history: list) -> list:
    """Earlier turns as the answer model gets them: every message at most HISTORY_MESSAGE_TOKENS, all together at most
    HISTORY_TOTAL_TOKENS — older turns are dropped first, and the window starts with a question."""
    per, total = int(HISTORY_MESSAGE_TOKENS * CHARS_PER_TOKEN), int(HISTORY_TOTAL_TOKENS * CHARS_PER_TOKEN)
    out, used = [], 0
    for m in reversed(history):
        text = _shorten(m["content"], per)
        if used + len(text) > total:
            break
        out.append({**m, "content": text})
        used += len(text)
    out.reverse()
    while out and out[0]["role"] == "assistant":
        out.pop(0)
    return out


def answer_was_gaps(history: list) -> bool:
    """The last earlier answer in this chat is a research-gap answer (its count table / collapsed blocks)."""
    last = next((m.get("content") or "" for m in reversed(history or []) if m.get("role") == "assistant"), "")
    return '<details type="memento_gaps">' in last or "| What the authors state |" in last \
        or "| Was die Autoren nennen |" in last


# a request to see the gaps (again): the table is shown
_SHOW_AGAIN_RE = re.compile(r"\b(?:show|list|repeat|give|display)\b[^.?!;]*?\b(?:again|once\s+more)\b|"
                            r"\b(?:zeig\w*|nenn\w*|wiederhol\w*|list\w*)\b[^.?!;]*?\b(?:nochmal|noch\s+(?:ein)?mal|erneut)\b",
                            re.I)


# a ban on repeating the gaps themselves, with "again" before or after them; checked before _SHOW_AGAIN_RE, which the
# ban contains
_NO_REPEAT_RE = re.compile(r"\b(?:don'?t|do\s+not|without|no\s+need\s+to)\s+(?:repeat(?:ing)?|show(?:ing)?|list(?:ing)?)"
                           r"\s+(?:again\s+)?(?:(?:the\s+|these\s+)?(?:research\s+)?(?:gaps?|table|limitations?|points?|"
                           r"list)\b|(?:those|these|them)\b)|"
                           r"\b(?:lücken|tabelle|einschränkungen|punkte|liste)\s+(?:bitte\s+)?nicht\s+(?:\w+\s+)?"
                           r"wiederhol\w*|\bohne\s+(?:die\s+|den\s+)?(?:lücken|tabelle|einschränkungen|punkte|liste)\s+"
                           r"(?:\w+\s+)?zu\s+wiederholen\b", re.I)


# the request is about the gaps, the table or its points too
_GAP_REF_RE = re.compile(r"\b(?:gaps?|limitations?|uncertaint\w*|untested|future\s+work|open\s+(?:questions?|problems?)|"
                         r"table|points?|those|them|lücken?|forschungslücke\w*|einschränkung\w*|unsicherheit\w*|"
                         r"zukünftig\w*|offene\w*\s+fragen|tabelle|punkt(?:e|en)?)\b", re.I)


def show_gap_table(*, suggest: bool, history: list, question: str = "", same_document: bool = True) -> bool:
    """Show the gap answer's table? Not when a request right after a gap answer about the same document only asks for
    extensions or bans repeating the gaps. Shown whenever it asks to see them, names the gaps or their points, or is
    about another document."""
    if not (suggest and answer_was_gaps(history)) or not same_document:
        return True
    q = question or ""
    if _NO_REPEAT_RE.search(q):
        return False
    if _SHOW_AGAIN_RE.search(q):
        return True
    return bool(_GAP_REF_RE.search(q))


def router_history(history: list) -> list:
    """The same turns, each cut to its gist: enough to resolve "it" / "the second paper", and a faster router call."""
    return [{**m, "content": _shorten(m["content"], ROUTER_TURN_CHARS)} for m in history]


def section_tag(section_raw, full: bool = False) -> str:
    """Short section marker for a citation name, so passages from one page are told apart: '2.4.2 | Heading' → '§2.4.2';
    an unnumbered heading → its first words (`full`: the whole heading, for a figure caption)."""
    raw = " ".join(str(section_raw or "").replace("|", " ").split())
    m = re.match(r"(\d+(?:\.\d+)*)\b", raw)
    if m:
        return f"§{m.group(1)}"
    return raw if full or len(raw) <= 28 else raw[:27].rsplit(" ", 1)[0] + "…"


# Document Chat's file names for display: its sources carry a random token ("Report-3fa2…"); within its answer this maps
# them to the attached file's name. Unset (Library Chat): names are shown unchanged.
SHOWN_NAMES: "contextvars.ContextVar[dict | None]" = contextvars.ContextVar("memento_shown_names", default=None)


CHARS_PER_TOKEN = 3.0      # measured with the Qwen3 tokenizer: 3.2 (English papers, raw PDF text), 3.6 (German thesis)


# the question's language ("de": our own fixed texts in German)
LANG: "contextvars.ContextVar[str]" = contextvars.ContextVar("memento_lang", default="en")


# this request's copy's model ids for reply links (None: the default ids), set in pipe()
LINK_IDS: "contextvars.ContextVar[tuple | None]" = contextvars.ContextVar("memento_link_ids", default=None)


def _reset(var: contextvars.ContextVar, token) -> None:
    try:
        var.reset(token)
    except ValueError:                 # closed from another context (a client that went away)
        pass


@contextlib.contextmanager
def scope(*, lang: str, link_ids: tuple | None = None) -> Iterator[None]:
    """A message's language and its copy's link ids, reset however the message ends (cancellation included)."""
    tokens = ((LANG, LANG.set(lang)), (LINK_IDS, LINK_IDS.set(link_ids)))
    try:
        yield
    finally:
        for var, token in tokens:
            _reset(var, token)


@contextlib.contextmanager
def shown_names(names: dict | None) -> Iterator[None]:
    """Document Chat's names for display during one answer, reset however it ends."""
    token = SHOWN_NAMES.set(names)
    try:
        yield
    finally:
        _reset(SHOWN_NAMES, token)


THOROUGH_FILTER_ID = "memento_thorough"   # openwebui/functions/thorough_filter.py (toggle button)


_FILE_TYPE_RE = re.compile(r'<file [^>]*?content_type="([^"]*)"')


def attachment_kinds(raw, raw_text: str) -> tuple[bool, bool]:
    """(images attached to this message, other files attached to it), from OpenWebUI's <attached_files> tags and
    the message's image parts."""
    block = re.search(r"<attached_files>(.*?)</attached_files>", raw_text or "", re.S)
    types = _FILE_TYPE_RE.findall(block.group(1)) if block else []
    n_files = len(re.findall(r"<file ", block.group(1))) if block else 0
    parts = raw if isinstance(raw, list) else []
    images = any(t.startswith("image/") for t in types) or any(
        isinstance(x, dict) and x.get("type") == "image_url" for x in parts)
    others = n_files > sum(t.startswith("image/") for t in types)
    return images, others


# Asking to transform or reason about content means normal Q&A, never verbatim (substring match, so word forms match
# too). Also used by figures.detect_image_intent below.
ANALYSIS_WORDS = [
    # English
    "explain", "why", "how", "summar", "interpret", "compar", "analy",
    "discuss", "elaborat", "simpl", "mean", "derive", "signific",
    "implication", "versus", " vs", "difference", "relate", "evaluat",
    # German
    "erklär", "erläuter", "warum", "wieso", "weshalb", "zusammenfass",
    "fasse", "vergleich", "analysier", "interpretier", "bedeut",
    "unterschied", "bewert",
]


# ── One message on its way through Library Chat (service.run's steps share it) ──
@dataclass
class Turn:
    body: dict
    user: dict
    files: list | None
    metadata: dict | None
    emit: Callable | None                   # sends an event to the chat (none when Open WebUI gave no emitter)
    function_id: str                        # this copy's Open WebUI function id
    valves: Any                             # this copy's Valves
    engine: Callable[[], Any]               # Open WebUI's database (a getter)
    rec: "cache.Recorder"                   # what is stored as this message's answer
    doc_chat: "DocChat | None"              # Document Chat's files, None in Library Chat
    status: Callable | None = None          # shows a status line under the question
    done: bool = False                      # a step has finished the reply: nothing more is sent
    # set by the steps, in their order
    api_key: str | None = None              # the answer model's key
    llm: Any = None                         # the answer model's client (the router asks it)
    dom: dict | None = None                 # the library's domain profile
    variant: Any = None                     # the model that writes the answer
    local_variant: Any = None               # the local model (Thorough check's notes; Document Chat)
    collection: str | None = None           # the passages searched: the library's or Document Chat's
    thorough_on: bool | None = None         # Thorough check is on (never for the background warm-up)
    budget: Any = None                      # the reply's time budget
    chat_messages: list | None = None       # the chat as Open WebUI sent it
    query: str | None = None                # the question as searched (the router's rewrite)
    library_empty: bool | None = None       # the library has no documents yet
    scope: str | None = None                # what the search covers: document | documents | library | none
    history: list | None = None             # earlier turns, sanitized and within budget
    answer_history: list | None = None      # the same, the earlier answers without their citations (for answer models)
    search_all: str | None = None           # a yes to "search the whole library instead?": the question asked again
    original_query: str | None = None       # the question as the reader wrote it
    query_intent: str | None = None         # the route: ranked_retrieval, summary, gap_query, …
    think: bool | None = None               # the answer model thinks first
    router_ok: bool | None = None           # the router classified this message (its labels decide)
    suggest: bool | None = None             # the question wants ideas beyond the documents
    element: str | None = None              # the part of a paper asked for: figure | table | equation | none
    author: str | None = None               # the person whose documents are asked for
    author_docs: list | None = None         # their documents, when the question adds a topic
    list_topic: str | None = None           # a list question's topic
    list_kind: str | None = None            # the kind of document a list question asks for
    list_capped: bool | None = None         # the list scan had more documents than it may add
    gap_base_think: bool | None = None      # thinking for a research-gap answer (before the live-answer rule)
    client: Any = None                      # the Qdrant client
    paper: str | None = None                # the one document the question is about
    named_docs: list | None = None          # two or more documents named in the question
    sources: set | None = None              # the library's (or the chat's) documents, ready ones
    in_progress: set | None = None          # documents being uploaded or re-embedded
    picked: Any = None                      # the document picked on the start screen, with its route
    docs: dict | None = None                # the library's rows, by source
    resolve_failed: bool | None = None      # the lookup of a requested document failed
    image_intent: bool | None = None        # a request to see figures
    file_order: list | None = None          # several documents, each searched on its own
    even_files: bool | None = None          # each of file_order gets an equal share of the passages
    qvec: list | None = None                # the question's embedding
    hits: list | None = None                # the passages the answer is written from
    related_task: Any = None                # the search for related documents, running while the answer is written
    verbatim: bool | None = None            # the reply reproduces a section as it stands
    verbatim_total: int | None = None       # the section's passages in all
    whole_doc: bool | None = None           # the answer reads whole documents (a summary)
    summary_mode: str | None = None         # whole | partial
    multi_parts: list | None = None         # Document Chat: one summary per file
    ctx_names: dict | None = None           # the titles the model reads
    verify: bool | None = None              # Thorough check runs on this answer
    check_prompt: str | None = None         # the instructions the check's revisions keep
    system_prompt: str | None = None        # the answer model's instructions
    user_content: Any = None                # the sources and the question, with images
    seen_images: set | None = None          # the passages the model saw as images
