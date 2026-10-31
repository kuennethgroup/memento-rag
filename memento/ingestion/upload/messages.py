"""Library Upload's replies: the fixed lines, the links that ask about new documents in Library Chat, a typed
question handed over, the chat's title, the batch's status and progress lines, each file's summary line, and the
same in German for a German message.
"""
import re
from pathlib import Path

from pypdf import PdfReader

from memento.query import texts
from memento.query.texts import STUDY_NOT_READY_MSG


def last_text(body: dict) -> str:
    """The text of the latest user message (OpenWebUI sends a string, or a list of parts for multimodal input)."""
    content = ((body.get("messages") or [{}])[-1] or {}).get("content") or ""
    if isinstance(content, list):
        text = " ".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    else:
        text = str(content)
    # Open WebUI prepends an <attached_files> block to messages with files (for its own tools); it is not the uploader's
    # words, so an upload without typed text stays "no text"
    return re.sub(r"<attached_files>.*?</attached_files>\s*", "", text, flags=re.S).strip()


# a member's message without new files: what they typed first, then how to add, then earlier uploads; Document Chat is
# named for files kept out of the library, and a typed question gets a link that asks it in Library Chat (ask_line). A
# copy of the pipes that is not set up uploads nothing (STUDY_NOT_READY_MSG: Library Chat's own text, memento/query/texts.py).


# an admin's message with neither a usable file nor a path: no text is no path (never the working directory)
ADMIN_NO_PATH_MSG = ("Type the path of a folder or file on the server to add it, or attach the files with "
                     "**+ → Upload Files**.\n")


ONLY_UPLOADS_MSG = ("Library Upload doesn't answer messages. It only adds documents to the library.\n\n"
                    "To add documents, attach files with **+ → Upload Files** and send. To ask about a file without "
                    "adding it to the library, use [Document Chat](/?model=memento-document-chat).\n")


OPEN_CHAT_LINE = "To ask a question, open a new chat with [Library Chat](/?model=memento-query).\n"


ASK_URL = "/?model=memento-query"         # a new Library Chat; with &q= OpenWebUI sends the question at once


ASK_EACH = 3                              # up to this many new documents get their own links


ASK_NAMED_MAX = 10                        # a question handed over names up to this many uploaded documents


# summaries and limitations are written after the upload (warm-up)
WARMUP_NOTE = ("\n_Summaries and limitations are being prepared in the background. Please try again in a few "
               "minutes._\n")


def ask_links(titles: list) -> str:
    """After an upload: links that open a new Library Chat and ask the start screen's own questions about each new
    document, so stored answers replay; more than ASK_EACH documents get one link to Library Chat."""
    from urllib.parse import quote
    titles = [t for t in titles or [] if t]
    if not titles:
        return ""
    if len(titles) > ASK_EACH:
        return f"\nAsk about them: [open Library Chat]({ASK_URL})\n" + WARMUP_NOTE
    esc = lambda t: re.sub(r"([\\`*_{}\[\]<>#|])", r"\\\1", t)
    q = lambda text: f"{ASK_URL}&q={quote(text, safe='')}"
    return "\n" + "\n\n".join(
        f"Ask about **{esc(t)}**: [Summarize it]({q(f'Summarize the document \"{t}\"')}) · "
        f"[Limitations & future work]({q(f'What research gaps does the document \"{t}\" name?')})"
        for t in titles) + "\n" + WARMUP_NOTE


# a typed message offered to Library Chat must look like a question or a request: a question mark, a question or request
# word first (English or German), or five words or more; a greeting or a thank-you gets the plain pointer to Library
# Chat
_ASK_WORDS = frozenset((
    "what which how why when where who whom whose is are was were does do did can could should would will "
    "explain summarize summarise describe list compare find show give tell "
    "welche welcher welches welchen wie warum wieso weshalb wann wo wer wem wen gibt ist sind kann können "
    "erkläre erklären fasse liste zeige nenne beschreibe vergleiche").split())


def _is_request(text: str) -> bool:
    words = re.findall(r"[^\W\d_]+", (text or "").lower())
    return "?" in (text or "") or (bool(words) and words[0] in _ASK_WORDS) or len(words) >= 5


def ask_line(text: str, titles: list | None = None, *, uploaded: bool = False) -> str:
    """A typed question in Library Upload, handed over: a link that opens a new Library Chat and asks it; the question
    names the uploaded documents, up to ASK_NAMED_MAX ("this paper" means nothing in a new chat)."""
    from urllib.parse import quote
    text = " ".join((text or "").split())
    if not text:
        return OPEN_CHAT_LINE
    if not _is_request(text):
        return OPEN_CHAT_LINE                # where questions go, without treating a greeting as one
    titles = [t for t in titles or [] if t]
    if len(titles) == 1:
        text = f'{text} (the document "{titles[0]}")'
    elif 1 < len(titles) <= ASK_NAMED_MAX:     # "these papers" names them all
        names = ", ".join(f'"{t}"' for t in titles)
        text = f"{text} (the documents {names})"
    head = "Your message isn't answered here: Library Upload only adds the files." if uploaded else \
        "Library Upload doesn't answer questions."
    return f"\n_{head}_ [Ask it in Library Chat]({ASK_URL}&q={quote(text, safe='')})\n"


EARLIER_UPLOADS_NOTE = "\n_The files uploaded earlier in this chat are already in the library._\n"


EARLIER_ATTACHMENTS_NOTE = ("\n_Files attached to earlier messages in this chat were not added. To add one, attach it "
                            "with **+ → Upload Files** to the message you send to Library Upload._\n")


# ── German for a German message ──
# The reply's lines in German when the typed text is German (an upload without text stays English). Not the "Uploading N
# files:" header and the status line: loader.js reads both (the upload bar; "Uploaded" once done).
_DE_UPLOAD_FIXED = {
    STUDY_NOT_READY_MSG: texts.DE_FIXED[STUDY_NOT_READY_MSG],
    ADMIN_NO_PATH_MSG: ("Geben Sie den Pfad eines Ordners oder einer Datei auf dem Server ein, um sie hinzuzufügen, "
                        "oder hängen Sie die Dateien mit **+ → Upload Files** an.\n"),
    ONLY_UPLOADS_MSG: ("Library Upload beantwortet keine Nachrichten. Es fügt nur Dokumente zur Bibliothek hinzu.\n\n"
                       "Um Dokumente hinzuzufügen, hängen Sie Dateien mit **+ → Upload Files** an und senden Sie sie. "
                       "Um nach einer Datei zu fragen, ohne sie zur Bibliothek hinzuzufügen, nutzen Sie "
                       "[Document Chat](/?model=memento-document-chat).\n"),
    OPEN_CHAT_LINE: "Um eine Frage zu stellen, öffnen Sie einen neuen Chat mit [Library Chat](/?model=memento-query).\n",
    WARMUP_NOTE: ("\n_Zusammenfassungen und Einschränkungen werden im Hintergrund vorbereitet. Bitte versuchen Sie es "
                  "in ein paar Minuten noch einmal._\n"),
    EARLIER_UPLOADS_NOTE: "\n_Die früher in diesem Chat hochgeladenen Dateien sind schon in der Bibliothek._\n",
    EARLIER_ATTACHMENTS_NOTE: ("\n_Dateien, die an frühere Nachrichten in diesem Chat angehängt wurden, wurden nicht "
                               "hinzugefügt. Um eine hinzuzufügen, hängen Sie sie mit **+ → Upload Files** an "
                               "die Nachricht an, die Sie an Library Upload senden._\n"),
    "Uploading documents is available to contributors only. Please ask the Memento admin for access.":
        "Dokumente hochladen können nur Mitwirkende. Bitte fragen Sie die Memento-Administration nach Zugang.",
    "Nothing to process. See the messages above, or try attaching the file(s) again.":
        "Nichts zu verarbeiten. Bitte sehen Sie sich die Meldungen oben an, oder hängen Sie die Datei(en) erneut an.",
    "_Your message isn't answered here: Library Upload only adds the files._":
        "_Ihre Nachricht wird hier nicht beantwortet: Library Upload fügt nur die Dateien hinzu._",
    "_Library Upload doesn't answer questions._": "_Library Upload beantwortet keine Fragen._",
    "[Ask it in Library Chat](": "[Im Library Chat fragen](",
    "\nAsk about them: [open Library Chat](": "\nFragen dazu: [Library Chat öffnen](",
    "<summary>Error details</summary>": "<summary>Fehlerdetails</summary>",
}


_DE_TYPES = {"text passage": "Textabschnitt", "text passages": "Textabschnitte", "list": "Liste", "lists": "Listen",
             "equation": "Gleichung", "equations": "Gleichungen", "table": "Tabelle", "tables": "Tabellen",
             "figure/image": "Abbildung/Bild", "figures/images": "Abbildungen/Bilder", "chart/plot": "Diagramm",
             "charts/plots": "Diagramme", "code block": "Codeblock", "code blocks": "Codeblöcke",
             "nothing stored": "nichts gespeichert"}


def _de_stored(m) -> str:
    parts = [re.sub(r"^(\d+) (.+)$", lambda p: f"{p.group(1)} {_DE_TYPES.get(p.group(2), p.group(2))}", x)
             if x != "nothing stored" else _DE_TYPES[x] for x in m.group(1).split(", ")]
    return "- Gespeichert: " + ", ".join(parts) + "\n"


_DE_UPLOAD_PATTERNS = [
    (re.compile(r"Ask about \*\*(.+?)\*\*: \[Summarize it\]\("), r"Fragen zu **\1**: [Zusammenfassen]("),
    (re.compile(r"(\[\d+/\d+\] ✅ [^\n]+?) · (\d+) page(s?) · "),
     lambda m: f"{m.group(1)} · {m.group(2)} Seite{'n' if m.group(3) else ''} · "),
    (re.compile(r"- Stored: ([^\n]*)\n"), _de_stored),
    (re.compile(r"\*\*(.+?)\*\*: skipped, already in the library and unchanged"),
     r"**\1**: übersprungen, schon in der Bibliothek und unverändert"),
    (re.compile(r"\*\*(.+?)\*\*: parse failed"), r"**\1**: Lesen fehlgeschlagen"),
    (re.compile(r"\*\*(.+?)\*\*: store failed"), r"**\1**: Speichern fehlgeschlagen"),
    (re.compile(r"⏭ \*\*(.+?)\*\*: already in the library as \*\*(.+?)\*\*, not added again"),
     r"⏭ **\1**: schon in der Bibliothek als **\2**, nicht erneut hinzugefügt"),
    (re.compile(r"✅ Done · (\d+) stored(?: · (\d+) skipped \(already in the library, unchanged\))? · (\d+) failed · "),
     lambda m: (f"✅ Fertig · {m.group(1)} gespeichert"
                + (f" · {m.group(2)} übersprungen (schon in der Bibliothek, unverändert)" if m.group(2) else "")
                + f" · {m.group(3)} fehlgeschlagen · ")),
    (re.compile(r"❌ Failed: "), "❌ Fehlgeschlagen: "),
    (re.compile(r"⚠ \*\*(.+?)\*\*: unable to access the uploaded file, please try attaching it again"),
     r"⚠ **\1**: Auf die hochgeladene Datei kann nicht zugegriffen werden, bitte hängen Sie sie erneut an"),
    (re.compile(r"⏭ \*\*(.+?)\*\*: skipped, file type not supported"),
     r"⏭ **\1**: übersprungen, Dateityp nicht unterstützt"),
    (re.compile(r"⏭ \*\*(.+?)\*\*: skipped, this upload was already added"),
     r"⏭ **\1**: übersprungen, dieser Upload wurde schon hinzugefügt"),
    (re.compile(r"⚠ \*\*(.+?)\*\*: file type not supported, skipped"), r"⚠ **\1**: Dateityp nicht unterstützt, übersprungen"),
    (re.compile(r"🔄 \*\*(.+?)\*\*: this file has changed since last time, re-reading it"),
     r"🔄 **\1**: diese Datei hat sich seit dem letzten Mal geändert, sie wird neu gelesen"),
]


def in_german(text: str) -> str:
    """The reply's text in German (our own lines only; file names and titles unchanged)."""
    if not text:
        return text
    for en in sorted(_DE_UPLOAD_FIXED, key=len, reverse=True):
        if en in text:
            text = text.replace(en, _DE_UPLOAD_FIXED[en])
    for rx, rep in _DE_UPLOAD_PATTERNS:
        text = rx.sub(rep, text)
    return text


def upload_lang(body: dict) -> str:
    """ "de" when the typed message is German; an upload without text is "en"."""
    try:
        text = last_text(body or {})
        if not text.strip():
            return "en"
        import importlib
        return "de" if importlib.import_module("memento.common.text_language").question_language(text) == "de" else "en"
    except Exception:
        return "en"


def upload_title(names: list, now=None) -> str:
    """Sidebar title of a Library Upload chat. Open WebUI's own fallback is the first message's text, blank for an
    upload without text, so the pipe answers the title task itself without a model call: the uploaded file names, or the
    date and time when the chat has none."""
    stems = list(dict.fromkeys(Path(n).stem for n in names if n))
    if not stems:
        from datetime import datetime
        return f"Library upload, {(now or datetime.now()):%d %b %Y %H:%M}"
    first = stems[0] if len(stems[0]) <= 60 else stems[0][:59].rstrip() + "…"
    return f"Upload: {first}" + (f" (+{len(stems) - 1} more)" if len(stems) > 1 else "")


# ── friendly chat-output helpers ──
# What a document contributed, by MinerU's own item type: (singular, plural), in this order.
_FRIENDLY_TYPE_LABELS = {
    "text": ("text passage", "text passages"), "list": ("list", "lists"), "equation": ("equation", "equations"),
    "table": ("table", "tables"), "image": ("figure/image", "figures/images"), "chart": ("chart/plot", "charts/plots"),
    "code": ("code block", "code blocks"),
}


def _friendly_counts(counts: dict) -> str:
    order = list(_FRIENDLY_TYPE_LABELS) + sorted(set(counts) - set(_FRIENDLY_TYPE_LABELS))
    parts = [f"{counts[t]} {_FRIENDLY_TYPE_LABELS.get(t, (t, t))[counts[t] != 1]}" for t in order if counts.get(t)]
    return ", ".join(parts) if parts else "nothing stored"


def details_block(summary: str, body: str) -> str:
    # The blank lines around the block are required: CommonMark starts a new HTML block only after a blank line, and
    # back-to-back <details> blocks would otherwise merge and show raw tags.
    return f"\n<details>\n<summary>{summary}</summary>\n\n{body}\n</details>\n\n"


# Uploads: every document streams as it finishes, so the newest is always the bottom line. A rolling window is
# impossible: Open WebUI ignores message rewrites while a reply streams. The class keeps its interface for the two call
# sites.
class DocLines:
    def add(self, line: str, ok: bool) -> str:
        """What to yield now for one finished document: its line, at once."""
        return line

    def rest(self) -> str:
        return ""


def batch_header(n: int) -> str:
    """The upload reply's first line. A streamed reply cannot be rewritten, so it says "Uploading N files:" and
    loader.js shows "Uploaded N files:" once the upload is done."""
    return f"\nUploading **{n}** file{'s' if n != 1 else ''}:\n\n"


def _counts(counts) -> str:
    """" · S stored · K unchanged · F failed" (parse only: "read") for the status lines; "" without counts."""
    if not counts:
        return ""
    stored, unchanged, failed, verb = counts
    return f" · {stored} {verb} · {unchanged} unchanged · {failed} failed"


def batch_status(completed: int, n: int, parsing: int | None, storing: int, mins: float,
                  pct: int | None = None, counts: tuple | None = None) -> str:
    """The live status line; loader.js draws it as the upload bar (keep the two in step). parsing=None: storing only
    (documents parsed before). `pct`: how far the whole upload is. `counts` (stored, unchanged, failed, "stored" |
    "read"): the finished documents apart, so the bar never calls a failed one stored."""
    phases = f"{parsing} parsing, {storing} storing" if parsing is not None else f"{storing} storing"
    done = f" · {pct}% done" if pct is not None else ""
    return f"Uploading... {completed} of {n} ready ({phases}){_counts(counts)} · {int(mins)}m elapsed{done}"


def took(elapsed: float) -> str:
    """The upload's time in the completion summary: minutes and seconds from the rounded total (rounding each part on
    its own gives "2 min 60 s")."""
    m, s = divmod(round(elapsed), 60)
    return f"{m} min {s} s" if m else f"{s} s"


def done_status(completed: int, n: int, elapsed: float, counts: tuple | None = None) -> str:
    """The final status line (loader.js reads it as the finished bar). Whole minutes floored, like batch_status, so
    "0m" (the bar's "under 1 min") means under a minute. With `counts`: "Done · S of N stored · K unchanged · F failed";
    without: "N of M processed"."""
    mins = f"{int(elapsed // 60)}m elapsed"
    if not counts:
        return f"Done · {completed} of {n} processed · {mins}"
    stored, unchanged, failed, verb = counts
    return f"Done · {stored} of {n} {verb} · {unchanged} unchanged · {failed} failed · {mins}"


# progress inside a document: MinerU reports no reading percentage, so reading is estimated from the page count at
# READ_SECONDS_PER_PAGE (measured on a 282-page book), slower when several documents are read at once, and shown at most
# READ_SHOWN_MAX until the reading is done. Reading is READ_SHARE of a document's time; storing has a real percentage.
READ_SECONDS_PER_PAGE = 1.05


READ_SHOWN_MAX = 0.95


READ_SHARE = 0.6


READ_DEFAULT_PAGES = 20           # a Word / PowerPoint file without a page count


STATUS_EVERY_S = 10               # the status line at least this often while documents are in progress


def doc_progress(phase: str, *, elapsed: float = 0, pages: int | None = None, reading: int = 1,
                  store_pct: float | None = None) -> float:
    """How far one document is (0–1): reading estimated, storing as reported."""
    if phase == "embed":
        return round(READ_SHARE + (1 - READ_SHARE) * min(100, max(0, store_pct or 0)) / 100, 3)
    est = elapsed / ((pages or READ_DEFAULT_PAGES) * READ_SECONDS_PER_PAGE * max(1, reading))
    return round(READ_SHARE * min(READ_SHOWN_MAX, max(0.0, est)), 3)


def batch_percent(*, completed: int, n: int, in_progress: list) -> int:
    """The whole upload in percent: finished documents plus the share of those in progress (100 only when done)."""
    if not n:
        return 0
    pct = round(100 * (completed + sum(in_progress)) / n)
    return pct if completed >= n else min(99, pct)


def shown_percent(*, previous: int, pct: int) -> int:
    """The bar never goes back: when more documents start reading, each reading estimate is divided by more, so the
    computed percent can drop for a while; the shown one holds."""
    return max(previous, pct)


def page_count(path) -> int | None:
    try:
        return len(PdfReader(str(path)).pages) if str(path).lower().endswith(".pdf") else None
    except Exception:
        return None


class Status:
    """An ephemeral progress marker: it flows through the same yield chain as body strings (workflow.run_batch →
    pipe_sync → run's relay loop) but is sent through the event emitter as a status line, not appended to the chat."""
    __slots__ = ("text", "done")

    def __init__(self, text: str, done: bool = False):
        self.text = text
        self.done = done


def file_summary_line(idx: int, total: int, phase: str, name: str, result: dict | None,
                        pages: int | None = None) -> str:
    """One compact, click-to-expand transcript line for a finished file. `phase` is the phase this event came from
    ("parse" for parse-only/SKIP_EMBED runs, "embed" for the normal parse+embed path, where a successful parse gets no
    line of its own; see workflow.run_batch's drain loop)."""
    prefix = f"[{idx}/{total}]"

    if not result or not result.get("ok"):
        err = (result or {}).get("error", "no result returned")
        label = "parse failed" if phase == "parse" else "store failed"
        return (
            f"❌ {prefix} **{name}**: {label}\n"
            + details_block("Error details", err)
        )

    if result.get("skipped"):
        return f"⏭ {prefix} **{name}**: skipped, already in the library and unchanged\n"

    notices    = result.get("notices") or []
    notice_lns = "".join(f"- {note}\n" for note in notices)

    if phase == "parse":
        pairs     = sorted(result.get("counts", {}).items(), key=lambda x: -x[1])
        breakdown = ", ".join(f"{t}: {c}" for t, c in pairs)
        summary   = f"{prefix} ✅ {name} · {result.get('pages', '?')}p · {result.get('elapsed', 0):.0f}s"
        body      = (f"- {result.get('chunks', '?')} chunks: {breakdown}\n"
                     f"- Output: `data/output/{Path(name).stem}/`\n"
                     + notice_lns)
        return details_block(summary, body)

    # visible: name, pages, time; expanded: what was stored, in plain words (technical notices go to the server log)
    pages_s = f" · {pages} page{'s' if pages != 1 else ''}" if pages is not None else ""
    summary = f"{prefix} ✅ {name}{pages_s} · {result.get('elapsed', 0):.0f} s"
    body    = f"- Stored: {_friendly_counts(result.get('type_counts', {}))}\n" + notice_lns
    return details_block(summary, body)
