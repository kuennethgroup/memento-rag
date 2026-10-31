"""Our own texts in Library Chat: user errors, fixed notes and messages, the German twins of every fixed text
(`tr` on the way out, `untr` where the code recognises its own notes), the language addenda, and the copy's model links
in replies.
"""
import re

from memento.query import request


def sanitize_assistant_text(text: str) -> str:
    text = untr(text)                # our German notes back to English, so they are recognised below
    text = request.THINK_BLOCK_RE.sub("", text)
    text = request.BASE64_IMAGE_RE.sub("", text)
    return text.strip()


RELATED_NONE_BODY = "Memento compared the closest passages with your question; none is close enough to show."


RELATED_WEAK = "The closest matches in the library were weak. Check the cited passages."


SUGGEST_FAILED_NOTE = "\n\n_Suggestions could not be written this time._\n\n"


# the local model answered instead of a cloud model, or a model stopped
QUOTA_NOTE = "_Cloud quota reached, so the local model answered._\n\n"


# first in a report or proposal that drew on papers' full texts: publisher text stays on this server, and this note
# keeps the rest of the chat on the local model (holds_full_text)
FULLTEXT_NOTE = "_This chat stays on the local model, because the text below draws on the full text of papers._\n\n"


UNREACHABLE_NOTE = "_Cloud model unreachable, so the local model answered._\n\n"


STOPPED_NOTE = ("_The cloud model stopped (quota reached or unreachable). "
                "Please ask again with another model or the local one._")


TIMEOUT_NOTE = ("_⚠ The answer took too long, so Memento stopped it. This can happen when the model is busy. "
                "Please try again in a moment, or pick another model._")


CHECK_TIMEOUT_NOTE = "_⚠ The thorough check could not finish in time, so this answer was not checked._"


# Thorough check made visible: the first status line while it is on, and a line under an answer it checked (no counts;
# the outcome is logged).
THOROUGH_ON_STATUS = "Thorough check is on: answers take longer."


CHECKED_NOTE = "_Checked with Thorough check._"


# some statements could not be judged (a judge call failed; they stay in the text): never called "checked"
PARTLY_CHECKED_NOTE = "_Partly checked with Thorough check: some statements could not be checked._"


# no statement was extracted: the check says nothing about the answer, so it is not called checked
NO_CLAIMS_NOTE = "_Not checked with Thorough check: it found no statements to check._"


# statements still failing after the check are marked ⚠ where their sentence is known. Copies of
# memento/agentic_validation/strategies.py (the pipe imports nothing from the repository when Open WebUI loads it)
CHECKED_UNCONFIRMED_NOTE = "_Checked with Thorough check: some statements could not be confirmed and are marked ⚠._"


CHECKED_UNCONFIRMED_UNMARKED_NOTE = "_Checked with Thorough check: some statements could not be confirmed._"


PARTLY_UNCONFIRMED_NOTE = ("_Partly checked with Thorough check: some statements could not be checked, and some could "
                           "not be confirmed and are marked ⚠._")


PARTLY_UNCONFIRMED_UNMARKED_NOTE = ("_Partly checked with Thorough check: some statements could not be checked, and "
                                    "some could not be confirmed._")


EXTENSIONS_UNCHECKED_NOTE = "_Not checked with Thorough check._"      # under the AI's own extensions of a checked answer


# Answers within a time budget: one budget per response; thinking that has not turned into answer text by the cut-off is
# closed and the request goes again without thinking.
SLOW_NOTE = "_The cloud model was slow, so the local model answered._\n\n"


# a requested document cannot be searched: still being uploaded, or the lookup failed
SCOPE_FAILED_MSG = "Sorry, I could not check the requested documents just now. Please try again."


# a research-gap answer the parser rejected twice
GAP_FAILED_MSG = "Sorry, I could not write the research gaps this time. Please try again."


# a Thorough revision of a research-gap answer that is not in its layout: the first draft is shown, unchecked
GAP_REVISION_UNREADABLE_NOTE = "_⚠ Verification unavailable for this answer (revision unreadable)._"


# a requested document that matches nothing: asked, with the offer to search the whole library; "yes" in the next
# message does (resolve.search_all_request). Never a silent library search.
NAMED_NOT_FOUND_MSG = ("I could not find that document in the library. Could you check the title? If you like, reply "
                       "**yes** and I will search the whole library instead.")


NAMED_NOT_FOUND_DE = ("Ich konnte dieses Dokument nicht in der Bibliothek finden. Können Sie den Titel prüfen? Wenn "
                      "Sie möchten, antworten Sie mit **ja**, dann durchsuche ich stattdessen die ganze Bibliothek.")


NO_UPLOADS_MSG = ("I could not find any documents you uploaded to the library. You can add files with "
                  "[Library Upload](/?model=memento-ingestion-prod), or ask about the library's most recent documents.")


NO_UPLOADS_DE = ("Ich konnte in der Bibliothek keine Dokumente finden, die Sie hochgeladen haben. Sie können Dateien mit "
                 "[Library Upload](/?model=memento-ingestion-prod) hinzufügen oder nach den neuesten Dokumenten der "
                 "Bibliothek fragen.")


DOC_CHAT_FILES_NONE = ("No file is attached yet. To attach one: + next to the message box, then Upload Files (PDF, "
                       "Word, PowerPoint or Excel).")


# Attached images: Library Chat does not look at images. Any message with an image gets only this reply, with or without
# text, pointing to Document Chat or Library Upload; one wording for every attachment.
ATTACH_CORE = "Library Chat can't read attachments directly. To ask about {what} privately, use **Document Chat**; " \
              "to add it to the shared library, use **Library Upload**, then ask here."


IMAGE_MSG = ATTACH_CORE.format(what="this image")


FILE_NOTE = "_" + ATTACH_CORE.format(what="a file") + "_\n\n"


# Document Chat reads only its attached files: a question about the library gets this short reply, with the question
# ready in Library Chat.
DOC_CHAT_LIBRARY_NOTE = ("Document Chat answers questions about your attached files and does not search the library. "
                         "For the library, please [use Library Chat](/?model=memento-query&q={q}).")


GENERAL_NOTE = "\n\n---\n_Answered directly, not from the library, so there are no sources._"


DOC_CHAT_GENERAL_NOTE = "\n\n---\n_Answered directly, not from your files, so there are no sources._"


# nothing in the library matches the question: the best passage scores below RELATED_LOW_SCORE, so the answer model is
# not called and the reply says what the search found
NOTHING_IN_LIBRARY_MSG = ("I could not find anything in the library that answers this. You could rephrase the question, "
                          "or add the document with **Library Upload** first.")


EMPTY_LIBRARY_MSG = ("**The library is empty.** No documents have been added yet. "
                     "Add documents with **Library Upload**, then ask again.")


# ── User-facing errors ──
# Users see a plain sentence and a short code; the technical detail goes to the Open WebUI log under the same code.
_ERROR_TEXT = {
    "library": "the document library cannot be reached right now",
    "search":  "searching the library failed",
    "answer":  "the answer model could not finish the answer",
    "gap":     "the research-gap answer could not be finished",
}


def user_error(kind: str, exc: Exception) -> str:
    import logging
    import time as _t
    code = f"E-{kind} {_t.strftime('%H:%M')}"
    logging.getLogger("memento.errors").warning("%s: %r", code, exc)
    text = _ERROR_TEXT.get(kind, "something went wrong")
    e = str(exc).lower()
    if kind in ("answer", "gap") and any(k in e for k in ("429", "rate limit", "overloaded", "503", "busy")):
        text = "the answer model is busy right now"
    elif kind in ("answer", "gap") and any(k in e for k in ("timed out", "timeout", "connection", "connect")):
        text = "the answer model is not reachable right now"
    return (f"⚠️ **Sorry, {text}.** Please try again in a minute. If it keeps happening, tell the study "
            f"organiser and mention `{code}`.")


DOC_CHAT_PRIVACY = ("This file is stored only by Memento on the chair's server and used only in this chat. It is not "
                    "added to the library and is deleted when you delete the chat. Document Chat uses only the local "
                    "model for reasoning and answering.")


DOC_CHAT_NO_FILE_MSG = ("To ask about a file, please attach it with **+** next to the message box, then **Upload Files** "
                        "(PDF, Word, PowerPoint or Excel).")


DOC_CHAT_READY_MSG = "Your file is ready. What would you like to know about it?"


DOC_CHAT_READING_MSG = "Your file is still being read. Please ask again once it is ready."


# a copy of the pipes that is not set up (memento/library/copies.py): never the default settings
STUDY_NOT_READY_MSG = "This library is not set up yet. Please tell the administrator."


# a chat whose Document Chat files belong to the other library's copy
DOC_CHAT_OTHER_LIBRARY_MSG = "This chat already uses the other library's Document Chat. Please start a new chat."


DOC_CHAT_SAVED_CHAT_MSG = ("Document Chat works only in saved chats, so that your file can be deleted together with "
                           "the chat. Please turn off the temporary chat and attach the file again.")


GAP_REPORT_EMPTY_MSG = ("The overview of open research questions is not ready for this library yet: its documents "
                        "are still being checked. Please try again later, or ask about one document with "
                        "**Limitations & future work**.")


GAP_REPORT_STEPS = {
    "extract": "Reading the open questions the new document names...",
    "against": "Checking the library's open questions against the new document...",
    "library": "Checking the new questions against the other documents...",
    "literature": "Searching published papers for the new questions...",
}


GAP_REPORT_NO_MORE_MSG = ("Those were all the open research questions here. To look at others, use **Draft a proposal** "
                          "on the start screen with a topic or a document.")


# ── German around the answer ──
# The question's language decides: our own fixed texts (notes, messages, footers, status lines) are shown in German,
# model text is never changed. Applied where text leaves the pipe (Pipe.pipe, _typed) and in _status_event; untr turns
# them back where the code recognises its own notes (history, storing). The interface's own names (Library Chat,
# Document Chat, Library Upload, Thorough check, the start-screen buttons) stay as the UI shows them.


def copy_link_ids(valves) -> tuple:
    from memento.library.copies import link_ids
    return link_ids(valves)


def relink_for_copy(text, valves=None):
    """Links to the default models in `text` pointing at this copy's (memento/library/copies.py): the given Valves', else
    this request's. After _tr: the German twins are found by the English text with the default ids."""
    ids = copy_link_ids(valves) if valves is not None else request.LINK_IDS.get()
    if not ids or not isinstance(text, str) or "/?model=" not in text:
        return text
    from memento.library.copies import relink
    return relink(text, ids)


DE_FIXED = {
    STUDY_NOT_READY_MSG: "Diese Bibliothek ist noch nicht eingerichtet. Bitte informieren Sie die Administration.",
    DOC_CHAT_OTHER_LIBRARY_MSG: ("Dieser Chat nutzt bereits Document Chat der anderen Bibliothek. Bitte beginnen Sie "
                                 "einen neuen Chat."),
    CHECKED_UNCONFIRMED_NOTE: ("_Geprüft mit Thorough check: einige Aussagen konnten nicht bestätigt werden und sind mit "
                               "⚠ markiert._"),
    CHECKED_UNCONFIRMED_UNMARKED_NOTE: "_Geprüft mit Thorough check: einige Aussagen konnten nicht bestätigt werden._",
    PARTLY_UNCONFIRMED_NOTE: ("_Teilweise geprüft mit Thorough check: einige Aussagen konnten nicht geprüft werden, und "
                              "einige konnten nicht bestätigt werden und sind mit ⚠ markiert._"),
    PARTLY_UNCONFIRMED_UNMARKED_NOTE: ("_Teilweise geprüft mit Thorough check: einige Aussagen konnten nicht geprüft "
                                       "werden, und einige konnten nicht bestätigt werden._"),
    EXTENSIONS_UNCHECKED_NOTE: "_Nicht mit Thorough check geprüft._",
    SCOPE_FAILED_MSG: ("Entschuldigung, ich konnte die angefragten Dokumente gerade nicht prüfen. Bitte "
                       "versuchen Sie es noch einmal."),
    NAMED_NOT_FOUND_MSG: NAMED_NOT_FOUND_DE,
    NO_UPLOADS_MSG: NO_UPLOADS_DE,
    GAP_FAILED_MSG: ("Entschuldigung, ich konnte die Forschungslücken diesmal nicht schreiben. Bitte "
                     "versuchen Sie es noch einmal."),
    GAP_REVISION_UNREADABLE_NOTE: "_⚠ Prüfung für diese Antwort nicht verfügbar (Überarbeitung unlesbar)._",
    CHECKED_NOTE: "_Geprüft mit Thorough check._",
    PARTLY_CHECKED_NOTE: "_Teilweise geprüft mit Thorough check: einige Aussagen konnten nicht geprüft werden._",
    NO_CLAIMS_NOTE: "_Nicht mit Thorough check geprüft: Es wurden keine prüfbaren Aussagen gefunden._",
    CHECK_TIMEOUT_NOTE: ("_⚠ Thorough check konnte nicht rechtzeitig fertig werden, daher wurde diese Antwort nicht "
                         "geprüft._"),
    TIMEOUT_NOTE: ("_⚠ Die Antwort hat zu lange gedauert, daher hat Memento sie abgebrochen. Das kann passieren, wenn "
                   "das Modell ausgelastet ist. Bitte versuchen Sie es gleich noch einmal oder wählen Sie ein anderes Modell._"),
    QUOTA_NOTE: "_Cloud-Kontingent erreicht, daher hat das lokale Modell geantwortet._\n\n",
    FULLTEXT_NOTE: ("_Dieser Chat bleibt beim lokalen Modell, weil der folgende Text auf Volltexten von Publikationen "
                    "beruht._\n\n"),
    UNREACHABLE_NOTE: "_Cloud-Modell nicht erreichbar, daher hat das lokale Modell geantwortet._\n\n",
    SLOW_NOTE: "_Das Cloud-Modell war langsam, daher hat das lokale Modell geantwortet._\n\n",
    STOPPED_NOTE: ("_Das Cloud-Modell hat abgebrochen (Kontingent erreicht oder nicht erreichbar). Bitte fragen Sie "
                   "noch einmal mit einem anderen Modell oder dem lokalen._"),
    SUGGEST_FAILED_NOTE: "\n\n_Die Vorschläge konnten diesmal nicht geschrieben werden._\n\n",
    GENERAL_NOTE: "\n\n---\n_Direkt beantwortet, nicht aus der Bibliothek, daher gibt es keine Quellen._",
    DOC_CHAT_GENERAL_NOTE: "\n\n---\n_Direkt beantwortet, nicht aus Ihren Dateien, daher gibt es keine Quellen._",
    FILE_NOTE: ("_Library Chat kann Anhänge nicht direkt lesen. Um privat nach einer Datei zu fragen, nutzen Sie "
                "**Document Chat**; um sie zur gemeinsamen Bibliothek hinzuzufügen, nutzen Sie **Library Upload** und "
                "fragen Sie dann hier._\n\n"),
    IMAGE_MSG: ("Library Chat kann Anhänge nicht direkt lesen. Um privat nach diesem Bild zu fragen, nutzen Sie "
                "**Document Chat**; um es zur gemeinsamen Bibliothek hinzuzufügen, nutzen Sie **Library Upload** und "
                "fragen Sie dann hier."),
    NOTHING_IN_LIBRARY_MSG: ("Ich konnte in der Bibliothek nichts finden, das das beantwortet. Sie können die Frage "
                             "umformulieren oder das Dokument zuerst mit **Library Upload** hinzufügen."),
    EMPTY_LIBRARY_MSG: ("**Die Bibliothek ist leer.** Es wurden noch keine Dokumente hinzugefügt. Sie können "
                        "Dokumente mit **Library Upload** hinzufügen und dann noch einmal fragen."),
    DOC_CHAT_PRIVACY: ("Diese Datei wird nur von Memento auf dem Server des Lehrstuhls gespeichert und nur in diesem "
                       "Chat verwendet. Sie wird nicht zur Bibliothek hinzugefügt und gelöscht, wenn Sie den Chat "
                       "löschen. Document Chat nutzt nur das lokale Modell zum Nachdenken und Antworten."),
    DOC_CHAT_NO_FILE_MSG: ("Um nach einer Datei zu fragen, hängen Sie sie bitte mit **+** neben dem Eingabefeld an, dann "
                           "**Upload Files** (PDF, Word, PowerPoint oder Excel)."),
    DOC_CHAT_READY_MSG: "Ihre Datei ist bereit. Was möchten Sie darüber wissen?",
    DOC_CHAT_READING_MSG: "Ihre Datei wird noch gelesen. Bitte fragen Sie noch einmal, sobald sie bereit ist.",
    DOC_CHAT_SAVED_CHAT_MSG: ("Document Chat funktioniert nur in gespeicherten Chats, damit Ihre Datei zusammen mit "
                              "dem Chat gelöscht werden kann. Bitte schalten Sie den temporären Chat aus und hängen "
                              "Sie die Datei erneut an."),
    GAP_REPORT_EMPTY_MSG: ("Die Übersicht der offenen Forschungsfragen ist für diese Bibliothek noch nicht fertig: "
                           "ihre Dokumente werden noch geprüft. Bitte versuchen Sie es später noch einmal, oder fragen Sie mit "
                           "**Limitations & future work** nach einem einzelnen Dokument."),
    GAP_REPORT_NO_MORE_MSG: ("Das waren alle offenen Forschungsfragen hier. Für weitere nutzen Sie **Draft a proposal** auf "
                             "dem Startbildschirm mit einem Thema oder einem Dokument."),
    THOROUGH_ON_STATUS: "Thorough check ist an: Antworten dauern länger.",
    # status lines
    "Done": "Fertig",
    "Stopped": "Abgebrochen",
    "Searching your files...": "Durchsuche Ihre Dateien...",
    "Searching the library...": "Durchsuche die Bibliothek...",
    "Suggesting extensions...": "Schlage Erweiterungen vor...",
    "Writing the overview...": "Schreibe die Übersicht...",
    "Thorough check: refining the answer...": "Thorough check: verbessere die Antwort...",
    "Reworking the previous answer...": "Überarbeite die vorige Antwort...",
    "Reading the whole document...": "Lese das ganze Dokument...",
    "Reading the files...": "Lese die Dateien...",
    "No gap tags found, using the Discussion and Conclusion sections...":
        "Keine markierten Forschungslücken gefunden, nutze die Abschnitte Diskussion und Fazit...",
    "New paper almost ready...": "Neues Paper fast fertig...",
    "Looking for answer...": "Suche die Antwort...",
    "Generating answer...": "Schreibe die Antwort...",
    "Finding the open questions on this topic...": "Suche die offenen Fragen zu diesem Thema...",
    "Finding the most open question on this topic...": "Suche die offenste Frage zu diesem Thema...",
    "Drafting the proposal...": "Entwerfe den Vorschlag...",
    "Collecting the evidence for the proposal...": "Sammle die Belege für den Vorschlag...",
    "Checking the overview against its sources...": "Prüfe die Übersicht an ihren Quellen...",
    "Checking the draft against its sources...": "Prüfe den Entwurf an seinen Quellen...",
    "Answering directly...": "Antworte direkt...",
    "Analyzing research gaps...": "Analysiere Forschungslücken...",
    "Adding the new paper first...": "Füge zuerst das neue Paper hinzu...",
    GAP_REPORT_STEPS["extract"]: "Lese die offenen Fragen, die das neue Dokument nennt...",
    GAP_REPORT_STEPS["against"]: "Prüfe die offenen Fragen der Bibliothek am neuen Dokument...",
    GAP_REPORT_STEPS["library"]: "Prüfe die neuen Fragen an den anderen Dokumenten...",
    GAP_REPORT_STEPS["literature"]: "Durchsuche veröffentlichte Paper nach den neuen Fragen...",
    # footers and parts of messages
    "_The numbers match the links in the text._": "_Die Nummern entsprechen den Links im Text._",
    RELATED_NONE_BODY:
        "Memento hat die nächsten Abschnitte mit Ihrer Frage verglichen; keiner ist nah genug, um ihn zu zeigen.",
    RELATED_WEAK: "Die nächsten Treffer in der Bibliothek waren schwach. Bitte prüfen Sie die zitierten Abschnitte.",
    "Searched the library: no passage came close to this question.":
        "Bibliothek durchsucht: kein Abschnitt kam dieser Frage nah genug.",
    " · [open](": " · [öffnen](",
    "\n\nTo ask the group's library instead, [ask it in Library Chat](":
        "\n\nUm stattdessen die Bibliothek der Gruppe zu fragen, [fragen Sie im Library Chat](",
    "Document Chat answers questions about your attached files and does not search the library. For the library, "
    "please [use Library Chat](":
        "Document Chat beantwortet Fragen zu Ihren angehängten Dateien und durchsucht die Bibliothek nicht. Für die "
        "Bibliothek [nutzen Sie bitte Library Chat](",
}


# the same texts in the informal "du" form older chats carry, so the code that recognises its own notes reads them too
# (untr, the store guard)
_DE_FIXED_DU = {
    SCOPE_FAILED_MSG: ("Entschuldige, ich konnte die angefragten Dokumente gerade nicht prüfen. Bitte versuch es "
        "noch einmal."),
    NAMED_NOT_FOUND_MSG: ("Ich konnte dieses Dokument nicht in der Bibliothek finden. Kannst du den Titel prüfen? "
        "Wenn du möchtest, antworte mit **ja**, dann durchsuche ich stattdessen die ganze "
        "Bibliothek."),
    NO_UPLOADS_MSG: ("Ich konnte in der Bibliothek keine Dokumente finden, die du hochgeladen hast. Du kannst "
        "Dateien mit [Library Upload](/?model=memento-ingestion-prod) hinzufügen oder nach den "
        "neuesten Dokumenten der Bibliothek fragen."),
    GAP_FAILED_MSG: ("Entschuldige, ich konnte die Forschungslücken diesmal nicht schreiben. Bitte versuch es "
        "noch einmal."),
    TIMEOUT_NOTE: ("_⚠ Die Antwort hat zu lange gedauert, daher hat Memento sie abgebrochen. Das kann "
        "passieren, wenn das Modell ausgelastet ist. Bitte versuch es gleich noch einmal oder "
        "wähle ein anderes Modell._"),
    STOPPED_NOTE: ("_Das Cloud-Modell hat abgebrochen (Kontingent erreicht oder nicht erreichbar). Bitte "
        "frag noch einmal mit einem anderen Modell oder dem lokalen._"),
    DOC_CHAT_GENERAL_NOTE: "\n\n---\n_Direkt beantwortet, nicht aus deinen Dateien, daher gibt es keine Quellen._",
    FILE_NOTE: ("_Library Chat kann Anhänge nicht direkt lesen. Um privat nach einer Datei zu fragen, "
        "nutze **Document Chat**; um sie zur gemeinsamen Bibliothek hinzuzufügen, nutze **Library "
        "Upload** und frag dann hier._\n\n"),
    IMAGE_MSG: ("Library Chat kann Anhänge nicht direkt lesen. Um privat nach diesem Bild zu fragen, "
        "nutze **Document Chat**; um es zur gemeinsamen Bibliothek hinzuzufügen, nutze **Library "
        "Upload** und frag dann hier."),
    NOTHING_IN_LIBRARY_MSG: ("Ich konnte in der Bibliothek nichts finden, das das beantwortet. Du kannst die Frage "
        "umformulieren oder das Dokument zuerst mit **Library Upload** hinzufügen."),
    EMPTY_LIBRARY_MSG: ("**Die Bibliothek ist leer.** Es wurden noch keine Dokumente hinzugefügt. Füge Dokumente "
        "mit **Library Upload** hinzu und frag dann noch einmal."),
    DOC_CHAT_PRIVACY: ("Diese Datei wird nur von Memento auf dem Server des Lehrstuhls gespeichert und nur in "
        "diesem Chat verwendet. Sie wird nicht zur Bibliothek hinzugefügt und gelöscht, wenn du "
        "den Chat löschst. Document Chat nutzt nur das lokale Modell zum Nachdenken und "
        "Antworten."),
    DOC_CHAT_NO_FILE_MSG: ("Um nach einer Datei zu fragen, hänge sie bitte mit **+** neben dem Eingabefeld an, dann "
        "**Upload Files** (PDF, Word, PowerPoint oder Excel)."),
    DOC_CHAT_READY_MSG: "Deine Datei ist bereit. Was möchtest du darüber wissen?",
    DOC_CHAT_READING_MSG: "Deine Datei wird noch gelesen. Bitte frag noch einmal, sobald sie bereit ist.",
    DOC_CHAT_SAVED_CHAT_MSG: ("Document Chat funktioniert nur in gespeicherten Chats, damit deine Datei zusammen mit "
        "dem Chat gelöscht werden kann. Bitte schalte den temporären Chat aus und hänge die Datei "
        "erneut an."),
    GAP_REPORT_EMPTY_MSG: ("Die Übersicht der offenen Forschungsfragen ist für diese Bibliothek noch nicht fertig: "
        "ihre Dokumente werden noch geprüft. Bitte versuch es später noch einmal, oder frag mit "
        "**Limitations & future work** nach einem einzelnen Dokument."),
    GAP_REPORT_NO_MORE_MSG: ("Das waren alle offenen Forschungsfragen hier. Für weitere nutze **Draft a proposal** auf "
        "dem Startbildschirm mit einem Thema oder einem Dokument."),
    "Searching your files...": "Durchsuche deine Dateien...",
    RELATED_NONE_BODY: ("Memento hat die nächsten Abschnitte mit deiner Frage verglichen; keiner ist nah genug, "
        "um ihn zu zeigen."),
    RELATED_WEAK: "Die nächsten Treffer in der Bibliothek waren schwach. Prüfe die zitierten Abschnitte.",
    "\n\nTo ask the group's library instead, [ask it in Library Chat](":
        "\n\nUm stattdessen die Bibliothek der Gruppe zu fragen, [frag im Library Chat](",
    "Document Chat answers questions about your attached files and does not search the library. For the library, "
    "please [use Library Chat](":
        "Document Chat beantwortet Fragen zu deinen angehängten Dateien und durchsucht die Bibliothek nicht. Für die "
        "Bibliothek [nutze bitte Library Chat](",
}


_DE_REASONS = {
    "this file type is not supported": "dieser Dateityp wird nicht unterstützt",
    "it could not be read": "sie konnte nicht gelesen werden",
    "it could not be stored": "sie konnte nicht gespeichert werden",
    "the uploaded file could not be found. Please attach it again":
        "die hochgeladene Datei wurde nicht gefunden. Bitte hängen Sie sie erneut an",
    "reading it was interrupted. Please attach it again": "das Lesen wurde unterbrochen. Bitte hängen Sie sie erneut an",
}


_DE_ERRORS = {
    "the document library cannot be reached right now": "die Dokumentbibliothek ist gerade nicht erreichbar",
    "searching the library failed": "die Suche in der Bibliothek ist fehlgeschlagen",
    "the answer model could not finish the answer": "das Antwortmodell konnte die Antwort nicht fertigstellen",
    "the research-gap answer could not be finished": "die Antwort zu den Forschungslücken konnte nicht fertig werden",
    "the answer model is busy right now": "das Antwortmodell ist gerade ausgelastet",
    "the answer model is not reachable right now": "das Antwortmodell ist gerade nicht erreichbar",
    "something went wrong": "etwas ist schiefgelaufen",
}


def _de_read(m) -> str:
    """ "the 1 source" / "all N sources" / "K of the N sources" → German."""
    one, all_n, k, n = m.group(1), m.group(2), m.group(3), m.group(4)
    what = ("die 1 Quelle" if one else f"alle {all_n} Quellen" if all_n else f"{k} von {n} Quellen")
    return f"In dieser Antwort zitiert: {what}, die Memento gelesen hat"


_DE_PATTERNS = [
    # the check failed (strategies.finalize_for_display; memento/agentic_validation/strategies.py)
    (re.compile(r"_⚠ Verification unavailable for this answer \(([^)\n]*)\)\._"),
     r"_⚠ Prüfung für diese Antwort nicht verfügbar (\1)._"),
    (re.compile(r"_Answered by ([^\n]+?), checked by ([^\n]+?)\._"), r"_Beantwortet von \1, geprüft von \2._"),
    (re.compile(r"_Answered by ([^\n]+?)\._"), r"_Beantwortet von \1._"),
    (re.compile(r"Cited in this answer: (?:(the 1 source)|all (\d+) sources|(\d+) of the (\d+) sources) Memento read"),
     _de_read),
    # the list for one document
    (re.compile(r"Cited in this answer: (\d+) passages? of this document"),
     lambda m: f"In dieser Antwort zitiert: {m.group(1)} Abschnitt{'e' if m.group(1) != '1' else ''} dieses Dokuments"),
    (re.compile(r"^All from \*\*(.+?)\*\*\.$", re.M), lambda m: f"Alle aus **{m.group(1)}**."),
    # the block under an answer
    (re.compile(r"Searched (\d+) documents?: no passage came close to this question\."),
     lambda m: f"{m.group(1)} Dokument{'e' if m.group(1) != '1' else ''} durchsucht: kein Abschnitt kam dieser Frage nah "
               f"genug."),
    (re.compile(r"Searched (\d+) documents?: the closest passages \((\d+)\)"),
     lambda m: f"{m.group(1)} Dokument{'e' if m.group(1) != '1' else ''} durchsucht: die nächsten Abschnitte "
               f"({m.group(2)})"),
    (re.compile(r"Searched the library: the closest passages \((\d+)\)"), r"Bibliothek durchsucht: die nächsten Abschnitte (\1)"),
    (re.compile(r"This answer cites no sources\. The closest passages \((\d+)\)"),
     r"Diese Antwort zitiert keine Quellen. Die nächsten Abschnitte (\1)"),
    (re.compile(r"Other documents with related passages \((\d+)\), not used in this answer"),
     r"Weitere Dokumente mit verwandten Abschnitten (\1), in dieser Antwort nicht verwendet"),
    (re.compile(r"\*Figure \((\d+)\): "), r"*Abbildung (\1): "),
    (re.compile(r"_(\d+) more figures are in the cited sources\._"),
     r"_\1 weitere Abbildungen sind in den zitierten Quellen._"),
    (re.compile(r"\*\*(.+?)\*\* was not added: (.+?)\.\n"),
     lambda m: f"**{m.group(1)}** wurde nicht hinzugefügt: {_de_reason(m.group(2))}.\n"),
    (re.compile(r"⚠️ \*\*Sorry, (.+?)\.\*\* Please try again in a minute\. If it keeps happening, tell the study "
                r"organiser and mention `([^`]+)`\."),
     lambda m: (f"⚠️ **Entschuldigung, {_DE_ERRORS.get(m.group(1), m.group(1))}.** Bitte versuchen Sie es gleich "
                f"noch einmal. Wenn es wieder passiert, sagen Sie bitte der Studienleitung Bescheid und nennen Sie "
                f"`{m.group(2)}`.")),
    (re.compile(r"^Summarizing file (\d+) of (\d+)\.\.\.$"), r"Fasse Datei \1 von \2 zusammen..."),
    (re.compile(r"^Reproducing '(.*)' section of (.*)\.\.\.$"), r"Gebe den Abschnitt '\1' aus \2 wieder..."),
    (re.compile(r"The open questions of \*\*(.+?)\*\* are still being checked against the library\. Please try again "
                r"later\."),
     r"Die offenen Fragen von **\1** werden noch mit der Bibliothek abgeglichen. Bitte versuchen Sie es später noch "
     r"einmal."),
    (re.compile(r"The open questions of \*\*(.+?)\*\* could not be read this time\. To see its limitations and future "
                r"work, pick it under \*\*Limitations & future work\*\* on the start screen\."),
     r"Die offenen Fragen von **\1** konnten diesmal nicht gelesen werden. Um seine Einschränkungen und zukünftigen "
     r"Arbeiten zu sehen, wählen Sie es auf dem Startbildschirm unter **Limitations & future work** aus."),
    (re.compile(r"\*\*(.+?)\*\* names no open questions that could be checked against the library\. To see its "
                r"limitations and future work, pick it under \*\*Limitations & future work\*\* on the start screen\."),
     r"**\1** nennt keine offenen Fragen, die mit der Bibliothek abgeglichen werden konnten. Um seine Einschränkungen "
     r"und zukünftigen Arbeiten zu sehen, wählen Sie es auf dem Startbildschirm unter **Limitations & future work** aus."),
    (re.compile(r"\*\*(.+?)\*\*: Paper still processing, try again later\."),
     r"**\1**: Das Paper wird noch verarbeitet, bitte versuchen Sie es später noch einmal."),
    (re.compile(r"\*\*\"(.+?)\"\*\* is not in the library \(it may have been removed or renamed\)\. Pick it again under "
                r"\*\*Draft a proposal\*\* on the start screen\."),
     r"**\"\1\"** ist nicht in der Bibliothek (es wurde vielleicht entfernt oder umbenannt). Wählen Sie es auf dem "
     r"Startbildschirm unter **Draft a proposal** erneut aus."),
    (re.compile(r"No open questions on \*\*(.+?)\*\* were found among the questions the documents of this library "
                r"name\. Try a broader topic, or the whole library\. To see one document's limitations and future work, "
                r"pick it under \*\*Limitations & future work\*\* on the start screen, or name it\."),
     r"Zu **\1** wurden unter den Fragen, die die Dokumente dieser Bibliothek nennen, keine offenen Fragen gefunden. "
     r"Versuchen Sie ein breiteres Thema oder die ganze Bibliothek. Um die Einschränkungen und zukünftigen Arbeiten eines "
     r"Dokuments zu sehen, wählen Sie es auf dem Startbildschirm unter **Limitations & future work** aus oder nennen Sie es."),
]


# the way back, for the code that recognises its own notes in German answers (history, storing)
_EN_FIXED = {de: en for en, de in list(_DE_FIXED_DU.items()) + list(DE_FIXED.items())}


_EN_PATTERNS = [
    (re.compile(r"_⚠ Prüfung für diese Antwort nicht verfügbar \(([^)\n]*)\)\._"),
     r"_⚠ Verification unavailable for this answer (\1)._"),
    (re.compile(r"_Beantwortet von ([^\n]+?), geprüft von ([^\n]+?)\._"), r"_Answered by \1, checked by \2._"),
    (re.compile(r"_Beantwortet von ([^\n]+?)\._"), r"_Answered by \1._"),
    (re.compile(r"\*Abbildung \((\d+)\): "), r"*Figure (\1): "),
    (re.compile(r"_(\d+) weitere Abbildungen sind in den zitierten Quellen\._"),
     r"_\1 more figures are in the cited sources._"),
]


# German answers: the model's own words in German too
DE_ANSWER_ADDENDUM = """

The question is in German: write the answer in German, and call the library "die Bibliothek" (never "Library")."""


# An English summary's language stated too, so a German document's summary for an English question comes back in English
# (gap answers state their language as well). Summaries only: an ordinary answer keeps the prompt the RQ3 evaluation
# pins (memento/retrieval/answer_pipeline.py). Outside the stored answers' key (cache.summary_prompt_version): stored
# answers keep their keys.
EN_SUMMARY_ADDENDUM = """

The question is in English: write the summary in English, also when the document is in another language, unless the \
question asks for another language. Keep document titles and quotations as written."""


# An ordinary English answer's language stated when a passage is in another language, so German sources do not turn the
# answer German. Only then: an English question on English passages keeps exactly the prompt the RQ3 evaluation pins.
# Outside the stored answers' key (_answer_prompt only), like the lines above.
EN_ANSWER_ADDENDUM = """

The question is in English: write the answer in English, also when the passages are in another language, unless the \
question asks for another language. Keep document titles and quotations as written."""


def answer_language_addendum(passages: list | None = None, own_line: bool = False) -> str:
    """The answer's language for the model: German for a German question; English for an English one when one of
    `passages` (the texts the model reads) is German, unless a summary's or list's own line states it (`own_line`)."""
    if request.LANG.get() == "de":
        return DE_ANSWER_ADDENDUM
    if own_line or not passages:
        return ""
    from memento.common.text_language import guess_language
    return EN_ANSWER_ADDENDUM if any(guess_language(t) == "de" for t in passages) else ""


def summary_language_addendum() -> str:
    """For a whole-document summary: English stated for an English question (a German one has DE_ANSWER_ADDENDUM)."""
    return "" if request.LANG.get() == "de" else EN_SUMMARY_ADDENDUM


# The answer's instructions after an earlier turn; the earlier answers come without their citations
# (scope.conversation.history_text, cite=False), and this says how to cite. The same text as
# memento.retrieval.answer_pipeline.HISTORY_NOTE, which the RQ3 builder uses
HISTORY_RULE = ("\nEarlier turns of this conversation are context only, not evidence: support every claim in this "
                "answer with the numbered sources below. Cite them only as [N], never by document name, page or "
                "section.")


# German addressed formally (Sie): added to the model's instructions for a German question wherever an answer is written
# (generation.open_answer, Thorough check) and always to a rework ("translate it into German" asked in English)
DE_ADDRESS_RULE = ("\n\nIn German prose you write yourself, address the reader formally using Sie, Ihnen and Ihr, even "
                   "when the user addresses you informally. Preserve quotations, titles, names and third-person pronouns.")


def address_prompt(prompt: str) -> str:
    """A system prompt with DE_ADDRESS_RULE for a German question, once; unchanged otherwise."""
    if request.LANG.get() != "de" or DE_ADDRESS_RULE.strip() in (prompt or ""):
        return prompt
    return (prompt or "") + DE_ADDRESS_RULE


def address_messages(messages: list) -> list:
    """The messages for the model with DE_ADDRESS_RULE in the system message (one is added when there is none), for a
    German question; a copy, the caller's list stays as it is."""
    if request.LANG.get() != "de":
        return messages
    out = [dict(m) for m in messages]
    first = next((m for m in out if m.get("role") == "system"), None)
    if first is None:
        return [{"role": "system", "content": DE_ADDRESS_RULE.strip()}] + out
    if isinstance(first.get("content"), str):
        first["content"] = address_prompt(first["content"])
    elif isinstance(first.get("content"), list) and not any(
            DE_ADDRESS_RULE.strip() in (p.get("text") or "") for p in first["content"] if isinstance(p, dict)):
        first["content"] = list(first["content"]) + [{"type": "text", "text": DE_ADDRESS_RULE}]
    return out


def question_lang(body: dict) -> str:
    """ "de" when the user's last message is German (a file-context wrapper reduced to its question), else "en"."""
    try:
        msgs = [m for m in (body or {}).get("messages") or [] if m.get("role") == "user"]
        text = request.message_text(msgs[-1].get("content")) if msgs else ""
        m = re.search(r"<user_query>(.*?)</user_query>", text, re.S)
        from memento.common.text_language import question_language
        return "de" if question_language(m.group(1) if m else text) == "de" else "en"
    except Exception:
        return "en"


def _de_reason(why: str) -> str:
    """A Document Chat reason in German (the ones with numbers by pattern)."""
    if why in _DE_REASONS:
        return _DE_REASONS[why]
    m = re.fullmatch(r"it has (\S+) pages, and Document Chat takes files of up to (\d+) pages", why)
    if m:
        return f"sie hat {m.group(1)} Seiten, und Document Chat nimmt Dateien mit bis zu {m.group(2)} Seiten"
    m = re.fullmatch(r"a chat takes up to (\d+) files\. Please start a new chat for more", why)
    if m:
        return f"ein Chat nimmt bis zu {m.group(1)} Dateien. Bitte starte für mehr einen neuen Chat"
    return why


_DE_ORDER = sorted(DE_FIXED, key=len, reverse=True)      # longest first: "Done" must not hit inside a longer text


def tr(text: str) -> str:
    """Our own fixed texts in `text` in the question's language (German when _LANG is "de"); model text unchanged."""
    if not text or request.LANG.get() != "de":
        return text
    if text in DE_FIXED:
        return DE_FIXED[text]
    for en in _DE_ORDER:
        if len(en) > 12 and en in text:          # short ones ("Done", "Stopped") only as a whole status line
            text = text.replace(en, DE_FIXED[en])
    for rx, rep in _DE_PATTERNS:
        text = rx.sub(rep, text)
    return text


def untr(text: str) -> str:
    """German notes back to English, where the code recognises its own notes (any language of the question)."""
    # every text is checked: a shortcut could skip notes the store guard relies on
    if not text:
        return text
    for de in sorted(_EN_FIXED, key=len, reverse=True):
        if len(de) > 12 and de in text:
            text = text.replace(de, _EN_FIXED[de])
    for rx, rep in _EN_PATTERNS:
        text = rx.sub(rep, text)
    return text


def holds_full_text(chat_messages: list) -> bool:
    """A report or proposal in this chat drew on papers' full texts: its note, in either language."""
    notes = (FULLTEXT_NOTE.strip(), DE_FIXED[FULLTEXT_NOTE].strip())
    return any(m.get("role") == "assistant" and any(n in request.message_text(m.get("content")) for n in notes)
               for m in chat_messages)


MODEL_NOTE_RE = re.compile(r"\n*_(?:Answered by [^\n]*?|Cloud quota reached, so the local model answered\."
                            r"|This chat stays on the local model[^\n]*?\."
                            r"|Cloud model unreachable, so the local model answered\."
                            r"|The cloud model stopped \(quota reached or unreachable\)\.[^\n]*?"
                            r"|⚠ The answer took too long, so Memento stopped it\.[^\n]*?"
                            r"|⚠ The thorough check could not finish in time[^\n]*?"
                            r"|Checked with Thorough check(?:: [^\n]*?)?\."
                            r"|Partly checked with Thorough check:[^\n]*?"
                            r"|Not checked with Thorough check(?:: [^\n]*?)?\."
                            r"|⚠ Verification unavailable for this answer \([^)\n]*\)\."
                            r"|The cloud model was slow, so the local model answered\.)_[ \t]*\n*")
