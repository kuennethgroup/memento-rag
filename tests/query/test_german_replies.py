# A German question gets German around the answer too: status lines, notes, footers and fixed messages. Model text is
# never changed; an English question stays English; German notes are recognised in history and before storing like the
# English ones.
import re

import pytest

import openwebui.functions.query_pipe as qp
from memento.query import cache, generation, presentation, request, texts
from memento.query.scope import conversation
from memento.document_chat import chat
from memento.ingestion.upload import messages

FIXED = [(m, n) for m in (qp, texts) for n in dir(m)
         if re.fullmatch(r"[A-Z_]+(_NOTE|_MSG|_STATUS)", n) and isinstance(getattr(m, n), str)]


@pytest.fixture
def de():
    token = request.LANG.set("de")
    yield
    request.LANG.reset(token)


def _statuses() -> set:
    import inspect
    src = inspect.getsource(qp)
    return set(re.findall(r'status\("([^"]+)"', src)) | {texts.THOROUGH_ON_STATUS} | set(texts.GAP_REPORT_STEPS.values())


def test_every_fixed_text_and_status_has_a_german_version():
    fixed = {getattr(m, n) for m, n in FIXED} | _statuses() | {texts.DOC_CHAT_PRIVACY}
    missing = [t for t in fixed if "{" not in t and t not in texts.DE_FIXED]
    assert not missing, missing


def test_german_texts_follow_the_ui_rules():
    for en, de in texts.DE_FIXED.items():
        assert de and de != en, en
        assert "—" not in de, de
        assert not re.search(r"\b(the|please|answer|library|could not)\b", de, re.I) or "Library" in de, de


@pytest.mark.usefixtures("de")
def test_fixed_texts_are_german():
    assert texts.tr(texts.CHECKED_NOTE) == "_Geprüft mit Thorough check._"
    assert texts.tr("Searching the library...") == "Durchsuche die Bibliothek..."
    assert texts.tr(texts.NOTHING_IN_LIBRARY_MSG).startswith("Ich konnte in der Bibliothek nichts finden")
    assert texts.tr(texts.GENERAL_NOTE) == "\n\n---\n_Direkt beantwortet, nicht aus der Bibliothek, daher gibt es keine Quellen._"


@pytest.mark.usefixtures("de")
def test_footers_built_from_templates_are_german():
    assert texts.tr(generation.answered_by("Qwen 3.8 27B (Local)")) == "\n\n_Beantwortet von Qwen 3.8 27B (Local)._\n"
    assert texts.tr(generation.answered_by("Mistral (Academic Cloud)", "Qwen 3.8 27B (Local)")) == \
        "\n\n_Beantwortet von Mistral (Academic Cloud), geprüft von Qwen 3.8 27B (Local)._\n"
    hits = [{"source": "P", "page": 0}, {"source": "Q", "page": 1}, {"source": "R", "page": 2}]
    marker = texts.tr(presentation.sources_marker(hits, {0, 2}))
    assert "In dieser Antwort zitiert: 2 von 3 Quellen, die Memento gelesen hat" in marker
    assert "_Die Nummern entsprechen den Links im Text._" in marker
    # one cited document: named once, its passages counted
    assert "In dieser Antwort zitiert: 1 Abschnitt dieses Dokuments" in texts.tr(presentation.sources_marker(hits[:1], {0}))
    assert "alle 3 Quellen, die Memento gelesen hat" in texts.tr(presentation.sources_marker(hits, {0, 1, 2}))
    none = texts.tr(presentation.related_footer(items=[], titles={}, searched=35, cited=False, weak=True, files_url=""))
    assert "35 Dokumente durchsucht: kein Abschnitt kam dieser Frage nah genug." in none
    assert texts.tr("\n\n*Figure (2): PolyBERT, p. 3*") == "\n\n*Abbildung (2): PolyBERT, p. 3*"
    assert "wurde nicht hinzugefügt: dieser Dateityp wird nicht unterstützt." in \
        texts.tr(chat.doc_chat_failure("a.xyz", "unsupported"))
    assert texts.tr(chat.doc_chat_no_file("Was ist SEI?")).endswith(
        "[fragen Sie im Library Chat](/?model=memento-query&q=Was%20ist%20SEI%3F).")
    err = texts.tr(texts.user_error("search", RuntimeError("x")))
    assert err.startswith("⚠️ **Entschuldigung, die Suche in der Bibliothek ist fehlgeschlagen.**")


@pytest.mark.usefixtures("de")
def test_model_text_is_never_changed():
    text = "The library contains polyBERT [1]. Answered by the authors, it is checked. Figure 2 shows it."
    assert texts.tr(text) == text


def test_english_questions_stay_english():
    assert request.LANG.get() == "en"
    assert texts.tr(texts.CHECKED_NOTE) == texts.CHECKED_NOTE
    assert presentation.status_event("Searching the library...", False)["data"]["description"] == "Searching the library..."


@pytest.mark.usefixtures("de")
def test_status_lines_are_german_and_done_stays_hidden():
    assert presentation.status_event("Searching the library...", False)["data"]["description"] == "Durchsuche die Bibliothek..."
    done = presentation.status_event("Done", True)["data"]
    assert done["hidden"] is True and done["description"] == "Fertig"
    assert presentation.status_event("Summarizing file 2 of 3...", False)["data"]["description"] == "Fasse Datei 2 von 3 zusammen..."


@pytest.mark.usefixtures("de")
def test_german_notes_are_recognised_in_history_and_before_storing():
    shown = texts.tr("Answer [1].\n\n" + texts.CHECKED_NOTE + generation.answered_by("Qwen 3.8 27B (Local)"))
    assert "Geprüft" in shown and "Beantwortet von" in shown
    assert conversation.history_text(texts.sanitize_assistant_text(shown)) == "Answer ."
    stored = texts.tr("### Limitations\n1. A [1]." + texts.SUGGEST_FAILED_NOTE)
    assert "Vorschläge" in stored and "Vorschläge" not in conversation.without_suggestions(stored)
    assert any(n in texts.untr(texts.tr(texts.TIMEOUT_NOTE)) for n in cache.UNSTORED_NOTES)


def test_a_german_question_gets_german_end_to_end(monkeypatch):
    from test_nothing_found_skip import _library_run
    mod, out, streamed = _library_run(monkeypatch, [0.016, 0.012, 0.009], question="Was weißt du über Gummimischungen?",
                                      NOTHING_FOUND_REPLY=False)
    assert out.strip().startswith("Ich konnte in der Bibliothek nichts finden")


# ── Library Upload (ingestion_pipe): German when the typed message is German. The status line and the "Uploading N
# files:" header stay English: loader.js reads them (upload bar, "Uploaded").
import openwebui.functions.ingestion_pipe as ip


def test_upload_language_comes_from_the_typed_message():
    assert messages.upload_lang({"messages": [{"role": "user", "content": "fasse kurz zusammen was in diesen Dateien steht"}]}) == "de"
    assert messages.upload_lang({"messages": [{"role": "user", "content": "summarize these files"}]}) == "en"
    assert messages.upload_lang({"messages": [{"role": "user", "content": ""}]}) == "en"       # a plain upload


def test_upload_texts_are_german():
    assert messages.in_german(messages.ONLY_UPLOADS_MSG).startswith("Library Upload beantwortet keine Nachrichten.")
    assert "Bitte versuchen Sie es in ein paar Minuten noch einmal." in messages.in_german(messages.WARMUP_NOTE)
    assert messages.in_german("\n---\n✅ Done · 7 stored · 0 failed · 3 min 42 s\n") == \
        "\n---\n✅ Fertig · 7 gespeichert · 0 fehlgeschlagen · 3 min 42 s\n"
    assert messages.in_german("✅ Done · 2 stored · 1 skipped (already in the library, unchanged) · 0 failed · 40 s") == \
        "✅ Fertig · 2 gespeichert · 1 übersprungen (schon in der Bibliothek, unverändert) · 0 fehlgeschlagen · 40 s"
    line = messages.file_summary_line(1, 7, "embed", "Ax.pdf", {"ok": True, "elapsed": 26,
                                                            "type_counts": {"text": 30, "table": 1}}, pages=25)
    de = messages.in_german(line)
    assert "[1/7] ✅ Ax.pdf · 25 Seiten · 26 s" in de and "- Gespeichert: 30 Textabschnitte, 1 Tabelle" in de
    links = messages.in_german(messages.ask_links(["Ax"]))
    assert "Fragen zu **Ax**: [Zusammenfassen](" in links and "[Limitations & future work](" in links
    assert "q=Summarize%20the%20document" in links            # the link still asks the start screen's own question
    assert messages.in_german("⏭ **a.pdf**: already in the library as **A**, not added again\n") == \
        "⏭ **a.pdf**: schon in der Bibliothek als **A**, nicht erneut hinzugefügt\n"
    handed = messages.in_german(messages.ask_line("fasse zusammen", ["Ax"], uploaded=True))
    assert "_Ihre Nachricht wird hier nicht beantwortet: Library Upload fügt nur die Dateien hinzu._" in handed
    assert "[Im Library Chat fragen](" in handed


def test_upload_parts_the_browser_reads_stay_english():
    assert messages.in_german(messages.batch_header(7)) == messages.batch_header(7)
    status = messages.batch_status(3, 7, 2, 1, 1.0)
    assert messages.in_german(status) == status


@pytest.mark.parametrize("q", ["fasse kurz zusammen was in diesen Dateien steht", "Liste mir alle ressourcen auf",
                               "liste mir die paper als link auf", "Servus, sag mir was ich hier machen kann",
                               "kannst du die paper kurz erklären", "Was steht in diesem Paper?"])
def test_short_german_questions_are_german(q):
    # a tie ("was", "in" count as English) is English
    from memento.common.text_language import question_language
    assert question_language(q) == "de"


@pytest.mark.parametrize("q", ["What is polyBERT?", "Which papers use multitask learning?", "list all documents",
                               "Summarize the document \"Einfluss der Temperatur auf die Härte\""])
def test_english_questions_stay_english_in_the_detector(q):
    from memento.common.text_language import question_language
    assert question_language(q) == "en"
