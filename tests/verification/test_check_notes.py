# The Thorough check's closing notes, one definition for every place that recognises them: the research-gap layout, the
# history cleaner and the German texts all know "_⚠ Verification unavailable for this answer (…)._", the note
# strategies.finalize_for_display writes when the check fails. The notes' texts live in memento/agentic_validation/strategies.py; the
# pipe keeps its own copies (it imports nothing from the repository when Open WebUI loads it), pinned to that module
# here.
import re

import pytest

import openwebui.functions.query_pipe as qp
from memento.agentic_validation.strategies import finalize_for_display
from memento.agentic_validation import strategies as cn
from memento.research_gaps.passages import calm_gap_answer
from memento.library import answers as ac
from memento.query import cache, generation, request, texts
from memento.query.scope import conversation

GAP_ANSWER = "### Limitations\n1. Only one material was tested [1]\n\n### Future work\n2. Test more materials [2]\n"


def _unavailable() -> str:
    """The note exactly as production writes it (the answer's text plus the note)."""
    out = finalize_for_display("x", None, RuntimeError("judge down"))
    return out[len("x"):].strip()


def test_production_note_matches_the_shared_definition():
    note = _unavailable()
    assert note == "_⚠ Verification unavailable for this answer (RuntimeError)._"
    assert re.fullmatch(cn.UNAVAILABLE_RE, note)
    assert re.fullmatch(cn.NOTE_RE, note)


def test_pipe_copies_equal_the_shared_texts():
    assert texts.CHECKED_NOTE == cn.CHECKED
    assert texts.PARTLY_CHECKED_NOTE == cn.PARTLY
    assert texts.CHECK_TIMEOUT_NOTE == cn.TIMEOUT
    for en in (cn.CHECKED, cn.PARTLY, cn.TIMEOUT):
        assert texts.DE_FIXED[en] == cn.DE[en]


def test_every_note_is_a_note_for_the_gap_layout():
    for note in (cn.CHECKED, cn.PARTLY, cn.TIMEOUT, _unavailable()):
        assert re.fullmatch(cn.NOTE_RE, note), note


def test_gap_layout_puts_the_unavailable_note_after_the_blocks():
    shown = calm_gap_answer(f"{GAP_ANSWER}\n{_unavailable()}")
    assert shown.rstrip().endswith(_unavailable())
    assert shown.rindex("</details>") < shown.index("Verification unavailable")


def test_stored_formatted_answer_moves_the_note_out_of_the_last_block():
    # a formatted answer stored with the note inside its last block is fixed on replay
    formatted = calm_gap_answer(GAP_ANSWER)
    inside = formatted.rstrip()
    assert inside.endswith("</details>")
    stored = inside[: -len("</details>")] + f"\n{_unavailable()}\n\n</details>\n"
    shown = calm_gap_answer(stored)
    assert shown.rindex("</details>") < shown.index("Verification unavailable")


def test_history_drops_the_note_in_english_and_german():
    answer = f"Alumina reaches 18 GPa [1].\n\n{_unavailable()}"
    assert "Verification unavailable" not in conversation.history_text(texts.sanitize_assistant_text(answer))
    token = request.LANG.set("de")
    try:
        german = texts.tr(answer)
    finally:
        request.LANG.reset(token)
    assert "Verification unavailable" not in german                     # shown in German …
    cleaned = conversation.history_text(texts.sanitize_assistant_text(german))
    assert "nicht verfügbar" not in cleaned and "unavailable" not in cleaned     # … and dropped from the history


def test_german_round_trip():
    token = request.LANG.set("de")
    try:
        de = texts.tr(_unavailable())
    finally:
        request.LANG.reset(token)
    assert de == "_⚠ Prüfung für diese Antwort nicht verfügbar (RuntimeError)._"
    assert texts.untr(de) == _unavailable()


def test_an_answer_with_the_note_is_never_stored():
    assert any(n in texts.untr(f"text\n\n{_unavailable()}") for n in cache.UNSTORED_NOTES)


# ── an answer from which no claim was extracted ───────────────

def _without_claims():
    return cn.ArmResult("A", "An answer.")


def test_an_answer_without_extracted_claims_is_not_called_checked():
    out = generation.checked_text("An answer.", _without_claims(), None, False)
    assert out.endswith(texts.NO_CLAIMS_NOTE) and texts.CHECKED_NOTE not in out


def test_an_answer_without_extracted_claims_is_not_stored_as_checked():
    assert generation.check_ok(_without_claims(), None, False) is False


def test_report_prose_without_extracted_claims_stays_and_says_it_was_not_checked():
    from memento.research_gaps import proposal as gp
    assert generation.report_checked(_without_claims()) == "no_claims"
    footer = gp._how_made("no_claims")
    assert "were not checked against the sources" in footer and "were checked against the sources" not in footer
    assert "left out" not in footer


def test_the_no_claims_note_is_known_everywhere():
    note = cn.NO_CLAIMS
    assert texts.NO_CLAIMS_NOTE == note and texts.DE_FIXED[note] == cn.DE[note]
    assert note in cn.STATUS and note in generation._CHECK_STATUS_NOTES
    assert re.fullmatch(cn.NOTE_RE, note)
    assert texts.MODEL_NOTE_RE.sub("", f"An answer.\n\n{note}\n") == "An answer."
    assert generation._strip_check_status(f"An answer.\n\n{cn.DE[note]}") == "An answer."
    assert note != cn.EXTENSIONS_UNCHECKED and not re.fullmatch(cn.UNAVAILABLE_RE, note)


def test_answers_stored_as_checked_under_the_old_meaning_replay_without_the_claim():
    assert ac._effective(True, "2026-10-11") is False
    assert ac._effective(True, ac.CHECK_CONTRACT) is True
