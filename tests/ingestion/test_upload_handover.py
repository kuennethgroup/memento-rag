# After an upload, Library Chat knows which documents "these / the papers I just uploaded" are, and the upload reply
# says that summaries are prepared in the background.
from datetime import datetime, timedelta, timezone
from urllib.parse import unquote

import pytest

import openwebui.functions.ingestion_pipe as ip
import openwebui.functions.query_pipe as qp
from memento.query.scope import documents
from memento.ingestion.upload import messages

TITLES = ["Ax: A Platform for Adaptive Experimentation", "BayBE: a Bayesian Back End", "BOHB: Robust and Efficient"]


def test_a_question_typed_with_several_uploads_names_them_in_library_chat():
    link = unquote(messages.ask_line("fasse kurz zusammen", TITLES, uploaded=True))
    assert 'fasse kurz zusammen (the documents "Ax: A Platform for Adaptive Experimentation", "BayBE: a Bayesian ' \
           'Back End", "BOHB: Robust and Efficient")' in link
    one = unquote(messages.ask_line("summarize this", TITLES[:1], uploaded=True))
    assert 'summarize this (the document "Ax: A Platform for Adaptive Experimentation")' in one
    many = unquote(messages.ask_line("summarize these", [f"Doc {i}" for i in range(messages.ASK_NAMED_MAX + 1)], uploaded=True))
    assert "Doc 0" not in many and "summarize these" in many


def test_the_upload_reply_says_summaries_come_later():
    assert messages.WARMUP_NOTE.strip() == ("_Summaries and limitations are being prepared in the background. Please try "
                                      "again in a few minutes._")
    assert messages.WARMUP_NOTE in messages.ask_links(TITLES[:1]) and messages.WARMUP_NOTE in messages.ask_links(TITLES * 3)
    assert messages.ask_links([]) == ""


@pytest.mark.parametrize("q, mine", [
    ("fasse die soeben upgeloadeten Paper zusammen", True),
    ("Summarize the papers I just uploaded", True),
    ("summarize my uploads", True),
    ("What do my uploaded documents say about BO?", True),
    ("fasse die hochgeladenen Dateien zusammen", True),
    ("Which documents were uploaded most recently?", False),
    ("Summarize the polyBERT paper", False),
])
def test_my_uploads_phrases(q, mine):
    assert documents.mentions_my_uploads(q) is mine


def test_the_latest_upload_batch_of_this_user():
    t = datetime(2026, 10, 9, 20, 31, tzinfo=timezone.utc)
    rows = [("Optuna", "u1", t - timedelta(minutes=1)), ("Ax", "u1", t), ("BOHB", "u1", t),
            ("Old", "u1", t - timedelta(days=2)), ("Theirs", "u2", t + timedelta(minutes=5))]
    assert documents._upload_batch(rows, "u1") == ["Ax", "BOHB", "Optuna"]
    assert documents._upload_batch(rows, "u3") == []
    assert documents._upload_batch(rows, None) == []


# ── a greeting is not handed over to Library Chat ──
# Only a question or a request is offered to Library Chat; a greeting gets the plain pointer.

@pytest.mark.parametrize("text", ["Hello", "hi", "Danke!", "ok", "Hallo zusammen", "thanks a lot"])
def test_a_greeting_gets_no_hand_over_to_library_chat(text):
    assert messages.ask_line(text) == messages.OPEN_CHAT_LINE == messages.ask_line(text, ["Doc"], uploaded=True)
    assert "Ask it in Library Chat" not in messages.ask_line(text)


@pytest.mark.parametrize("text", ["What does the paper say about creep?", "Welche Methoden werden genutzt",
                                  "summarize this", "fasse kurz zusammen", "Hello, can you help me?",
                                  "the strand foam results for the new resin please"])
def test_a_question_or_request_is_still_handed_over(text):
    assert "Ask it in Library Chat" in messages.ask_line(text)


def test_an_empty_message_still_gets_the_open_chat_line():
    assert messages.ask_line("") == messages.OPEN_CHAT_LINE
