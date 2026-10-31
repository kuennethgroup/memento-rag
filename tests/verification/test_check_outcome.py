# What "checked" means: the outcome counts the statements still failing (verdicts and scientific-check issues), marks
# them ⚠ where their sentence is known, says so in the note, and such an answer is not stored; the extensions section is
# cut out of the check only up to the next heading and comes after the note, labelled as not checked; a stored answer is
# "checked" only under the current contract.
import re
import sqlite3
import threading

import pytest
from sqlalchemy import create_engine, inspect, text

import openwebui.functions.query_pipe as qp
from memento.agentic_validation.claims import Claim
from memento.agentic_validation.scientific_checks import CheckIssue
from memento.agentic_validation.strategies import ArmResult, remaining_failures
from memento.agentic_validation.verifiers import Verdict
from memento.library import answers as ac
from memento.agentic_validation import strategies as cn
from memento.query import cache, generation, prompts, request, texts
from memento.query.scope import conversation

S1, S2 = "Alumina reaches 18 GPa [1].", "Zirconia reaches 99 GPa [2]."
ANSWER = f"{S1} {S2}"


def _result(labels, *, issues=None, spans=None, arm="A", removed=(), answer=ANSWER, unjudged=None):
    spans = spans or [(0, len(S1)), (len(S1) + 1, len(ANSWER))]
    claims = [Claim(f"c{i + 1}", f"claim {i + 1}", (i + 1,), spans[i]) for i in range(len(labels))]
    verdicts = {c.id: Verdict(c.id, lab) for c, lab in zip(claims, labels)}
    return ArmResult(arm, answer, claims, verdicts, issues or {}, removed=list(removed),
                     unjudged=sum(lab == "unjudged" for lab in labels) if unjudged is None else unjudged)


# ── outcome and marks ────────────────────────────────────────────────────────────────────────────────────────────────

def test_all_supported_is_checked():
    r = _result(["supported", "supported"])
    assert remaining_failures(r) == []
    assert generation.checked_text(r.answer, r, None, False) == f"{ANSWER}\n\n{cn.CHECKED}"
    assert generation.check_ok(r, None, False)


def test_a_contradicted_statement_is_marked_and_not_stored():
    r = _result(["supported", "contradicted"])
    out = generation.checked_text(r.answer, r, None, False)
    assert out == f"{S1} {S2} ⚠\n\n{cn.CHECKED_UNCONFIRMED}"
    assert not generation.check_ok(r, None, False)


def test_a_scientific_check_issue_counts_as_failing():
    r = _result(["supported", "supported"], issues={"c1": [CheckIssue("c1", "unit_mismatch", "MPa vs GPa")]})
    assert [c.id for c in remaining_failures(r)] == ["c1"]
    assert generation.checked_text(r.answer, r, None, False).startswith(f"{S1} ⚠ {S2}")


def test_partial_and_contradicted_get_one_combined_note():
    r = _result(["unjudged", "contradicted"])
    assert generation.checked_text(r.answer, r, None, False).endswith(cn.PARTLY_UNCONFIRMED)


def test_partial_alone_keeps_the_partly_note():
    r = _result(["unjudged", "supported"])
    assert generation.checked_text(r.answer, r, None, False).endswith(cn.PARTLY)


def test_two_claims_in_one_sentence_get_one_mark():
    r = _result(["contradicted", "unsupported"], spans=[(0, len(S1)), (0, len(S1))])
    assert generation.checked_text(r.answer, r, None, False).count("⚠") == 1 + cn.CHECKED_UNCONFIRMED.count("⚠")


def test_a_claim_without_a_sentence_is_not_marked_and_the_note_says_so():
    r = _result(["supported", "contradicted"], spans=[(0, len(S1)), (-1, -1)])
    out = generation.checked_text(r.answer, r, None, False)
    assert out == f"{ANSWER}\n\n{cn.CHECKED_UNCONFIRMED_UNMARKED}"


def test_offsets_that_do_not_fit_the_shown_text_mark_nothing():
    r = _result(["supported", "contradicted"], answer=f"<think>x</think>{ANSWER}")    # spans refer to the cleaned text
    out = generation.checked_text(r.answer, r, None, False)
    assert "⚠ " not in out.replace(cn.CHECKED_UNCONFIRMED_UNMARKED, "") and out.endswith(cn.CHECKED_UNCONFIRMED_UNMARKED)


def test_arm_c_removed_failures_are_not_marked():
    r = _result(["supported", "contradicted"], arm="C", removed=["c2"], answer=S1)
    assert remaining_failures(r) == []
    assert generation.checked_text(r.answer, r, None, False) == f"{S1}\n\n{cn.CHECKED}"
    r = _result(["supported", "contradicted"], arm="C", spans=[(0, len(S1)), (-1, -1)], answer=ANSWER)
    assert generation.checked_text(r.answer, r, None, False).endswith(cn.CHECKED_UNCONFIRMED_UNMARKED)   # never marks for C


def test_unavailable_is_unchanged():
    assert generation.checked_text("x\n\n_⚠ Verification unavailable for this answer (RuntimeError)._", None,
                            RuntimeError("x"), False).endswith("(RuntimeError)._")


# ── the notes ────────────────────────────────────────────────────────────────────────────────────────────────────────

def test_new_notes_are_notes_everywhere():
    for note in (cn.CHECKED_UNCONFIRMED, cn.CHECKED_UNCONFIRMED_UNMARKED, cn.PARTLY_UNCONFIRMED,
                 cn.PARTLY_UNCONFIRMED_UNMARKED):
        assert re.fullmatch(cn.NOTE_RE, note), note
        assert note not in conversation.history_text(texts.sanitize_assistant_text(f"Answer [1].\n\n{note}"))
        assert texts.DE_FIXED[note] == cn.DE[note]
        token = request.LANG.set("de")
        try:
            assert texts.tr(note) == cn.DE[note]
        finally:
            request.LANG.reset(token)
        assert texts.untr(cn.DE[note]) == note


# ── the extensions section ───────────────────────────────────────────────────────────────────────────────────────────

EXT = "### Possible extensions (suggested by the AI)\nAn idea of the model."


def test_extensions_end_at_the_next_heading():
    text_ = f"Fact [1].\n\n{EXT}\n\n## Evidence\nMore fact [2]."
    body, exts = prompts.split_extensions(text_)
    assert "More fact [2]." in body and "An idea" not in body               # the later section is checked
    assert exts == [("## Evidence", EXT)]


def test_two_extension_sections_stay_two():
    text_ = f"Fact [1].\n\n{EXT}\n\n## Evidence\nMore fact [2].\n\n{EXT.replace('An idea', 'Another idea')}"
    body, exts = prompts.split_extensions(text_)
    assert "More fact [2]." in body and "idea" not in body
    assert [a for a, _ in exts] == ["## Evidence", None]


def test_extensions_come_after_the_note_and_say_they_are_not_checked():
    final = f"Fact [1].\n\n{cn.CHECKED}"
    out = prompts.with_extensions(final, [(None, EXT)])
    assert out.index(cn.CHECKED) < out.index("### Possible extensions")
    assert cn.EXTENSIONS_UNCHECKED in out.split("### Possible extensions", 1)[1]


def test_an_anchored_extension_goes_back_before_its_heading():
    final = f"Fact [1].\n\n## Evidence\nMore fact [2].\n\n{cn.CHECKED}"
    out = prompts.with_extensions(final, [("## Evidence", EXT)])
    assert out.index("### Possible extensions") < out.index("## Evidence") < out.index(cn.CHECKED)


def test_no_extensions_no_change():
    assert prompts.with_extensions("Body [1].", []) == "Body [1]."
    assert prompts.split_extensions("No section [1].") == ("No section [1].", [])


# ── stored answers: the check contract ───────────────────────────────────────────────────────────────────────────────

FIELDS = dict(collection="col", source="doc", content_hash="h1", action="summary", language="en",
              model="local:qwen3.8-27b", mode="whole", prompt_version="p1")


@pytest.fixture
def engine():
    e = create_engine("sqlite://")
    ac.ensure_table(e)
    return e


def _legacy_checked(engine, key, answer):
    ac.put(engine, key, fields=FIELDS, answer=answer, citations=[], generated_by="m", checked=True)
    with engine.begin() as c:
        c.execute(text("UPDATE memento_answer_cache SET check_contract = NULL WHERE key = :k"), {"k": key})


def test_a_checked_answer_records_the_contract(engine):
    key = ac.cache_key(**FIELDS)
    ac.put(engine, key, fields=FIELDS, answer="a", citations=[], generated_by="m", checked=True)
    row = ac.get(engine, key)
    assert row["check_contract"] == ac.CHECK_CONTRACT and row["effective_checked"] is True


def test_a_legacy_checked_row_is_not_effectively_checked(engine):
    key = ac.cache_key(**FIELDS)
    _legacy_checked(engine, key, f"Old [1].\n\n{cn.CHECKED}")
    row = ac.get(engine, key)
    assert row["checked"] is True and row["effective_checked"] is False


def test_a_new_checked_warmup_replaces_a_legacy_checked_row(engine):
    key = ac.cache_key(**FIELDS)
    _legacy_checked(engine, key, "legacy")
    ac.put(engine, key, fields=FIELDS, answer="new checked", citations=[], generated_by="m", replace=False, checked=True)
    assert ac.get(engine, key)["answer"] == "new checked"
    ac.put(engine, key, fields=FIELDS, answer="another", citations=[], generated_by="m", replace=False, checked=True)
    assert ac.get(engine, key)["answer"] == "new checked"                     # a current checked one stays


def test_replay_of_a_legacy_row(engine):
    key = ac.cache_key(**FIELDS)
    _legacy_checked(engine, key, f"Old [1].\n\n{cn.CHECKED}")
    row = ac.get(engine, key)
    assert not cache.replay_allowed(checked=row["effective_checked"], regenerated=False, thorough=True)   # regenerated
    assert cache.replay_allowed(checked=row["effective_checked"], regenerated=False, thorough=False)
    assert generation.replay_text(row) == "Old [1]."                                 # the old claim of a check removed


def test_replay_keeps_warnings_and_current_notes(engine):
    assert generation._strip_check_status(f"A [1].\n\n{cn.DE[cn.CHECKED]}") == "A [1]."
    assert generation._strip_check_status(f"A [1].\n\n{cn.PARTLY}") == "A [1]."
    warn = "_⚠ Verification unavailable for this answer (RuntimeError)._"
    assert generation._strip_check_status(f"A [1].\n\n{warn}") == f"A [1].\n\n{warn}"
    key = ac.cache_key(**FIELDS)
    ac.put(engine, key, fields=FIELDS, answer=f"New [1].\n\n{cn.CHECKED}", citations=[], generated_by="m", checked=True)
    assert generation.replay_text(ac.get(engine, key)) == f"New [1].\n\n{cn.CHECKED}"   # current contract: unchanged


def test_the_column_is_added_once_even_by_concurrent_first_callers(tmp_path, monkeypatch):
    db = tmp_path / "c.db"
    with sqlite3.connect(db) as c:                                            # the table in an older form
        c.execute("CREATE TABLE memento_answer_cache (key VARCHAR(64) PRIMARY KEY, collection VARCHAR(200) NOT NULL, "
                  "source VARCHAR(500) NOT NULL, content_hash VARCHAR(128) NOT NULL, action VARCHAR(20) NOT NULL, "
                  "language VARCHAR(10) NOT NULL, model VARCHAR(200) NOT NULL, mode VARCHAR(20) NOT NULL, "
                  "prompt_version VARCHAR(32) NOT NULL, generated_by VARCHAR(200) NOT NULL, answer TEXT NOT NULL, "
                  "citations TEXT NOT NULL, status VARCHAR(10) NOT NULL, created_at TIMESTAMP, hits INTEGER NOT NULL, "
                  "last_hit_at TIMESTAMP, checked BOOLEAN NOT NULL DEFAULT 0)")
    e = create_engine(f"sqlite:///{db}")
    stale = [c["name"] for c in inspect(e).get_columns("memento_answer_cache")]
    # both callers read the columns before either adds one (the race): the second ALTER meets a duplicate column
    monkeypatch.setattr(ac, "_columns", lambda engine: set(stale))
    ac.ensure_table(e)
    ac.ensure_table(e)
    assert "check_contract" in {c["name"] for c in inspect(e).get_columns("memento_answer_cache")}


def test_other_migration_errors_propagate(monkeypatch):
    e = create_engine("sqlite://")
    ac.ensure_table(e)
    monkeypatch.setattr(ac, "_columns", lambda engine: set())                  # claims the table lacks its columns
    real = ac._alter

    def broken(engine, sql):
        raise RuntimeError("disk full")
    monkeypatch.setattr(ac, "_alter", broken)
    with pytest.raises(RuntimeError):
        ac.ensure_table(e)
    monkeypatch.setattr(ac, "_alter", real)
