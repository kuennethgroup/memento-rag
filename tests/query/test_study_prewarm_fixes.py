# Pre-warm fixes: picker summaries written in the question's language, the warm-up names a document as the picker does,
# and --sources takes "|" for document names with commas.
import asyncio
import sys
from pathlib import Path
from memento.config.paths import PROJECT_ROOT
from _owui_loader import load_function
from memento.query import library_state, prompts, request, service, texts, workers
from memento.query.scope import documents

ROOT = PROJECT_ROOT
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
qp = load_function(ROOT / "openwebui/functions/query_pipe.py", "memento_query")
from memento.library import documents as db  # noqa: E402
from memento.library import warmup_queue as wq  # noqa: E402
from memento.query.flows import answer as answer_flow


# ── a summary's language is stated for English questions too ──
# Otherwise the summary of a German document can come back in German for the English picker question and is not stored.
# Summaries only: an ordinary answer keeps the prompt the RQ3 evaluation pins (memento/retrieval/answer_pipeline.py; tests
# test_pipe_cli / test_rq3_answer_pipeline); it states English only when a passage is German
# (tests/query/test_english_answer_language.py).

def test_an_english_summary_question_states_english():
    text = texts.summary_language_addendum()                # an English question (the default)
    assert "write the summary in English" in text
    assert "another language" in text                     # a German document does not pull the summary into German
    assert "titles" in text                               # document titles and quotations stay as written


def test_an_ordinary_english_answer_keeps_the_prompt_the_evaluation_pins():
    import inspect
    en = "The density of the samples was measured with the Archimedes method and was higher than 98 % for all of them."
    assert texts.answer_language_addendum() == "" and texts.answer_language_addendum([en, en]) == ""
    src = inspect.getsource(answer_flow.build_prompt)
    assert "if turn.whole_doc:\n        turn.system_prompt += texts.summary_language_addendum()" in src


def test_a_german_question_still_states_german():
    token = request.LANG.set("de")
    try:
        text, summary = texts.answer_language_addendum(), texts.summary_language_addendum()
    finally:
        request.LANG.reset(token)
    assert "write the answer in German" in text and "Bibliothek" in text and "English" not in text
    assert summary == ""                                   # the German line covers a German summary


def test_the_language_line_stays_out_of_the_stored_answers_key():
    # a summary's key is built from the base prompt and the summary addendum (cache.summary_prompt_version): with the
    # language line outside them, every stored answer of both libraries keeps its key
    assert texts.EN_SUMMARY_ADDENDUM.strip() not in prompts.answer_prompt(None)
    assert texts.DE_ANSWER_ADDENDUM.strip() not in prompts.answer_prompt(None)


# ── the warm-up names a document as the picker does ──
# The picker sends the source when two documents of the library share a title (loader.js `shared`); a title the pipe
# cannot resolve (two matches) reports no result.

class StubPipe:
    """The copy's answer path (service.handle) as its worker calls it."""
    def __init__(self):
        self.calls = []

    async def handle(self, body, user, emit, files, metadata, task, **k):
        self.calls.append((body, metadata))
        metadata["warmup_result"].update(result="stored", reason="", key="k")
        yield "an answer"


def _library(e, collection, docs):
    db.ensure_tables(e)
    for source, title in docs:
        db.add_document(e, collection=collection, entry={"source": source, "title": title}, content_hash="h-" + source,
                        uploaded_by="u", output_root=None)


def test_the_warmup_names_a_document_as_the_picker_does():
    e = qp._db_engine()
    _library(e, "col", [("talk", "Same title"), ("thesis", "Same title"), ("paper", "Own title")])
    assert workers.warmup_name(e, "col", db.get_document(e, "col", "talk")) == "talk"
    assert workers.warmup_name(e, "col", db.get_document(e, "col", "thesis")) == "thesis"
    assert workers.warmup_name(e, "col", db.get_document(e, "col", "paper")) == "Own title"
    _library(e, "other", [("x", "Own title")])            # another library's titles do not count
    assert workers.warmup_name(e, "col", db.get_document(e, "col", "paper")) == "Own title"


def test_a_warmup_job_for_a_shared_title_asks_by_source(monkeypatch):
    e = qp._db_engine()
    _library(e, "col", [("talk", "Same title"), ("thesis", "Same title")])
    wq.enqueue(e, "col", "talk", ["summary"])
    monkeypatch.setattr(workers, "vllm_metrics", lambda url=None, key_env=None: None)
    monkeypatch.setattr(library_state, "in_progress_sources", lambda *a: set())
    pipe = StubPipe()
    pipe.valves = qp.Pipe.Valves(COLLECTION="col")
    monkeypatch.setattr(service, "handle", pipe.handle)
    assert asyncio.run(workers.warmup_step(pipe.valves, e, function_id="memento_query", db_engine=qp._db_engine)) == "done"
    assert pipe.calls[0][0]["messages"][-1]["content"] == 'Summarize the document "talk"'


def test_the_picker_question_by_source_resolves_to_that_document():
    # the live path the warm-up uses: a name that is a source resolves before any title is compared
    assert documents.picked_document('Summarize the document "talk"', {"talk", "thesis"}, "col") == ("talk", "summary")


# ── a stored "no gaps" answer replays ──
# The check passes before the pipe adds its "_Answered by …_" line, so the replay check skips the pipe's own lines, as
# the calm layout's branch already does.

ANSWERED = "\n\n_Answered by Qwen 3.8 Flash-Next (Local)._"


def test_a_stored_no_gap_answer_replays_with_the_pipes_own_line():
    from memento.research_gaps.passages import stored_gap_ok
    assert stored_gap_ok("No genuine research gaps are stated in the provided passages." + ANSWERED)


def test_leaked_reasoning_still_fails_next_to_the_pipes_own_line():
    from memento.research_gaps.passages import stored_gap_ok
    assert not stored_gap_ok("Could [9] be a limitation? Let's think about it." + ANSWERED)
    assert not stored_gap_ok("1. Some point [1]\nWait, maybe the second one is not a gap." + ANSWERED)


def test_an_empty_stored_gap_answer_never_replays():
    # once the pipe's own lines are skipped, an answer with nothing else in it must not pass (gap_answer_clean takes
    # empty text as clean); with the Thorough check off an empty answer could be stored
    from memento.research_gaps.passages import stored_gap_ok
    assert not stored_gap_ok(ANSWERED.strip())
    assert not stored_gap_ok("")
