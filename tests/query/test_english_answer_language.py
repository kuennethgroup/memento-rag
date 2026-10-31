"""An English answer's language is stated when a passage is in another language (many German sources can turn an English
answer German). An English question on English passages keeps exactly the prompt the RQ3 evaluation pins
(tests/retrieval/test_pipe_cli.py, tests/retrieval/test_rq3_answer_pipeline.py)."""
from openwebui.functions import query_pipe as qp
from memento.query import prompts, request, texts
from memento.query.flows import answer as answer_flow

EN = "The quality of the foam depends on the melt strength of the polymer and on the crystallization rate."
DE = "Die Expansion der Partikelschäume wird durch die geringe Schmelzefestigkeit und die langsame Kristallisation begrenzt."


def test_an_english_question_on_a_german_passage_states_english():
    text = texts.answer_language_addendum([EN, DE])
    assert text == texts.EN_ANSWER_ADDENDUM
    assert "write the answer in English" in text and "another language" in text and "titles" in text


def test_an_english_question_on_english_passages_keeps_the_prompt():
    assert texts.answer_language_addendum([EN, EN]) == ""
    assert texts.answer_language_addendum([]) == ""
    assert texts.answer_language_addendum() == ""


def test_a_passage_too_short_to_tell_does_not_add_the_line():
    assert texts.answer_language_addendum([EN, "Tab. 3: 1400 °C, 99 %"]) == ""


def test_a_german_question_states_german_whatever_the_passages():
    token = request.LANG.set("de")
    try:
        for passages in ([DE], [EN], None):
            text = texts.answer_language_addendum(passages)
            assert text == texts.DE_ANSWER_ADDENDUM and "English" not in text
    finally:
        request.LANG.reset(token)


def test_a_summary_or_list_line_already_states_english():
    # whole-document summaries have EN_SUMMARY_ADDENDUM, English list answers the list's own language line: once only
    assert texts.answer_language_addendum([DE], own_line=True) == ""
    import inspect
    src = inspect.getsource(answer_flow.build_prompt)
    assert "texts.answer_language_addendum([h.get(\"text\") or \"\" for h in turn.hits],\n" in src
    assert "own_line=turn.whole_doc or bool(list_add))" in src
    assert "if turn.whole_doc:\n        turn.system_prompt += texts.summary_language_addendum()" in src


def test_the_english_line_stays_out_of_the_stored_answers_key():
    # stored answers are keyed by the base prompt (cache.prompt_version → prompts.answer_prompt): no key changes
    assert texts.EN_ANSWER_ADDENDUM.strip() not in prompts.answer_prompt(None)
