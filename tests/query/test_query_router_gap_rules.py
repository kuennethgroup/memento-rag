# Two router rules, additive (the categories and the JSON fields are unchanged): a how-to question is not a gap report,
# and a question about the previous gap answer is not a new gap answer.
from memento.query import router as qr


def test_how_to_solve_a_task_is_a_content_question():
    p = qr.ROUTER_SYSTEM_PROMPT
    assert "how would you solve the fingerprinting of a copolymer?" in p
    assert "wie würdest du" in p.lower()
    rule = p[p.index("how would you solve the fingerprinting of a copolymer?") - 400:]
    assert '"ranked_retrieval"' in rule[:900] and "NOT gap_query" in rule[:900]
    assert '"how could this work be extended?"' in p                      # extending a study: still gap_query


def test_a_question_about_the_previous_gap_answer_is_not_a_new_gap_answer():
    p = qr.ROUTER_SYSTEM_PROMPT
    assert "those in the last section are not limitations, are they?" in p
    rule = p[p.index("those in the last section are not limitations, are they?") - 500:]
    assert '"ranked_retrieval"' in rule[:1000] and "previous research-gap answer" in rule[:1000]


def test_categories_and_json_shape_unchanged():
    p = qr.ROUTER_SYSTEM_PROMPT
    for intent in qr._VALID_INTENTS:
        assert f'"{intent}"' in p, intent
    assert '"element": "none"' in p and '"scope": "none"' in p
