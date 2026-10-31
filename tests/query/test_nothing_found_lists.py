# The "nothing found" check (unchanged for list questions, measured) and the router keeping every part of a message, not
# only the topic it rewrote.
import openwebui.functions.query_pipe as qp
from memento.query import router as qr
from memento.query.flows import lists

ARGS = dict(rerank_enabled=True, low=0.1, dc=False, paper=None, named_docs=[])


def test_list_questions_still_go_to_the_model():
    # measured: real list questions score near the 0.1 mark, so a refusal there would hit them; they keep the model
    # (LIST_ADDENDUM: honest wording)
    assert lists.nothing_relevant([0.03, 0.01], "set_query", **ARGS) is False
    assert lists.nothing_relevant([0.03, 0.01], "ranked_retrieval", **ARGS) is True        # as before
    assert lists.nothing_relevant([0.4, 0.01], "ranked_retrieval", **ARGS) is False
    assert lists.nothing_relevant([0.03], "gap_query", **ARGS) is False                    # other routes: as before
    assert lists.nothing_relevant([0.03], "ranked_retrieval", **{**ARGS, "dc": True}) is False   # Document Chat: never
    assert lists.nothing_relevant([0.03], "ranked_retrieval", **{**ARGS, "paper": "PolyBERT"}) is False
    assert lists.nothing_relevant([], "ranked_retrieval", **ARGS) is False                  # no scores: no judgement


def test_the_router_keeps_every_part_of_the_message():
    p = qr.ROUTER_SYSTEM_PROMPT
    assert "Keep every part of the message" in p
    assert "AI in rubber compounding" in p
