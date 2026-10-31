# The router names the part of a paper a question asks for (figure, table, equation), so a table request searches
# tables. Additive: the intents and the other fields are unchanged.
import asyncio
import json
import types

import pytest

from memento.query import router as qr


class _FakeLLM:
    def __init__(self, reply):
        self.reply, self.kwargs = reply, None
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    async def _create(self, **kw):
        self.kwargs = kw
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=self.reply))])


def _route(reply):
    return asyncio.run(qr.route(_FakeLLM(reply), "m", "q", []))


@pytest.mark.parametrize("element, want", [("table", "table"), ("figure", "figure"), ("equation", "equation"),
                                           ("none", "none"), ("TABLE", "table"), ("diagram", "none"), (None, "none"),
                                           (3, "none")])
def test_the_element_field_is_read_and_coerced(element, want):
    reply = {"intent": "ranked_retrieval", "query": "results table", "reasoning": "light", "suggest": False}
    if element is not None:
        reply["element"] = element
    r = _route(json.dumps(reply))
    assert r.element == want and r.intent == "ranked_retrieval" and r.query == "results table"


def test_a_reply_without_the_field_and_a_failed_call_mean_no_element():
    assert _route('{"intent": "image_request", "query": "pipeline figure", "reasoning": "light"}').element == "none"
    assert qr.RouterResponse(ok=False).element == "none"


def test_the_instructions_describe_the_content_and_keep_the_categories():
    p = qr.ROUTER_SYSTEM_PROMPT
    for t in ("tables", "equations", "figures", "charts", "lists"):
        assert t in p, t
    assert '"element"' in p and '"figure"' in p and '"table"' in p and '"equation"' in p
    for intent in qr._VALID_INTENTS:                     # every existing category still described
        assert f'"{intent}"' in p, intent
    assert '"element": "none"' in p                       # the JSON shape the model is told to output
