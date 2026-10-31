# The router says whether a question is about named documents or the whole library (a method's name can also be a
# document's short name). Additive: the intents and the other fields are unchanged.
import asyncio
import json
import types

import pytest

from memento.query import router as qr


class _FakeLLM:
    def __init__(self, reply):
        self.reply = reply
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    async def _create(self, **kw):
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=self.reply))])


def _route(reply):
    return asyncio.run(qr.route(_FakeLLM(reply), "m", "q", []))


@pytest.mark.parametrize("scope, want", [("document", "document"), ("documents", "documents"), ("library", "library"),
                                         ("none", "none"), ("Library", "library"), ("corpus", "none"), (None, "none"),
                                         (1, "none")])
def test_the_scope_field_is_read_and_coerced(scope, want):
    reply = {"intent": "set_query", "query": "papers that use polyBERT", "reasoning": "deep", "suggest": False,
             "element": "none"}
    if scope is not None:
        reply["scope"] = scope
    r = _route(json.dumps(reply))
    assert r.scope == want and r.intent == "set_query" and r.query == "papers that use polyBERT"


def test_a_reply_without_the_field_and_a_failed_call_mean_no_scope():
    assert _route('{"intent": "ranked_retrieval", "query": "x", "reasoning": "light"}').scope == "none"
    assert qr.RouterResponse(ok=False).scope == "none"


def test_the_instructions_describe_scope_and_keep_the_categories():
    p = qr.ROUTER_SYSTEM_PROMPT
    assert '"scope"' in p and '"document"' in p and '"documents"' in p and '"library"' in p
    assert '"scope": "none"' in p                          # the JSON shape the model is told to output
    assert "polyBERT" in p                                 # the case of a method that is also a document's name
    for intent in qr._VALID_INTENTS:                       # every existing category still described
        assert f'"{intent}"' in p, intent
    assert '"element": "none"' in p
