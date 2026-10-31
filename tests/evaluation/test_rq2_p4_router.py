# tests/evaluation/test_rq2_p4_router.py
import json
import sys

sys.argv = [sys.argv[0], "--domain", "ceramics"]
from memento.config.cli_domain import apply_domain_arg  # noqa: E402
apply_domain_arg()
from evaluation.RQ2.harness.router import effective_arm, route_real, route_true  # noqa: E402
from memento.query.router import ROUTER_SYSTEM_PROMPT, RouterResponse  # noqa: E402


def test_router_knows_multi_hop():
    assert RouterResponse(intent="multi_hop").intent == "multi_hop"
    assert '"multi_hop"' in ROUTER_SYSTEM_PROMPT


class LLM:
    def __init__(self, intent):
        self.intent, self.chat, self.completions = intent, self, self

    def create(self, **kw):
        msg = type("M", (), {"content": json.dumps({"intent": self.intent, "query": "q"})})
        return type("R", (), {"choices": [type("C", (), {"message": msg})]})


def test_route_real_maps_intents():
    assert route_real(LLM("set_query"), "m", "q") == "corpus"
    assert route_real(LLM("corpus_listing"), "m", "q") == "corpus"
    assert route_real(LLM("multi_hop"), "m", "q") == "multihop"
    assert route_real(LLM("image_request"), "m", "q") == "single"


def test_effective_arm_routes_by_design():
    assert route_true("corpus_count") == "corpus" and route_true("control") == "single"
    assert effective_arm("g2", "multihop") == ("g2", "multihop")
    assert effective_arm("g2", "corpus") == ("g2", "corpus")
    assert effective_arm("g2", "single") == ("g0", "rag")
    assert effective_arm("d2", "corpus") == ("g0", "rag") and effective_arm("lx", "multihop") == ("g0", "rag")


def test_pipe_treats_multi_hop_as_ranked_retrieval():
    # query_pipe gates RQ3 validation on intent == "ranked_retrieval"; the new label must not change production.
    from memento.query.router import pipe_intent
    assert pipe_intent("multi_hop") == "ranked_retrieval"
    assert pipe_intent("set_query") == "set_query" and pipe_intent("ranked_retrieval") == "ranked_retrieval"
    src = open("memento/query/service.py").read()                      # the router step
    assert "query_intent = routed.query, pipe_intent(routed.intent)" in src


class Boom:
    def __init__(self):
        self.chat = self.completions = self
        self.calls = 0

    def create(self, **kw):
        self.calls += 1
        raise RuntimeError("server down")


def test_route_real_raises_instead_of_silently_routing_single():
    import pytest
    b = Boom()
    with pytest.raises(RuntimeError):
        route_real(b, "m", "q")
    assert b.calls == 2                       # one retry, then loud
