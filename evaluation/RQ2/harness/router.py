"""Routing policies: the routing evaluation and its robustness checks."""
from __future__ import annotations

import json

from memento.query.router import ROUTER_SYSTEM_PROMPT, RouterResponse

POLICIES = ("true", "real", "allgraph")
_MULTI = {"g1", "g2", "g3", "d2", "g2l", "g2lg"}
_CORPUS = {"g1", "g2", "g3", "lx", "g2l", "g2lg", "fs"}      # g2l/g2lg: G2 on the no-LLM graphs; fs: fact sheets


def route_true(qtype):
    return "multihop" if qtype == "multihop" else "corpus" if qtype.startswith("corpus") else "single"


def _classify_strict(llm, model, question):
    """The production router's prompt and schema, but failures raise: classify_and_rewrite_sync falls
    back to ranked_retrieval on any error, which would silently route every question to G0."""
    r = llm.chat.completions.create(
        model=model, temperature=0.0, max_tokens=256,
        messages=[{"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                  {"role": "user", "content": f"Latest message: {question}"}],
        extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    raw = (r.choices[0].message.content or "").strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return RouterResponse.model_validate(json.loads(raw)).intent


def route_real(llm, model, question):
    for attempt in range(2):                  # one retry, then loud
        try:
            intent = _classify_strict(llm, model, question)
            break
        except Exception:
            if attempt == 1:
                raise
    return {"set_query": "corpus", "corpus_listing": "corpus", "multi_hop": "multihop"}.get(intent, "single")


def route_for(policy, qtype, real_route):
    if policy == "real":
        return real_route
    r = route_true(qtype)
    return "multihop" if policy == "allgraph" and r == "single" else r


def effective_arm(arm, route):
    if arm == "g0" or route == "single":
        return ("g0", "rag")
    if route == "multihop":
        return (arm, "multihop") if arm in _MULTI else ("g0", "rag")
    return (arm, "corpus") if arm in _CORPUS else ("g0", "rag")
