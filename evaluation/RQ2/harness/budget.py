"""The one merge rule for every second-stream arm: keep G0's top 20−K, fill K slots
from the stream (already reranked), drop duplicates, and stay within G0's token budget with any block
capped at BLOCK_SHARE of it."""
from __future__ import annotations

import tiktoken

_ENC = tiktoken.get_encoding("cl100k_base")
BLOCK_SHARE = 0.25


def n_tokens(text: str) -> int:
    return len(_ENC.encode(text or ""))


def hit_tokens(hits) -> int:
    return sum(n_tokens(h.get("text", "")) for h in hits)


def _key(h):
    return h.get("chunk_id") or (h.get("source"), h.get("page"), (h.get("text") or "")[:80])


def _truncate(text: str, max_tokens: int) -> str:
    ids = _ENC.encode(text or "")
    return text if len(ids) <= max_tokens else _ENC.decode(ids[:max(0, max_tokens)])


def fit_budget(hits, block, budget):
    """Cap the block at BLOCK_SHARE of the budget, then drop the lowest-ranked hits until it fits."""
    block = _truncate(block, int(BLOCK_SHARE * budget))
    hits = list(hits)
    while hits and hit_tokens(hits) + n_tokens(block) > budget:
        hits.pop()
    return hits, block


def merge_streams(g0_hits, stream_hits, *, k_slots, block="", origin):
    """The stream replaces K/20 of G0's list (at least one chunk) from the bottom; the result never
    exceeds G0's token budget. Trimming drops G0's lowest chunks while at least half of the kept G0
    chunks remain, then the stream's lowest chunks, then G0 again (so a small G0 is never fully
    replaced and the context never grows)."""
    from evaluation.RQ2.questions.pooling import RERANK_TOP
    budget = hit_tokens(g0_hits)
    k_eff = min(k_slots, max(1, round(len(g0_hits) * k_slots / RERANK_TOP))) if g0_hits else 0
    seen, new = {_key(h) for h in g0_hits}, []
    for h in stream_hits:
        k = _key(h)
        if k not in seen and len(new) < k_eff:
            seen.add(k)
            new.append(dict(h, origin=origin))
    keep = [dict(h, origin="g0") for h in g0_hits[:max(0, len(g0_hits) - len(new))]]
    block = _truncate(block, int(BLOCK_SHARE * budget))
    floor = (len(keep) + 1) // 2
    while keep and len(keep) > floor and hit_tokens(keep + new) + n_tokens(block) > budget:
        keep.pop()
    while new and hit_tokens(keep + new) + n_tokens(block) > budget:
        new.pop()
    while keep and hit_tokens(keep + new) + n_tokens(block) > budget:
        keep.pop()
    return keep + new, block
