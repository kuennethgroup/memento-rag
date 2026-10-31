"""Pooled gold evidence (never from the graph; all arms pool). Fact-level judging:
the judge says which of the given facts each candidate page states."""
from __future__ import annotations

import base64
import json
import re
from pathlib import Path

FACT_JUDGE_SYSTEM = (
    "You check which facts a document page states. For each numbered candidate page you get its text (and "
    "sometimes its figure). Return, for every candidate, the keys of the listed FACTS that the page itself "
    "states (same information; wording and language may differ). A page that only mentions the topic without "
    "stating the fact does not count. Output ONLY a JSON array: [{\"ref\": \"c0\", \"facts\": [\"a\"]}, ...] "
    "with one object per candidate, an empty list when it states none.")


# G0 = RQ1 production retrieval: the query pipe's Valve values (the same constants
# evaluation/RQ1/auto_annotate.py used); a test pins them to query_pipe.py.
RETRIEVE_K = 80
RERANK_TOP = 20
RERANK_MIN_SCORE = 0.03
IMAGE_RERANK_RATIO = 0.2


def g0_candidates(client, collection, question, allowed, *, k=20):
    from memento.config.settings import DOMAIN, EMBEDDER_URL, RERANKER_URL
    from memento.config.settings import HYBRID_ENABLED, HYBRID_PREFETCH_K
    from memento.retrieval.search import search
    QUERY_INSTRUCTION = DOMAIN["instruction"]
    hits, _ = search(client, collection, question, embedder_url=EMBEDDER_URL, query_instruction=QUERY_INSTRUCTION,
                     retrieve_k=RETRIEVE_K, hybrid=HYBRID_ENABLED, hybrid_prefetch_k=HYBRID_PREFETCH_K,
                     rerank_enabled=True, reranker_url=RERANKER_URL, rerank_top=RERANK_TOP,
                     rerank_min_score=RERANK_MIN_SCORE, image_rerank_ratio=IMAGE_RERANK_RATIO)
    out = [h for h in hits if allowed is None or h.get("source") in allowed]
    return out[:k]


def _key(c):
    return (c["source"], c.get("page", 0))


def _parts(facts, batch):
    parts = [{"type": "text", "text": "FACTS:\n" + "\n".join(f"{k}: {v}" for k, v in facts.items()) + "\n"}]
    for i, c in enumerate(batch):
        parts.append({"type": "text", "text": f"\n--- candidate c{i} --- ({c['source']}, page {c.get('page', 0) + 1})\n"
                                              f"{(c.get('text') or '')[:1500]}\n"})
        img = c.get("img_path")
        if c.get("type") in ("image", "chart") and img and Path(img).is_file():
            data = base64.b64encode(Path(img).read_bytes()).decode()
            parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}})
    return parts


def judge_facts(client, model, facts, candidates, *, batch=8):
    labels, unparsed = {}, []
    for s in range(0, len(candidates), batch):
        chunk = candidates[s:s + batch]
        r = client.chat.completions.create(
            model=model, temperature=0, max_tokens=2000,
            messages=[{"role": "system", "content": FACT_JUDGE_SYSTEM},
                      {"role": "user", "content": _parts(facts, chunk)}],
            extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        raw = r.choices[0].message.content or ""
        m = re.search(r"\[.*\]", raw, re.S)
        try:
            rows = json.loads(m.group(0)) if m else []
        except json.JSONDecodeError:
            rows = []
        got = {}
        for row in rows if isinstance(rows, list) else []:
            if isinstance(row, dict) and isinstance(row.get("facts"), list):
                got[row.get("ref")] = {f for f in row["facts"] if f in facts}
        for i, c in enumerate(chunk):
            if f"c{i}" in got:
                if got[f"c{i}"]:
                    labels[_key(c)] = got[f"c{i}"]
            else:
                unparsed.append(_key(c))
    return {"labels": labels, "unparsed": unparsed}


def pool_and_judge(client, model, facts, candidate_lists, known=None):
    found_by, first = {}, {}
    for name, lst in candidate_lists.items():
        for c in lst:
            k = _key(c)
            found_by.setdefault(k, [])
            if name not in found_by[k]:
                found_by[k].append(name)
            first.setdefault(k, c)
    judged_before = {}
    if known:
        judged_before = {(g["source"], g["page"]): set(g["fact"].split("+")) for g in known["gold"]}
        done = set(judged_before) | {tuple(u) for u in known.get("unparsed", [])} | set(known.get("judged", []))
    else:
        done = set()
    todo = [first[k] for k in first if k not in done]
    res = judge_facts(client, model, facts, todo) if todo else {"labels": {}, "unparsed": []}
    labels = judged_before | res["labels"]
    gold = [{"source": k[0], "page": k[1], "fact": "+".join(sorted(v)), "found_by": found_by.get(k, [])}
            for k, v in sorted(labels.items())]
    judged = sorted(set(first) | set(done))
    return {"gold": gold, "unparsed": res["unparsed"], "n_candidates": len(first), "judged": judged}
