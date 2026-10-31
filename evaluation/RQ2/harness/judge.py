"""Answer judging: absolute vs the reference, pairwise vs G0 in both orders, cross-family check."""
from __future__ import annotations

import re
import time

from evaluation.RQ2.questions.common import llm_json

ABS_SYSTEM = ("You grade an answer to a research question against a reference answer. Output ONLY JSON "
              "{\"score\": 1 | 0.5 | 0}: 1 if the answer states the reference's key facts (wording, language and "
              "extra correct detail may differ); 0.5 if it states part of them; 0 if it misses them, contradicts "
              "them or says the information is not available. For a list or count, 1 needs the same items or number.")
PAIR_SYSTEM = ("You compare two answers to a research question. If a reference answer is given, prefer the answer "
               "that states its facts more completely and correctly; otherwise prefer the more correct, complete and "
               "well-supported answer. Ignore length and style. Output ONLY JSON {\"winner\": \"A\"|\"B\"|\"tie\"}.")


_CITE = re.compile(r"\[(\d+)\]")


def _cited(answer, context):
    return [n for n in dict.fromkeys(int(m) for m in _CITE.findall(answer or "")) if 1 <= n <= len(context)]


def cited_docs(answer, context):
    """Documents the answer cites, in first-citation order ([N] → context[N-1].source)."""
    return list(dict.fromkeys(context[n - 1]["source"] for n in _cited(answer, context)))


def with_citations(answer, context):
    """Corpus-level judging: references list document ids, answers cite [N] (the production prompt
    forbids naming documents), so the judge gets the citation → document map."""
    ns = _cited(answer, context)
    if not ns:
        return answer
    return answer + "\n\nDocuments cited: " + "; ".join(f"[{n}] {context[n - 1]['source']}" for n in ns)


def judge_absolute(llm, model, question, reference, answer):
    if not (reference or "").strip():
        return None
    r = llm_json(llm, model, ABS_SYSTEM, f"Question: {question}\n\nReference answer: {reference}\n\nAnswer: {answer}")
    s = r.get("score") if isinstance(r, dict) else None
    return float(s) if s in (0, 0.5, 1) else None


def _pair_once(llm, model, question, reference, a, b):
    ref = f"Reference answer: {reference}\n\n" if (reference or "").strip() else ""
    r = llm_json(llm, model, PAIR_SYSTEM, f"Question: {question}\n\n{ref}Answer A: {a}\n\nAnswer B: {b}")
    w = r.get("winner") if isinstance(r, dict) else None
    return w if w in ("A", "B", "tie") else None


def judge_pair(llm, model, question, reference, ans_arm, ans_g0):
    if ans_arm.strip() == ans_g0.strip():
        return "tie"
    first = _pair_once(llm, model, question, reference, ans_arm, ans_g0)
    second = _pair_once(llm, model, question, reference, ans_g0, ans_arm)
    if first is None or second is None:
        return None
    if first == "A" and second == "B":
        return "win"
    if first == "B" and second == "A":
        return "loss"
    return "tie"


def cross_family(cloud, model, items, *, max_calls, min_gap_s=6.0, judge_fn=judge_absolute):
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("ga", Path(__file__).resolve().parents[1] / "graph_audit.py")
    ga = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ga)
    a, b, last, errors = [], [], 0.0, 0
    for it in items[:max_calls]:
        wait = min_gap_s - (time.time() - last)
        if wait > 0:
            time.sleep(wait)
        last = time.time()
        try:
            got = judge_fn(cloud, model, it["question"], it["reference"], it["answer"])
        except Exception:       # a cloud outage must not discard the local judgements
            errors += 1
            continue
        if got is not None and it["local"] is not None:
            a.append(str(it["local"])); b.append(str(got))
    return {"n": len(a), "errors": errors, "agreement": round(sum(x == y for x, y in zip(a, b)) / len(a), 4) if a else None,
            "kappa": round(ga.cohen_kappa(a, b), 4) if a else None}
