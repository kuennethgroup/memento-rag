"""RQ3 evaluation judge (Gemma 4 12B, blind to the arm). Scores one final answer against the frozen sources and the
references. Primary metrics: correctness, completeness, faithfulness; secondary: citation precision/recall
(list questions), abstention (unanswerables).

The judge splits the answer into claims itself (memento.agentic_validation.claims with the Gemma client — the system's own
split is never reused), checks faithfulness with the production verifier components driven by Gemma, and asks Gemma
for contradictions with the gold evidence and for completeness. List/count completeness is deterministic (cited
documents vs the gold document set).
"""
from __future__ import annotations

import json
import re

from memento.agentic_validation.jsonfix import loads_lenient
from memento.agentic_validation.claims import NUMBER_WORDS, Claim, clean_answer, markers, split_claims
from memento.agentic_validation.verifiers import LLMJudgeVerifier, VerdictCache, VLMFigureVerifier, verify_claims

CONTRA = """\
Question: {q}
{ref}
Gold evidence:
{gold}

Claims from an answer:
{claims}

Which claims CONTRADICT the gold evidence or the reference answer (a different value, unit, material, condition, \
direction or attribution)? A claim the gold evidence does not mention is NOT a contradiction. A statement about \
what the answer's own sources do or do not contain ("the sources do not mention X", "no value is given") is NOT a \
contradiction either — a missed fact counts against completeness, not correctness. \
Output ONLY JSON: {{"contradicted": [<claim numbers>], "why": "short"}}"""

HAS_REF = """\
Question: {q}
Reference answer: {ref}

Answer to grade:
{a}

Does the answer give the reference answer (same value/entity; wording and notation may differ)? \
Output ONLY JSON: {{"grade": "yes" | "partial" | "no", "why": "short"}}"""

NUGGETS = """\
Question: {q}

Answer to grade:
{a}

Key facts (nuggets):
{n}

For each nugget: does the answer state it (same meaning; wording may differ; values must match)? \
Output ONLY JSON: [{{"id": 1, "present": true}}]"""

ABSTAIN = """\
Question: {q}

Answer to grade:
{a}

The document collection does NOT contain the answer to this question. Classify the answer:
- "declines": it says the information is not available / not reported in the sources (it may mention related \
information, as long as it does not present a specific answer to the question as fact).
- "answers": it presents a specific answer to the question (a value, name, result) as if the sources gave it.
Output ONLY JSON: {{"label": "declines" | "answers", "why": "short"}}"""


def stated_counts(answer: str) -> set[int]:
    """Numbers the answer states as digits or number words (two–twenty), citation markers removed."""
    t = re.sub(r"\[\d[\d,\s–-]*\]", " ", clean_answer(answer))
    nums = {int(x) for x in re.findall(r"(?<![\d.,])(\d{1,3})(?![\d.,]\d)", t)}
    return nums | {NUMBER_WORDS[w.lower()] for w in re.findall(r"[A-Za-zäöüÄÖÜß]+", t) if w.lower() in NUMBER_WORDS}


def _json(raw: str, kind="obj"):
    m = re.search(r"\{.*\}" if kind == "obj" else r"\[.*\]", raw or "", re.S)
    return loads_lenient(m.group(0)) if m else None


class Judge:
    def __init__(self, llm, model="gemma-4-12b"):
        self.llm, self.model = llm, model
        self.text_v = LLMJudgeVerifier(llm, model, workers=1)
        self.fig_v = VLMFigureVerifier(llm, model)

    def ask(self, prompt, kind="obj", max_tokens=600):
        r = self.llm.chat.completions.create(model=self.model, temperature=0.0, max_tokens=max_tokens,
                                             messages=[{"role": "user", "content": prompt}])
        return _json(r.choices[0].message.content or "", kind)

    # ── claims + faithfulness ─────────────────────────────────────────────────
    def claims(self, answer: str) -> list[Claim]:
        return split_claims(self.llm, self.model, answer)

    def faithfulness(self, claims, hits):
        v = verify_claims(claims, hits, text_verifier=self.text_v, figure_verifier=self.fig_v, cache=VerdictCache())
        judged = [x for x in v.values() if x.label != "unjudged"]
        sup = sum(x.label == "supported" for x in judged)
        return {"faithfulness": sup / len(judged) if judged else None, "n_claims": len(claims),
                "unjudged": len(v) - len(judged), "verdicts": {k: x.label for k, x in v.items()}}

    # ── correctness ───────────────────────────────────────────────────────────
    def correctness(self, q, claims, gold_texts, reference=None):
        if not claims or not gold_texts:
            return {"contradicted": [], "correct": None if not gold_texts else True}
        gold = "\n\n".join(f"[{i + 1}] {t[:2500]}" for i, t in enumerate(gold_texts[:8]))
        listing = "\n".join(f"{i + 1}. {c.text}" for i, c in enumerate(claims))
        ref = f"Reference answer: {reference}\n" if reference else ""
        d = self.ask(CONTRA.format(q=q, ref=ref, gold=gold, claims=listing)) or {}
        bad = [int(x) for x in d.get("contradicted", []) if str(x).isdigit() and 1 <= int(x) <= len(claims)]
        return {"contradicted": bad, "correct": not bad, "claim_error_rate": len(bad) / len(claims),
                "why": d.get("why", "")}

    # ── completeness ──────────────────────────────────────────────────────────
    def completeness_reference(self, q, answer, reference):
        d = self.ask(HAS_REF.format(q=q, ref=reference, a=clean_answer(answer))) or {}
        g = str(d.get("grade", "")).lower()
        return {"completeness": {"yes": 1.0, "partial": 0.5, "no": 0.0}.get(g), "grade": g, "why": d.get("why", "")}

    def completeness_nuggets(self, q, answer, nuggets):
        if not nuggets:
            return {"completeness": None}
        listing = "\n".join(f"{i + 1}. {n['text']}" for i, n in enumerate(nuggets))
        d = self.ask(NUGGETS.format(q=q, a=clean_answer(answer), n=listing), kind="list") or []
        present = {int(x["id"]): bool(x.get("present")) for x in d if isinstance(x, dict) and str(x.get("id")).isdigit()}
        vital = [i + 1 for i, n in enumerate(nuggets) if n.get("importance") == "vital"] or \
                list(range(1, len(nuggets) + 1))
        return {"completeness": sum(present.get(i, False) for i in vital) / len(vital),
                "all_nuggets": sum(present.get(i + 1, False) for i in range(len(nuggets))) / len(nuggets),
                "present": present}

    @staticmethod
    def completeness_docs(answer, hits, gold_docs, subtype=None, gold_count=None):
        cited = {hits[n - 1].get("source") for n in markers(clean_answer(answer)) if 1 <= n <= len(hits)}
        gold = {d for d in (gold_docs or []) if d and d.strip()}
        rec = len(cited & gold) / len(gold) if gold else None
        prec = len(cited & gold) / len(cited) if cited else None
        out = {"completeness": rec, "doc_precision": prec, "cited_docs": sorted(cited)}
        if subtype == "corpus_count" and gold_count is not None:
            out["count_correct"] = gold_count in stated_counts(answer)
        return out

    def abstention(self, q, answer):
        d = self.ask(ABSTAIN.format(q=q, a=clean_answer(answer))) or {}
        label = str(d.get("label", "")).strip().lower()
        return {"abstains": True if label == "declines" else (False if label == "answers" else None),
                "why": d.get("why", "")}
