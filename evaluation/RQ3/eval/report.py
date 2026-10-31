"""RQ3 report: per group × arm metrics from judgements.jsonl + answers.jsonl, harm vs P1, cost, judge agreement.
  .venv/bin/python evaluation/RQ3/eval/report.py pilot"""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
run = sys.argv[1] if len(sys.argv) > 1 else "pilot"
ARMS = ("P0", "P1", "C", "A")


def jl(p):
    return [json.loads(l) for l in open(p)] if Path(p).exists() else []


def m(xs):
    xs = [x for x in xs if x is not None]
    return f"{np.mean(xs):.3f}" if xs else "–"


lines = [f"# RQ3 {run} report\n"]
for corpus in ("ceramics", "polymer-engineering"):
    d = ROOT / f"data/rq3/runs/{run}/{corpus}"
    J_all = {r["key"]: r for r in jl(d / "judgements.jsonl") if not r.get("error")}
    A = {r["key"]: r for r in jl(d / "answers.jsonl") if not r.get("error")}
    gen_err = {r["key"]: r["error"] for r in jl(d / "answers.jsonl") if r.get("error") and r["key"] not in A}
    # paired: only questions with a judgement for every arm, so arm means compare the same questions
    full = {q for q in {k.split("|")[0] for k in J_all} if all(f"{q}|{arm}" in J_all for arm in ARMS)}
    J = {k: r for k, r in J_all.items() if k.split("|")[0] in full}
    X = {r["key"]: r for r in jl(d / "crossjudge_qwen.jsonl") if not r.get("error")}
    if not J:
        continue
    dropped = sorted({k.split("|")[0] for k in J_all} - full) + sorted({k.split("|")[0] for k in gen_err} - set(
        k.split("|")[0] for k in J_all))
    lines.append(f"\n## {corpus} ({len(full)} questions with all arms judged, {len(J)} answers; "
                 f"left out: {', '.join(q.split(':', 1)[1] for q in dropped) or 'none'})\n")
    if gen_err:
        lines.append("Generation failures: " + "; ".join(f"{k.split(':', 1)[1]}: {e[:60]}" for k, e in sorted(gen_err.items())) + "\n")
    groups = sorted({r["group"] for r in J.values()})
    lines.append("| group | arm | n | faithfulness | correct (no contradiction) | completeness | abstention / doc precision "
                 "| count correct |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for g in groups:
        for arm in ARMS:
            rs = [r for r in J.values() if r["group"] == g and r["arm"] == arm]
            if not rs:
                continue
            extra = m([r.get("abstains") for r in rs]) if g == "unanswerable" else \
                (m([r.get("doc_precision") for r in rs]) if g == "list" else "")
            lines.append(f"| {g} | {arm} | {len(rs)} | {m([r.get('faithfulness') for r in rs])} | "
                         f"{m([r.get('corr_correct') for r in rs])} | {m([r.get('completeness') for r in rs])} | {extra} | "
                         f"{m([r.get('count_correct') for r in rs]) if g == 'list' else ''} |")
    # harm vs P1 (per answer): completeness lower or correctness lost
    lines.append("\n| arm vs P1 | n | completeness ↓ | completeness ↑ | correctness lost | correctness gained | faithfulness Δ (mean) |")
    lines.append("|---|---|---|---|---|---|---|")
    for arm in ("C", "A", "P0"):
        pairs = [(J[f"{q}|P1"], J[f"{q}|{arm}"]) for q in {k.split("|")[0] for k in J}
                 if f"{q}|P1" in J and f"{q}|{arm}" in J]
        if not pairs:
            continue
        cd = sum((b.get("completeness") or 0) < (a.get("completeness") or 0) for a, b in pairs)
        cu = sum((b.get("completeness") or 0) > (a.get("completeness") or 0) for a, b in pairs)
        cl = sum(a.get("corr_correct") is True and b.get("corr_correct") is False for a, b in pairs)
        cg = sum(a.get("corr_correct") is False and b.get("corr_correct") is True for a, b in pairs)
        fd = [b["faithfulness"] - a["faithfulness"] for a, b in pairs
              if a.get("faithfulness") is not None and b.get("faithfulness") is not None]
        lines.append(f"| {arm} | {len(pairs)} | {cd} | {cu} | {cl} | {cg} | {np.mean(fd):+.3f} |")
    # cost
    lines.append("\n| arm | time own (median / p90 s) | time incl. P1 (median s) | model calls (median) | rounds (A) | removed (C, median) |")
    lines.append("|---|---|---|---|---|---|")
    p1 = {r["qid"]: r for r in A.values() if r["arm"] == "P1"}
    for arm in ARMS:
        rs = [r for r in A.values() if r["arm"] == arm]
        if not rs:
            continue
        own = [r["latency_s"] for r in rs]
        tot = [r["latency_s"] + (p1[r["qid"]]["latency_s"] if arm in ("A", "C") and r["qid"] in p1 else 0) for r in rs]
        lines.append(f"| {arm} | {np.median(own):.0f} / {np.percentile(own, 90):.0f} | {np.median(tot):.0f} | "
                     f"{np.median([r.get('llm_calls', 0) for r in rs]):.0f} | "
                     f"{m([r.get('rounds') for r in rs]) if arm == 'A' else ''} | "
                     f"{np.median([len(r.get('removed') or []) for r in rs]) if arm == 'C' else ''} |")
    # self-verification leniency: qwen's verdicts on its own P1 answer vs the Gemma judge's faithfulness of that answer
    sv = []
    for r in A.values():
        v = r.get("p1_self_verdicts")
        j = J.get(f"{r['qid']}|P1")
        if r["arm"] == "C" and v and j and j.get("faithfulness") is not None:
            judged = [x for x in v.values() if x["label"] != "unjudged"]
            if judged:
                sv.append((sum(x["label"] == "supported" for x in judged) / len(judged), j["faithfulness"]))
    if sv:
        q, g = np.array([a for a, _ in sv]), np.array([b for _, b in sv])
        lines.append(f"\nSelf-verification (P1 answers, n={len(sv)}): qwen rates its own claims supported {q.mean():.3f} "
                     f"vs Gemma judge faithfulness {g.mean():.3f} (mean difference {np.mean(q - g):+.3f}; qwen higher in "
                     f"{int((q > g + 1e-9).sum())}, lower in {int((q < g - 1e-9).sum())} answers)")

    # judge agreement (qwen cross-judge)
    if X:
        both = [(J[k], X[k]) for k in X if k in J]
        agree_c = [abs((a.get("completeness") or 0) - (b.get("completeness") or 0)) < 0.5 for a, b in both
                   if a.get("completeness") is not None and b.get("completeness") is not None]
        agree_k = [a.get("corr_correct") == b.get("corr_correct") for a, b in both
                   if a.get("corr_correct") is not None and b.get("corr_correct") is not None]
        fdiff = [abs(a["faithfulness"] - b["faithfulness"]) for a, b in both
                 if a.get("faithfulness") is not None and b.get("faithfulness") is not None]
        lines.append(f"\nJudge agreement Gemma vs qwen on {len(both)} real answers: completeness "
                     f"{np.mean(agree_c):.2f} (n={len(agree_c)}), correctness {np.mean(agree_k):.2f} (n={len(agree_k)}), "
                     f"faithfulness mean |Δ| {np.mean(fdiff):.3f} (n={len(fdiff)})")
text = "\n".join(lines)
print(text)
(ROOT / f"data/rq3/runs/{run}/report.md").write_text(text + "\n")
