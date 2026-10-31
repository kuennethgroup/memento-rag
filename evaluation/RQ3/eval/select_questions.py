"""RQ3 step 4.1 — the question set per corpus (frozen before any arm runs).
multihop: all RQ2 multi-hop questions · list: 20 RQ2 corpus questions (7 list, 7 count, 6 compare) · control: 30 RQ2
control questions (= RQ1 GT v4) stratified by RQ1 target type. Unanswerables are added by build_unanswerable.py."""
import json, random, sys
from collections import defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
GT = {"ceramics": "evaluation/RQ1/ground_truth_auto_v4_qwenjudge.json",
      "polymer-engineering": "evaluation/RQ1/polymer_ground_truth_auto_v4_qwenjudge.json"}
rng = random.Random(3)
for corpus in ("ceramics", "polymer-engineering"):
    gold = [json.loads(l) for l in open(ROOT / f"data/rq2/runs/final/{corpus}/gold.jsonl") if "qid" in json.loads(l)]
    rq1 = {q["query"]: q for q in json.load(open(ROOT / GT[corpus]))["queries"]}
    out = [dict(qid=g["qid"], group="multihop", question=g["question"], lang=g.get("lang") or "en",
                reference=g.get("reference_answer"), gold_pages=g.get("gold_pages"), gold_docs=None)
           for g in gold if g["qtype"] == "multihop"]
    for t, n in (("corpus_list", 7), ("corpus_count", 7), ("corpus_compare", 6)):
        pool = [g for g in gold if g["qtype"] == t]; rng.shuffle(pool)
        out += [dict(qid=g["qid"], group="list", subtype=t, question=g["question"], lang=g.get("lang") or "en",
                     reference=g.get("reference_answer"), gold_pages=None, gold_docs=g.get("gold_docs"))
                for g in pool[:n]]
    by_type = defaultdict(list)
    for g in gold:
        if g["qtype"] == "control":
            by_type[(rq1.get(g["question"]) or {}).get("target_type", "?")].append(g)
    types = sorted(by_type); picked = []
    while len(picked) < 30 and any(by_type.values()):           # round-robin over target types
        for t in types:
            if by_type[t] and len(picked) < 30:
                g = by_type[t].pop(rng.randrange(len(by_type[t]))); picked.append((t, g))
    out += [dict(qid=g["qid"], group="control", target_type=t, question=g["question"], lang="en",
                 reference=None, gold_pages=g.get("gold_pages"), gold_docs=None) for t, g in picked]
    p = ROOT / f"data/rq3/eval/questions_{corpus}.jsonl"
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out))
    from collections import Counter
    print(corpus, len(out), Counter(r["group"] for r in out), "control types:", Counter(t for t, _ in picked))
