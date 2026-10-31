"""RQ3 evaluation runner (pilot and full run).

  gen:    per question, from the frozen routing/sources: P0, P1 (production prompt / strict citation, production thinking
          decision), then C and A from the saved P1 answer (qwen verifier + checks, figure rule A, A re-retrieves with
          the production search). Checkpointed per (qid, arm).
  judge:  Gemma (blind) per answer: claims (own split), faithfulness vs the sources the arm used, correctness vs gold
          evidence, completeness (multi-hop: reference · control: vital nuggets · list: cited gold documents),
          abstention (unanswerable). A qwen cross-judge on a sample for agreement.
  report: per group × arm tables.

  .venv/bin/python evaluation/RQ3/eval/run_eval.py --corpus ceramics --run pilot --stage gen
"""
import argparse, json, os, random, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).parent))
for l in (ROOT / "config/.env").read_text().splitlines():
    l = l.strip()
    if l and not l.startswith("#") and "=" in l:
        k, v = l.split("=", 1); os.environ.setdefault(k, v.strip().strip('"').strip("'"))
import yaml
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue
from memento.agentic_validation.strategies import ArmContext, run_strategy
from memento.agentic_validation.verifiers import LLMJudgeVerifier, VerdictCache, VLMFigureVerifier
from memento.retrieval.search import search
from judge import Judge

QWEN_URL, QWEN = "http://localhost:8090/v1", "qwen3.8-27b"
COLL = {"ceramics": "memento-ceramics-rag", "polymer-engineering": "memento-polymer-rag"}
PILOT = {"multihop": 8, "list": 4, "control": 5, "unanswerable": 3}
ARMS = ("P0", "P1", "C", "A")


def jl(path):
    return [json.loads(l) for l in open(path)] if Path(path).exists() else []


def questions(corpus, run):
    frozen = {r["qid"]: r for r in jl(ROOT / f"data/rq3/eval/frozen_{corpus}.jsonl") if not r["excluded"]}
    meta = {q["qid"]: q for f in (f"questions_{corpus}.jsonl", f"unanswerable_{corpus}.jsonl")
            for q in jl(ROOT / f"data/rq3/eval/{f}")}
    qs = [{**meta[q], **frozen[q]} for q in frozen if q in meta]
    if run.startswith("pilot"):          # pilot, pilot2, …: the same seeded sample
        rng, out = random.Random(27), []
        for g, n in PILOT.items():
            pool = sorted((q for q in qs if q["group"] == g), key=lambda q: q["qid"]); rng.shuffle(pool)
            out += pool[:n]
        (ROOT / f"data/rq3/eval/pilot_qids_{corpus}.json").write_text(json.dumps([q["qid"] for q in out], indent=1))
        return out
    return qs


def outdir(corpus, run):
    d = ROOT / f"data/rq3/runs/{run}/{corpus}"; d.mkdir(parents=True, exist_ok=True); return d


# ── generation ────────────────────────────────────────────────────────────────
def make_retrieve(corpus, dom):
    client = QdrantClient(url="http://localhost:6333", timeout=60)

    def fn(q):
        hits, _ = search(client, COLL[corpus], q, embedder_url="http://localhost:8021",
                         query_instruction=dom["instruction"], retrieve_k=80, hybrid=True, hybrid_prefetch_k=80,
                         rerank_enabled=True, reranker_url="http://localhost:8022", rerank_top=20,
                         rerank_min_score=0.03, image_rerank_ratio=0.2)
        return hits
    return fn


def gen_one(q, corpus, dom, done, lock, out_f, qwen_url=QWEN_URL):
    llm = OpenAI(base_url=qwen_url, api_key=os.environ.get("REASONING_API_KEY", "none"), timeout=1800)
    ctx = ArmContext(llm, QWEN, dom["system_prompt"], q["question"], q["hits"], LLMJudgeVerifier(llm, QWEN),
                     VLMFigureVerifier(llm, QWEN), VerdictCache(), retrieve_fn=make_retrieve(corpus, dom),
                     enable_thinking=q["thinking"])
    res = {}
    for arm in ARMS:
        key = f"{q['qid']}|{arm}"
        if key in done:
            res[arm] = done[key]; continue
        t0 = time.time()
        try:
            if arm in ("P0", "P1"):
                r = run_strategy(arm, ctx)
                if not r.answer.strip():          # model produced no answer (e.g. thinking budget) — not a real arm output
                    raise RuntimeError(f"empty answer after {r.latency_s}s")
            else:
                ctx.cache = VerdictCache()
                r = run_strategy(arm, ctx, p1_answer=res["P1"]["answer"])
            rec = {"key": key, "qid": q["qid"], "arm": arm, "answer": r.answer, "latency_s": r.latency_s,
                   "llm_calls": r.llm_calls, "verifier_calls": r.verifier_calls, "rounds": r.rounds,
                   "removed": r.removed, "collateral": r.collateral, "unjudged": r.unjudged,
                   "subqueries": r.subqueries, "n_hits": len(r.hits),
                   "extra_hits": r.hits[len(q["hits"]):] if len(r.hits) > len(q["hits"]) else [],
                   "claims_n": len(r.claims), "failing_trace": r.trace, "error": None}
            if arm == "C":   # qwen's own verdicts on its P1 answer — to measure self-verification leniency vs the judge
                rec["p1_self_verdicts"] = {c.id: {"text": c.text, "label": r.verdicts[c.id].label,
                                                  "issues": [i.kind for i in r.issues.get(c.id, [])]} for c in r.claims}
        except Exception as e:
            rec = {"key": key, "qid": q["qid"], "arm": arm, "answer": None, "error": repr(e)[:300],
                   "latency_s": round(time.time() - t0, 1)}
        res[arm] = rec
        with lock:
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n"); out_f.flush()
        print(key, "ERROR " + rec["error"] if rec["error"] else f"{rec['latency_s']}s", flush=True)
        if rec["error"] and arm == "P1":      # C and A start from P1
            break
    return res


# ── judging ───────────────────────────────────────────────────────────────────
def gold_texts(q, corpus, client, nuggets):
    if q["group"] == "control":
        return [p["text"] for p in (nuggets.get(q["qid"]) or {}).get("gold_passages", [])]
    if q["group"] == "multihop":
        out = []
        for g in q.get("gold_pages") or []:
            pts, _ = client.scroll(COLL[corpus], limit=30, with_payload=["text"], scroll_filter=Filter(must=[
                FieldCondition(key="source", match=MatchValue(value=g["source"])),
                FieldCondition(key="page", match=MatchValue(value=g["page"]))]))
            out += [p.payload["text"] for p in pts if (p.payload or {}).get("text")]
        return out[:10]
    return []


def judge_one(judge, q, a, gold, nug):
    if not a.get("answer"):
        return {"key": a["key"], "error": "no answer"}
    hits = q["hits"] + (a.get("extra_hits") or [])
    claims = judge.claims(a["answer"])
    r = {"key": a["key"], "qid": q["qid"], "arm": a["arm"], "group": q["group"]}
    r.update(judge.faithfulness(claims, hits))
    r.pop("verdicts", None)
    r.update({f"corr_{k}": v for k, v in judge.correctness(q["question"], claims, gold, q.get("reference")).items()})
    if q["group"] == "multihop":
        r.update(judge.completeness_reference(q["question"], a["answer"], q["reference"]))
    elif q["group"] == "control":
        ns = [n for n in (nug.get(q["qid"]) or {}).get("nuggets", []) if n.get("qwen_ok") is not False]
        r.update(judge.completeness_nuggets(q["question"], a["answer"], ns))
    elif q["group"] == "list":
        gd = q.get("gold_docs") or []
        r.update(judge.completeness_docs(a["answer"], hits, gd, q.get("subtype"), len([d for d in gd if d.strip()])))
    elif q["group"] == "unanswerable":
        r.update(judge.abstention(q["question"], a["answer"]))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True); ap.add_argument("--run", default="pilot")
    ap.add_argument("--stage", choices=["gen", "judge", "crossjudge"], required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--qwen-url", default=QWEN_URL, help="qwen3.8 endpoint for generation (a second identical instance)")
    a = ap.parse_args()
    dom = yaml.safe_load((ROOT / f"config/domains/{a.corpus}.yaml").read_text())
    qs, d = questions(a.corpus, a.run), outdir(a.corpus, a.run)
    import threading
    lock = threading.Lock()
    if a.stage == "gen":
        done = {r["key"]: r for r in jl(d / "answers.jsonl") if not r.get("error")}
        with open(d / "answers.jsonl", "a") as f, ThreadPoolExecutor(a.workers) as pool:
            list(pool.map(lambda q: gen_one(q, a.corpus, dom, done, lock, f, a.qwen_url), qs))
        print(f"GEN DONE {a.corpus} {a.run}", flush=True)
        return
    answers = {r["key"]: r for r in jl(d / "answers.jsonl") if not r.get("error")}
    nug = {r["qid"]: r for r in jl(ROOT / f"data/rq3/eval/nuggets_{a.corpus}.jsonl")}
    client = QdrantClient(url="http://localhost:6333", timeout=60)
    golds = {q["qid"]: gold_texts(q, a.corpus, client, nug) for q in qs}
    qmap = {q["qid"]: q for q in qs}
    if a.stage == "judge":
        judge = Judge(OpenAI(base_url="http://localhost:8095/v1", api_key="none", timeout=600), "gemma-4-12b")
        path, workers = d / "judgements.jsonl", 8
        todo_keys = [k for k in answers if k.split("|")[0] in qmap]
    else:
        judge = Judge(OpenAI(base_url=QWEN_URL, api_key=os.environ.get("REASONING_API_KEY", "none"), timeout=600), QWEN)
        path, workers = d / "crossjudge_qwen.jsonl", 4
        rng = random.Random(4); keys = sorted(k for k in answers if k.split("|")[0] in qmap); rng.shuffle(keys)
        todo_keys = keys[:40 if a.run.startswith("pilot") else 80]      # agreement sample: pilots 40, full run 80
    done = {r["key"] for r in jl(path) if not r.get("error")}
    todo = [answers[k] for k in todo_keys if k not in done]

    def one(ans):
        q = qmap[ans["qid"]]
        try:
            return judge_one(judge, q, ans, golds[q["qid"]], nug)
        except Exception as e:
            return {"key": ans["key"], "error": repr(e)[:300]}
    with open(path, "a") as f, ThreadPoolExecutor(workers) as pool:
        for r in pool.map(one, todo):
            f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()
            print(r["key"], r.get("error") or "ok", flush=True)
    print(f"{a.stage.upper()} DONE {a.corpus} {a.run}", flush=True)


if __name__ == "__main__":
    main()
