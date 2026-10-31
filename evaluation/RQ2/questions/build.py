#!/usr/bin/env python3
"""RQ2 question builder."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from memento.config.cli_domain import apply_domain_arg  # noqa: E402
apply_domain_arg()
from memento.config.settings import COLLECTION, QDRANT_URL, TEXT_COLLECTION  # noqa: E402
from evaluation.RQ2.questions import corpus_level as cl, multihop as mh  # noqa: E402
from evaluation.RQ2.questions.common import (Question, content_chunks, load_chunks, question_problem, read_jsonl,  # noqa: E402
                                             universe_ids, write_jsonl)
from evaluation.RQ2.questions.pooling import g0_candidates, pool_and_judge  # noqa: E402
from evaluation.RQ2.questions.hardness import bm25_top  # noqa: E402
from evaluation.RQ2.questions.terms import gate_terms, term_doc_freq  # noqa: E402
from memento.graph.documents import extracts_entities, load_document_registry  # noqa: E402

MODEL, URL = "qwen3.8-27b", "http://localhost:8091/v1"


def _prov(seed, *prompts):
    from memento.graph.build.manifest import git_state
    return {"model": MODEL, "prompt_sha": hashlib.sha1("\n".join(prompts).encode()).hexdigest()[:12],
            "seed": seed, "git": git_state()}


def cmd_multihop(a, ctx):
    llm, qd, reg, allowed = ctx["llm"], ctx["qdrant"], ctx["registry"], ctx["allowed"]
    excl = {tuple(q.meta["pair_key"]) for f in a.exclude_from for q in read_jsonl(f) if "pair_key" in q.meta}
    pairs = mh.anchor_pairs(ctx["universe_chunks"], ctx["df_full"], reg, n_research_docs=ctx["n_research"],
                            seed=a.seed, max_pairs=a.n * 12, exclude_pairs=excl)
    kept, rejects = [], []
    ok_terms, n_unparsed = gate_terms(llm, MODEL, [p["bridge"] for p in pairs])
    rejects += [{"pair_key": p["pair_key"], "bridge": p["bridge"], "reason": "term_gate"}
                for p in pairs if p["bridge"] not in ok_terms]
    print(json.dumps({"term_gate": {"pairs": len(pairs), "kept": sum(p["bridge"] in ok_terms for p in pairs),
                                    "unparsed_terms": n_unparsed}}), flush=True)
    pairs = [p for p in pairs if p["bridge"] in ok_terms]
    for pair in pairs:
        if len(kept) >= a.n:
            break
        c = mh.compose(llm, MODEL, pair)
        if c is None:
            rejects.append({"pair_key": pair["pair_key"], "bridge": pair["bridge"], "reason": "compose_failed"}); continue
        bad = question_problem(c["question"], c["lang"])
        if bad:
            rejects.append({"pair_key": pair["pair_key"], "bridge": pair["bridge"], "reason": bad,
                            "question": c["question"]}); continue
        if mh.bridge_leaks(c["question"], pair["bridge"], c["bridge_forms"]):
            rejects.append({"pair_key": pair["pair_key"], "bridge": pair["bridge"], "reason": "bridge_leak",
                            "question": c["question"]}); continue
        nec = mh.necessity(llm, MODEL, c, pair["a"]["text"], pair["b"]["text"])
        if not nec["passes"]:
            rejects.append({"pair_key": pair["pair_key"], "bridge": pair["bridge"], "question": c["question"],
                            "reason": "necessity:" + ",".join(f"{k}={nec[k]}" for k in ("ab", "a_only", "b_only"))})
            continue
        pool = pool_and_judge(llm, MODEL, {"a": c["fact_a"], "b": c["fact_b"]},
                              {"g0": g0_candidates(qd, COLLECTION, c["question"], allowed),
                               "bm25": [{"source": s, "page": p, "type": "text",
                                         "text": ctx["page_text"].get((s, p), "")}
                                        for s, p in bm25_top(qd, COLLECTION, c["question"], 20, allowed)]})
        gold = [{"source": pair["a"]["source"], "page": pair["a"]["page"], "role": "A", "fact": "a"},
                {"source": pair["b"]["source"], "page": pair["b"]["page"], "role": "B", "fact": "b"}]
        seen = {(g["source"], g["page"]) for g in gold}
        gold += [{"source": g["source"], "page": g["page"], "role": "pooled", "fact": g["fact"]}
                 for g in pool["gold"] if (g["source"], g["page"]) not in seen]
        q = Question(f"{a.domain}:multihop:{len(kept) + 1}", a.domain, "multihop", c["question"], c["answer"],
                     c["lang"], a.universe, gold, [], [],
                     {"bridge": pair["bridge"], "bridge_forms": c["bridge_forms"], "fact_a": c["fact_a"],
                      "fact_b": c["fact_b"], "pair_key": list(pair["pair_key"]), "necessity": nec,
                      "pool": {"n_candidates": pool["n_candidates"], "unparsed": pool["unparsed"]},
                      "provenance": _prov(a.seed, mh.COMPOSE_SYSTEM, mh.ANSWER_SYSTEM, mh.JUDGE_SYSTEM)})
        kept.append(q)
        print(json.dumps({"accepted": q.qid, "question": q.question}, ensure_ascii=False), flush=True)
    write_jsonl(Path(a.out_dir) / f"{a.domain}_multihop.jsonl", kept)
    Path(a.out_dir, f"{a.domain}_multihop_rejects.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False, default=list) + "\n" for r in rejects))


def cmd_corpus_level(a, ctx):
    llm, qd, reg, allowed = ctx["llm"], ctx["qdrant"], ctx["registry"], ctx["allowed"]
    used = {q.meta.get("seed_term") for f in a.exclude_from for q in read_jsonl(f)}
    lo, hi = (3, 10) if a.universe == "pilot" else (3, 30)
    df_u = term_doc_freq(ctx["universe_chunks"])
    seeds = [t for t in cl.seed_terms(df_u, lo=lo, hi=hi, seed=a.seed, n=a.n_per_type * len(a.types) * 6)
             if t not in used]
    kept, rejects, i = [], [], 0
    ok_terms, n_unparsed = gate_terms(llm, MODEL, seeds)
    rejects += [{"seed_term": t, "reason": "term_gate"} for t in seeds if t not in ok_terms]
    print(json.dumps({"term_gate": {"seeds": len(seeds), "kept": len(ok_terms), "unparsed_terms": n_unparsed}}),
          flush=True)
    seeds = [t for t in seeds if t in ok_terms]
    per_type = {t: 0 for t in a.types}
    for term in seeds:
        qtype = next((t for t in a.types if per_type[t] < a.n_per_type), None)
        if qtype is None:
            break
        passages = [c for c in ctx["universe_chunks"] if term in c["text"]][:6]
        c = cl.compose_corpus_question(llm, MODEL, term, passages, qtype)
        if c is None:
            rejects.append({"seed_term": term, "type": qtype, "reason": "compose_failed"}); continue
        bad = question_problem(c["question"], c["lang"])
        if bad:
            rejects.append({"seed_term": term, "type": qtype, "reason": bad, "question": c["question"]}); continue
        docs = cl.variant_scan(ctx["universe_chunks"], c["variants"])
        for s, cs in cl.dense_top_docs(qd, COLLECTION, c["question"], allowed).items():
            docs.setdefault(s, cs)
        sc = cl.screen(llm, MODEL, c["criterion"], docs)
        if len(sc["gold_docs"]) < 2:
            rejects.append({"seed_term": term, "type": qtype, "reason": f"gold_docs={len(sc['gold_docs'])}",
                            "question": c["question"]}); continue
        n, works = cl.count_answer(sc["gold_docs"], reg, c["unit"])
        ref = (f"{n} works: " + "; ".join(works)) if c["unit"] == "works" else "; ".join(sc["gold_docs"])
        i += 1
        q = Question(f"{a.domain}:corpus_{qtype}:{i}", a.domain, f"corpus_{qtype}", c["question"], ref,
                     c["lang"], a.universe, [], sc["gold_docs"], works,
                     {"seed_term": term, "criterion": c["criterion"], "variants": c["variants"], "unit": c["unit"],
                      "screen": {"n_screened": sc["n_screened"], "unparsed_docs": sc["unparsed_docs"]},
                      "provenance": _prov(a.seed, cl.SEED_SYSTEM)})
        per_type[qtype] += 1
        kept.append(q)
        print(json.dumps({"accepted": q.qid, "question": q.question, "n_gold": len(sc["gold_docs"])},
                         ensure_ascii=False), flush=True)
    write_jsonl(Path(a.out_dir) / f"{a.domain}_corpus.jsonl", kept)
    Path(a.out_dir, f"{a.domain}_corpus_rejects.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rejects))


GT_V4 = {"ceramics": "evaluation/RQ1/ground_truth_auto_v4_qwenjudge.json",
         "polymer-engineering": "evaluation/RQ1/polymer_ground_truth_auto_v4_qwenjudge.json"}


def cmd_control(a):
    from evaluation.RQ2.questions.control import from_rq1_v4
    qs = from_rq1_v4(Path(GT_V4[a.domain]), a.domain, "full")
    write_jsonl(Path(a.out_dir) / f"{a.domain}_control.jsonl", qs)
    print(json.dumps({"control": len(qs)}), flush=True)


def cmd_hardness(a):
    from qdrant_client import QdrantClient
    from evaluation.RQ2.questions.hardness import hardness, text_dense_top
    from evaluation.RQ1.collections.local_text_embed import local_text_embed_query
    qd, allowed = QdrantClient(url=QDRANT_URL), universe_ids(a.domain, a.universe)
    path = Path(a.out_dir) / f"{a.domain}_multihop.jsonl"
    qs = read_jsonl(path)
    for q in qs:
        b = next(g for g in q.gold_pages if g["role"] == "B")
        bm = bm25_top(qd, COLLECTION, q.question, 20, allowed)
        de = text_dense_top(qd, TEXT_COLLECTION, q.question, local_text_embed_query, 20, allowed)
        q.meta["hardness"] = hardness((b["source"], b["page"]), bm, de)
        q.meta["hardness_hits"] = {"bm25": (b["source"], b["page"]) in {tuple(x) for x in bm},
                                   "text_dense": (b["source"], b["page"]) in {tuple(x) for x in de}}
    write_jsonl(path, qs)
    print(json.dumps({"hardness": {h: sum(q.meta["hardness"] == h for q in qs) for h in ("easy", "hard")}}), flush=True)


def cmd_crosscheck(a):
    """Mistral re-judges a seeded sample: half multi-hop A+B verdicts, half corpus-level screening decisions.
    One cloud call per judgement, >= 6 s apart; agreement and kappa per kind."""
    import os
    import random
    import time
    import importlib.util
    from openai import OpenAI
    from qdrant_client import QdrantClient
    from evaluation.RQ2.questions.pooling import judge_facts
    spec = importlib.util.spec_from_file_location("ga", Path(__file__).resolve().parents[1] / "graph_audit.py")
    ga = importlib.util.module_from_spec(spec)
    sys_argv = sys.argv[:]
    spec.loader.exec_module(ga)
    sys.argv = sys_argv
    cloud = OpenAI(base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                   api_key=ga.cloud_api_key(), timeout=120, max_retries=2)
    mistral = "mistral-medium-3.5-128b"
    rng = random.Random(a.seed)
    out, last = {}, [0.0]

    def pace():
        wait = 6.0 - (time.time() - last[0])
        if wait > 0:
            time.sleep(wait)
        last[0] = time.time()

    def nothink(client):          # the cloud judge takes no chat_template_kwargs
        class W:
            def __init__(self, c): self.c = c; self.chat = self; self.completions = self
            def create(self, **kw):
                kw.pop("extra_body", None)
                pace()
                return self.c.chat.completions.create(**kw)
        return W(client)
    mc = nothink(cloud)
    mh_qs = read_jsonl(Path(a.out_dir) / f"{a.domain}_multihop.jsonl")
    pick = rng.sample(mh_qs, min(a.n // 2, len(mh_qs)))
    la, lb = [], []
    for q in pick:
        nec = q.meta["necessity"]
        v = mh.judge_answer(mc, mistral, q.question, q.reference_answer, nec["answers"]["ab"])
        if v is not None:
            la.append(nec["ab"]); lb.append(v)
    out["multihop_ab"] = {"n": len(la), "agreement": round(sum(x == y for x, y in zip(la, lb)) / len(la), 4) if la else None,
                          "kappa": round(ga.cohen_kappa(la, lb), 4) if la else None}
    cq = read_jsonl(Path(a.out_dir) / f"{a.domain}_corpus.jsonl")
    reg = load_document_registry(a.domain)
    allowed = universe_ids(a.domain, a.universe)
    uni = [ch for ch in content_chunks(load_chunks(COLLECTION), reg, None) if allowed is None or ch["source"] in allowed]
    qd = QdrantClient(url=QDRANT_URL)
    decisions = []
    for q in cq:
        docs = cl.variant_scan(uni, q.meta["variants"])
        for s_, cs in cl.dense_top_docs(qd, COLLECTION, q.question, allowed).items():
            docs.setdefault(s_, cs)
        decisions += [(q, s_, cs) for s_, cs in sorted(docs.items())]
    la, lb = [], []
    for q, src, cs in rng.sample(decisions, min(a.n - a.n // 2, len(decisions))):
        r = judge_facts(mc, mistral, {"c": q.meta["criterion"]}, cs, batch=3)
        if len(r["unparsed"]) == len(cs):
            continue
        la.append("yes" if src in q.gold_docs else "no"); lb.append("yes" if r["labels"] else "no")
    out["corpus_screen"] = {"n": len(la), "agreement": round(sum(x == y for x, y in zip(la, lb)) / len(la), 4) if la else None,
                            "kappa": round(ga.cohen_kappa(la, lb), 4) if la else None}
    Path(a.out_dir, f"{a.domain}_crosscheck.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out), flush=True)


def cmd_report(a):
    from collections import Counter
    d = Path(a.out_dir)
    rep = {}
    for kind in ("multihop", "corpus"):
        acc = read_jsonl(d / f"{a.domain}_{kind}.jsonl") if (d / f"{a.domain}_{kind}.jsonl").exists() else []
        rj = [json.loads(l) for l in (d / f"{a.domain}_{kind}_rejects.jsonl").read_text().splitlines()
              if l.strip()] if (d / f"{a.domain}_{kind}_rejects.jsonl").exists() else []
        reasons = Counter(r["reason"].split(":")[0] if kind == "multihop" else r["reason"].split("=")[0] for r in rj)
        rep[kind] = {"accepted": len(acc), "rejected": len(rj), "reasons": dict(reasons),
                     "by_type": dict(Counter(q.qtype for q in acc)),
                     "hardness": dict(Counter(q.meta.get("hardness") for q in acc)) if kind == "multihop" else None,
                     "gold_sizes": [len(q.gold_pages) if kind == "multihop" else len(q.gold_docs) for q in acc]}
    Path(d, f"{a.domain}_report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(json.dumps(rep, ensure_ascii=False), flush=True)


def main():
    ap = argparse.ArgumentParser(description="RQ2 question builder")
    ap.add_argument("--domain", required=True)
    ap.add_argument("--universe", choices=["pilot", "full"], default="pilot")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out-dir", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("multihop"); m.add_argument("--n", type=int, default=15)
    m.add_argument("--exclude-from", nargs="*", default=[])
    c = sub.add_parser("corpus-level"); c.add_argument("--n-per-type", type=int, default=5)
    c.add_argument("--types", nargs="+", default=["list", "count", "compare"])
    c.add_argument("--exclude-from", nargs="*", default=[])
    sub.add_parser("control"); sub.add_parser("hardness")
    x = sub.add_parser("crosscheck"); x.add_argument("--n", type=int, default=40)
    sub.add_parser("report")
    a = ap.parse_args()
    a.out_dir = a.out_dir or f"data/rq2/questions/{a.universe}"
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    if a.cmd in ("multihop", "corpus-level"):
        from openai import OpenAI
        from qdrant_client import QdrantClient
        reg = load_document_registry(a.domain)
        allowed = universe_ids(a.domain, a.universe)
        full = content_chunks(load_chunks(COLLECTION), reg, None)
        ctx = {"llm": OpenAI(base_url=URL, api_key="none", timeout=600), "qdrant": QdrantClient(url=QDRANT_URL),
               "registry": reg, "allowed": allowed, "df_full": term_doc_freq(full),
               "universe_chunks": [ch for ch in full if allowed is None or ch["source"] in allowed],
               "n_research": sum(1 for v in reg.values() if extracts_entities(v)),
               "page_text": {(ch["source"], ch["page"]): ch["text"] for ch in full}}
        (cmd_multihop if a.cmd == "multihop" else cmd_corpus_level)(a, ctx)
    elif a.cmd == "control":
        cmd_control(a)
    elif a.cmd == "hardness":
        cmd_hardness(a)
    elif a.cmd == "crosscheck":
        cmd_crosscheck(a)
    else:
        cmd_report(a)


if __name__ == "__main__":
    main()
