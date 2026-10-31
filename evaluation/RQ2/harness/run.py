#!/usr/bin/env python3
"""RQ2 harness CLI. Stages: arms → pool (gold frozen) → answers → judge → score → stats → report;
`tune` is a retrieval-only grid on the pilot.

  python evaluation/RQ2/harness/run.py --domain ceramics --universe pilot --run pilot arms
"""
from __future__ import annotations

import os

os.environ.setdefault("GRAPH_STRICT", "1")
os.environ.setdefault("MEMENTO_STRICT_RERANK", "1")

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from memento.config.cli_domain import apply_domain_arg  # noqa: E402

apply_domain_arg()
from evaluation.RQ2.harness.router import POLICIES, effective_arm, route_for, route_true  # noqa: E402

ARMS = ("g0", "g1", "g2", "g3", "d2", "lx", "g2l", "g2lg", "fs")
GRAPH_ARMS = ("g1", "g2", "g3")
# G2-lite rungs: the G2 arm on the no-LLM graphs (registry / registry + GLiNER), own Neo4j each
LITE_PORTS = {"ceramics": {"g2l": 7691, "g2lg": 7693}, "polymer-engineering": {"g2l": 7692, "g2lg": 7694}}
TERM_ARMS = ("g2", "g3", "lx", "g2l", "g2lg", "fs")
# secondary comparisons (exploratory, own BH family; the pre-registered primary family is unchanged):
# what LLM extraction buys over the no-LLM graphs, and the fact sheets against the other set-query arms
SECONDARY = {"multihop": (("g2", "g2l"), ("g2", "g2lg"), ("g2lg", "g2l")),
             "corpus": (("g2", "g2l"), ("g2", "g2lg"), ("g2lg", "g2l"), ("fs", "lx"), ("fs", "g2"), ("fs", "g3"))}
MODEL = "qwen3.8-27b"
# One or more identical local servers (comma-separated: extra servers for the answer and judge stages). Requests go
# round-robin; RQ2_LLM_WORKERS sets the stage parallelism.
LOCAL_URLS = [u.strip() for u in os.getenv("RQ2_LLM_URLS", "http://localhost:8091/v1").split(",") if u.strip()]
LOCAL_URL = LOCAL_URLS[0]
LLM_WORKERS = int(os.getenv("RQ2_LLM_WORKERS", "8"))
# arms and pool handle questions one after another by default; RQ2_QUESTION_WORKERS > 1 runs several questions at
# once (each question's own steps stay sequential; rows are saved under the Checkpoint lock)
QUESTION_WORKERS = int(os.getenv("RQ2_QUESTION_WORKERS", "1"))


def for_each_question(fn, items):
    from concurrent.futures import ThreadPoolExecutor
    if QUESTION_WORKERS <= 1:
        for x in items:
            fn(x)
        return
    with ThreadPoolExecutor(max_workers=QUESTION_WORKERS) as pool:
        list(pool.map(fn, items))


class RoundRobinLLM:
    """OpenAI-client stand-in (`.chat.completions.create`) that spreads calls over identical servers."""

    def __init__(self, clients):
        import itertools
        import threading
        self._clients, self._next, self._lock = clients, itertools.count(), threading.Lock()
        self.chat = self.completions = self

    def create(self, **kw):
        import openai
        with self._lock:
            start = next(self._next)
        last = None
        for k in range(len(self._clients)):         # a dead server moves the call to the next one
            c = self._clients[(start + k) % len(self._clients)]
            try:
                return c.chat.completions.create(**kw)
            except (openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError) as exc:
                last = exc
        raise last


def sheets_points(qdrant, collection):
    """Point count of the fact-sheets collection for the manifest; None when the corpus has none."""
    return qdrant.count(collection, exact=True).count if qdrant.collection_exists(collection) else None


def local_llm(timeout):
    from openai import OpenAI
    clients = [OpenAI(base_url=u, api_key="none", timeout=timeout) for u in LOCAL_URLS]
    return clients[0] if len(clients) == 1 else RoundRobinLLM(clients)
CLOUD_MODEL = "mistral-medium-3.5-128b"


# ── file helpers ─────────────────────────────────────────────────────────────
def write_rows(path, rows, manifest):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False, default=str) + "\n"
                           for r in [{"_manifest": manifest}] + list(rows)))
    tmp.replace(path)


def read_rows(path):
    return [r for r in (json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()) if "_manifest" not in r]


def policy_table(routes, arms=ARMS):
    table = []
    for r in routes:
        for policy in POLICIES:
            route = route_for(policy, r["qtype"], r["route_real"])
            for arm in arms:
                run_arm, design = effective_arm(arm, route)
                table.append({"qid": r["qid"], "qtype": r["qtype"], "policy": policy, "arm": arm,
                              "key": f"{r['qid']}|{run_arm}|{design}"})
    return table


# ── run context ──────────────────────────────────────────────────────────────
def out_dir(a):
    return Path("data/rq2/runs") / a.run / a.domain


def manifest(a, stage, **extra):
    from memento.graph.build.manifest import make_manifest
    return make_manifest(domain=a.domain, args={k: v for k, v in vars(a).items() if k != "func"}, summaries=[],
                         extra={"stage": stage, "k_slots": a.k_slots, "max_df": a.max_df, "min_hits": a.min_hits}
                         | extra)


def load_questions(a):
    from evaluation.RQ2.questions.common import read_jsonl, universe_ids
    qdir = Path("data/rq2/questions") / a.universe
    qs = []
    for kind in ("multihop", "corpus", "control"):
        p = qdir / f"{a.domain}_{kind}.jsonl"
        if not p.exists():
            continue
        rows = read_jsonl(p)
        if kind == "control" and a.universe == "pilot":
            allowed = universe_ids(a.domain, "pilot")          # control mechanics only
            rows = [q for q in rows if any(g["source"] in allowed for g in q.gold_pages)]
        qs += rows[:a.limit] if a.limit else rows
    return qs


def make_ctx(a, *, store=None):
    from openai import OpenAI
    from qdrant_client import QdrantClient
    from memento.config.settings import COLLECTION, QDRANT_URL, RERANKER_URL
    from evaluation.RQ2.harness.scope import ArmContext
    from evaluation.RQ2.questions.common import load_chunks, universe_ids
    from memento.graph.documents import load_document_registry
    from memento.graph.store import get_store
    chunks = load_chunks(COLLECTION)
    from memento.config.settings import NEO4J_URI
    from memento.graph.store import Neo4jStore
    from memento.graph.build.fact_sheets import sheets_collection
    store = store or get_store()
    lite = {arm: Neo4jStore(uri=f"bolt://localhost:{port}") for arm, port in LITE_PORTS.get(a.domain, {}).items()}
    assert all(f":{p}" not in NEO4J_URI for p in LITE_PORTS.get(a.domain, {}).values()), "main graph is a lite one"
    ctx = ArmContext(domain=a.domain, qdrant=QdrantClient(url=QDRANT_URL, timeout=120), collection=COLLECTION,
                     allowed=universe_ids(a.domain, a.universe),
                     llm=local_llm(900), model=MODEL,
                     registry=load_document_registry(a.domain), store=store, k_slots=a.k_slots,
                     max_df=resolve_max_df(a.max_df, store), min_hits=a.min_hits, reranker_url=os.getenv("RQ2_RERANKER_URLS") or RERANKER_URL,
                     chunk_source={c["point_id"]: c["source"] for c in chunks}, stores=lite,
                     sheets=sheets_collection(COLLECTION))
    return ctx, chunks


def close_ctx(ctx):
    ctx.store.close()
    for s in ctx.stores.values():
        s.close()


def resolve_max_df(spec, store):
    """None/'none' → no hub cap; an integer → that chunk count; 'cap2' → 2% of the graph's chunks
    (the graph's df counts chunks, not documents)."""
    if spec in (None, "none"):
        return None
    if str(spec).isdigit():
        return int(spec)
    if spec == "cap2" and store is not None:
        return max(2, int(0.02 * store._n_chunks()))
    return None


def _arm_row(o):
    d = dataclasses.asdict(o)
    d["info"] = {k: v for k, v in d["info"].items() if k != "qvec"}
    return d


def _arm_of(row):
    from evaluation.RQ2.harness.arms import ArmOut
    return ArmOut(**row)


# ── stages ───────────────────────────────────────────────────────────────────
def run_arms_for(ctx, q, keys, terms_cache, chunks):
    from evaluation.RQ2.harness import arms
    from evaluation.RQ2.harness.qterms import query_terms
    from evaluation.RQ2.harness.view import view_of
    v = view_of(q)
    g0 = arms.arm_g0(ctx, v)
    outs = {g0.key: g0}
    for key in sorted(keys):
        _, arm, design = key.split("|")
        if arm == "g0" or key in outs:
            continue
        needs_terms = arm in TERM_ARMS
        terms = query_terms(ctx.llm, ctx.model, v, terms_cache) if needs_terms else {"terms": [], "combine": "or",
                                                                                      "relation": None}
        lite = arm in ctx.stores
        if design == "multihop":
            if arm == "d2":
                o = arms.arm_d2(ctx, v, g0)
            elif lite:
                o = arms.arm_graph(arms.lite_ctx(ctx, arm), v, g0, "g2", terms, name=arm)
            else:
                o = arms.arm_graph(ctx, v, g0, arm, terms)
        elif arm == "g1":
            o = arms.corpus_g1(ctx, v, g0)
        elif arm == "lx":
            o = arms.corpus_lx(ctx, v, g0, terms, chunks)
        elif arm == "fs":
            o = arms.corpus_fs(ctx, v, g0, terms)
        elif lite:
            o = arms.corpus_lookup(arms.lite_ctx(ctx, arm), v, g0, "g2", terms, name=arm)
        else:
            o = arms.corpus_lookup(ctx, v, g0, arm, terms)
        outs[o.key] = o
    return outs


def stage_arms(a):
    from evaluation.RQ2.harness.router import route_real
    ctx, chunks = make_ctx(a)
    qs = load_questions(a)
    od = out_dir(a)
    tpath = od / "terms.json"
    terms_cache = json.loads(tpath.read_text()) if tpath.exists() else {}
    rck = Checkpoint(od / "routes.jsonl", key="qid")          # resumable: a relaunch skips finished work
    for q in qs:
        if q.qid not in rck.done:
            rck.add({"qid": q.qid, "qtype": q.qtype, "route_true": route_true(q.qtype),
                     "route_real": route_real(ctx.llm, ctx.model, q.question)})
    routes = rck.rows
    table = policy_table(routes)
    need = defaultdict(set)
    for r in table:
        need[r["qid"]].add(r["key"])
    ack = Checkpoint(od / "arms.jsonl", key=("qid", "arm", "design"))
    todo = [q for q in qs if any(tuple(k.split("|")) not in ack.done for k in need[q.qid])]
    import threading
    tlock = threading.Lock()

    def one(q):
        outs = run_arms_for(ctx, q, need[q.qid], terms_cache, chunks)
        for o in outs.values():
            if (o.qid, o.arm, o.design) not in ack.done:
                ack.add(_arm_row(o))
        with tlock:
            # dict() copies atomically under the GIL; another thread may add terms meanwhile
            tpath.write_text(json.dumps(dict(terms_cache), ensure_ascii=False, indent=1))
        print(json.dumps({"arms": q.qid, "n": len(outs), "of": len(qs)}), flush=True)
    for_each_question(one, todo)
    m = manifest(a, "arms", graph_build_id=ctx.store.get_meta("build_id"), resolved_max_df=ctx.max_df,
                 lite_build_ids={arm: s.get_meta("build_id") for arm, s in ctx.stores.items()},
                 sheets_points=sheets_points(ctx.qdrant, ctx.sheets))
    ack.finish(m)
    rck.finish(m)
    write_rows(od / "table.jsonl", table, m)
    close_ctx(ctx)


def stage_pool(a):
    from memento.config.settings import DOMAIN, EMBEDDER_URL
    from evaluation.RQ2.harness.pool import extend_corpus_gold, extend_multihop_gold
    from evaluation.RQ2.harness.scope import best_chunk_per_doc
    from memento.retrieval.search import embed_query
    ctx, _ = make_ctx(a)
    od = out_dir(a)
    by_q = defaultdict(list)
    for r in read_rows(od / "arms.jsonl"):
        by_q[r["qid"]].append(_arm_of(r))
    ck = Checkpoint(od / "gold.jsonl", key="qid")             # resumable: a relaunch skips pooled questions

    def one(q):
        outs = by_q.get(q.qid, [])
        if q.qtype == "multihop":
            q = extend_multihop_gold(ctx.llm, ctx.model, q, outs)
        elif q.qtype.startswith("corpus"):
            qvec = embed_query(q.question, EMBEDDER_URL, DOMAIN["instruction"])
            q = extend_corpus_gold(ctx.llm, ctx.model, q, outs, registry=ctx.registry,
                                   fetch=lambda d, qv=qvec: best_chunk_per_doc(ctx, [d], qv))
        ck.add(dataclasses.asdict(q))
        print(json.dumps({"pooled": q.qid}), flush=True)
    for_each_question(one, [q for q in load_questions(a) if q.qid not in ck.done])
    ck.finish(manifest(a, "pool"))
    close_ctx(ctx)


def _gold(a):
    from evaluation.RQ2.questions.common import Question
    return {r["qid"]: Question(**r) for r in read_rows(out_dir(a) / "gold.jsonl")}


def stage_answers(a):
    from concurrent.futures import ThreadPoolExecutor
    from openai import OpenAI
    from evaluation.RQ2.harness.answer import generate_answer
    od = out_dir(a)
    llm = local_llm(1800)
    gold = _gold(a)
    outs = {(o := _arm_of(r)).key: o for r in read_rows(od / "arms.jsonl")}
    keys = sorted({r["key"] for r in read_rows(od / "table.jsonl")})

    ck = Checkpoint(od / "answers.jsonl", key="key")

    def one(k):
        res = generate_answer(llm, MODEL, gold[outs[k].qid].question, outs[k])
        ck.add({"key": k} | res)
        print(json.dumps({"answered": k, "chars": len(res["answer"])}), flush=True)
    with ThreadPoolExecutor(max_workers=LLM_WORKERS) as pool:
        list(pool.map(one, [k for k in keys if k not in ck.done]))
    ck.finish(manifest(a, "answers"))


def _cloud_key():
    """The Academic Cloud key (CLOUD_API_KEY, else OPENAI_API_KEY), from the environment or
    config/.env — stages that never import memento.config.settings would otherwise miss it."""
    key = os.getenv("CLOUD_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not key:
        from dotenv import dotenv_values
        env = dotenv_values(Path(__file__).resolve().parents[3] / "config" / ".env")
        key = env.get("CLOUD_API_KEY") or env.get("OPENAI_API_KEY")
    return key or "none"


def _cloud():
    from openai import OpenAI
    cloud = OpenAI(base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                   api_key=_cloud_key(), timeout=120, max_retries=2)

    class NoThink:          # the cloud judge takes no chat_template_kwargs
        def __init__(self, c):
            self.c, self.chat, self.completions = c, self, self

        def create(self, **kw):
            kw.pop("extra_body", None)
            return self.c.chat.completions.create(**kw)
    return NoThink(cloud)


def judge_text(qtype, answer, out):
    """What the judge reads: corpus-level answers carry their [N] → document map."""
    from evaluation.RQ2.harness.judge import with_citations
    return with_citations(answer, out.context) if qtype.startswith("corpus") else answer


def stage_judge(a):
    from concurrent.futures import ThreadPoolExecutor
    from openai import OpenAI
    from evaluation.RQ2.harness.judge import cross_family, judge_absolute, judge_pair
    od = out_dir(a)
    llm = local_llm(600)
    gold = _gold(a)
    raw = {r["key"]: r["answer"] for r in read_rows(od / "answers.jsonl")}
    outs = {(o := _arm_of(r)).key: o for r in read_rows(od / "arms.jsonl")}
    ans = {k: judge_text(gold[k.split("|")[0]].qtype, v, outs[k]) for k, v in raw.items()}
    table = read_rows(od / "table.jsonl")
    abs_keys = sorted({r["key"] for r in table if not gold[r["qid"]].qtype == "control"})
    pair_keys = sorted({r["key"] for r in table if r["arm"] != "g0"})

    ck = Checkpoint(od / "judgements.jsonl", key=("kind", "key"))

    def do_abs(k):
        q = gold[k.split("|")[0]]
        ck.add({"kind": "abs", "key": k, "score": judge_absolute(llm, MODEL, q.question, q.reference_answer, ans[k])})

    def do_pair(k):
        qid = k.split("|")[0]
        q, g0k = gold[qid], f"{qid}|g0|rag"
        v = "tie" if k == g0k else judge_pair(llm, MODEL, q.question, q.reference_answer, ans[k], ans[g0k])
        ck.add({"kind": "pair", "key": k, "verdict": v})
    with ThreadPoolExecutor(max_workers=LLM_WORKERS) as pool:
        list(pool.map(do_abs, [k for k in abs_keys if ("abs", k) not in ck.done]))
        list(pool.map(do_pair, [k for k in pair_keys if ("pair", k) not in ck.done]))
    rows = [r for r in ck.rows if r["kind"] in ("abs", "pair")]
    if a.cross_family:
        items = [{"question": gold[r["key"].split("|")[0]].question,
                  "reference": gold[r["key"].split("|")[0]].reference_answer,
                  "answer": ans[r["key"]], "local": r["score"]} for r in rows if r["kind"] == "abs"]
        random.Random(a.seed).shuffle(items)
        ck.rows.append({"kind": "cross_family", "model": CLOUD_MODEL,
                        **cross_family(_cloud(), CLOUD_MODEL, items, max_calls=a.cross_family)})
    ck.finish(manifest(a, "judge"))


def stage_score(a):
    from evaluation.RQ2.harness.budget import hit_tokens
    from evaluation.RQ2.harness.judge import cited_docs
    from evaluation.RQ2.harness.metrics import control_hits, doc_prf, fact_coverage
    from memento.graph.documents import load_document_registry
    od = out_dir(a)
    reg = load_document_registry(a.domain)
    gold = _gold(a)
    outs = {(o := _arm_of(r)).key: o for r in read_rows(od / "arms.jsonl")}
    jud = read_rows(od / "judgements.jsonl")
    absd = {r["key"]: r["score"] for r in jud if r["kind"] == "abs"}
    paird = {r["key"]: r["verdict"] for r in jud if r["kind"] == "pair"}
    ans = {r["key"]: r for r in read_rows(od / "answers.jsonl")}
    rows = []
    for r in read_rows(od / "table.jsonl"):
        q, o = gold[r["qid"]], outs[r["key"]]
        if q.qtype == "multihop":
            m = fact_coverage(o.context, q.gold_pages)
        elif q.qtype.startswith("corpus"):
            m = doc_prf(o.docs, q.gold_docs, q.meta.get("unit", "documents"), reg)
            cited = doc_prf(cited_docs(ans[r["key"]].get("answer", ""), o.context), q.gold_docs,
                            q.meta.get("unit", "documents"), reg)
            m |= {"cited_p": cited["p"], "cited_r": cited["r"], "cited_f1": cited["f1"]}
        else:
            m = control_hits(o.context, q.gold_pages)
        rows.append(r | m | {"abs": absd.get(r["key"]), "pair": None if r["arm"] == "g0" else paird.get(r["key"]),
                             "retrieval_s": round(o.latency_s, 2), "answer_s": ans[r["key"]].get("latency_s"),
                             "context_tokens": hit_tokens(o.context), "abstained": bool(o.info.get("abstained")),
                             "empty_answer": not (ans[r["key"]].get("answer") or "").strip(),
                             "hardness": q.meta.get("hardness")})
    write_rows(od / "scores.jsonl", rows, manifest(a, "score"))


PRIMARY = {"multihop": "fact_recall", "corpus": "f1", "control": "recall"}
PAIR_VAL = {"win": 1.0, "tie": 0.5, "loss": 0.0}


def _family(qtype):
    return "multihop" if qtype == "multihop" else "corpus" if qtype.startswith("corpus") else "control"


def arms_for_family(rows, fam):
    """Arms that actually run their own design for this family (not routed to G0)."""
    return sorted({r["arm"] for r in rows if _family(r["qtype"]) == fam and r["arm"] != "g0"
                   and r["key"].split("|")[1] != "g0"})


PRIMARY_METRICS = {"fact_recall", "f1", "abs"}   # H1/H3 retrieval measure + H2/H4 answer quality; cited_f1 is secondary


def primary_family(comparisons):
    """Spec §6 primary family per corpus: rung steps (G1−G0, G2−G1, G3−G2), best rung − G0 and H6,
    on the hypothesis measures only."""
    fam = {}
    for c in comparisons:
        if c["family"] not in ("multihop", "corpus") or c["metric"] not in PRIMARY_METRICS:
            continue
        if c["kind"] in ("rung_step", "best_vs_g0", "h6") or (c["kind"] == "vs_g0" and c["arm"] == "g1"):
            fam[f"{c['family']}:{c['kind']}:{c['arm']}-{c['base']}:{c['metric']}"] = c["p"]
    return fam


class Checkpoint:
    """Row-level resume for long stages: each finished row is appended to <out>.partial.jsonl; a
    relaunch skips done keys; finish() writes the manifest-headed output and drops the partial."""

    def __init__(self, path, key):
        self.path, self.key = Path(path), key
        self.partial = self.path.with_name(self.path.name + ".partial")
        self.rows = [json.loads(l) for l in self.partial.read_text().splitlines() if l.strip()] \
            if self.partial.exists() else []
        self.done = {self._k(r) for r in self.rows}
        import threading
        self._lock = threading.Lock()

    def _k(self, row):
        return tuple(row[k] for k in self.key) if isinstance(self.key, tuple) else row[self.key]

    def add(self, row):
        with self._lock:
            self.partial.parent.mkdir(parents=True, exist_ok=True)
            with self.partial.open("a") as f:
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            self.rows.append(row)
            self.done.add(self._k(row))

    def finish(self, manifest):
        write_rows(self.path, self.rows, manifest)
        self.partial.unlink(missing_ok=True)


def compare(rows, fam, arm, base, metric):
    from evaluation.RQ2.harness import stats as st
    x = {r["qid"]: r for r in rows if r["arm"] == arm and _family(r["qtype"]) == fam}
    y = {r["qid"]: r for r in rows if r["arm"] == base and _family(r["qtype"]) == fam}
    qids = sorted(q for q in set(x) & set(y) if x[q].get(metric) is not None and y[q].get(metric) is not None)
    if not qids:
        return None
    xs, ys = [float(x[q][metric]) for q in qids], [float(y[q][metric]) for q in qids]
    diffs = [u - v for u, v in zip(xs, ys)]
    return {"arm": arm, "base": base, "metric": metric, "n": len(qids), "mean_arm": sum(xs) / len(xs),
            "mean_base": sum(ys) / len(ys), "diff": sum(diffs) / len(diffs), "ci": st.bootstrap_ci(diffs, seed=0),
            "p": st.wilcoxon_paired(xs, ys)["p"]}


def stage_stats(a):
    from evaluation.RQ2.harness import stats as st
    od = out_dir(a)
    rows = read_rows(od / "scores.jsonl")
    out = {"policies": {}}
    for policy in POLICIES:
        pr = [r for r in rows if r["policy"] == policy]
        res = {"comparisons": [], "binary": [], "pairwise": {}}
        for fam in ("multihop", "corpus", "control"):
            metric = PRIMARY[fam]
            present = set(arms_for_family(pr, fam)) | {"g0"}
            extra = ("cited_f1",) if fam == "corpus" else ()
            for arm in sorted(present - {"g0"}):
                for m in (metric, "abs") + extra:
                    c = compare(pr, fam, arm, "g0", m)
                    if c:
                        res["comparisons"].append({"family": fam, "kind": "vs_g0"} | c)
            for hi, lo in (("g2", "g1"), ("g3", "g2")):
                if not {hi, lo} <= present:
                    continue
                for m in (metric, "abs"):
                    c = compare(pr, fam, hi, lo, m)
                    if c:
                        res["comparisons"].append({"family": fam, "kind": "rung_step"} | c)
            for hi, lo in SECONDARY.get(fam, ()):
                if not {hi, lo} <= present:
                    continue
                for m in (metric, "abs") + extra:
                    c = compare(pr, fam, hi, lo, m)
                    if c:
                        res["comparisons"].append({"family": fam, "kind": "fs" if "fs" in (hi, lo) else "lite"} | c)
            means = {arm: sum(float(r[metric]) for r in pr if r["arm"] == arm and _family(r["qtype"]) == fam
                              and r.get(metric) is not None) /
                     max(1, sum(1 for r in pr if r["arm"] == arm and _family(r["qtype"]) == fam
                                and r.get(metric) is not None)) for arm in GRAPH_ARMS if arm in present}
            if means:
                best = max(means, key=means.get)
                res.setdefault("best_graph", {})[fam] = best
                comp = {"multihop": "d2", "corpus": "lx"}.get(fam)
                for base, kind in (("g0", "best_vs_g0"), (comp, "h6")):
                    if base and base in present:
                        for m in (metric, "abs") + extra:
                            c = compare(pr, fam, best, base, m)
                            if c:
                                res["comparisons"].append({"family": fam, "kind": kind} | c)
                binkey = "complete" if fam == "multihop" else "hit" if fam == "control" else None
                if binkey:
                    b = {r["qid"]: r[binkey] for r in pr if r["arm"] == best and _family(r["qtype"]) == fam}
                    c0 = {r["qid"]: r[binkey] for r in pr if r["arm"] == "g0" and _family(r["qtype"]) == fam}
                    q = sorted(set(b) & set(c0))
                    n10 = sum(1 for k in q if b[k] and not c0[k])
                    n01 = sum(1 for k in q if c0[k] and not b[k])
                    power = (st.mcnemar_n(n10 / len(q), n01 / len(q)) if q and n10 != n01 else None)
                    res["binary"].append({"family": fam, "arm": best, "metric": binkey, "n": len(q), "n10": n10,
                                          "n01": n01, "p": st.mcnemar_exact(n10, n01), "n_for_power80": power})
            for arm in sorted(present - {"g0"}):
                v = Counter(r["pair"] for r in pr if r["arm"] == arm and _family(r["qtype"]) == fam)
                res["pairwise"][f"{fam}:{arm}"] = {"win": v["win"], "tie": v["tie"], "loss": v["loss"],
                                                   "unjudged": v[None], "p": st.sign_test(v["win"], v["loss"])}
                if fam == "control" and policy == "real":
                    vals = [PAIR_VAL[r["pair"]] - 0.5 for r in pr if r["arm"] == arm and r["qtype"] == "control"
                            and r["pair"] in PAIR_VAL]
                    if len(vals) > 1:
                        res["pairwise"][f"{fam}:{arm}"]["tost_p"] = st.tost_paired(vals, 0.03)
        if policy == "true":
            fam_p = primary_family(res["comparisons"])
            res["bh"] = st.bh(fam_p) if fam_p else {}
            sec = {f"{c['family']}:{c['kind']}:{c['arm']}-{c['base']}:{c['metric']}": c["p"]
                   for c in res["comparisons"] if c["kind"] in ("lite", "fs") and c["metric"] in PRIMARY_METRICS}
            res["bh_secondary"] = st.bh(sec) if sec else {}
        out["policies"][policy] = res
    write_rows(od / "stats.jsonl", [out], manifest(a, "stats"))


def stage_tune(a):
    from evaluation.RQ2.harness import arms
    from evaluation.RQ2.harness.metrics import doc_prf, reaches_b, stream_reaches_b
    from evaluation.RQ2.harness.qterms import query_terms
    from evaluation.RQ2.harness.view import view_of
    ctx, _ = make_ctx(a)
    od = out_dir(a)
    terms_cache = {}
    qs = load_questions(a)
    g0s = {q.qid: arms.arm_g0(ctx, view_of(q)) for q in qs if q.qtype != "control"}
    grid = []
    for k in (3, 5, 8):
        for spec in ("none", "cap2"):
            ctx.k_slots, ctx.max_df = k, resolve_max_df(spec, ctx.store)
            for rung in ("g2", "g3"):
                top_k, anywhere, added = [], [], []
                for q in (q for q in qs if q.qtype == "multihop"):
                    b_docs = {g["source"] for g in q.gold_pages if "b" in (g.get("fact") or "")}
                    first_hop = arms.without_docs(g0s[q.qid], b_docs)       # bridge check: B not handed over
                    o = arms.arm_graph(ctx, view_of(q), first_hop, rung,
                                       query_terms(ctx.llm, ctx.model, view_of(q), terms_cache))
                    sp = o.info.get("stream_pages", [])
                    top_k.append(stream_reaches_b(sp, q.gold_pages, k))
                    anywhere.append(stream_reaches_b(sp, q.gold_pages, len(sp)))
                    added.append(reaches_b(o.context, q.gold_pages, rung))
                n = len(top_k)
                grid.append({"k_slots": k, "max_df": spec, "resolved_max_df": ctx.max_df, "rung": rung, "n": n,
                             "reaches_b": sum(top_k) / n if n else None,          # selection criterion
                             "reaches_b_anywhere": sum(anywhere) / n if n else None,
                             "b_added_to_context": sum(added) / n if n else None})
                print(json.dumps(grid[-1]), flush=True)
    ctx.k_slots, ctx.max_df = a.k_slots, resolve_max_df(a.max_df, ctx.store)
    mh = []
    for mhits in (1, 2, 3):
        ctx.min_hits = mhits
        f1 = [doc_prf(arms.corpus_lookup(ctx, view_of(q), g0s[q.qid], "g2",
                                         query_terms(ctx.llm, ctx.model, view_of(q), terms_cache)).docs,
                      q.gold_docs, q.meta.get("unit", "documents"), ctx.registry)["f1"]
              for q in qs if q.qtype.startswith("corpus")]
        mh.append({"min_hits": mhits, "n": len(f1), "mean_f1": sum(f1) / len(f1) if f1 else None})
        print(json.dumps(mh[-1]), flush=True)
    write_rows(od / "tune.jsonl", [{"reach_b_grid": grid, "min_hits": mh}], manifest(a, "tune"))
    close_ctx(ctx)


def stage_report(a):
    od = out_dir(a)
    st_ = read_rows(od / "stats.jsonl")[0]
    scores = read_rows(od / "scores.jsonl")
    routes = read_rows(od / "routes.jsonl")
    lines = [f"# RQ2 harness run `{a.run}` — {a.domain}", ""]
    conf = Counter((r["route_true"], r["route_real"]) for r in routes)
    lines += ["## Router (true → real)", "", "| true | real | n |", "|---|---|---:|"]
    lines += [f"| {t} | {r} | {n} |" for (t, r), n in sorted(conf.items())]
    for policy, res in st_["policies"].items():
        lines += ["", f"## Policy `{policy}`", "", "| family | kind | arm − base | metric | n | mean arm | mean base "
                  "| diff [95% CI] | p |", "|---|---|---|---|---:|---:|---:|---|---:|"]
        for c in res["comparisons"]:
            lines.append(f"| {c['family']} | {c['kind']} | {c['arm']} − {c['base']} | {c['metric']} | {c['n']} | "
                         f"{c['mean_arm']:.3f} | {c['mean_base']:.3f} | {c['diff']:+.3f} "
                         f"[{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}] | {c['p']:.3g} |")
        lines += ["", "| pairwise vs G0 | win | tie | loss | unjudged | sign p |", "|---|---:|---:|---:|---:|---:|"]
        lines += [f"| {k} | {v['win']} | {v['tie']} | {v['loss']} | {v['unjudged']} | {v['p']:.3g} |"
                  for k, v in res["pairwise"].items()]
        for b in res["binary"]:
            lines.append(f"\nBinary `{b['metric']}` ({b['family']}, {b['arm']} vs g0): n10={b['n10']} n01={b['n01']} "
                         f"p={b['p']:.3g}; n for 80% power: {b['n_for_power80']}")
    ab = Counter((r["arm"], r["abstained"]) for r in scores if r["policy"] == "true" and r["qtype"].startswith("corpus"))
    lines += ["", "## Corpus-level abstentions (true routing)", ""]
    lines += [f"- {arm}: {ab[(arm, True)]} of {ab[(arm, True)] + ab[(arm, False)]}" for arm in ("g2", "g3", "lx", "g2l", "g2lg", "fs")]
    empty = Counter(r["arm"] for r in scores if r["policy"] == "true" and r.get("empty_answer"))
    lines += ["", "## Empty answers (reasoning used the whole token budget; production shows a note)", ""]
    lines += [f"- {arm}: {n}" for arm, n in sorted(empty.items())] or ["- none"]
    lat = defaultdict(list)
    for r in scores:
        if r["policy"] == "true":
            lat[r["arm"]].append((r["retrieval_s"], r["answer_s"] or 0, r["context_tokens"]))
    lines += ["", "## Latency and context (true routing, means)", "", "| arm | retrieval s | answer s | context tokens |",
              "|---|---:|---:|---:|"]
    for arm, v in sorted(lat.items()):
        lines.append(f"| {arm} | {sum(x[0] for x in v) / len(v):.2f} | {sum(x[1] for x in v) / len(v):.1f} | "
                     f"{sum(x[2] for x in v) / len(v):.0f} |")
    (od / "report.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {od / 'report.md'}")


STAGES = {"arms": stage_arms, "pool": stage_pool, "answers": stage_answers, "judge": stage_judge,
          "score": stage_score, "stats": stage_stats, "tune": stage_tune, "report": stage_report}
OUTPUT = {"arms": "arms.jsonl", "pool": "gold.jsonl", "answers": "answers.jsonl", "judge": "judgements.jsonl",
          "score": "scores.jsonl", "stats": "stats.jsonl", "tune": "tune.jsonl", "report": "report.md"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--universe", choices=["pilot", "full"], required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--k-slots", type=int, default=5)
    ap.add_argument("--max-df", default="none", help="none | cap2 | integer chunk count")
    ap.add_argument("--min-hits", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="first N questions per type (smoke tests)")
    ap.add_argument("--cross-family", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("stage", choices=list(STAGES))
    a = ap.parse_args()
    out = out_dir(a) / OUTPUT[a.stage]
    if out.exists() and not a.force and a.stage not in ("stats", "report"):
        print(f"skip {a.stage}: {out} exists (use --force)")
        return
    if a.force:                                  # a forced rerun starts fresh, never from this stage's old partial
        for name in [OUTPUT[a.stage]] + (["routes.jsonl"] if a.stage == "arms" else []):
            (out_dir(a) / f"{name}.partial").unlink(missing_ok=True)
    STAGES[a.stage](a)


if __name__ == "__main__":
    main()
