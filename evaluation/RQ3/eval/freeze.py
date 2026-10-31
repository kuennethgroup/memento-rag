"""RQ3 step 4.1 — freeze routing + retrieval per question by running the REAL query pipe up to the generation call.

Recorded per question: the router result (intent, rewritten query, reasoning → thinking), the final ranked hits (the
output of retrieval/search.rerank, which the answer step uses), and the exact generation request. Fidelity check: that
request must equal retrieval/answer_pipeline.build_messages(domain prompt, hits, question). Gap questions are routed to
the normal answer path (intent recorded); questions the pipe answers without the answer model (library listing,
verbatim section, chitchat) are marked excluded. Output: data/rq3/eval/frozen_<corpus>.jsonl (resumable).
"""
import asyncio, json, os, sys, types
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import openai, yaml
import memento.query.router as qr
import memento.retrieval.search as search
from memento.retrieval.answer_pipeline import build_messages, system_prompt_for

LIVE_VALVES = dict(  # the Query pipe's Valves for the runs; corpus set per run (main)
    EMBEDDER_URL="http://localhost:8021", QDRANT_URL="http://localhost:6333", RERANKER_URL="http://localhost:8022",
    FILES_URL="http://localhost:8024", LLM_BASE_URL=os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:8090/v1"),
    LLM_MODEL="qwen3.8-27b", RETRIEVE_K=80, RERANK_TOP=20, RERANK_MIN_SCORE=0.03, IMAGE_RERANK_RATIO=0.2,
    TEMPERATURE=0.15, MAX_TOKENS=16384, ENABLE_THINKING=True, RERANK_ENABLED=True, MAX_IMAGES_TO_LLM=6,
    HYBRID_ENABLED=True, HYBRID_PREFETCH_K=80, GRAPH_RETRIEVAL=False, VALIDATION_ENABLED=False,
    ENABLE_QUERY_REWRITE=True, HISTORY_TURNS=3,
    DOMAIN_NAME="polymer-engineering", COLLECTION="memento-polymer-rag",
)


def load_env():
    for line in (ROOT / "config/.env").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    # the model proxy (LLM_BASE_URL) accepts REASONING_API_KEY; the pipe reads OPENAI_API_KEY
    os.environ["OPENAI_API_KEY"] = os.environ.get("REASONING_API_KEY", "")


def load_pipe():
    """The Library Chat pipe as Open WebUI loads it, with LIVE_VALVES."""
    from open_webui.utils.plugin import replace_imports
    src = replace_imports((ROOT / "openwebui/functions/query_pipe.py").read_text())
    mod = types.ModuleType("memento_query_freeze")
    exec(compile(src, "query_pipe.py", "exec"), mod.__dict__)
    pipe = mod.Pipe()
    pipe.valves = pipe.Valves(**LIVE_VALVES)
    return pipe


load_env()
COLL = {"ceramics": "memento-ceramics-rag", "polymer-engineering": "memento-polymer-rag"}
_RealAsync = openai.AsyncOpenAI


class _Stop(Exception):
    pass


async def freeze_one(corpus, q, dom_prompt):
    rec = {"qid": q["qid"], "group": q["group"], "question": q["question"]}
    state = {}
    real_route, real_rerank = qr.route, search.rerank

    async def route_rec(llm, model, query, history):
        r = await real_route(llm, model, query, history)
        state["route"] = {"intent": r.intent, "query": r.query, "reasoning": r.reasoning, "ok": r.ok,
                          "needs_thinking": r.needs_thinking}
        if r.intent == "gap_query":                     # RQ3: gap questions take the normal answer path,
            r.intent, r.reasoning = "ranked_retrieval", "deep"   # with production's thinking decision (gap → think)
        return r

    def rerank_rec(*a, **k):
        hits = real_rerank(*a, **k)
        state["hits"] = hits
        return hits

    class Proxy:
        def __init__(self, *a, **k):
            self._real = _RealAsync(*a, **k); self.chat = self; self.completions = self

        async def create(self, **kw):
            if kw.get("stream"):
                state["request"] = kw
                raise _Stop()
            return await self._real.chat.completions.create(**kw)

    qr.route, search.rerank, openai.AsyncOpenAI = route_rec, rerank_rec, Proxy
    try:
        p = load_pipe()
        async for _ in p.pipe({"messages": [{"role": "user", "content": q["question"]}]}, {"id": "rq3-freeze"}):
            pass
    finally:
        qr.route, search.rerank, openai.AsyncOpenAI = real_route, real_rerank, _RealAsync
    rec["route"] = state.get("route")
    req = state.get("request")
    if not req or "hits" not in state:
        rec.update(excluded=True, reason=f"no answer-model call (intent {(state.get('route') or {}).get('intent')})")
        return rec
    hits = state["hits"]
    expected = build_messages(system_prompt_for(dom_prompt, False), hits, q["question"],
                              max_images=LIVE_VALVES["MAX_IMAGES_TO_LLM"])
    rec.update(excluded=False, hits=hits,
               thinking=req["extra_body"]["chat_template_kwargs"]["enable_thinking"],
               fidelity=req["messages"] == expected)
    return rec


async def main(corpus):
    dom = yaml.safe_load((ROOT / f"config/domains/{corpus}.yaml").read_text())
    LIVE_VALVES.update(DOMAIN_NAME=corpus, COLLECTION=COLL[corpus], THINKING_AUTO=True, VALIDATION_ENABLED=False,
                         GRAPH_RETRIEVAL=False)
    qs = [json.loads(l) for l in open(ROOT / f"data/rq3/eval/questions_{corpus}.jsonl")]
    extra = ROOT / f"data/rq3/eval/unanswerable_{corpus}.jsonl"
    if extra.exists():
        qs += [json.loads(l) for l in open(extra)]
    out = ROOT / f"data/rq3/eval/frozen_{corpus}.jsonl"
    done = {json.loads(l)["qid"] for l in open(out)} if out.exists() else set()
    with open(out, "a") as f:
        for q in qs:                                   # sequential: monkeypatches are process-global
            if q["qid"] in done:
                continue
            rec = await freeze_one(corpus, q, dom["system_prompt"])
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
            print(q["qid"], "excluded" if rec["excluded"] else f"{len(rec['hits'])} hits, think={rec['thinking']}, "
                  f"fidelity={rec['fidelity']}", (rec.get("route") or {}).get("intent"), flush=True)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))
