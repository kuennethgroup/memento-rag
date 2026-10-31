#!/usr/bin/env python3
# scripts/build_gap_ledger.py: build the gap ledger of the served library, step 1 (logic memento/research_gaps/jobs.py, storage
# memento/research_gaps/ledger.py)
#
# Per document of the library (memento_documents): its gap-tagged passages → extract (field question / study scope)
# → judge each question's claim against its passage (RQ3 judge; only "supported" enters) → merge into the ledger
# (same question → a further mention, else a new gap + its vector in <collection>-gaps). Resumable: a document whose
# content fingerprint is already done is skipped (--force redoes it). One model request at a time, so users of the
# local model keep its other slots. Run detached, env from config/.env SOURCED:
#   ( set -a; source config/.env; set +a; nohup setsid .venv/bin/python scripts/build_gap_ledger.py \
#       > logs/gap_ledger_<date>.log 2>&1 & )
#   options: --doc <source> (one document), --limit N, --force, --dry-run (counts only, no model calls)
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "openwebui/functions"))

# the same instruction for stored questions and new ones: symmetric similarity between research questions
GAP_EMBED_INSTRUCTION = "Represent this research question so that the same open question, however worded, is found."


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def saved_valves(engine, function_id: str = "memento_query") -> dict | None:
    """A copy's saved Valves; None when the copy is not installed (never another copy's)."""
    from sqlalchemy import text
    with engine.connect() as c:
        row = c.execute(text("SELECT valves FROM function WHERE id = :id"), {"id": function_id}).first()
    if row is None:
        return None
    v = json.loads(row[0]) if isinstance(row[0], str) else (row[0] or {})
    return v if isinstance(v, dict) else {}


def tagged_passages(qc, collection: str, source: str) -> list[dict]:
    from qdrant_client.models import FieldCondition, Filter, MatchValue
    flt = Filter(must=[FieldCondition(key="source", match=MatchValue(value=source)),
                       FieldCondition(key="gap_relevant", match=MatchValue(value=True))])
    out, offset = [], None
    while True:
        pts, offset = qc.scroll(collection, scroll_filter=flt, limit=256, offset=offset, with_payload=True,
                                with_vectors=False)
        for p in pts:
            pl = p.payload or {}
            if (pl.get("text") or "").strip():
                out.append({"point_id": str(p.id), "page": pl.get("page") or 0, "text": pl["text"],
                            "gap_type": pl.get("gap_type"), "gap_signal": pl.get("gap_signal")})
        if offset is None:
            break
    return sorted(out, key=lambda p: (p["page"], p["point_id"]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--floor", type=float, default=0.6)
    args = ap.parse_args()

    from openai import OpenAI
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointIdsList, PointStruct, VectorParams
    from sqlalchemy import create_engine
    import query_pipe as qp
    from memento.query import request
    from memento.agentic_validation.verifiers import LLMJudgeVerifier
    from memento.research_gaps import jobs as gj, ledger as gl
    from memento.library import documents as ldb
    from memento.common.model_servers import vl_embed

    engine = create_engine(os.environ["DATABASE_URL"])
    gl.ensure_tables(engine)
    V = qp.Pipe.Valves()
    for k, v in (saved_valves(engine) or {}).items():
        if v is not None and hasattr(V, k):
            setattr(V, k, v)
    collection = V.COLLECTION
    gaps_collection = f"{collection}-gaps"
    qc = QdrantClient(url=V.QDRANT_URL, timeout=60)
    client = OpenAI(base_url=V.LLM_BASE_URL, api_key=request.llm_key(V), timeout=600)

    def llm(messages, max_tokens=8192):
        r = client.chat.completions.create(model=V.LLM_MODEL, messages=messages, temperature=0.0,
                                           max_tokens=max_tokens,
                                           extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        return r.choices[0].message.content or ""

    verifier = LLMJudgeVerifier(client, V.LLM_MODEL, workers=1)     # one call at a time

    def embed(text):
        return vl_embed(V.EMBEDDER_URL, V.EMBEDDER_MODEL, GAP_EMBED_INSTRUCTION, text, None, 60)

    def ensure_vectors(size: int):
        if not qc.collection_exists(gaps_collection):
            qc.create_collection(gaps_collection, vectors_config=VectorParams(size=size, distance=Distance.COSINE))

    def search(vec, k):
        if not qc.collection_exists(gaps_collection):
            return []
        res = qc.query_points(gaps_collection, query=vec, limit=k, with_payload=True).points
        return [(int(p.payload["gap_id"]), p.payload["statement"], p.score) for p in res]

    def upsert(gid, vec, statement):
        ensure_vectors(len(vec))
        qc.upsert(gaps_collection, [PointStruct(id=int(gid), vector=vec,
                                                payload={"gap_id": int(gid), "statement": statement})])

    def relate(question, near):
        return gj.parse_relation(llm(gj.relation_messages(question, near), max_tokens=96), [g for g, _ in near])

    docs = ldb.list_documents(engine, collection)
    if args.doc:
        docs = [d for d in docs if d["source"] == args.doc]
    log(f"library {collection}: {len(docs)} documents; model {V.LLM_MODEL}; ledger vectors {gaps_collection}")
    done = new_gaps = merged = 0
    t_all = time.time()
    for d in docs:
        if args.limit and done >= args.limit:
            break
        src, chash = d["source"], d.get("content_hash")
        prev = gl.get_document(engine, collection, src)
        if prev and prev["status"] == "done" and prev["content_hash"] == chash and not args.force:
            continue
        passages = tagged_passages(qc, collection, src)
        if args.dry_run:
            log(f"would process {src[:70]}: {len(passages)} tagged passages")
            done += 1
            continue
        t0 = time.time()
        try:
            orphans = gl.delete_for_source(engine, collection, src)          # a redo starts clean
            if orphans and qc.collection_exists(gaps_collection):
                qc.delete(gaps_collection, PointIdsList(points=orphans))
            items = gj.extract_document(llm, d.get("title") or src, passages) if passages else []
            for i, claim, reason in gj.left_out(items):          # study scope only / solved in this document
                p = passages[i]
                gl.add_scope_only(engine, collection, src, point_id=p["point_id"], page=p["page"],
                                  signal=p.get("gap_signal") or "", claim=claim, reason=reason)
            kept, dropped = gj.ground(verifier, items, passages)
            for x in dropped:
                p = passages[x["i"]]
                gl.add_dropped(engine, collection, src, point_id=p["point_id"], page=p["page"],
                               question=x["question"], claim=x["claim"], reason=x["reason"])

            def add_mention(gid, x, src=src):
                p = passages[x["i"]]
                gl.add_mention(engine, gid, collection, source=src, point_id=p["point_id"], page=p["page"],
                               signal=p.get("gap_signal") or "", claim=x["claim"], passage=p["text"],
                               judge=x["judge"], judge_reason=x.get("judge_reason", ""), question=x["question"])
            for x in kept:
                _, created = gj.merge(x, embed=embed, search=search, relate=relate,
                                      add_gap=lambda s, k: gl.add_gap(engine, collection, s, k),
                                      add_mention=add_mention, upsert=upsert, floor=args.floor,
                                      add_relation=lambda a, b: gl.add_relation(engine, collection, a, b))
                new_gaps += created
                merged += not created
            n_scope = len(gj.left_out(items))
            gl.set_document(engine, collection, src, content_hash=chash, status="done", n_questions=len(kept),
                            n_scope=n_scope, n_dropped=len(dropped))
            log(f"done {src[:60]}: {len(passages)} passages → {len(kept)} questions kept, {len(dropped)} dropped by "
                f"the judge, {n_scope} left out (scope only / solved here) ({time.time() - t0:.0f} s)")
        except Exception as e:
            gl.set_document(engine, collection, src, content_hash=chash, status="failed", error=repr(e)[:500])
            log(f"FAILED {src[:60]}: {e!r}")
        done += 1
    gaps = gl.list_gaps(engine, collection)
    linked = sum(1 for g in gaps if gl.related(engine, collection, g["id"]))
    log(f"finished: {done} documents in {time.time() - t_all:.0f} s; new gaps {new_gaps}, merged mentions {merged}; "
        f"ledger now {len(gaps)} gaps, {sum(1 for g in gaps if len(g['documents']) > 1)} stated by 2+ documents, "
        f"{linked} with related questions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
