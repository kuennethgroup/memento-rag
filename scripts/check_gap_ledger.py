#!/usr/bin/env python3
# scripts/check_gap_ledger.py: gap ledger step 2, the library check (logic memento/research_gaps/jobs.py, storage
# memento/research_gaps/ledger.py)
#
# Per open question of the ledger: passages of the OTHER documents (normal hybrid search + rerank, and the best passage
# of every document, reranked), merged and capped; each labelled by the local model (addresses / partly / contradicts /
# not_concern, with a reason); the status follows fixed rules from the labels; evidence stored (non-"not_concern"
# passages; for open questions also the 3 closest near-misses). Resumable (a checked gap is skipped; --force redoes),
# one model request at a time, temperature 0. Unreadable model output: one retry, else the gap stays unchecked.
#   ( set -a; source config/.env; set +a; nohup setsid .venv/bin/python scripts/check_gap_ledger.py \
#       > logs/gap_check_<date>.log 2>&1 & )      options: --limit N, --gap ID, --force
import argparse
import os
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "openwebui/functions"))
sys.path.insert(0, str(ROOT / "scripts"))

CAP, HITS_K, HEADS_K = 15, 15, 30


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gap", type=int)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--against", help="check the questions already checked against this NEW document only (step 5)")
    args = ap.parse_args()

    from openai import OpenAI
    from qdrant_client import QdrantClient
    from qdrant_client.models import FieldCondition, Filter, MatchAny
    from sqlalchemy import create_engine
    import query_pipe as qp
    from memento.query import library_state, request
    from build_gap_ledger import saved_valves
    from memento.research_gaps import jobs as gj, ledger as gl
    from memento.library import documents as ldb
    from memento.retrieval.search import point_to_hit, qdrant_query, related_heads, rerank
    from memento.common.model_servers import vl_embed

    engine = create_engine(os.environ["DATABASE_URL"])
    gl.ensure_tables(engine)
    V = qp.Pipe.Valves()
    for k, v in (saved_valves(engine) or {}).items():
        if v is not None and hasattr(V, k):
            setattr(V, k, v)
    collection = V.COLLECTION
    dom = request.domain(V.DOMAIN_NAME)
    qc = QdrantClient(url=V.QDRANT_URL, timeout=60)
    client = OpenAI(base_url=V.LLM_BASE_URL, api_key=os.environ.get("OPENAI_API_KEY", ""), timeout=600)
    n_docs = len(ldb.list_documents(engine, collection))

    def llm(messages):
        r = client.chat.completions.create(model=V.LLM_MODEL, messages=messages, temperature=0.0, max_tokens=4096,
                                           extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        return r.choices[0].message.content or ""

    def with_id(h):
        return {**h, "point_id": h.get("chunk_id")}

    gaps = gl.list_gaps(engine, collection)
    if args.against:
        return against(args.against, gaps, engine=engine, V=V, qc=qc, dom=dom, llm=llm, collection=collection)
    if args.gap:
        gaps = [g for g in gaps if g["id"] == args.gap]
    log(f"library {collection}: {len(gaps)} questions, {n_docs} documents; model {V.LLM_MODEL}")
    done, counts, t_all = 0, Counter(), time.time()
    for g in gaps:
        if args.limit and done >= args.limit:
            break
        if g.get("library_checked_at") and not args.force:
            continue
        t0 = time.time()
        stating = set(g["documents"])
        claim = g["mentions"][0]["claim"] if g["mentions"] else ""
        try:
            others = Filter(must_not=[FieldCondition(key="source", match=MatchAny(any=sorted(stating)))])
            flt = library_state.exclude_keywords_section(others)
            qvec = vl_embed(V.EMBEDDER_URL, V.EMBEDDER_MODEL, dom.get("instruction") or request.FALLBACK_INSTRUCTION,
                            g["statement"], None, 60)
            pts = qdrant_query(qc, collection, qvec, g["statement"], flt, V.RETRIEVE_K, V.HYBRID_ENABLED,
                               V.HYBRID_PREFETCH_K).points
            hits = [point_to_hit(p.payload or {}, p.score, p.id) for p in pts if (p.payload or {}).get("text", "").strip()]
            hits = [with_id(h) for h in rerank(g["statement"], hits, V.RERANKER_URL, HITS_K, 0.0, 0.0)]
            heads = related_heads(qc, collection, qvec, flt, 200)
            heads = [with_id(h) for h in rerank(g["statement"], heads[:HEADS_K], V.RERANKER_URL, HEADS_K, 0.0, 0.0)]
            cands = gj.check_candidates(hits, heads, stating=stating, cap=CAP)
            labels = gj.judge_labels(llm, lambda sub: gj.check_messages(g["statement"], claim, sub), cands)
            if not gj.complete(labels):        # not all judged: no status (never a false "open")
                counts["incomplete"] += 1
                log(f"gap {g['id']}: incomplete judgment ({sum(lb['relation'] == gj.UNKNOWN for lb in labels)} "
                    f"of {len(labels)} unjudged) {g['statement'][:90]}")
                continue
            status = gj.library_status(labels)
            ev = [dict(source=h["source"], point_id=h["point_id"], page=h.get("page"), relation=lb["relation"],
                       reason=lb["reason"]) for h, lb in zip(cands, labels) if lb["relation"] != "not_concern"]
            if status == "open":
                ev += [dict(source=h["source"], point_id=h["point_id"], page=h.get("page"), relation="near_miss",
                            reason=lb["reason"]) for h, lb in gj.near_misses(cands, labels)]
            gl.set_library_status(engine, g["id"], status=status, docs_checked=n_docs - len(stating), evidence=ev)
            counts[status] += 1
            log(f"gap {g['id']}: {status} ({len(cands)} passages, "
                f"{sum(1 for lb in labels if lb['relation'] != 'not_concern')} relevant, {time.time() - t0:.0f} s) "
                f"{g['statement'][:90]}")
        except Exception as e:
            counts["failed"] += 1
            log(f"FAILED gap {g['id']}: {e!r}")
        done += 1
    log(f"finished: {done} questions in {time.time() - t_all:.0f} s; " + ", ".join(f"{k} {v}" for k, v in counts.most_common()))
    return 0


def against(source: str, gaps: list, *, engine, V, qc, dom, llm, collection) -> int:
    """Step 5: a new document checked against every question already checked that it does not raise itself (open,
    partly, contested): its best passages for the question (hybrid search within it, reranked); only questions with a
    passage scoring at least memento.research_gaps.jobs.NEW_DOC_FLOOR get a model call; the labels combine with the
    stored evidence; the question counts the document as checked either way."""
    import query_pipe as qp
    from memento.query import library_state, request
    from qdrant_client.models import FieldCondition, Filter, MatchValue
    from memento.research_gaps import jobs as gj, ledger as gl
    from memento.retrieval.search import point_to_hit, qdrant_query, rerank
    from memento.common.model_servers import vl_embed
    todo = [g for g in gaps if g.get("library_checked_at") and g.get("library_status") in ("open", "partly", "contested")
            and source not in set(g["documents"])]
    flt = library_state.exclude_keywords_section(Filter(must=[FieldCondition(key="source", match=MatchValue(value=source))]))
    log(f"new document {source}: {len(todo)} questions to check against it")
    counts, t_all = Counter(), time.time()
    for g in todo:
        try:
            qvec = vl_embed(V.EMBEDDER_URL, V.EMBEDDER_MODEL, dom.get("instruction") or request.FALLBACK_INSTRUCTION,
                            g["statement"], None, 60)
            pts = qdrant_query(qc, collection, qvec, g["statement"], flt, 20, V.HYBRID_ENABLED, 20).points
            hits = [{**point_to_hit(p.payload or {}, p.score, p.id), "point_id": p.id} for p in pts
                    if (p.payload or {}).get("text", "").strip()]
            cands = gj.worth_checking(rerank(g["statement"], hits, V.RERANKER_URL, 5, 0.0, 0.0) if hits else [])
            old = [e["relation"] for e in gl.evidence(engine, g["id"], origin="library")]
            if not cands:
                gl.add_library_evidence(engine, g["id"], status=g["library_status"], new_docs=1, evidence=[])
                counts["no passage"] += 1
                continue
            claim = g["mentions"][0]["claim"] if g["mentions"] else ""
            labels = gj.judge_labels(llm, lambda sub: gj.check_messages(g["statement"], claim, sub), cands)
            if not gj.complete(labels):        # not all judged: this document changes and counts nothing
                counts["incomplete"] += 1
                log(f"gap {g['id']}: incomplete judgment against {source} "
                    f"({sum(lb['relation'] == gj.UNKNOWN for lb in labels)} of {len(labels)} unjudged)")
                continue
            status = gj.combined_status(old, labels)
            ev = [dict(source=h["source"], point_id=h["point_id"], page=h.get("page"), relation=lb["relation"],
                       reason=lb["reason"]) for h, lb in zip(cands, labels) if lb["relation"] != "not_concern"]
            gl.add_library_evidence(engine, g["id"], status=status, new_docs=1, evidence=ev)
            counts["changed" if status != g["library_status"] else "unchanged"] += 1
            if status != g["library_status"]:
                log(f"gap {g['id']}: {g['library_status']} → {status} ({len(ev)} passages) {g['statement'][:90]}")
        except Exception as e:
            counts["failed"] += 1
            log(f"FAILED gap {g['id']}: {e!r}")
    log(f"finished against {source}: {len(todo)} questions in {time.time() - t_all:.0f} s; "
        + ", ".join(f"{k} {v}" for k, v in counts.most_common()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
