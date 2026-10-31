#!/usr/bin/env python3
# scripts/check_gap_literature.py: gap ledger step 3, the literature check (search memento/research_gaps/literature.py,
# judging memento/research_gaps/jobs.py)
#
# For each question of the ledger: a short search phrase from the local model (the only text that leaves the server),
# the literature services, our reranker, and the local model's label for each of the best abstracts. The status follows
# the library check's rules; papers that concern the question are stored as evidence.
# Resumable: a checked question is skipped (--force redoes it; --add-openalex and --add-alexandria redo questions
# checked without that source). An unreadable model answer: one retry, else the question stays unchecked (never a false
# "nothing found").
#   ( set -a; source config/.env; set +a; nohup setsid .venv/bin/python scripts/check_gap_literature.py \
#       > logs/gap_literature_<date>.log 2>&1 & )      options: --help
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

TOP_K, PER_SERVICE, S2_PAUSE, OA_RESERVE = 10, 25, 1.1, 3


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gap", type=int)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--add-openalex", action="store_true", help="redo questions checked without OpenAlex")
    ap.add_argument("--add-alexandria", action="store_true", help="redo questions checked without the offline index")
    ap.add_argument("--fill-text", action="store_true", help="only fill in stored papers' abstracts from the cache")
    args = ap.parse_args()

    from openai import OpenAI
    from sqlalchemy import create_engine
    import query_pipe as qp
    from build_gap_ledger import saved_valves
    from memento.research_gaps import fulltext as ft, literature as al, jobs as gj, ledger as gl, literature as ls
    from memento.library import documents as ldb
    from memento.retrieval.search import rerank

    if os.getenv("GAP_WEB_CHECK", "true").strip().lower() in ("0", "false", "no", "off"):
        log("GAP_WEB_CHECK is off: nothing searched")
        return 0
    s2_key = os.getenv("S2_API_KEY", "")
    oa_mailto, oa_key = os.getenv("OPENALEX_MAILTO", ""), os.getenv("OPENALEX_API_KEY", "")
    engine = create_engine(os.environ["DATABASE_URL"])
    gl.ensure_tables(engine)
    ls.ensure_tables(engine)
    if args.fill_text:
        log(f"abstracts filled in: {gl.fill_web_text(engine, ls.cached_abstracts(engine))}")
        return 0
    V = qp.Pipe.Valves()
    for k, v in (saved_valves(engine) or {}).items():
        if v is not None and hasattr(V, k):
            setattr(V, k, v)
    collection = V.COLLECTION
    client = OpenAI(base_url=V.LLM_BASE_URL, api_key=os.environ.get("OPENAI_API_KEY", ""), timeout=600)

    def llm(messages, max_tokens=4096):
        r = client.chat.completions.create(model=V.LLM_MODEL, messages=messages, temperature=0.0,
                                           max_tokens=max_tokens,
                                           extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        return r.choices[0].message.content or ""

    docs = ldb.list_documents(engine, collection)
    lib_titles = [t for d in docs for t in (d.get("parsed_title"), d.get("pdf_title"), d.get("source")) if t]
    gaps = gl.list_gaps(engine, collection)
    if args.gap:
        gaps = [g for g in gaps if g["id"] == args.gap]
    log(f"library {collection}: {len(gaps)} questions, {len(docs)} documents; model {V.LLM_MODEL}; "
        f"Semantic Scholar {'with key' if s2_key else 'WITHOUT key'}; OpenAlex "
        f"{'with key' if oa_key else 'without key (about 100 searches a day)'}")
    oa_left: int | None = None                    # unknown until the first OpenAlex answer
    done, counts, t_all, last_s2 = 0, Counter(), time.time(), 0.0
    for g in gaps:
        if args.limit and done >= args.limit:
            break
        checked = bool(g.get("web_checked_at"))
        lacks_oa = "openalex" not in (g.get("web_sources") or "")
        lacks_al = "alexandria" not in (g.get("web_sources") or "")
        if checked and not args.force and not (args.add_openalex and lacks_oa) and not (args.add_alexandria and lacks_al):
            continue
        t0 = time.time()
        claim = g["mentions"][0]["claim"] if g["mentions"] else ""
        try:
            query = gj.parse_query(llm(gj.query_messages(g["statement"]), max_tokens=100))
            works, searched = [], set()

            def search(q):
                nonlocal last_s2, oa_left
                found = []
                if s2_key:
                    time.sleep(max(0.0, S2_PAUSE - (time.time() - last_s2)))
                    try:
                        found += ls.search_s2(q, engine=engine, api_key=s2_key, n=PER_SERVICE)
                        searched.add("semantic_scholar")
                    except ls.SearchFailed as e:
                        log(f"  gap {g['id']}: Semantic Scholar failed: {str(e)[:120]}")
                    last_s2 = time.time()
                if oa_left is None or oa_left > OA_RESERVE:
                    try:
                        oa, left = ls.search_openalex(q, engine=engine, n=PER_SERVICE, mailto=oa_mailto,
                                                      api_key=oa_key)
                        found += oa
                        searched.add("openalex")
                        oa_left = left if left is not None else oa_left
                    except ls.SearchFailed as e:
                        log(f"  gap {g['id']}: OpenAlex failed: {str(e)[:120]}")
                        if "429" in str(e):
                            oa_left = 0
                if al.available():                         # the offline literature index: local, no limits
                    found += al.search(q, n=25)
                    searched.add("alexandria")
                return found
            works = search(query)
            short = gj.shorter_query(query) if len(ls.merge(works)) < 5 else None
            if short:                                  # every term must match: a long phrase can find nothing
                works += search(short)
                query = f"{query} | {short}"
            if not searched:
                raise RuntimeError("no literature service answered")
            stating_titles = [t for d in docs if d["source"] in set(g["documents"])
                              for t in (d.get("parsed_title"), d.get("pdf_title"))]
            works = ls.drop_library(ls.merge(works), lib_titles + stating_titles)
            hits = [{"text": f"{w.title}. {w.abstract}".strip(), "type": "text", "work": w} for w in works]
            ranked = rerank(g["statement"], hits, V.RERANKER_URL, TOP_K, 0.0, 0.0) if hits else []
            top = [h["work"] for h in ranked[:TOP_K]]
            if top and ft.enabled():
                ft.attach(top, g["statement"], query, rank=ft.ranker(g["statement"], V.RERANKER_URL))
            if top:
                labels = gj.judge_labels(llm, lambda sub: gj.web_check_messages(g["statement"], claim, sub), top)
                if not gj.complete(labels):    # not all judged: no status (never a false "nothing found")
                    counts["incomplete"] += 1
                    log(f"gap {g['id']}: incomplete judgment ({sum(lb['relation'] == gj.UNKNOWN for lb in labels)} "
                        f"of {len(labels)} unjudged) {g['statement'][:80]}")
                    continue
            else:
                labels = []
            status = gj.web_status(labels)
            ev = [dict(origin=w.source, source=w.work_id, title=w.title, year=w.year, doi=w.doi, url=w.url,
                       relation=lb["relation"], reason=lb["reason"], text=ft.judged_text(w))
                  for w, lb in zip(top, labels) if lb["relation"] != "not_concern"]
            if status == "nothing_found":               # what was read, so "nothing found" can be inspected
                ev += [dict(origin=w.source, source=w.work_id, title=w.title, year=w.year, doi=w.doi, url=w.url,
                            relation="near_miss", reason=lb["reason"], text=ft.judged_text(w))
                       for w, lb in gj.web_near_misses(top, labels)]
            gl.set_web_status(engine, g["id"], status=status, query=query, sources=sorted(searched), evidence=ev)
            counts[status] += 1
            n_rel = sum(1 for x in ev if x["relation"] != "near_miss")
            log(f"gap {g['id']}: {status} ({len(works)} papers, {n_rel} relevant of {len(top)} read; "
                f"{'+'.join(sorted(searched))}; OpenAlex left {oa_left}; {time.time() - t0:.0f} s) [{query}] "
                f"{g['statement'][:80]}")
        except Exception as e:
            counts["failed"] += 1
            log(f"FAILED gap {g['id']}: {e!r}")
        done += 1
    log(f"finished: {done} questions in {time.time() - t_all:.0f} s; "
        + ", ".join(f"{k} {v}" for k, v in counts.most_common()) + f"; OpenAlex searches left today {oa_left}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
