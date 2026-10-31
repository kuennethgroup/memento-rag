"""Gap ledger: processing one new document, fast. Shared by the background updater (scripts/update_gap_ledger.py, 2
model calls at a time, only while the model is idle) and by the open-research-questions report when a document is still
pending (Query pipe, 3 at a time while the reader waits).

For one document: its own questions (extract → judge → merge, as scripts/build_gap_ledger.py), the questions already
checked against it (one search per question within the document; the model only above the reranker threshold), then
the library check and the literature check of its new questions (as scripts/check_gap_ledger.py and
scripts/check_gap_literature.py), each step in parallel. A per-document lock (one row) keeps the updater
and a report from doing the same document twice."""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from sqlalchemy import Column, DateTime, MetaData, String, Table, delete, insert, select
from sqlalchemy.exc import IntegrityError

GAP_EMBED_INSTRUCTION = "Represent this research question so that the same open question, however worded, is found."
LOCK_STALE = timedelta(minutes=20)
RERANK_IN_DOC = 8           # search results per question the reranker reads within the new document
SEARCH_THREADS = 6          # searches and reranking side by side (model calls stay limited by Ctx.workers)

METADATA = MetaData()
JOBS = Table(
    "memento_gap_doc_jobs", METADATA,
    Column("collection", String(200), primary_key=True),
    Column("source", String(500), primary_key=True),
    Column("owner", String(100)),                         # "updater" | "report"
    Column("started_at", DateTime(timezone=True), nullable=False),
)


def ensure_tables(engine) -> None:
    METADATA.create_all(engine, tables=[JOBS])


def claim(engine, collection: str, source: str, owner: str) -> bool:
    """Take the document for processing; False when someone else has it (a lock older than LOCK_STALE is taken over:
    its process died)."""
    now = datetime.now(timezone.utc)
    for _ in range(2):
        try:
            with engine.begin() as c:
                c.execute(insert(JOBS).values(collection=collection, source=source, owner=owner, started_at=now))
            return True
        except IntegrityError:
            with engine.begin() as c:
                row = c.execute(select(JOBS.c.started_at).where(JOBS.c.collection == collection,
                                                                JOBS.c.source == source)).first()
                at = row[0] if row else None
                if at is not None and at.tzinfo is None:
                    at = at.replace(tzinfo=timezone.utc)
                if at is None or now - at > LOCK_STALE:
                    c.execute(delete(JOBS).where(JOBS.c.collection == collection, JOBS.c.source == source))
                    continue
            return False
    return False


def release(engine, collection: str, source: str) -> None:
    with engine.begin() as c:
        c.execute(delete(JOBS).where(JOBS.c.collection == collection, JOBS.c.source == source))


def in_progress(engine, collection: str, source: str) -> bool:
    with engine.connect() as c:
        return c.execute(select(JOBS.c.source).where(JOBS.c.collection == collection,
                                                     JOBS.c.source == source)).first() is not None


@dataclass
class Ctx:
    """What the steps need; built by the updater (from the saved valves) and by the Query pipe (its own valves)."""
    engine: object
    qdrant: object
    llm_client: object                 # OpenAI-compatible, sync
    model: str
    collection: str
    embedder_url: str
    embedder_model: str
    reranker_url: str
    instruction: str                   # the domain's retrieval instruction
    retrieve_k: int = 80
    hybrid: bool = True
    hybrid_prefetch_k: int = 80
    workers: int = 2                   # model calls at a time
    s2_key: str = ""
    oa_mailto: str = ""
    oa_key: str = ""
    log: object = print
    rerank_instruction: str | None = None   # the copy's domain; None: the process domain's
    _slots: threading.Semaphore = field(default=None, repr=False)
    _s2_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _s2_last: list = field(default_factory=lambda: [0.0], repr=False)

    def __post_init__(self):
        self._slots = threading.Semaphore(max(1, self.workers))

    def llm(self, messages, max_tokens=8192) -> str:      # as scripts/build_gap_ledger.py (long extraction lists)
        with self._slots:
            r = self.llm_client.chat.completions.create(
                model=self.model, messages=messages, temperature=0.0, max_tokens=max_tokens,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        return r.choices[0].message.content or ""

    def s2_wait(self, pause: float = 1.1) -> None:
        """Semantic Scholar: one search per `pause` seconds across all threads."""
        with self._s2_lock:
            time.sleep(max(0.0, pause - (time.time() - self._s2_last[0])))
            self._s2_last[0] = time.time()


def _pmap(fn, items, workers):
    if not items:
        return []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return list(pool.map(fn, items))


# ── step 1: the document's own questions ──────────────────────────────────────

def extract(ctx: Ctx, source: str, title: str, content_hash: str | None) -> list[int]:
    """Extract → judge → merge (scripts/build_gap_ledger.py); the ids of the questions it raises."""
    from qdrant_client.models import Distance, PointIdsList, PointStruct, VectorParams
    from memento.agentic_validation.verifiers import LLMJudgeVerifier
    from memento.research_gaps import jobs as gj, ledger as gl
    from memento.common.model_servers import vl_embed
    coll, qc, e = ctx.collection, ctx.qdrant, ctx.engine
    gaps_coll = f"{coll}-gaps"
    passages = tagged_passages(qc, coll, source)
    orphans = gl.delete_for_source(e, coll, source)                       # a redo starts clean
    if orphans and qc.collection_exists(gaps_coll):
        qc.delete(gaps_coll, PointIdsList(points=orphans))
    verifier = LLMJudgeVerifier(ctx.llm_client, ctx.model, workers=max(1, ctx.workers))

    def embed(text):
        return vl_embed(ctx.embedder_url, ctx.embedder_model, GAP_EMBED_INSTRUCTION, text, None, 60)

    def search(vec, k):
        if not qc.collection_exists(gaps_coll):
            return []
        return [(int(p.payload["gap_id"]), p.payload["statement"], p.score)
                for p in qc.query_points(gaps_coll, query=vec, limit=k, with_payload=True).points]

    def upsert(gid, vec, statement):
        if not qc.collection_exists(gaps_coll):
            qc.create_collection(gaps_coll, vectors_config=VectorParams(size=len(vec), distance=Distance.COSINE))
        qc.upsert(gaps_coll, [PointStruct(id=int(gid), vector=vec, payload={"gap_id": int(gid), "statement": statement})])

    def relate(question, near):
        return gj.parse_relation(ctx.llm(gj.relation_messages(question, near), max_tokens=96), [g for g, _ in near])

    try:
        items = gj.extract_document(ctx.llm, title or source, passages) if passages else []
        for i, claim_, reason in gj.left_out(items):
            p = passages[i]
            gl.add_scope_only(e, coll, source, point_id=p["point_id"], page=p["page"],
                              signal=p.get("gap_signal") or "", claim=claim_, reason=reason)
        kept, dropped = gj.ground(verifier, items, passages)
        for x in dropped:
            p = passages[x["i"]]
            gl.add_dropped(e, coll, source, point_id=p["point_id"], page=p["page"], question=x["question"],
                           claim=x["claim"], reason=x["reason"])
        ids = []

        def add_mention(gid, x):
            p = passages[x["i"]]
            ids.append(gid)
            gl.add_mention(e, gid, coll, source=source, point_id=p["point_id"], page=p["page"],
                           signal=p.get("gap_signal") or "", claim=x["claim"], passage=p["text"], judge=x["judge"],
                           judge_reason=x.get("judge_reason", ""), question=x["question"])
        for x in kept:                                                     # merging stays one at a time (order)
            gj.merge(x, embed=embed, search=search, relate=relate,
                     add_gap=lambda s, k: gl.add_gap(e, coll, s, k), add_mention=add_mention, upsert=upsert,
                     add_relation=lambda a, b: gl.add_relation(e, coll, a, b))
        gl.set_document(e, coll, source, content_hash=content_hash, status="done", n_questions=len(kept),
                        n_scope=len(gj.left_out(items)), n_dropped=len(dropped))
        return sorted(set(ids))
    except Exception as ex:
        gl.set_document(e, coll, source, content_hash=content_hash, status="failed", error=repr(ex)[:500])
        raise


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


# ── step 2: the existing questions, checked against the new document ──────────

def check_against(ctx: Ctx, source: str, gaps: list[dict]) -> dict:
    """Every checked open / partly / contested question it does not raise: its best passages within the document
    (all questions encoded in one batch; searches and reranking side by side), the model only for passages scoring at
    least jobs.NEW_DOC_FLOOR; the labels combine with the stored evidence. Counts by outcome."""
    from qdrant_client.models import FieldCondition, Filter, MatchValue
    from memento.research_gaps import jobs as gj, ledger as gl
    from memento.retrieval.search import point_to_hit, qdrant_query, rerank
    from memento.common.model_servers import vl_embed_many
    others = [g for g in gaps if g.get("library_checked_at") and source not in set(g["documents"])]
    # an answered question cannot become more answered: counted, not checked (deletion counts the document off
    # every question, so adding must count it on every question too)
    answered = [g for g in others if g.get("library_status") not in ("open", "partly", "contested")]
    for g in answered:
        gl.add_library_evidence(ctx.engine, g["id"], status=g["library_status"], new_docs=1, evidence=[])
    todo = [g for g in others if g.get("library_status") in ("open", "partly", "contested")]
    if not todo:
        return {"counted": len(answered)} if answered else {}
    flt = Filter(must=[FieldCondition(key="source", match=MatchValue(value=source))],
                 must_not=[FieldCondition(key="section", match=MatchValue(value="Keywords"))])
    t0 = time.time()
    vecs = []
    for k in range(0, len(todo), 32):
        vecs += vl_embed_many(ctx.embedder_url, ctx.embedder_model,
                              [{"instruction": ctx.instruction, "text": g["statement"]} for g in todo[k:k + 32]])

    def candidates(i):
        g = todo[i]
        pts = qdrant_query(ctx.qdrant, ctx.collection, vecs[i], g["statement"], flt, 20, ctx.hybrid, 20).points
        hits = [{**point_to_hit(p.payload or {}, p.score, p.id), "point_id": p.id} for p in pts
                if (p.payload or {}).get("text", "").strip()]
        # the reranker reads the 8 best search results of a document's passages (more queue the reranker for little
        # gain)
        return gj.worth_checking(rerank(g["statement"], hits[:RERANK_IN_DOC], ctx.reranker_url, 5, 0.0, 0.0,
                                        instruction=ctx.rerank_instruction) if hits else [])
    t1 = time.time()
    found = _pmap(candidates, list(range(len(todo))), SEARCH_THREADS)
    t2 = time.time()

    def judge(i):
        g, cands = todo[i], found[i]
        if not cands:
            gl.add_library_evidence(ctx.engine, g["id"], status=g["library_status"], new_docs=1, evidence=[])
            return "no passage"
        claim_ = g["mentions"][0]["claim"] if g["mentions"] else ""
        labels = gj.judge_labels(ctx.llm, lambda sub: gj.check_messages(g["statement"], claim_, sub), cands)
        if not gj.complete(labels):            # not all judged: this document changes and counts nothing
            ctx.log(f"gap {g['id']}: incomplete judgment against {source[:40]} "
                    f"({sum(lb['relation'] == gj.UNKNOWN for lb in labels)} of {len(labels)} unjudged)")
            return "incomplete"
        old = [x["relation"] for x in gl.evidence(ctx.engine, g["id"], origin="library")]
        status = gj.combined_status(old, labels)
        ev = [dict(source=h["source"], point_id=h["point_id"], page=h.get("page"), relation=lb["relation"],
                   reason=lb["reason"]) for h, lb in zip(cands, labels) if lb["relation"] != "not_concern"]
        gl.add_library_evidence(ctx.engine, g["id"], status=status, new_docs=1, evidence=ev)
        if status != g["library_status"]:
            ctx.log(f"gap {g['id']}: {g['library_status']} → {status} {g['statement'][:90]}")
            return "changed"
        return "unchanged"
    outcomes = _pmap(judge, list(range(len(todo))), SEARCH_THREADS) + ["counted"] * len(answered)
    ctx.log(f"against {source[:50]}: encode {t1 - t0:.0f} s, search + rerank {t2 - t1:.0f} s, "
            f"model {time.time() - t2:.0f} s ({sum(1 for f in found if f)} questions read)")
    return {k: outcomes.count(k) for k in set(outcomes)}


# ── step 3: the new questions' library check ──────────────────────────────────

def check_library(ctx: Ctx, gaps: list[dict], n_docs: int) -> dict:
    """The library check of scripts/check_gap_ledger.py for these questions, side by side."""
    from qdrant_client.models import FieldCondition, Filter, MatchAny
    from memento.research_gaps import jobs as gj, ledger as gl
    from memento.retrieval.search import point_to_hit, qdrant_query, related_heads, rerank
    from memento.common.model_servers import vl_embed

    def one(g):
        stating = set(g["documents"])
        claim_ = g["mentions"][0]["claim"] if g["mentions"] else ""
        flt = Filter(must_not=[FieldCondition(key="source", match=MatchAny(any=sorted(stating))),
                               FieldCondition(key="section", match=MatchAny(any=["Keywords"]))])
        qvec = vl_embed(ctx.embedder_url, ctx.embedder_model, ctx.instruction, g["statement"], None, 60)
        pts = qdrant_query(ctx.qdrant, ctx.collection, qvec, g["statement"], flt, ctx.retrieve_k, ctx.hybrid,
                           ctx.hybrid_prefetch_k).points
        hits = [{**point_to_hit(p.payload or {}, p.score, p.id), "point_id": p.id} for p in pts
                if (p.payload or {}).get("text", "").strip()]
        hits = [{**h, "point_id": h.get("chunk_id") or h["point_id"]}
                for h in rerank(g["statement"], hits, ctx.reranker_url, 15, 0.0, 0.0,
                                instruction=ctx.rerank_instruction)]
        heads = related_heads(ctx.qdrant, ctx.collection, qvec, flt, 200)
        heads = [{**h, "point_id": h.get("chunk_id")} for h in rerank(g["statement"], heads[:30], ctx.reranker_url,
                                                                       30, 0.0, 0.0,
                                                                       instruction=ctx.rerank_instruction)]
        cands = gj.check_candidates(hits, heads, stating=stating, cap=15)
        labels = gj.judge_labels(ctx.llm, lambda sub: gj.check_messages(g["statement"], claim_, sub), cands)
        if not gj.complete(labels):            # not all judged: no status (stays "not checked yet")
            ctx.log(f"gap {g['id']}: incomplete library judgment "
                    f"({sum(lb['relation'] == gj.UNKNOWN for lb in labels)} of {len(labels)} unjudged)")
            return "incomplete"
        status = gj.library_status(labels)
        ev = [dict(source=h["source"], point_id=h["point_id"], page=h.get("page"), relation=lb["relation"],
                   reason=lb["reason"]) for h, lb in zip(cands, labels) if lb["relation"] != "not_concern"]
        if status == "open":
            ev += [dict(source=h["source"], point_id=h["point_id"], page=h.get("page"), relation="near_miss",
                        reason=lb["reason"]) for h, lb in gj.near_misses(cands, labels)]
        gl.set_library_status(ctx.engine, g["id"], status=status, docs_checked=n_docs - len(stating), evidence=ev)
        return status
    out = _pmap(one, gaps, SEARCH_THREADS)
    return {k: out.count(k) for k in set(out)}


# ── step 4: the new questions' literature check ───────────────────────────────

def check_literature(ctx: Ctx, gaps: list[dict], lib_titles: list[str], docs: list[dict]) -> dict:
    """The literature check of scripts/check_gap_literature.py for these questions, side by side (Semantic Scholar
    paced across threads; OpenAlex while its daily budget lasts)."""
    from memento.research_gaps import fulltext as ft, literature as al, jobs as gj, ledger as gl, literature as ls
    from memento.retrieval.search import rerank
    ls.ensure_tables(ctx.engine)
    oa = {"left": None}

    def search(q, searched):
        found = []
        if ctx.s2_key:
            ctx.s2_wait()
            try:
                found += ls.search_s2(q, engine=ctx.engine, api_key=ctx.s2_key, n=25)
                searched.add("semantic_scholar")
            except ls.SearchFailed:
                pass
        if oa["left"] is None or oa["left"] > 3:
            try:
                works, left = ls.search_openalex(q, engine=ctx.engine, n=25, mailto=ctx.oa_mailto, api_key=ctx.oa_key)
                found += works
                searched.add("openalex")
                oa["left"] = left if left is not None else oa["left"]
            except ls.SearchFailed as e:
                if "429" in str(e):
                    oa["left"] = 0
        if al.available():                                 # the offline literature index: local, no rate limits
            found += al.search(q, n=25)
            searched.add("alexandria")
        return found

    def one(g):
        claim_ = g["mentions"][0]["claim"] if g["mentions"] else ""
        query = gj.parse_query(ctx.llm(gj.query_messages(g["statement"]), max_tokens=100))
        searched: set = set()
        works = search(query, searched)
        short = gj.shorter_query(query) if len(ls.merge(works)) < 5 else None
        if short:
            works += search(short, searched)
            query = f"{query} | {short}"
        if not searched:
            raise RuntimeError("no literature service answered")
        stating = [t for d in docs if d["source"] in set(g["documents"]) for t in (d.get("parsed_title"),
                                                                                    d.get("pdf_title"))]
        works = ls.drop_library(ls.merge(works), lib_titles + stating)
        hits = [{"text": f"{w.title}. {w.abstract}".strip(), "type": "text", "work": w} for w in works]
        top = [h["work"] for h in (rerank(g["statement"], hits, ctx.reranker_url, 10, 0.0, 0.0,
                                          instruction=ctx.rerank_instruction) if hits else [])][:10]
        labels = []
        if top and ft.enabled():
            ft.attach(top, g["statement"], query, rank=ft.ranker(g["statement"], ctx.reranker_url,
                                                                 ctx.rerank_instruction))
        if top:
            labels = gj.judge_labels(ctx.llm, lambda sub: gj.web_check_messages(g["statement"], claim_, sub), top)
            if not gj.complete(labels):        # not all judged: no status (stays "not checked yet")
                ctx.log(f"gap {g['id']}: incomplete literature judgment "
                        f"({sum(lb['relation'] == gj.UNKNOWN for lb in labels)} of {len(labels)} unjudged)")
                return "incomplete"
        status = gj.web_status(labels)
        ev = [dict(origin=w.source, source=w.work_id, title=w.title, year=w.year, doi=w.doi, url=w.url,
                   relation=lb["relation"], reason=lb["reason"], text=ft.judged_text(w))
              for w, lb in zip(top, labels) if lb["relation"] != "not_concern"]
        if status == "nothing_found":
            ev += [dict(origin=w.source, source=w.work_id, title=w.title, year=w.year, doi=w.doi, url=w.url,
                        relation="near_miss", reason=lb["reason"], text=ft.judged_text(w))
                   for w, lb in gj.web_near_misses(top, labels)]
        gl.set_web_status(ctx.engine, g["id"], status=status, query=query, sources=sorted(searched), evidence=ev)
        return status
    out = _pmap(one, gaps, SEARCH_THREADS)
    return {k: out.count(k) for k in set(out)}


# ── one document, all steps ───────────────────────────────────────────────────

def process_document(ctx: Ctx, source: str, *, owner: str, status=None, literature: bool = True) -> dict | None:
    """All steps for one new or changed document, under its lock; None when another process has it. `status(text)`
    is called before each step (the report's progress lines)."""
    from memento.research_gaps import ledger as gl
    from memento.library import documents as ldb
    say = status or (lambda _t: None)
    ensure_tables(ctx.engine)
    gl.ensure_tables(ctx.engine)
    if not claim(ctx.engine, ctx.collection, source, owner):
        return None
    t0, result = time.time(), {"s": {}}
    lap = [time.time()]

    def took(step):
        result["s"][step] = round(time.time() - lap[0])
        lap[0] = time.time()
    try:
        docs = ldb.list_documents(ctx.engine, ctx.collection)
        row = next((d for d in docs if d["source"] == source), {})
        say("extract")
        new_ids = extract(ctx, source, row.get("title") or source, row.get("content_hash"))
        result["questions"] = len(new_ids)
        took("extract")
        gaps = gl.list_gaps(ctx.engine, ctx.collection)
        mine = [g for g in gaps if g["id"] in set(new_ids) and not g.get("library_checked_at")]

        def new_questions():                      # its own questions: the library, then the literature
            say("library")
            t = time.time()
            # counted: the documents the ledger has processed, and this one. Documents still waiting (uploaded in the
            # same batch, say) are counted once each when they are processed (check_against), not here as well
            done = gl.processed_sources(ctx.engine, ctx.collection) & {d["source"] for d in docs}
            result["library"] = check_library(ctx, mine, len(done | {source}))
            result["s"]["library"] = round(time.time() - t)
            if literature:
                say("literature")
                t = time.time()
                lib_titles = [x for d in docs for x in (d.get("parsed_title"), d.get("pdf_title"), d["source"]) if x]
                result["literature"] = check_literature(
                    ctx, [g for g in gl.list_gaps(ctx.engine, ctx.collection)
                          if g["id"] in set(new_ids) and not g.get("web_checked_at")], lib_titles, docs)
                result["s"]["literature"] = round(time.time() - t)
        # the questions already there, checked against it, at the same time (independent; the model slots are shared)
        say("against")
        with ThreadPoolExecutor(max_workers=1) as pool:
            side = pool.submit(new_questions)
            t = time.time()
            result["against"] = check_against(ctx, source, [g for g in gaps if g["id"] not in set(new_ids)])
            result["s"]["against"] = round(time.time() - t)
            side.result()
        result["seconds"] = round(time.time() - t0)
        ctx.log(f"processed {source}: {result}")
        return result
    finally:
        release(ctx.engine, ctx.collection, source)
