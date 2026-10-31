"""Open research questions across the library: the report (for a topic, or one picked document, page by page), the
proposal drafted from its questions or from a topic, the outside papers' cards, the documents still to be read for
it, and the check of the report's prose against its sources.
"""
import asyncio
import os
import re
import time
from pathlib import Path

from memento.config.paths import PROJECT_ROOT
from memento.query import cache, generation, library_state, presentation, request, texts
from memento.research_gaps.update import GAP_EMBED_INSTRUCTION


# ── Open research questions across the library (memento/research_gaps/report.py) ──
# Reached by an exact phrase (the start-screen button, or typed): "Research gaps [on <topic>]" ("Open research …" too).
# Built from the stored ledger only; the model's prose is checked against the sent sources (no new searches) before it
# is shown, and unchecked prose is left out.
_GAP_REPORT_RE = re.compile(r"^\s*(?:open )?research (?:gaps|questions)(?:\s+(?:on|about)\s+(.*?))?"
                            r"(?:\s+(?:across|in) the (?:whole )?library)?\s*[.?!]?\s*$", re.I | re.S)


GAP_REPORT_NONE_MSG = ("No open questions on **{topic}** were found among the questions the documents of this library "
                       "name. Try a broader topic, or the whole library. To see one document's limitations and future "
                       "work, pick it under **Limitations & future work** on the start screen, or name it.")


def shown_topic(topic: str | None) -> str:
    """A report topic as the messages show it (the question framing removed)."""
    from memento.research_gaps.report import display_topic
    return display_topic(topic) or "the whole library"



# documents uploaded but not yet in the gap ledger when a report is asked: processed in the request, at most this many
# (others are named as pending), with this many model calls at a time; a document the background updater is processing
# is waited for, at most GAP_REPORT_WAIT_S
GAP_REPORT_PROCESS, GAP_REPORT_WORKERS, GAP_REPORT_WAIT_S = 2, 3, 300


def gap_report_request(text: str):
    """(topic or None,) for the report phrase, else None."""
    m = _GAP_REPORT_RE.match(text or "")
    if not m:
        return None
    topic = (m.group(1) or "").strip().strip('"“”\'').strip()
    return (topic or None,)


# the start screen's "Draft a proposal": a quoted topic; the proposal builds on its best question
_PROPOSAL_TOPIC_RE = re.compile(r'^\s*draft an? (?:research )?proposal (?:on|about) ["“](.+?)["”]\s*[.?!]?\s*$', re.I | re.S)


def proposal_topic_request(text: str) -> str | None:
    m = _PROPOSAL_TOPIC_RE.match(text or "")
    return m.group(1).strip() if m and m.group(1).strip() else None


# One document's open questions: the start screen's picker sends this phrase with the name as it lists it (the title, or
# the file name when two documents share a title); the report is the usual one, limited to that document's questions, so
# its proposal buttons work unchanged. Straight or curly quotes: the chat box may turn typed quotes into curly ones.
_DOC_REPORT_RE = re.compile(r'^\s*research gaps in the document ["“](.*)["”]\s*$', re.I | re.S)


DOC_REPORT_UNKNOWN_MSG = ('**"{name}"** is not in the library (it may have been removed or renamed). Pick it again under '
                          "**Draft a proposal** on the start screen.")


DOC_REPORT_PENDING_MSG = "**{name}**: Paper still processing, try again later."


DOC_REPORT_CHECKING_MSG = ("The open questions of **{name}** are still being checked against the library. Please try "
                           "again later.")


DOC_REPORT_FAILED_MSG = ("The open questions of **{name}** could not be read this time. To see its limitations and "
                         "future work, pick it under **Limitations & future work** on the start screen.")


DOC_REPORT_NONE_MSG = ("**{name}** names no open questions that could be checked against the library. To see its "
                       "limitations and future work, pick it under **Limitations & future work** on the start screen.")


# the next page of a report: only right after a report; "tell me more about …" stays a question
_MORE_RE = re.compile(r"^\s*(?:show(?: me)?(?: the)? )?(?:more|next)(?: (?:\d+|ten|page))?"
                      r"(?: (?:research gaps|open (?:research )?questions|questions|gaps))?(?: please)?\s*[.!?]?\s*$",
                      re.I)


def more_request(text: str) -> bool:
    return bool(_MORE_RE.match(text or ""))


def doc_report_request(text: str) -> str | None:
    """The picked document's name for the document-report phrase, else None."""
    m = _DOC_REPORT_RE.match(text or "")
    return m.group(1).strip() if m and m.group(1).strip() else None


def library_source(name: str, docs: dict) -> str | None:
    """The source of a document named as the picker lists it: its file name, or its title when exactly one document
    has that title (quotes in titles are sent as ')."""
    if name in docs:
        return name
    hits = [src for src, row in docs.items() if ((row or {}).get("title") or "").replace('"', "'") == name]
    return hits[0] if len(hits) == 1 else None


def is_gap_report(text: str) -> bool:
    """An earlier answer is an open-research-questions report (a proposal can be drafted from its questions)."""
    return bool(re.match(r"\s*## Open research questions\b", text or ""))


_PAPER_SERVICES = {"semantic_scholar": "Semantic Scholar", "openalex": "OpenAlex", "alexandria": "Offline index"}


def report_citation(i: int, s: dict, files_url: str, card_url: str | None = None) -> dict:
    """Citation event of report source [i+1]: a library passage opens in the side panel; an outside paper its card
    (`card_url`, the file server's /paper page), else its abstract with a link."""
    if s.get("kind") != "paper":
        return presentation.gap_citation(i, s, files_url)
    # the services that found it; the text shown is the abstract, the excerpts stay on this server
    from memento.research_gaps import fulltext as ft
    found = "; " + ", ".join(_PAPER_SERVICES[k] for k in s.get("services") or [] if k in _PAPER_SERVICES)
    name = (f"[{i + 1}] {' '.join(str(s.get('title') or s['source']).split())} ({s.get('year') or 'n.d.'}"
            f"{found if len(found) > 2 else ''})")
    src = {"name": name}
    if s.get("url"):
        src["url"] = s["url"]
    if card_url:
        src["embed_url"] = card_url
    return {"type": "citation", "data": {"document": [ft.public_text(s.get("text"))],
                                         "metadata": [{"source": f"paper#{i}", "name": name}], "source": src}}


async def gap_report(valves, topic: str | None, *, db_engine, variant, emit, status, document: str | None = None,
                      page: int = 1):
    """The open-research-questions report (memento/research_gaps/report.py): code renders every fact from the ledger; the
    chosen model writes the overview, "why it matters" and suggestions; the overview and "why it matters" are
    checked against the sent sources without new searches, and left out when the check cannot finish. `document`:
    the report of one document's questions (the name as the picker lists it). `page`: the next questions after
    earlier pages ("more research gaps"), numbered on."""
    import logging
    from qdrant_client import QdrantClient
    from memento.research_gaps import ledger as gl, report as gr
    from memento.retrieval.search import rerank
    log = logging.getLogger("memento.gapreport")
    t0 = time.monotonic()
    rbudget = generation.response_budget(valves, thorough=True)        # a long task: the longer budget
    engine, coll = db_engine(), valves.COLLECTION
    dom = request.domain(valves.DOMAIN_NAME)
    await asyncio.to_thread(gl.ensure_tables, engine)
    src = shown = None
    if document:
        docs = await asyncio.to_thread(library_state.load_documents, coll, dom, db_engine) or {}
        src = library_source(document, docs)
        if src is None:
            yield DOC_REPORT_UNKNOWN_MSG.format(name=document)
            return
        shown = (docs.get(src) or {}).get("title") or src                 # as the picker shows it
    # documents uploaded but not yet in the ledger: processed now, while the reader waits; for one document's report
    # only that document
    await process_pending(valves, engine, coll, dom, status, log, only=src, db_engine=db_engine)
    gaps = await asyncio.to_thread(gl.list_gaps, engine, coll)
    if document:
        row = await asyncio.to_thread(gl.get_document, engine, coll, src)
        fresh = not gl.pending_documents([{"source": src, **(docs.get(src) or {})}], [row] if row else [])["new"]
        mine = [g for g in gaps if src in (g.get("documents") or [])]
        if not row or not fresh:
            yield DOC_REPORT_PENDING_MSG.format(name=shown)
            return
        if row.get("status") == "failed":
            yield DOC_REPORT_FAILED_MSG.format(name=shown)
            return
        if not mine:
            yield DOC_REPORT_NONE_MSG.format(name=shown)
            return
        if not any(g.get("library_status") for g in mine):
            yield DOC_REPORT_CHECKING_MSG.format(name=shown)
            return
        gaps = mine
    elif not any(g.get("library_status") for g in gaps):
        yield texts.GAP_REPORT_EMPTY_MSG
        return
    await status("Finding the open questions on this topic..." if topic else
                 "Collecting the document's open questions..." if document else "Collecting the open questions...")
    qc = QdrantClient(url=valves.QDRANT_URL, timeout=30)
    rel = None
    if topic:
        from memento.common.model_servers import vl_embed
        vec = await asyncio.to_thread(vl_embed, valves.EMBEDDER_URL, valves.EMBEDDER_MODEL,
                                      GAP_EMBED_INSTRUCTION, topic, None, 30)
        nearest = lambda _t, k: [int(p.id) for p in qc.query_points(f"{coll}-gaps", query=vec, limit=k).points]
        scores = await asyncio.to_thread(
            gr.topic_relevance, topic, gaps, nearest=nearest,
            rerank_fn=lambda q, hits, n: rerank(q, hits, valves.RERANKER_URL, n, 0.0, 0.0,
                                                instruction=request.rerank_instruction(dom)))
        rel = gr.relevant_only(scores)
        if not rel:
            yield GAP_REPORT_NONE_MSG.format(topic=shown_topic(topic))
            return
    links = await asyncio.to_thread(gl.related_counts, engine, coll)
    # one document: no cap on its own questions (the cap spreads a library report over documents); pages: page 1 the
    # most open, "more research gaps" the next ones
    sel = gr.page_select(gaps, page=page, per_page=valves.GAP_REPORT_FULL, relevance=rel,
                         related_counts=links,
                         **({"per_document": valves.GAP_REPORT_FULL} if document else {}))
    if page > 1 and not sel["full"]:
        yield texts.GAP_REPORT_NO_MORE_MSG
        return
    answered_shown = sel["addressed"][:gr.TABLE_ROWS] if page == 1 else []
    about = f'the open questions of the document "{shown}"' if document else topic   # for the writer and the check
    evidence = await asyncio.to_thread(gl.evidence_many, engine,              # one query
                                       [g["id"] for g in sel["full"] + answered_shown])
    passages = await asyncio.to_thread(gr.load_passages, qc, coll,
                                       gr.needed_points(sel["full"], evidence, answered_shown))
    # the answering pages of "possibly already answered" numbered after the report's own sources; the writer and its
    # check see only the report's own
    sources, refs = gr.gather_sources(sel["full"], evidence, passages, answered=answered_shown)
    own = gr.own_source_count(refs, sel["full"])
    if gr.reads_full_text(evidence):       # what follows may carry publisher text: never a cloud model, in this chat too
        variant = next(v for v in generation.copy_variants(valves) if not v.is_cloud)
        yield texts.FULLTEXT_NOTE
    docs = await asyncio.to_thread(library_state.load_documents, coll, dom, db_engine) or {}
    names = {src: (row or {}).get("title") or src for src, row in docs.items()}      # as the picker shows them
    by_doc = None                          # topic and document reports: the further questions' table
    if not topic and not document:         # the whole library: documents instead (page 1), nothing after
        seen_titles: dict = {}
        for src, t in names.items():
            seen_titles.setdefault(t, []).append(src)
        pick = {src: (t if len(seen_titles[t]) == 1 else src) for src, t in names.items()}   # as the picker sends
        by_doc = gr.by_document(gaps, pick) if page == 1 else []
    # "checked against the other N documents": never more than the library has
    sel = {**sel, "full": gr.cap_docs_checked(sel["full"], len(docs) or None),
           "addressed": gr.cap_docs_checked(sel["addressed"], len(docs) or None) if page == 1 else []}
    cards = await paper_cards(valves, sources, refs, sel["full"],
                                    [f"Question {sel.get('start', 1) + k}" for k in range(len(sel["full"]))])
    for i, s in enumerate(sources):
        await emit(report_citation(i, s, valves.FILES_URL, cards.get(i)))
    questions = [{"n": k, "statement": g["statement"],
                  "facts": f"{gr.library_line(g, refs[g['id']])} {gr.literature_line(g, refs[g['id']], {})}",
                  "sources": refs[g["id"]]["stating"] + [n for n, _ in refs[g["id"]]["library"]]
                  + [n for n, _ in refs[g["id"]]["papers"]]} for k, g in enumerate(sel["full"], start=1)]
    await status("Writing the overview...")
    prose, answered = None, variant
    try:
        chunks, answered, note = await generation.open_answer(
            valves, variant, messages=gr.prose_messages(questions, sources[:own], about), think=False, max_tokens=6000,
            budget=rbudget, held=True)
        raw = await generation.drain_content(chunks, answered.is_cloud)
        prose = gr.parse_prose(cache.strip_think(raw), n_sources=own, n_questions=len(sel["full"]))
    except Exception as e:
        log.warning("prose failed: %r", e)
    final, checked = {"overview": "", "why": {}, "suggestions": []}, False
    t_prose = time.monotonic()
    if prose is not None:
        final["suggestions"] = prose["suggestions"]
        final["short"] = prose.get("short") or {}              # the table's short titles
        text = gr.checkable_text(prose)
        if text:
            await status("Checking the overview against its sources...")
            try:
                res = await asyncio.wait_for(asyncio.to_thread(check_report, valves, text, sources[:own], about),
                                             generation.check_seconds(valves, rbudget))
                final.update(gr.split_checked(res.answer))
                checked = generation.report_checked(res)
                log.info("check: claims=%d removed=%d unjudged=%d", len(res.claims), len(res.removed),
                         int(getattr(res, "unjudged", 0) or 0))
            except Exception as e:
                logging.getLogger("memento.errors").warning("E-gapreport-check: %r", e)
        else:
            checked = True
    log.info("report topic=%r document=%r questions=%d sources=%d prose=%.0fs check=%.0fs checked=%s", topic, src,
             len(sel["full"]), len(sources), t_prose - t0, time.monotonic() - t_prose, checked)
    try:                                   # uploaded but not yet through the updater: named at the top
        pending = gl.pending_documents([{"source": k, **(v or {})} for k, v in docs.items()],
                                       await asyncio.to_thread(gl.ledger_documents, engine, coll))["new"]
    except Exception:
        pending = []
    async for piece in presentation.type_out(gr.assemble(sel, sources, refs, final, names=names, topic=topic,
                                          n_documents=len(docs), evidence=evidence, checked=checked,
                                          pending=pending, document=shown, by_document=by_doc),
                              valves.REPLAY_CHARS_PER_S, valves.REPLAY_MAX_S,
                              valves.REPLAY_TYPED_CHARS):
        yield piece
    yield generation.answered_by(answered.label, None if not answered.is_cloud else
                       next(v for v in generation.copy_variants(valves) if not v.is_cloud).label)


async def gap_proposal(valves, nums: list, report_md: str, *, db_engine, variant, emit, status):
    """A draft research proposal on questions of the report (memento/research_gaps/proposal.py): their stored evidence as
    sources; facts and references by code; motivation and background checked against the sources (no new
    searches); the proposed work labelled as the model's suggestion; the text reply and a PDF link."""
    from memento.research_gaps import ledger as gl, proposal as gp
    asked = gp.report_questions(report_md)
    missing = [n for n in nums if n not in asked]
    if missing or not asked:
        have = ", ".join(str(n) for n in sorted(asked)) or "none"
        yield (f"The report above has no question {', '.join(map(str, missing or nums))}. Its questions are "
               f"numbered {have}: for example, reply *draft a proposal on question {min(asked) if asked else 1}*.")
        return
    engine, coll = db_engine(), valves.COLLECTION
    by_text = {g["statement"].strip(): g for g in await asyncio.to_thread(gl.list_gaps, engine, coll)}
    gaps = [by_text[asked[n]] for n in nums if asked[n] in by_text]
    if len(gaps) < len(nums):
        yield ("These questions are no longer in the library's list of open questions (a document was changed or "
               "removed). Please ask for the research gaps again.")
        return
    async for c in proposal_for(valves, gaps, variant=variant, emit=emit, status=status, db_engine=db_engine):
        yield c


async def gap_proposal_topic(valves, topic: str, *, db_engine, variant, emit, status):
    """"Draft a proposal" from the start screen: the topic's questions ranked as for the report (pending documents
    processed first), the proposal built on the best one, the reply naming it."""
    import logging
    from qdrant_client import QdrantClient
    from memento.research_gaps import ledger as gl, report as gr
    from memento.retrieval.search import rerank
    from memento.common.model_servers import vl_embed
    log = logging.getLogger("memento.gapreport")
    engine, coll = db_engine(), valves.COLLECTION
    dom = request.domain(valves.DOMAIN_NAME)
    await asyncio.to_thread(gl.ensure_tables, engine)
    await process_pending(valves, engine, coll, dom, status, log, db_engine=db_engine)
    await status("Finding the most open question on this topic...")
    gaps = await asyncio.to_thread(gl.list_gaps, engine, coll)
    qc = QdrantClient(url=valves.QDRANT_URL, timeout=30)
    vec = await asyncio.to_thread(vl_embed, valves.EMBEDDER_URL, valves.EMBEDDER_MODEL,
                                  GAP_EMBED_INSTRUCTION, topic, None, 30)
    nearest = lambda _t, k: [int(p.id) for p in qc.query_points(f"{coll}-gaps", query=vec, limit=k).points]
    rel = gr.relevant_only(await asyncio.to_thread(
        gr.topic_relevance, topic, gaps, nearest=nearest,
        rerank_fn=lambda q, hits, n: rerank(q, hits, valves.RERANKER_URL, n, 0.0, 0.0,
                                            instruction=request.rerank_instruction(dom))))
    sel = gr.select(gaps, relevance=rel, full=1, related_counts=await asyncio.to_thread(gl.related_counts,
                                                                                         engine, coll))
    if not sel["full"]:
        yield GAP_REPORT_NONE_MSG.format(topic=shown_topic(topic))
        return
    best = sel["full"][0]
    intro = (f"_This draft builds on the most open question on **{topic}** in the library: \"{best['statement']}\" "
             f"To pick another question, use **Draft a proposal** on the start screen with this topic, and click "
             f"the button under the question you want._\n\n")
    async for c in proposal_for(valves, [best], variant=variant, emit=emit, status=status, intro=intro, db_engine=db_engine):
        yield c


async def proposal_for(valves, gaps: list, *, db_engine, variant, emit, status, intro: str = ""):
    """The proposal on these ledger questions (shared by both entries)."""
    import logging
    import secrets
    from qdrant_client import QdrantClient
    from memento.research_gaps import ledger as gl, proposal as gp, report as gr
    log = logging.getLogger("memento.gapreport")
    engine, coll = db_engine(), valves.COLLECTION
    dom = request.domain(valves.DOMAIN_NAME)
    await status("Collecting the evidence for the proposal...")
    qc = QdrantClient(url=valves.QDRANT_URL, timeout=30)
    evidence = {g["id"]: await asyncio.to_thread(gl.evidence, engine, g["id"]) for g in gaps}
    if gr.reads_full_text(evidence):       # as in gap_report
        variant = next(v for v in generation.copy_variants(valves) if not v.is_cloud)
        yield texts.FULLTEXT_NOTE
    passages = await asyncio.to_thread(gr.load_passages, qc, coll, gr.needed_points(gaps, evidence))
    sources, refs = gr.gather_sources(gaps, evidence, passages)
    docs = await asyncio.to_thread(library_state.load_documents, coll, dom, db_engine) or {}
    names = {src: (row or {}).get("title") or src for src, row in docs.items()}
    gaps = gr.cap_docs_checked(gaps, len(docs) or None)        # never more documents than the library has
    cards = await paper_cards(valves, sources, refs, gaps, [f"Q{k}" for k in range(1, len(gaps) + 1)])
    for i, s in enumerate(sources):
        await emit(report_citation(i, s, valves.FILES_URL, cards.get(i)))
    # with pages for the writing model and the PDF; without in chat, where the citation pill shows the page
    facts, facts_chat = (["\n".join(gr.library_block(g, refs[g["id"]], sources, names, pages=pages)
                                     + gr.literature_block(g, refs[g["id"]], sources)) for g in gaps]
                         for pages in (True, False))
    questions = [g["statement"] for g in gaps]
    await status("Drafting the proposal...")
    prose, answered = None, variant
    try:
        chunks, answered, note = await generation.open_answer(
            valves, variant, messages=gp.proposal_messages(questions, facts, sources), think=True, max_tokens=8000,
            local_thinking_budget=valves.GAP_THINKING_BUDGET or None)
        prose = gp.parse_proposal(cache.strip_think(await generation.drain_content(chunks, answered.is_cloud)),
                                  n_sources=len(sources))
    except Exception as e:
        log.warning("proposal prose failed: %r", e)
    if prose is None:
        yield "⚠️ **Sorry, the proposal draft could not be written this time.** Please try again in a minute."
        return
    checked = False
    text = gp.checkable_text(prose)
    if text:
        await status("Checking the draft against its sources...")
        try:
            res = await asyncio.wait_for(asyncio.to_thread(check_report, valves, text, sources, prose["title"],
                                                           gp.PROPOSAL_SYSTEM), valves.VERIFY_TIMEOUT)
            prose.update(gp.split_checked(res.answer))
            checked = generation.report_checked(res)
            log.info("proposal check: claims=%d removed=%d unjudged=%d", len(res.claims), len(res.removed),
                     int(getattr(res, "unjudged", 0) or 0))
        except Exception as e:
            logging.getLogger("memento.errors").warning("E-proposal-check: %r", e)
    md = gp.assemble(prose, questions=questions, facts=facts, sources=sources, names=names, checked=checked)
    link = ""
    try:
        token = secrets.token_hex(12)
        root = os.getenv("OUTPUT_ROOT") or str(PROJECT_ROOT / "data/output")
        await asyncio.to_thread(gp.write_pdf, md, Path(root) / "_runtime" / "reports" / f"{token}.pdf")
        link = f"\n[Download as PDF]({valves.FILES_URL.rstrip('/')}/report/{token}.pdf)\n"
    except Exception as e:
        logging.getLogger("memento.errors").warning("E-proposal-pdf: %r", e)
    # the chat reply collapsed: title, summary and the PDF link visible; the PDF stays plain (above)
    shown = gp.assemble(prose, questions=questions, facts=facts_chat, sources=sources, names=names,
                        checked=checked, collapsed=True, link=link)
    async for piece in presentation.type_out(intro + shown, valves.REPLAY_CHARS_PER_S, valves.REPLAY_MAX_S,
                              valves.REPLAY_TYPED_CHARS):
        yield piece
    yield generation.answered_by(answered.label, None if not answered.is_cloud else
                       next(v for v in generation.copy_variants(valves) if not v.is_cloud).label)


async def paper_cards(valves, sources: list, refs: dict, gaps: list, labels: list) -> dict:
    """{source index: card url} for the outside papers of a report or proposal: per paper a JSON card in
    OUTPUT_ROOT/_runtime/papers/<24 hex>.json for the file server's /paper page: title, year, journal (offline
    literature index), DOI, the services that found it, per citing question its label, verdict and the checker's
    reason, and the abstract's sentences with each question's most related ones (reranker, scored now). Best effort:
    no reranker, no marks; a card that cannot be written, no card."""
    import json as _json
    import secrets
    from memento.research_gaps import fulltext as ft, literature as al, report as gr
    from memento.retrieval.search import rerank
    files = (valves.FILES_URL or "").rstrip("/")
    papers = [i for i, s in enumerate(sources) if s.get("kind") == "paper"]
    if not files or not papers:
        return {}
    root = Path(os.getenv("OUTPUT_ROOT") or str(PROJECT_ROOT / "data/output")) / "_runtime" / "papers"
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError:
        return {}

    def marks(statement: str, sentences: list) -> list:
        if not sentences:
            return []
        try:
            ranked = rerank(statement, [{"text": t, "type": "text", "i": k} for k, t in enumerate(sentences)],
                            valves.RERANKER_URL, len(sentences), 0.0, 0.0,
                            instruction=request.rerank_instruction(request.domain(valves.DOMAIN_NAME)))
        except Exception:
            return []
        scores = [0.0] * len(sentences)
        for h in ranked:                 # a failed reranker returns the hits without scores: nothing marked
            if isinstance(h.get("i"), int) and 0 <= h["i"] < len(sentences):
                scores[h["i"]] = float(h.get("rerank_score") or 0.0)
        return gr.pick_marks(scores)
    jobs, cards = [], {}
    for i in papers:
        s = sources[i]
        sentences = gr.paper_sentences(ft.public_text(s.get("text")))
        qs = []
        for g, label in zip(gaps, labels):
            r = refs.get(g["id"]) or {}
            for n, rel in r.get("papers") or []:
                if n == i + 1:
                    q = {"label": label, "statement": g["statement"], "verdict": gr.WEB_REL.get(rel, rel),
                         "reason": (r.get("reasons") or {}).get(n, ""), "marks": []}
                    qs.append(q)
                    jobs.append((q, g["statement"], sentences))
        try:
            journal = await asyncio.to_thread(al.journal, s.get("doi") or "")
        except Exception:
            journal = ""
        cards[i] = {"title": s.get("title") or s.get("source") or "", "year": str(s.get("year") or ""),
                    "journal": journal, "doi": gr.doi_key(s.get("doi")) or "", "url": s.get("url") or "",
                    "services": gr.service_links(s.get("ids") or {}), "sentences": sentences, "questions": qs,
                    "full_text": ft.read_in_full(s.get("text"))}
    found = await asyncio.gather(*[asyncio.to_thread(marks, st, sn) for _, st, sn in jobs])
    for (q, _, _), m in zip(jobs, found):
        q["marks"] = m
    out = {}
    for i, card in cards.items():
        token = secrets.token_hex(12)
        try:
            (root / f"{token}.json").write_text(_json.dumps(card, ensure_ascii=False))
        except OSError:
            continue
        out[i] = f"{files}/paper/{token}"
    return out


async def process_pending(valves, engine, coll: str, dom: dict, status, log, only: str | None = None, *, db_engine) -> None:
    """Documents of the library not yet through the gap ledger: up to GAP_REPORT_PROCESS of them processed in this
    request (retrieval/gap_update.process_document, 3 model calls at a time), or waited for when the background
    updater has one; any others stay pending and are named in the report. `only`: one document's report needs
    that document alone (processed when pending, nothing done when it is in the ledger). Never fails the report."""
    import logging
    try:
        from openai import OpenAI as SyncOpenAI
        from qdrant_client import QdrantClient
        from memento.research_gaps import ledger as gl, update as gu
        await asyncio.to_thread(gu.ensure_tables, engine)
        docs = await asyncio.to_thread(library_state.load_documents, coll, dom, db_engine) or {}
        pending = gl.pending_documents([{"source": k, **(v or {})} for k, v in docs.items()],
                                       await asyncio.to_thread(gl.ledger_documents, engine, coll))["new"]
    except Exception as e:
        log.warning("pending documents not read: %r", e)
        return
    if only is not None:
        pending = [p for p in pending if p == only]
    if not pending:
        return
    n = min(len(pending), GAP_REPORT_PROCESS)
    await status("Adding the new paper first..." if n == 1 else f"Adding {n} new papers first...")
    ctx = gu.Ctx(engine=engine, qdrant=QdrantClient(url=valves.QDRANT_URL, timeout=60),
                 llm_client=SyncOpenAI(base_url=valves.LLM_BASE_URL, api_key=request.llm_key(valves),
                                       timeout=600),
                 model=valves.LLM_MODEL, collection=coll, embedder_url=valves.EMBEDDER_URL,
                 embedder_model=valves.EMBEDDER_MODEL, reranker_url=valves.RERANKER_URL,
                 instruction=dom.get("instruction") or request.FALLBACK_INSTRUCTION, retrieve_k=valves.RETRIEVE_K,
                 rerank_instruction=request.rerank_instruction(dom),
                 hybrid=valves.HYBRID_ENABLED, hybrid_prefetch_k=valves.HYBRID_PREFETCH_K,
                 workers=GAP_REPORT_WORKERS, s2_key=os.getenv("S2_API_KEY", ""),
                 oa_mailto=os.getenv("OPENALEX_MAILTO", ""), oa_key=os.getenv("OPENALEX_API_KEY", ""),
                 log=log.info)
    for src in pending[:n]:
        step = {"now": None}
        t0 = time.monotonic()
        if await asyncio.to_thread(gu.in_progress, engine, coll, src):
            await status("New paper almost ready...")
            while (await asyncio.to_thread(gu.in_progress, engine, coll, src)
                   and time.monotonic() - t0 < GAP_REPORT_WAIT_S):
                await asyncio.sleep(3)
            continue
        job = asyncio.create_task(asyncio.to_thread(
            gu.process_document, ctx, src, owner="report", status=lambda k: step.__setitem__("now", k)))
        shown = None
        while not job.done():
            if step["now"] != shown and step["now"] in texts.GAP_REPORT_STEPS:
                shown = step["now"]
                await status(texts.GAP_REPORT_STEPS[shown])
            await asyncio.sleep(1)
        try:
            job.result()
        except Exception as e:
            logging.getLogger("memento.errors").warning("E-gap-pending %s: %r", src, e)
        log.info("pending document %s processed in %.0f s", src, time.monotonic() - t0)


def check_report(valves, text: str, sources: list, topic: str | None, system: str | None = None):
    """The after-draft check of the report's prose (strategy C: unsupported statements removed) against the sent
    sources only (no re-retrieval), by the local model or the configured verifier."""
    from openai import OpenAI as SyncOpenAI
    from memento.agentic_validation import strategies
    from memento.agentic_validation.verifiers import LLMJudgeVerifier, VerdictCache
    from memento.research_gaps import report as gr
    key = request.llm_key(valves)
    gen = SyncOpenAI(base_url=valves.LLM_BASE_URL, api_key=key)
    ver = SyncOpenAI(base_url=valves.VERIFIER_URL, api_key=key) if valves.VERIFIER_URL else gen
    ctx = strategies.ArmContext(gen, valves.LLM_MODEL, system or gr.PROSE_SYSTEM,
                                topic or "open research questions of the library", list(sources),
                                LLMJudgeVerifier(ver, valves.VERIFIER_MODEL or valves.LLM_MODEL), None,
                                VerdictCache(), retrieve_fn=None, enable_thinking=False,
                                claim_rules=request.domain_claim_rules(request.domain(valves.DOMAIN_NAME)))
    return strategies.run_arm_c(ctx, text)
