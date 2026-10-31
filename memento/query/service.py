"""One Library Chat or Document Chat message from start to end: set up, the start screen's own requests, the history and
the router, then the ordered checks that hand the reply to the flow that owns it, and the shared answer chain.
"""
import asyncio
import contextlib
import re

from memento.document_chat import chat
from memento.query import cache, generation, library_state, presentation, prompts, request, search_step, texts
from memento.query.flows import answer as answer_flow, gap_answer, gap_report, general, library_listing, lists, rework, summary, verbatim as verbatim_flow
from memento.query.scope import conversation, documents, resolve


def _choose_model(turn) -> None:
    """The model that writes this chat's answers (memento_query.<variant id>); unknown ids answer locally. Document
    Chat, and a chat holding text drawn from papers' full texts, use the local model only."""
    from memento.query.models import resolve as _resolve_variant
    variants = generation.copy_variants(turn.valves)
    turn.variant = _resolve_variant(turn.body.get("model"), variants, turn.valves.QUERY_MODEL_ID)
    turn.local_variant = next(v for v in variants if not v.is_cloud)
    if turn.doc_chat is not None or texts.holds_full_text(turn.body.get("messages") or []):
        turn.variant = turn.local_variant


async def _setup(turn):
    """The message set up: this copy's model and domain, the answer model, the time budget, the question from the
    message (Open WebUI's background tasks answered directly, attachments noted), the first status line, and whether
    the library has documents."""
    from openai import AsyncOpenAI
    turn.api_key = request.llm_key(turn.valves)
    turn.llm = AsyncOpenAI(base_url=turn.valves.LLM_BASE_URL, api_key=turn.api_key)
    turn.dom = request.domain(turn.valves.DOMAIN_NAME)
    _choose_model(turn)
    # the passages searched: the library, or Document Chat's temporary store (its client adds the chat's scope)
    turn.collection = turn.doc_chat.collection if turn.doc_chat is not None else turn.valves.COLLECTION
    # the response's time budget; none for the background warm-up
    turn.thorough_on = request.THOROUGH_FILTER_ID in ((turn.metadata or {}).get("filter_ids") or [])
    turn.budget = generation.response_budget(turn.valves, thorough=turn.thorough_on, warm=bool((turn.metadata or {}).get("warmup")))
    turn.thorough_on = turn.thorough_on and not (turn.metadata or {}).get("warmup")

    # the last user message's text (multimodal content handled)
    turn.chat_messages = turn.body.get("messages") or []
    raw = turn.chat_messages[-1]["content"] if turn.chat_messages else ""

    # Open WebUI's background tasks (chat title, tags, follow-up suggestions) are sent to the active model, prefixed
    # with "### Task:". They are answered directly by the LLM without retrieval or citations. A file attached to
    # this model makes Open WebUI wrap the message in its RAG template, which also starts with "### Task:": the
    # question is recovered instead of being answered as a background task.
    attached = bool(turn.files)
    # __files__ holds every file of the chat (also earlier uploads in the same chat); only the message that carries
    # files gets Open WebUI's <attached_files> block, and the note is for that case (with file context on, Open
    # WebUI wraps that message in its RAG template: <user_query>…)
    raw_text = raw if isinstance(raw, str) else " ".join(p.get("text", "") for p in (raw or []) if isinstance(p, dict))
    attached_now = "<attached_files>" in raw_text or (attached and "<user_query>" in raw_text)
    if attached and isinstance(raw, str) and "<user_query>" in raw:
        m_q = re.search(r"<user_query>(.*?)</user_query>", raw, re.S)
        raw = m_q.group(1).strip() if m_q else raw
    if isinstance(raw, str) and raw.startswith("### Task:") and not attached:
        from openai import AsyncOpenAI
        try:
            turn.llm = AsyncOpenAI(base_url=turn.valves.LLM_BASE_URL, api_key=turn.api_key)
            stream = await turn.llm.chat.completions.create(
                model       = turn.valves.LLM_MODEL,
                messages    = [{"role": "user", "content": raw}],
                temperature = turn.valves.TEMPERATURE,
                max_tokens  = 256,
                stream      = True,
                # vLLM/SGLang OpenAI-compatible APIs take Qwen3's thinking switch nested under chat_template_kwargs;
                # a top-level "enable_thinking" key is ignored.
                extra_body  = {"chat_template_kwargs": {"enable_thinking": False}},
            )
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as e:
            import logging
            logging.getLogger("memento.errors").warning("E-task: %r", e)   # no text: OpenWebUI keeps its default title
        turn.done = True
        return

    turn.query = request.message_text(raw).strip()
    images_now, files_now = request.attachment_kinds(raw, raw_text) if attached_now or isinstance(raw, list) else (False, False)
    if images_now and not files_now and not attached_now:
        attached_now = True

    if images_now and turn.doc_chat is None:      # Document Chat has read the attached image as one of its files
        yield texts.IMAGE_MSG
        turn.done = True
        return
    if not turn.query:
        yield ((texts.DOC_CHAT_READY_MSG if turn.doc_chat.documents else texts.DOC_CHAT_NO_FILE_MSG) if turn.doc_chat is not None
               else "Please enter a question.")
        # the finished bar of files read now stays; else hidden (emit sends nothing without an emitter)
        await turn.emit(presentation.status_event(getattr(turn.doc_chat, "read_line", None) or "Done", True))
        turn.done = True
        return
    if attached_now and (files_now or not images_now) and turn.doc_chat is None:
        yield texts.FILE_NOTE
        if conversation.attachment_only(turn.query):        # about the attached file alone: the note is the reply
            turn.done = True
            return

    # Show a status at once, before the rewrite, document detection and library scroll below, so the reader never
    # sees a bare spinner.
    if turn.thorough_on:                      # said first, so the wait is understood
        await turn.status(texts.THOROUGH_ON_STATUS)
    await turn.status("Looking for answer...")

    # Empty or missing library (a new deployment before the first upload): say so, spend no model call.
    try:
        from qdrant_client import QdrantClient as _QC
        state = "ok" if turn.doc_chat is not None else await asyncio.to_thread(
            library_state.collection_status, _QC(url=turn.valves.QDRANT_URL, timeout=10), turn.valves.COLLECTION)
    except Exception as e:
        yield texts.user_error("library", e)
        await turn.status("Done", done=True)
        turn.done = True
        return
    # an empty library still answers greetings and general questions (the router decides below); everything that
    # needs documents gets EMPTY_LIBRARY_MSG after the router
    turn.library_empty = state == "empty"


async def _special_requests(turn):
    """The start screen's own requests, by their exact phrases: a proposal on a topic or from the report just given, the
    report's next page, one document's open questions, the library's report."""
    topic_req = gap_report.proposal_topic_request(turn.query) if turn.doc_chat is None and not turn.library_empty else None
    if topic_req:
        try:
            async for c in gap_report.gap_proposal_topic(turn.valves, topic_req, variant=turn.variant, emit=turn.emit, status=turn.status, db_engine=turn.engine):
                yield c
        except Exception as e:
            yield texts.user_error("gap", e)
        await turn.status("Done", done=True)
        turn.done = True
        return

    # A draft research proposal from questions of the report just given
    if turn.doc_chat is None and not turn.library_empty:
        from memento.research_gaps.proposal import proposal_request
        previous = rework.previous_answer(turn.chat_messages)
        nums = proposal_request(turn.query) if gap_report.is_gap_report(previous) else None
        if nums:
            try:
                async for c in gap_report.gap_proposal(turn.valves, nums, previous, variant=turn.variant, emit=turn.emit, status=turn.status, db_engine=turn.engine):
                    yield c
            except Exception as e:
                yield texts.user_error("gap", e)
            await turn.status("Done", done=True)
            turn.done = True
            return

    # The next page of the report just given: "more research gaps"
    if turn.doc_chat is None and not turn.library_empty and gap_report.more_request(turn.query):
        from memento.research_gaps import report as _gr
        more_scope = _gr.report_scope(rework.previous_answer(turn.chat_messages))
        if more_scope:
            page = more_scope["last"] // max(1, turn.valves.GAP_REPORT_FULL) + 1
            try:
                async for c in gap_report.gap_report(turn.valves, more_scope["value"] if more_scope["kind"] == "topic" else None,
                                                document=more_scope["value"] if more_scope["kind"] == "document" else None,
                                                page=page, variant=turn.variant, emit=turn.emit, status=turn.status, db_engine=turn.engine):
                    yield c
            except Exception as e:
                yield texts.user_error("gap", e)
            await turn.status("Done", done=True)
            turn.done = True
            return

    # One document's open questions (the start screen's "Draft a proposal" on a picked document)
    doc_report = gap_report.doc_report_request(turn.query) if turn.doc_chat is None and not turn.library_empty else None
    if doc_report:
        try:
            async for c in gap_report.gap_report(turn.valves, None, document=doc_report, variant=turn.variant, emit=turn.emit, status=turn.status, db_engine=turn.engine):
                yield c
        except Exception as e:
            yield texts.user_error("gap", e)
        await turn.status("Done", done=True)
        turn.done = True
        return

    # Open research questions across the library or on a topic: the exact phrase
    report = gap_report.gap_report_request(turn.query) if turn.doc_chat is None and not turn.library_empty else None
    if report is not None:
        try:
            async for c in gap_report.gap_report(turn.valves, report[0], variant=turn.variant, emit=turn.emit, status=turn.status, db_engine=turn.engine):
                yield c
        except Exception as e:
            yield texts.user_error("gap", e)
        await turn.status("Done", done=True)
        turn.done = True
        return


async def _history(turn) -> None:
    """The conversation so far: the earlier turns, sanitized, each answer's [N] named by the citations Open WebUI stored
    with it, the last HISTORY_TURNS pairs within the history budget. `answer_history`: the same turns for a model that
    writes a cited answer, the earlier answers without their citations (it would copy them as its own); the router and
    the other checks read `history`."""
    turn.history, turn.answer_history = [], []
    # earlier answers' [N] → documents, from the citations OpenWebUI stored with them; aligned from the most recent
    stored_names = await presentation.stored_citation_names(turn.metadata)
    n_answers = sum(1 for m in turn.chat_messages[:-1] if m.get("role") == "assistant")
    stored_names = ([{}] * max(0, n_answers - len(stored_names)) + stored_names)[-n_answers:] if n_answers else []
    answer_idx = 0
    for m in turn.chat_messages[:-1]:
        role = m.get("role")
        if role not in ("user", "assistant"):
            continue
        text = answer_text = request.message_text(m.get("content"))
        if role == "assistant":
            names = stored_names[answer_idx] if answer_idx < len(stored_names) else {}
            answer_idx += 1
            if request.INGEST_REPORT_RE.search(text):
                continue
            clean = texts.sanitize_assistant_text(text)
            text, answer_text = conversation.history_text(clean, names), conversation.history_text(clean, cite=False)
        text, answer_text = text.strip(), answer_text.strip()
        if text:
            turn.history.append({"role": role, "content": text})
        if answer_text:
            turn.answer_history.append({"role": role, "content": answer_text})
    turn.history = request.budget_history(turn.history[-(turn.valves.HISTORY_TURNS * 2):])
    turn.answer_history = request.budget_history(turn.answer_history[-(turn.valves.HISTORY_TURNS * 2):])


async def _route(turn) -> None:
    """The route: a yes to "search the whole library instead?", then one router call for the rewritten query and the
    intent (with the part of a paper, the scope, the person, a list's topic and kind), and whether the answer model
    thinks first."""
    turn.search_all = resolve.search_all_request(turn.query, turn.history)
    if turn.search_all:
        import logging
        logging.getLogger("memento.query").info("named document not found earlier: the whole library for %r",
                                                turn.search_all[:200])
        turn.query = turn.search_all
    turn.original_query = turn.query
    turn.query_intent = "ranked_retrieval"
    # Thinking for this answer: ENABLE_THINKING, unless THINKING_AUTO lets the router turn it off for a simple
    # question (router failure / unsure / synthesis intents → keep thinking).
    turn.think = turn.valves.ENABLE_THINKING
    turn.router_ok = False   # True once the LLM router has classified this message; then its label decides
    turn.suggest = False     # the question wants ideas beyond the documents: "Possible extensions"
    turn.element = "none"    # the part of a paper asked for: figure | table | equation | none
    turn.scope = "none"      # what the search covers: document | documents | library | none
    turn.author = ""         # the person whose documents are asked for
    turn.author_docs = []   # their documents, when the question adds a topic: the search stays within them
    turn.list_topic = turn.list_kind = ""     # a list question's topic and the kind of document it asks for
    turn.list_capped = False             # the list scan had more documents than it may add
    if turn.valves.ENABLE_QUERY_REWRITE:
        try:
            from memento.query.router import route, pipe_intent
            routed = await route(turn.llm, turn.valves.LLM_MODEL, turn.query, request.router_history(turn.history))
            turn.query, turn.query_intent = routed.query, pipe_intent(routed.intent)
            turn.router_ok, turn.suggest = routed.ok, routed.suggest
            turn.element = getattr(routed, "element", "none") or "none"
            turn.scope = getattr(routed, "scope", "none") or "none"
            turn.author = getattr(routed, "author", "") or ""
            turn.list_topic = getattr(routed, "topic", "") or ""
            turn.list_kind = getattr(routed, "doc_kind", "") or ""
            if turn.think and turn.valves.THINKING_AUTO and not routed.needs_thinking:
                turn.think = False
            import logging
            logging.getLogger("memento.thinking").info(
                "intent=%s element=%s scope=%s reasoning=%s thinking=%s auto=%s suggest=%s history=%d query=%r",
                routed.intent, turn.element, turn.scope, routed.reasoning, turn.think, turn.valves.THINKING_AUTO, turn.suggest,
                len(turn.history), (turn.query or "")[:200])
        except ImportError:
            turn.query = await prompts.rewrite_query(turn.llm, turn.valves.LLM_MODEL, turn.query, request.router_history(turn.history))

    if turn.think and not generation.list_thinks(turn.think, turn.query_intent, turn.variant.is_cloud, turn.valves.LIST_THINKING):
        import logging
        logging.getLogger("memento.thinking").info("thinking off: a list question on the local model")
    turn.think = generation.list_thinks(turn.think, turn.query_intent, turn.variant.is_cloud, turn.valves.LIST_THINKING)
    # research gaps keep this decision (their stored answers are written with it, GAP_THINKING); every other live
    # answer thinks only with Thorough check on (generation.live_thinks)
    turn.gap_base_think = turn.think
    warm_q = bool((turn.metadata or {}).get("warmup"))
    if turn.think and not generation.live_thinks(turn.think, thorough=turn.thorough_on, warm=warm_q, enabled=turn.valves.LIVE_THINKING):
        import logging
        logging.getLogger("memento.thinking").info("thinking off: a live answer without Thorough check")
    turn.think = generation.live_thinks(turn.think, thorough=turn.thorough_on, warm=warm_q, enabled=turn.valves.LIVE_THINKING)


async def run(turn):
    """One message's reply, in this order: set up, the start screen's requests, history and router, then the ordered
    checks, each early reply handed to the flow that owns it, else the search and the shared answer chain."""
    async with contextlib.aclosing(_setup(turn)) as reply:
        async for piece in reply:
            yield piece
    if turn.done:
        return

    # The start screen's own requests: a proposal, the report's next page, a document's open questions, the report
    async with contextlib.aclosing(_special_requests(turn)) as reply:
        async for piece in reply:
            yield piece
    if turn.done:
        return

    # ── Conversation history (sanitized, last HISTORY_TURNS pairs) ──
    # Everything but the last message, trimmed to the most recent user/assistant pairs and cleaned of rendered
    # artifacts (inline figures, <think> blocks); used to rewrite this turn's query and as context for the answer.
    await _history(turn)

    # ── Query rewrite + intent classification ──
    # One LLM call rewrites the query (resolving pronouns and ellipsis) and classifies its intent; on any failure,
    # ranked_retrieval. A "yes" to "search the whole library instead?" re-asks the previous question over the whole
    # library.
    await _route(turn)

    # Document Chat asked about the whole library: it does not search it, said alone
    lib_note = chat.doc_chat_library_note(turn.doc_chat is not None, turn.scope, turn.original_query)
    if lib_note:
        yield lib_note
        await turn.status("Done", done=True)
        return

    # ── 0c. Greeting / thanks / not about the library: date/time, arithmetic, how-to, rewording pasted text ──
    # FLOW: GENERAL (router labels "general" and "chitchat"): no retrieval, one answer-model call without thinking,
    # no citations, no verification; a general answer is marked as not from the library (not a greeting)
    if turn.query_intent in ("general", "chitchat"):
        async with contextlib.aclosing(general.reply(turn)) as reply:
            async for piece in reply:
                yield piece
        return

    if turn.library_empty:
        yield texts.EMPTY_LIBRARY_MSG
        await turn.status("Done", done=True)
        return
    if turn.doc_chat is not None and not turn.doc_chat.documents:          # Document Chat without a file: nothing to search
        yield chat.doc_chat_no_file(turn.original_query)
        await turn.status("Done", done=True)
        return

    # ── 0b. Rework of the previous answer ──
    # FLOW: REWORK ("summarize that in three bullets", "as a table", "translate it"): no search; the previous answer
    # and its own sources (stored by Open WebUI with that answer) go to the model, and the same citations are sent
    # again, so every [N] still opens the same passage. Without them (API call, error, very long): the normal path.
    if turn.query_intent == "rework":
        async with contextlib.aclosing(rework.reply(turn)) as reply:
            async for piece in reply:
                yield piece
        if turn.done:
            return

    # ── 1. Detect named paper + in-progress exclusions ────────────────────
    async with contextlib.aclosing(resolve.detect(turn)) as reply:
        async for piece in reply:
            yield piece

    async with contextlib.aclosing(resolve.decide(turn)) as reply:
        async for piece in reply:
            yield piece
    if turn.done:
        return

    # ── 1a. "Is document X in the library?" ──
    # FLOW: DOCUMENT LOOKUP, from the question's own words (the router rewrites them), library chats only: the name
    # matched against the document list; found: Yes + its title, else No + related documents. No answer model.
    asked = documents.existence_name(turn.original_query) if turn.doc_chat is None and turn.sources and not turn.picked else None
    if asked:
        yield await answer_flow.document_lookup(turn.valves, *asked, sources=turn.sources, in_progress=turn.in_progress, client=turn.client,
                                          dom=turn.dom)
        await turn.status("Done", done=True)
        return

    # ── 1a2. Documents by a person: from the stored front matter, complete; with a topic, the person's documents
    # become the named documents of the normal path ──
    if (turn.author and turn.doc_chat is None and not turn.paper and not turn.named_docs and turn.sources
            and turn.query_intent in ("set_query", "corpus_listing", "ranked_retrieval", "multi_hop")):
        async with contextlib.aclosing(library_listing.person_reply(turn)) as reply:
            async for piece in reply:
                yield piece
        if turn.done:
            return

    # ── 1c. A named document that matches nothing: the document the previous answer cited, when there is one; else
    # ask, offering the whole library ("yes" next turn). After the lookup and person steps, which answer such names.
    # ──
    if (turn.doc_chat is None and not turn.search_all and not turn.picked and turn.scope in ("document", "documents") and not turn.paper
            and not turn.named_docs and not turn.author and not turn.author_docs
            and turn.query_intent not in ("general", "chitchat", "corpus_listing", "rework")):
        async with contextlib.aclosing(resolve.named_not_found(turn)) as reply:
            async for piece in reply:
                yield piece
        if turn.done:
            return

    # ── 1b. Set-query intent: "list all <entity> papers" ──
    # From the graph's MENTIONS aggregate, without a model: top-k retrieval cannot express "the complete set".
    scoped = await asyncio.to_thread(documents.entity_scoped_listing, turn.query, turn.paper, turn.dom,
                                     turn.valves.GRAPH_RETRIEVAL and turn.doc_chat is None)
    if scoped is not None:
        yield scoped
        await turn.status("Done", done=True)
        return

    # ── 1b2. Gap-query intent: "research gaps in paper X" ──
    # Gap passages are tagged at ingestion, so retrieval is a lookup; the model drops false-positive tags.
    async with contextlib.aclosing(gap_answer.reply(turn)) as reply:
        async for piece in reply:
            yield piece
    if turn.done:
        return

    # ── 1c. Corpus-listing intent: no retrieval, no LLM ──
    # Answered from the `sources` set in memory: per-chunk retrieval would surface only the closest matches.
    if turn.query_intent == "corpus_listing":
        async with contextlib.aclosing(library_listing.corpus_reply(turn)) as reply:
            async for piece in reply:
                yield piece
        return

    # ── 2. Build retrieval filter: named paper + figure intent + in-progress ──
    # A figure request comes from the router; keyword detectors only when it did not answer. Used in step 4.
    async with contextlib.aclosing(search_step.run(turn)) as reply:
        async for piece in reply:
            yield piece
    if turn.done:
        return

    # ── 5a. Nothing relevant ──
    # The reranker keeps its top passages even when none is close. When even the best scores below RELATED_LOW_SCORE
    # (the "nothing found" mark), a plain reply instead, without the model. Not for a question naming library
    # documents (they exist; the model says what they cover), and never in Document Chat (the reader picked the
    # files, which may answer with low scores). No scores (reranker failed): no judgement.
    best = [h["rerank_score"] for h in turn.hits if h.get("rerank_score") is not None]
    if lists.nothing_relevant(best, turn.query_intent, rerank_enabled=turn.valves.RERANK_ENABLED,
                         low=turn.valves.RELATED_LOW_SCORE, dc=turn.doc_chat is not None, paper=turn.paper,
                         named_docs=turn.named_docs or turn.author_docs):
        import logging
        logging.getLogger("memento.query").info("nothing relevant: best passage %.3f < %.2f",
                                                max(best), turn.valves.RELATED_LOW_SCORE)
        yield await answer_flow.nothing_found_reply(turn.valves, turn.variant, question=turn.original_query, client=turn.client, qvec=turn.qvec,
                                              query=turn.query, in_progress=turn.in_progress, sources=turn.sources, dom=turn.dom,
                                              budget=turn.budget)
        await turn.status("Done", done=True)
        return

    # ── 5a2. List / count questions: every matching document in the context ──
    # The topic alone (router "topic"), in the documents of the kind asked for (router "doc_kind", by their library
    # titles), LIST_SCAN_PER_DOC passages per document without table-of-contents lines, for up to LIST_SCAN_MAX
    # documents (list_capped: more matched)
    if lists.list_scan_applies(turn.query_intent, enabled=turn.valves.LIST_SCAN and turn.valves.RERANK_ENABLED,
                          dc=turn.doc_chat is not None, paper=turn.paper, named_docs=turn.named_docs, image_intent=turn.image_intent):
        await lists.scan(turn)

    # ── 5b. Per-document search for the "documents not used / closest passages" block, while the answer is written
    turn.related_task = None
    if presentation.related_applies(turn.valves.RELATED_DOCS and turn.valves.RERANK_ENABLED and turn.doc_chat is None, paper=turn.paper,
                        named_docs=turn.named_docs or turn.author_docs,
                        image_intent=turn.image_intent, whole_doc=False, query_intent=turn.query_intent):
        turn.related_task = asyncio.create_task(asyncio.to_thread(
            answer_flow.related_heads, turn.valves, turn.client, turn.qvec, turn.query, library_state.exclude_keywords_section(library_state.with_exclusions(None, turn.in_progress))))

    # ── 6. Verbatim section-reproduction decision ──
    # The usual search, then the full section in document order, without a model. A loose match can only change the
    # format, not the content.
    turn.verbatim = False
    # only on the router's explicit verbatim_section; ambiguous: the normal answer
    if not turn.image_intent and turn.query_intent == "verbatim_section":
        # the named document, else the documents of the best passages (at most 3 are read)
        await verbatim_flow.find_section(turn)

    # ── 6b. Whole-document summary (_document_passages) ──
    # All of the named document's passages in reading order; the normal search above is the fallback.
    turn.whole_doc = False
    turn.summary_mode = None
    if turn.query_intent == "summary" and turn.paper and not turn.verbatim and not turn.image_intent:
        async with contextlib.aclosing(summary.prepare(turn)) as reply:
            async for piece in reply:
                yield piece
        if turn.done:
            return
    # Document Chat with several files and none named: each file summarized from all its passages, one after the
    # other. One list of sources, each file's passages after the previous file's, so every [N] is global: the
    # citations stay right and a follow-up (a rework) gets all of them.
    turn.multi_parts = []
    if turn.query_intent == "summary" and not turn.verbatim and not turn.image_intent and len(turn.file_order) >= 2:
        await summary.prepare_files(turn)

    # ── 7. Emit one citation card per chunk ──
    # Open WebUI maps an inline [N] to sources[N-1], the Nth citation event by emission order. One event per chunk
    # lets each [N] carry its own page and bbox in embed_url. The frontend collapses entries sharing
    # metadata[].source into one slot while [N] still indexes by position, so each chunk gets a unique
    # metadata.source (document name + chunk index). It also de-duplicates the inline labels by source name;
    # metadata.name takes priority there, so it is set to a unique "[N] document (page)".
    for i, h in enumerate(turn.hits):
        await turn.emit(presentation.citation_event(i, h, turn.valves.FILES_URL))

    # ── 8/9. Verbatim section reproduction: direct concatenation, no LLM ──
    # _section_span returned the section's passages in reading order; each is emitted with its [N] (step 7 emitted
    # the citation events in the same order). No model, so nothing meant to be reproduced exactly can be cut or
    # changed.
    if turn.verbatim:
        async with contextlib.aclosing(verbatim_flow.reply(turn)) as reply:
            async for piece in reply:
                yield piece
        return

    # ── 8. Build prompt ──
    # The default, when no other flow matched.
    turn.ctx_names = {}      # Library Chat: the titles the model reads (filled before the context is built)


    # FLOW: SUMMARY OF SEVERAL FILES (Document Chat): one answer-model call per file, under its own heading, its
    # passages with their global numbers; Thorough check per file; one footer and one source list for all
    if turn.multi_parts:
        async with contextlib.aclosing(summary.reply_files(turn)) as reply:
            async for piece in reply:
                yield piece
        return

    await answer_flow.build_prompt(turn)

    # ── 9. Stream LLM response ────────────────────────────────────────────
    async with contextlib.aclosing(answer_flow.stream_answer(turn)) as reply:
        async for piece in reply:
            yield piece


async def handle(body: dict, user: dict, emit, files, metadata, task, *, function_id: str, valves, engine):
    """The answer as run writes it, recorded: a picker question's clean answer is stored for the next time
    (memento/library/answers.py). The reader's stream and events pass through unchanged."""
    import logging
    if function_id not in library_state.DEFAULT_IDS:          # another library's copy: only once it is set up
        state = await asyncio.to_thread(library_state.copy_state, "query", function_id=function_id,
                                        db_engine=engine)
        if not state.enabled:
            logging.getLogger("memento.query").warning("%s not set up: %s", function_id, state.reason)
            if not task:
                yield (texts.DE_FIXED[texts.STUDY_NOT_READY_MSG] if texts.question_lang(body) == "de" else texts.STUDY_NOT_READY_MSG)
            return
    if not task and request.continue_request(body):
        logging.getLogger("memento.query").info("continue response: nothing added")
        return
    rec = cache.Recorder()
    warm = bool((metadata or {}).get("warmup"))       # the upload warm-up's own question

    async def recording(event):
        if isinstance(event, dict) and event.get("type") == "citation":
            rec.citations.append(event)
        await emit(event)
    emitter = recording if emit is not None else None
    # one Turn for the message (Document Chat sets its files on it, once read)
    turn = request.Turn(body=body, user=user, files=files, metadata=metadata, emit=None, function_id=function_id,
                        valves=valves, engine=engine, rec=rec, doc_chat=None)

    async def send(event: dict):
        if emitter:
            await emitter(event)

    async def status(msg: str, done: bool = False):
        await send(presentation.status_event(msg, done))
    turn.emit, turn.status = send, status
    if chat.is_doc_chat(body.get("model"), valves.DOC_CHAT_MODEL_ID):   # Document Chat: never stored answers
        answer = chat.answer(turn, task, run=run)
    else:
        answer = run(turn)
    # the question's language for our own texts around the answer; this copy's models in links
    with request.scope(lang=texts.question_lang(body), link_ids=texts.copy_link_ids(valves)):
        try:
            async for chunk in answer:
                if isinstance(chunk, str):
                    chunk = texts.tr(chunk)
                    rec.text.append(chunk)          # stored with the default ids; each copy relinks at output
                    chunk = texts.relink_for_copy(chunk)
                yield chunk
        except (asyncio.CancelledError, GeneratorExit):
            # the reader's Stop (Open WebUI cancels the request) or a closed page; the thinking cut never gets here,
            # so logged stops are the readers' own
            logging.getLogger("memento.query").info("warm-up answer cancelled" if warm
                                                    else "answer stopped by the user; model connection closed")
            raise
    # a warm-up's report of what happened to its answer: hit | stored | ineligible | error; nothing written means
    # "error" for the worker (retried), never success
    wr = (metadata or {}).get("warmup_result")

    def report(result: str, reason: str = "", key=None) -> None:
        if isinstance(wr, dict):
            wr.update(result=result, reason=reason, key=key)
    if rec.hit:
        report("hit", key=rec.hit_key)
        return
    if not rec.key:
        if rec.skip_reason:             # a rule says this answer is never stored: final
            report("ineligible", rec.skip_reason)
        return                          # no decision reached (a polite error, say): nothing, so it is retried
    if not rec.clean:
        report("error", "the answer did not finish", rec.key)
        return
    text = conversation.without_suggestions("".join(rec.text))
    log = logging.getLogger("memento.query")
    if any(n in texts.untr(text) for n in cache.UNSTORED_NOTES):
        log.info("cache: skip %s %s (note in the answer)", rec.fields["action"], rec.fields["source"])
        report("ineligible", "note in the answer", rec.key)
        return
    # stored under a language but written in another; "unsure" does not block
    from memento.common.text_language import guess_language
    written = guess_language(cache.strip_think(text))
    if written and rec.fields.get("language") and written != rec.fields["language"]:
        log.info("cache: skip %s %s (answer in %s, question in %s)", rec.fields["action"], rec.fields["source"],
                 written, rec.fields["language"])
        report("ineligible", f"answer in {written}", rec.key)
        return
    if not await asyncio.to_thread(cache.still_in_library, engine(), rec.fields["collection"], rec.fields["source"]):
        log.info("cache: skip %s %s (document removed)", rec.fields["action"], rec.fields["source"])
        report("ineligible", "document removed", rec.key)
        return
    try:
        from memento.library import answers as ac
        # a warm-up answer never replaces a usable one already stored (a live answer or a Regenerate came first);
        # one its replay check rejects is replaced
        outcome = await asyncio.to_thread(ac.put, engine(), rec.key, fields=rec.fields,
                                          answer=cache.strip_think(text), citations=rec.citations,
                                          generated_by=rec.generated_by, replace=not warm, checked=rec.checked,
                                          keep_if=cache.gap_answer_replayable if rec.fields["action"] == "gaps" else None)
        log.info("%s: %s %s %s %s%s", "warmup" if warm else "cache",
                 "stored" if outcome == "written" else outcome, rec.fields["action"], rec.fields["source"],
                 rec.fields["model"], " (checked)" if rec.checked else "")
        if outcome in ("written", "kept_existing"):
            report("stored", "", rec.key)
        else:
            report("ineligible", outcome.replace("_", " "), rec.key)
    except Exception as e:
        logging.getLogger("memento.errors").warning("E-answer-cache: %r", e)
        report("error", repr(e), rec.key)
