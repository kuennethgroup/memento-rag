"""The default answer: the answer model's prompt, the answer streamed (held for Thorough check and checked, the figures
it cites, the footer, the related documents, stored when clean), nothing found (a short reply naming the closest
documents), and a named document looked up.
"""
import asyncio
import re
import time

from memento.document_chat import chat
from memento.query import cache, checking, generation, library_state, presentation, prompts, request, texts
from memento.query.flows import figures, lists
from memento.query.scope import documents, resolve


# nothing relevant in the library: the model writes a short reply pointing to the closest documents
NOTHING_FOUND_PROMPT = (
    "You are Memento, an assistant for a shared library of research documents. The search found no passage that "
    "answers the user's question. Below are the closest passages the search found, each with its document's title. "
    "Reply in the language of the question, in at most three sentences: first say plainly that you found nothing in "
    "the library that answers the question. Then, only if a document below is genuinely useful for this question (someone asking "
    "it would want to open it), name at most three such documents by their title, each with a few words on why, and "
    "say they are only related, not about the topic itself. If none is genuinely useful, name none and say they can "
    "add documents on the topic with Library Upload. Never say a document covers something its passage does not show. "
    "No citations.")


NOTHING_FOUND_TOP = 5


def nothing_found_messages(question: str, heads: list, titles: dict) -> list:
    """The model's input for a "nothing relevant" reply: NOTHING_FOUND_PROMPT and the best passage of the closest
    documents (NOTHING_FOUND_TOP), each under its title, cut to 600 characters."""
    ctx = "\n\n".join(f'Document "{titles.get(h["source"], h["source"])}":\n{(h.get("text") or "")[:600]}'
                       for h in heads[:NOTHING_FOUND_TOP])
    return [{"role": "system", "content": NOTHING_FOUND_PROMPT},
            {"role": "user", "content": f"Closest passages:\n\n{ctx}\n\nQuestion: {question}"}]


async def nothing_found_reply(valves, variant, *, question: str, client, qvec, query: str, in_progress: set,
                               sources, dom, budget) -> str:
    """Nothing relevant in the library: a short reply by the answer model (no thinking) that says so and names the
    closest documents only when they are genuinely useful; the fixed NOTHING_IN_LIBRARY_MSG when switched off, on
    any failure or after 20 s."""
    if not valves.NOTHING_FOUND_REPLY:
        return texts.NOTHING_IN_LIBRARY_MSG
    import logging
    try:
        heads = await asyncio.to_thread(related_heads, valves, client, qvec, query,
                                        library_state.exclude_keywords_section(library_state.with_exclusions(None, in_progress)))
        heads = sorted(heads, key=lambda h: -(h.get("rerank_score") or 0))[:NOTHING_FOUND_TOP]
        if not heads:
            return texts.NOTHING_IN_LIBRARY_MSG
        titles = await asyncio.to_thread(documents.display_titles, sorted({h["source"] for h in heads}), dom)

        async def write():
            chunks, answered, note = await generation.open_answer(
                valves, variant, think=False, max_tokens=400, budget=budget,
                messages=nothing_found_messages(question, heads, titles))
            text = await generation.drain_content(chunks, answered.is_cloud)
            return note, re.sub(r"(?s)<think>.*?</think>", "", text).strip(), answered
        note, text, answered = await asyncio.wait_for(write(), 20)
        if not text:
            return texts.NOTHING_IN_LIBRARY_MSG
        logging.getLogger("memento.query").info("nothing relevant: model reply, %d chars", len(text))
        return (note or "") + presentation.link_app_names(text) + generation.answered_by(answered.label)
    except Exception as e:
        logging.getLogger("memento.errors").warning("E-nothing-found-reply: %r", e)
        return texts.NOTHING_IN_LIBRARY_MSG


def related_heads(valves, client, qvec, query: str, query_filter) -> list:
    """The best passage of each document (up to RELATED_GROUPS), the best RELATED_RERANK_K reranked."""
    import logging
    from memento.retrieval import search
    t0 = time.monotonic()
    heads = search.related_heads(client, valves.COLLECTION, qvec, query_filter, valves.RELATED_GROUPS)
    k = valves.RELATED_RERANK_K
    out = search.rerank(query, heads[:k], valves.RERANKER_URL, k, 0.0, 0.0,
                        instruction=request.rerank_instruction(request.domain(valves.DOMAIN_NAME)))
    capped = bool(getattr(heads, "capped", False))
    logging.getLogger("memento.query").info("related: search %.2f s (%d documents%s, %d reranked)",
                                            time.monotonic() - t0, len(heads), ", capped" if capped else "",
                                            min(k, len(heads)))
    # what the search reached travels with the reranked few (the footer must not count the library's size)
    return search.RelatedHeads(out, capped=capped, found=len(heads))


async def document_lookup(valves, name: str, lang: str, *, sources: set, in_progress: set, client, dom: dict) -> str:
    """The reply to "is document X in the library?": the resolver's match, else the documents whose best passage for
    the name scores at least RELATED_LOW_SCORE (and RELATED_RATIO × the best), up to 5 (missing titles score far
    below related documents). Reranker down or any failure: a plain No."""
    import logging
    log = logging.getLogger("memento.query")
    src = await asyncio.to_thread(documents.lookup_document, name, sorted(sources), dom)
    if src:
        title = (await asyncio.to_thread(documents.display_titles, [src], dom)).get(src) or src
        log.info("document lookup: %r found (%s)", name, src)
        return documents.existence_reply(name, lang, title=title, n_docs=len(sources))
    related = []
    try:
        from memento.common.model_servers import vl_embed
        qvec = await asyncio.to_thread(vl_embed, valves.EMBEDDER_URL, valves.EMBEDDER_MODEL,
                                       dom.get("instruction") or request.FALLBACK_INSTRUCTION, name, None, 30)
        heads = await asyncio.to_thread(related_heads, valves, client, qvec, name,
                                        library_state.exclude_keywords_section(library_state.with_exclusions(None, in_progress)))
        if not any(h.get("rerank_fallback") for h in heads):
            top = max((h.get("rerank_score") or 0.0 for h in heads), default=0.0)
            floor = max(valves.RELATED_LOW_SCORE, valves.RELATED_RATIO * top)
            keep = [h["source"] for h in heads if (h.get("rerank_score") or 0.0) >= floor]
            keep = list(dict.fromkeys(keep))[:5]
            titles = await asyncio.to_thread(documents.display_titles, keep, dom) if keep else {}
            related = [titles.get(s) or s for s in keep]
    except Exception as e:
        logging.getLogger("memento.errors").warning("E-document-lookup: %r", e)
    log.info("document lookup: %r not found, %d related", name, len(related))
    return documents.existence_reply(name, lang, related=related, n_docs=len(sources))


async def related_block(valves, task, hits: list, cited: set, n_docs: int, dom: dict,
                         figure_asked: bool = False, list_answer: bool = False) -> str:
    """The block under the answer, from the per-document search started with the answer; empty if it is not ready
    within a few seconds after the answer (it never holds the answer up for long) or fails."""
    import logging
    try:
        heads = await asyncio.wait_for(task, timeout=5)
        if any(h.get("rerank_fallback") for h in heads):
            return ""
        scores = [h["rerank_score"] for h in hits if h.get("rerank_score") is not None]
        if not scores:                     # no scores: nothing was measured, nothing is said
            return ""
        top = max(scores)
        cites = bool({c for c in cited if 0 <= c < len(hits)})        # citations that map to a source
        weak = top < valves.RELATED_LOW_SCORE and not figure_asked
        near = not cites or (weak and not list_answer)
        items = presentation.related_documents(heads, used={h["source"] for h in hits}, nothing_found=near, top_score=top,
                                   ratio=valves.RELATED_RATIO, min_score=valves.RELATED_MIN_SCORE,
                                   limit=5, near=3)
        titles = await asyncio.to_thread(documents.display_titles, sorted({h["source"] for h in items}), dom) if items else {}
        searched = None if getattr(heads, "capped", False) else getattr(heads, "found", len(heads))
        logging.getLogger("memento.query").info("related: %s documents searched, %d listed (best %.3f%s%s)",
                                                searched if searched is not None else "capped", len(items), top,
                                                ", no citations" if not cites else "", ", weak" if weak else "")
        return presentation.related_footer(items=items, titles=titles, searched=searched, cited=cites, weak=weak,
                               files_url=valves.FILES_URL, list_answer=list_answer)
    except Exception as e:
        task.cancel()
        logging.getLogger("memento.errors").warning("E-related: %r", e)
        return ""


async def build_prompt(turn) -> None:
    """The answer model's instructions and message: the numbered sources with their labels, the addenda for the route,
    the language and the check, and the images."""
    if turn.doc_chat is None and turn.hits and not turn.whole_doc:
        try:
            turn.ctx_names.update(await asyncio.to_thread(documents.display_titles, sorted({h["source"] for h in turn.hits}), turn.dom))
        except Exception:
            pass
    # a table or an equation with the authors' text about it (answer_pipeline.source_text)
    from memento.retrieval.answer_pipeline import source_text
    context = "\n\n---\n\n".join(
        f"[{i + 1}] {presentation.chunk_label(h, turn.whole_doc, turn.ctx_names)}\n{source_text(h)}"
        for i, h in enumerate(turn.hits)
    )
    # Prior turns are plain conversational context (no re-attached sources); grounding stays on this turn's
    # `context`. The question keeps its original wording, not the rewritten retrieval query. The domain's own prompt
    # (config/domains/<DOMAIN_NAME>.yaml); SYSTEM_PROMPT only as a fallback.
    turn.system_prompt = prompts.answer_prompt(turn.dom)
    if turn.whole_doc:
        turn.system_prompt += prompts.SUMMARY_PARTIAL_ADDENDUM if turn.summary_mode == "partial" else prompts.SUMMARY_ADDENDUM
    list_add = lists.list_addendum(turn.query_intent, turn.whole_doc, turn.doc_chat is not None, capped=turn.list_capped)
    turn.system_prompt += list_add + prompts.element_addendum(turn.element)
    turn.system_prompt += chat.doc_chat_addendum(turn.doc_chat is not None)
    turn.system_prompt += texts.answer_language_addendum([h.get("text") or "" for h in turn.hits],
                                               own_line=turn.whole_doc or bool(list_add))
    # tables / formulas with the authors' text: values from the element, statements as the authors'
    from memento.retrieval.answer_pipeline import linked_text_addendum
    turn.system_prompt += linked_text_addendum(turn.hits)
    if turn.whole_doc:
        turn.system_prompt += texts.summary_language_addendum()
    # The "Thorough check" button switches the check on for this question (off by default: the model's first answer,
    # streamed as written). On: the RQ3 loop with the thinking round (revise_thinking="escalate").
    thorough = request.THOROUGH_FILTER_ID in ((turn.metadata or {}).get("filter_ids") or [])
    turn.verify = checking.verify_applies(turn.valves.VALIDATION_ENABLED, thorough=thorough,
                             query_intent="summary" if turn.whole_doc else turn.query_intent)
    turn.check_prompt = turn.system_prompt              # the revisions keep the answer's own instructions (summary layout)
    if turn.verify:
        try:
            from memento.agentic_validation.strategies import P1_ADDENDUM
            turn.system_prompt += P1_ADDENDUM          # the verification loop starts from the strict-citation answer
        except Exception:
            turn.verify = False
    # the question's wish for ideas beyond the sources: not in check_prompt, so the check's revisions never write
    # one
    turn.system_prompt += prompts.suggest_addendum(turn.suggest, turn.query_intent)
    if turn.history:
        turn.system_prompt += texts.HISTORY_RULE
    head = f'Sources (all from the document "{presentation.shown(turn.hits[0]["source"])}"):' if turn.whole_doc and turn.hits else "Sources:"
    if turn.even_files:
        names = ", ".join(f'"{presentation.shown(f)}"' for f in turn.file_order)
        what = "attached files" if turn.doc_chat is not None and not turn.named_docs else "documents named in the question"
        head = (f"Sources (from all {len(turn.file_order)} {what}: {names}). The question is about each of them: cover "
                f"each one, and say so if one has nothing on it.\n\nSources:")
    user_message = f"{head}\n{context}\n\nQuestion: {turn.original_query}"

    # Image and chart sources are attached as images, not just their captions (the answer model reads both). At most
    # MAX_IMAGES_TO_LLM, the first in best-first order. Only what the model receives changes; the figures under the
    # answer (step 10) and the citation events (step 7) do not.
    turn.user_content = user_message
    image_parts = []
    turn.seen_images = set()     # the passages the model saw as images: shown first among the cited figures (step 10)
    for i, h in enumerate(turn.hits):
        # count figures, not parts (each figure is a label part plus an image part)
        if len(image_parts) // 2 >= turn.valves.MAX_IMAGES_TO_LLM:
            break
        if h["type"] in ("image", "chart") and h.get("img_path") and not presentation.icon_image(h) and not presentation.reference_image(h):
            url = prompts.verified_image_url(h["img_path"])
            if url:   # missing or unreadable image → caption only
                turn.seen_images.add(i)
                image_parts.append({"type": "text", "text": presentation.image_label(i + 1)})
                image_parts.append({"type": "image_url", "image_url": {"url": url}})
    if image_parts:
        turn.user_content = [{"type": "text", "text": user_message}] + image_parts


async def stream_answer(turn):
    """The answer written: streamed (held for Thorough check), checked, then the footer, the sources, the related
    documents and the cited figures; the answer is stored when clean."""
    await turn.status("Generating answer...")
    in_reasoning = False
    try:
        # the chosen variant writes the answer (a cloud one out of quota or failing before output → local, with a note)
        answered = turn.variant
        # a whole-document summary reads every passage and needs no thinking first
        chunks, answered, note = await generation.open_answer(
            turn.valves, turn.variant, think=turn.think and not turn.whole_doc, budget=turn.budget, held=turn.verify,
            local_thinking_budget=generation.thorough_thinking_budget(turn.valves) if turn.verify else None,
            messages=[{"role": "system", "content": turn.system_prompt}] + turn.answer_history
                     + [{"role": "user", "content": turn.user_content}])
        if note:
            yield note
        # Some servers stream Qwen3's reasoning as delta.reasoning_content, apart from delta.content: it is wrapped
        # in <think>...</think> so Open WebUI shows a collapsible "Thinking..." block. If the stream ends
        # mid-reasoning (max_tokens reached before any answer), the tag is closed with a note, so the chat is not
        # left on "Thinking...".
        got_content = False
        stopped = timed_out = False
        answer_text = []
        # a figure request: its cited figures right below the first paragraph, not only at the end; not for a held
        # answer (checked before showing), whose figures stay at the end
        early_pending, early_shown = turn.image_intent and not turn.verify, set()
        figure_cap = turn.valves.MAX_FIGURES_SHOWN if turn.valves.MAX_FIGURES_SHOWN > 0 else None
        # a list answer: entries after LIST_FIRST fold like the library listing
        fold = lists.list_folds(turn.query_intent, whole_doc=turn.whole_doc, image_intent=turn.image_intent)
        folder = presentation.ListFold() if fold and not turn.verify else None
        try:
            async for chunk in chunks:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                reasoning = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
                if reasoning:
                    if not in_reasoning:
                        yield "<think>"
                        in_reasoning = True
                    yield reasoning
                content = delta.content
                if content:
                    if in_reasoning:
                        yield "</think>"
                        in_reasoning = False
                    got_content = True
                    before = sum(len(t) for t in answer_text)
                    answer_text.append(content)
                    if not turn.verify:                 # verify-before-showing: the answer is held until checked
                        first = figures.first_paragraph_figures("".join(answer_text)) if early_pending else None
                        if first is None:
                            shown = folder.feed(content) if folder else content
                            if shown:
                                yield shown
                        else:                      # the paragraph ended in this piece: split it there
                            early_pending = False
                            cut = min(max(figures.first_paragraph_end("".join(answer_text)) - before, 0), len(content))
                            yield content[:cut]
                            for i in sorted(first):
                                if figure_cap is not None and len(early_shown) >= figure_cap:
                                    break
                                if 0 <= i < len(turn.hits) and figures.shown_figure(turn.hits[i], figure_asked=True):
                                    url = prompts.verified_image_url(turn.hits[i]["img_path"])
                                    if url:
                                        early_shown.add(i)
                                        yield figures.figure_markdown(i, turn.hits[i], url).lstrip("\n") + "\n\n"
                            if content[cut:]:
                                yield content[cut:]
        except generation.AnswerTimeout as e:                 # over ANSWER_TIMEOUT or silent: the model is already stopped
            import logging
            logging.getLogger("memento.errors").warning("E-answer-timeout: %s %r", answered.id, e)
            timed_out = True
        except Exception as e:
            if not answered.is_cloud:
                raise                              # local model: the error message below
            import logging
            logging.getLogger("memento.errors").warning("E-cloud-stream: %s %r", answered.id, e)
            stopped = True
        if folder:                                 # the held end of a long list, folded
            rest = folder.close()
            if rest:
                yield rest
        if in_reasoning:
            yield "</think>"
            in_reasoning = False
        if stopped or timed_out:
            # the partial answer stays (held text is shown unchecked) and the reader is told what happened
            if turn.verify:
                yield "".join(answer_text)
            yield "\n\n" + (texts.TIMEOUT_NOTE if timed_out else texts.STOPPED_NOTE)
            turn.verify = False
        elif not got_content:
            yield ("\n\n_The model used its entire token budget on reasoning "
                   "and produced no answer. Try a more specific question or "
                   "increase MAX_TOKENS._")

        # ── 9b. RQ3 verification loop (VALIDATION_ENABLED, "Thorough check" on) ──
        check_ok = False
        if turn.verify and got_content:
            checked_part, extensions = prompts.split_extensions("".join(answer_text))
            final, result, error, late = await checking.thorough_check(turn.valves,
                answer=checked_part, hits=turn.hits, question=turn.original_query, check_prompt=turn.check_prompt,
                think=turn.think, client=turn.client, dom=turn.dom, api_key=turn.api_key, status=turn.status, collection=turn.collection,
                retrieve_filter=resolve.check_search_filter(resolve.scope_sources(turn.paper, turn.named_docs, turn.author_docs), turn.in_progress),
                background=bool((turn.metadata or {}).get("warmup")), seconds=generation.check_seconds(turn.valves, turn.budget))
            if result is not None and error is None:
                for i in range(len(turn.hits), len(result.hits)):          # sources added by re-retrieval (arm A)
                    await turn.emit(presentation.citation_event(i, result.hits[i], turn.valves.FILES_URL))
                turn.hits = result.hits
            check_ok = generation.check_ok(result, error, late)
            turn.rec.checked = check_ok
            final = prompts.with_extensions(final, extensions)
            answer_text = [final]
            async for piece in presentation.type_out(presentation.fold_list(final) if fold else final,
                                      *presentation.typing_speed(turn.valves, turn.metadata, at_once=True)):
                yield piece

        # ── 10. Show the figures behind the image/chart sources the model cited ──
        # Only sources whose [N] the answer uses ([N] maps to hits[N-1], step 8); a retrieved chart the answer does
        # not cite is never shown.
        cited = {int(n) - 1 for n in re.findall(r"\[(\d+)\]", "".join(answer_text))}
        # the collapsed list for the reader; follow-up turns take the [N] mapping from the citations OpenWebUI
        # stores with each answer (presentation.stored_citation_names), the list is only a fallback there
        if got_content:
            yield generation.answered_by(answered.label,
                               turn.local_variant.label if (turn.verify and answered.is_cloud) else None)
        yield presentation.sources_marker(turn.hits, cited)
        if turn.related_task is not None and got_content and not turn.whole_doc:
            related = await related_block(turn.valves, turn.related_task, turn.hits, cited, len(turn.sources), turn.dom,
                                                figure_asked=turn.image_intent or turn.element == "figure",
                                                list_answer=turn.query_intent == "set_query")
            if related:
                yield related
        # a checked answer replaces the stored one, but only if the check finished
        if cache.storable(keyed=bool(turn.rec.key), got_content=got_content, interrupted=stopped or timed_out, note=note,
                     by_chosen_model=answered.id == turn.variant.id, verify=turn.verify, check_ok=check_ok):
            turn.rec.clean, turn.rec.generated_by = True, answered.label
        figures_ok, figure_asked = prompts.figures_for(element=turn.element, image_intent=turn.image_intent,
                                                 query_intent=turn.query_intent)
        show_figures = figures_ok and figures.show_figures(turn.hits, cited, turn.valves.RELATED_LOW_SCORE,
                                                    figure_asked=figure_asked)
        if show_figures:
            # at most MAX_FIGURES_SHOWN in all, those the answer discusses first
            shown, more = figures.figures_to_show(
                turn.hits, cited, "".join(answer_text), turn.seen_images,
                None if figure_cap is None else figure_cap - len(early_shown), figure_asked=figure_asked,
                skip=early_shown)
            for i in shown:
                url = prompts.verified_image_url(turn.hits[i]["img_path"])
                if url:   # missing or unreadable image → text only
                    yield figures.figure_markdown(i, turn.hits[i], url)
            if more:
                from memento.common.text_language import question_language
                yield figures.more_figures_note(more, question_language(turn.original_query) == "de")

    except Exception as e:
        yield ("</think>" if in_reasoning else "") + "\n\n" + texts.user_error("answer", e)
    finally:
        # Stop pressed: no more answer text, only the status line ends (else "Generating answer..." keeps shimmering)
        await turn.status("Stopped" if generation.stopped() else "Done", done=True)
