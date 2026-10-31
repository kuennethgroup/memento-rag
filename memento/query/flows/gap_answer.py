"""A research-gap question answered in chat: the document it is about, its gap-tagged passages, the answer (one more
without thinking when it came out unreadable), the gap citations, and the suggested extensions.
"""
import asyncio
import re

from memento.query import cache, checking, generation, presentation, request, texts
from memento.query.flows import gap_report
from memento.query.scope import conversation, documents, resolve


def gap_passages(client, collection: str, paper: str, budget_tokens: float = 20000) -> tuple[list, bool]:
    """(passages, only_sections) for a document's research gaps: gap-tagged passages by type, then Discussion /
    Conclusion / Outlook passages, up to the budget, in reading order. only_sections: no gap tags at all, the sections
    alone were used."""
    import logging
    from memento.research_gaps import passages as gr
    from memento.query.doc_budget import select_gap_passages
    gaps = gr.fetch_gap_chunks(client, collection, paper)
    sections = gr.fetch_fallback_sections(client, collection, paper)
    chosen, st = select_gap_passages(gaps, sections, budget_tokens, request.CHARS_PER_TOKEN)
    logging.getLogger("memento.query").info("gaps: %s candidates=%d chosen=%d tokens=%d tagged=%d",
                                            paper, st["candidates"], st["chosen"], st["tokens"], len(gaps))
    return chosen, (not gaps and bool(sections))


async def gap_suggestions(valves, *, source: str, docs: dict, variant, authors: str, chunks: list, question: str,
                           think: bool, budget, db_engine, time_budget: "generation.Budget | None" = None) -> str:
    """"Possible extensions", clearly labelled as suggestions: up to five, each building on a point the authors
    state; stored per document and per authors' answer (action gaps_ext); "" when none holds."""
    import logging
    from memento.library import answers as ac
    from memento.research_gaps.passages import (SUGGEST_SYSTEM_PROMPT, parse_suggestions, study_points, suggest_messages,
                                         suggestions_block)
    from memento.common.text_language import question_language
    authors = study_points(authors)       # the study's own points, not the gaps in earlier work it addresses
    allowed = {int(n) for n in re.findall(r"\[(\d+)\]", authors or "")}
    if not allowed:
        return ""
    if time_budget is not None and time_budget.left() < valves.SUGGEST_MIN_S:     # optional: time left only
        logging.getLogger("memento.query").info("gaps_ext: skipped, %.0f s left", time_budget.left())
        return texts.SUGGEST_FAILED_NOTE
    german = question_language(question) == "de"
    h = ((docs or {}).get(source) or {}).get("content_hash") or ""
    fields = dict(collection=valves.COLLECTION, source=source, content_hash=h, action="gaps_ext",
                  language="de" if german else "en", model=cache.model_key(variant, valves), mode="capped",
                  prompt_version=cache.suggest_prompt_version(authors, german=german))
    key, engine = ac.cache_key(**fields), db_engine()
    row = await asyncio.to_thread(ac.get, engine, key) if h else None
    if row:
        logging.getLogger("memento.query").info("cache: hit gaps_ext %s", source)
        return row["answer"]
    chunks_, answered, note = await generation.open_answer(
        valves, variant, think=think, local_thinking_budget=budget, messages=suggest_messages(authors, chunks, question),
        budget=time_budget, held=True)
    raw = cache.strip_think(await generation.drain_content(chunks_, answered.is_cloud))
    items = parse_suggestions(raw, allowed)
    block = suggestions_block(items, german)
    logging.getLogger("memento.query").info("gaps_ext: %d suggestions kept for %s", len(items), source)
    if block and h and answered.id == variant.id:
        await asyncio.to_thread(ac.put, engine, key, fields=fields, answer=block, citations=[],
                                generated_by=answered.label)
    return block


async def gap_target(valves, paper, query: str, sources, dom, metadata) -> str | None:
    """The document a research-gap question is about: picked or named, else the one the conversation is about
    (the previous answer's citations), else None (then the topic's open questions are shown)."""
    if paper:
        return paper
    if not sources:
        return None
    named = await asyncio.to_thread(documents.resolve_reference, query, list(sources), dom)
    if named:
        return named
    return await conversation.conversation_target(valves, sources, dom, metadata)


async def gap_retry(valves, variant, messages: list, budget) -> tuple[str, bool]:
    """(one more research-gap answer, written without thinking, from its first heading on; whether text before that
    heading was dropped) after an answer the parser rejected, within the request's time budget; ("", False) when no
    time is left or it fails."""
    import logging
    if budget is not None and budget.left() < valves.SUGGEST_MIN_S:
        logging.getLogger("memento.query").info("gaps: no time left for a retry")
        return "", False
    try:
        chunks, answered, _ = await generation.open_answer(valves, variant, think=False, budget=budget, held=True,
                                                      messages=messages)
        text = await generation.drain_content(chunks, answered.is_cloud)
    except Exception as e:
        logging.getLogger("memento.errors").warning("E-gap-retry: %r", e)
        return "", False
    gate = presentation.HeadingGate()
    return gate.feed(text) + gate.finish(), gate.dropped > 0


async def reply(turn):
    """Research gaps of one document in chat: the stored answer, or the gap-tagged passages answered (checked with
    Thorough check), the suggested extensions; no document: the library's report."""
    try:
        from memento.research_gaps.passages import build_gap_prompt, calm_gap_answer, gap_focus, gap_system_prompt
        # one kind asked for: shown open first; extensions right after a gap answer: no table again
        focus, show_table = gap_focus(turn.original_query), request.show_gap_table(suggest=turn.suggest, history=turn.history)
        if turn.query_intent == "gap_query":
            gap_paper = await gap_target(turn.valves, turn.paper, turn.query, turn.sources, turn.dom, turn.metadata)
            if turn.suggest and request.answer_was_gaps(turn.history):
                # hidden only for the same document as the previous gap answer
                show_table = request.show_gap_table(
                    suggest=turn.suggest, history=turn.history, question=turn.original_query,
                    same_document=bool(gap_paper) and await conversation.previous_document(turn.sources, turn.metadata) == gap_paper)
            if not gap_paper and turn.doc_chat is None:
                # no document named or in focus: the open questions the library's documents name on this topic
                # (the ledger's report; its "none" reply points to the Limitations & future work button)
                try:
                    async for c in gap_report.gap_report(turn.valves, turn.query, variant=turn.variant, emit=turn.emit, status=turn.status, db_engine=turn.engine):
                        yield c
                except Exception as e:
                    yield texts.user_error("gap", e)
                await turn.status("Done", done=True)
                turn.done = True
                return
            if gap_paper:
                # the document's gap prompt: a talk's has the aims rule (gap_system_prompt)
                gap_title = ((turn.docs or {}).get(gap_paper) or {}).get("title")
                gap_prompt = gap_system_prompt(gap_title)
                gap_thorough = request.THOROUGH_FILTER_ID in ((turn.metadata or {}).get("filter_ids") or [])
                # one stored answer per document whatever the switch: with the switch on, a fresh answer is written,
                # checked and replaces the stored one
                row = None if turn.doc_chat is not None else await cache.prepare(
                    turn.valves, turn.rec, db_engine=turn.engine, picked=turn.picked, history=turn.history, docs=turn.docs, in_progress=turn.in_progress, source=gap_paper,
                    action="gaps", mode="capped", valid=cache.gap_answer_replayable,
                    prompt_version=cache.prompt_version("gaps", "capped", turn.dom, turn.valves, think=turn.gap_base_think,
                                                               title=gap_title),
                    variant=turn.variant, metadata=turn.metadata, thorough=gap_thorough)
                if row:
                    async for c in cache.replay({**row, "answer": calm_gap_answer(row["answer"], focus)
                                            if show_table else ""}, turn.emit,
                                            *presentation.typing_speed(turn.valves, turn.metadata)):
                        yield c
                    if turn.suggest:
                        await turn.status("Suggesting extensions...")
                        try:
                            block = await gap_suggestions(turn.valves,
                                source=gap_paper, docs=turn.docs, variant=turn.variant, authors=row["answer"],
                                chunks=conversation.chunks_from_citations(row.get("citations") or []), question=turn.original_query,
                                think=turn.think, budget=generation.gap_thinking_budget(False, turn.valves.GAP_THINKING_BUDGET),
                                db_engine=turn.engine, time_budget=turn.budget)
                        except Exception as e:
                            import logging
                            logging.getLogger("memento.errors").warning("E-gap-suggest: %r", e)
                            block = texts.SUGGEST_FAILED_NOTE
                        from memento.research_gaps.passages import unfold_suggestions
                        block = block if focus else unfold_suggestions(block)    # asked to extend: open
                        async for piece in presentation.type_out(block, *presentation.typing_speed(turn.valves, turn.metadata)):
                            yield piece
                    await turn.status("Done", done=True)
                    turn.done = True
                    return
                if focus or not show_table:     # a partial display (one kind open, or no table) is never stored
                    turn.rec.key, turn.rec.skip_reason = None, "a partial display"
                await turn.status("Analyzing research gaps...")
                # gap-tagged passages by type, then the Discussion / Conclusion / Outlook passages, within the
                # budget (all tagged passages would overflow on very long documents)
                gap_chunks, only_sections = await asyncio.to_thread(
                    gap_passages, turn.client, turn.collection, gap_paper, turn.valves.GAP_BUDGET_TOKENS)
                if gap_chunks and only_sections:
                    await turn.status("No gap tags found, using the Discussion and Conclusion sections...")
                if not gap_chunks:
                    # the fallback below is no research-gap answer: never stored
                    turn.rec.key, turn.rec.skip_reason = None, "no research-gap passages"

                if gap_chunks:
                    # Emit citation events for gap chunks
                    for i, h in enumerate(gap_chunks):
                        await turn.emit(presentation.gap_citation(i, h, turn.valves.FILES_URL))

                    # LLM reasoning over gap passages: thinking while GAP_THINKING is on, capped at
                    # GAP_THINKING_BUDGET on the local model, also with "Thorough check"; leaked lines are not shown
                    # or stored.
                    thorough = request.THOROUGH_FILTER_ID in ((turn.metadata or {}).get("filter_ids") or [])
                    gap_think = generation.gap_thinks(turn.gap_base_think, thorough, turn.valves.GAP_THINKING)
                    gap_budget = generation.gap_thinking_budget(thorough, turn.valves.GAP_THINKING_BUDGET)
                    import logging
                    logging.getLogger("memento.query").info("gaps: thinking=%s local_budget=%s thorough=%s",
                                                            gap_think, gap_budget, thorough)
                    user_message = build_gap_prompt(gap_chunks, conversation.gap_question(turn.original_query))
                    answered, in_reasoning = None, False
                    # "Thorough check" on: the answer is held, checked against the gap passages, then shown
                    gap_verify = checking.verify_applies(turn.valves.VALIDATION_ENABLED, thorough=thorough,
                                                 query_intent="gap_query")
                    held = []
                    gate = presentation.HeadingGate()           # the visible answer from its first heading on
                    try:
                        gap_messages = ([{"role": "system", "content": gap_prompt}]
                                        + turn.answer_history + [{"role": "user", "content": user_message}])
                        chunks, answered, note = await generation.open_answer(
                            turn.valves, turn.variant, think=gap_think, local_thinking_budget=gap_budget, budget=turn.budget, held=True,
                            messages=gap_messages)
                        if note:
                            yield note
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
                                shown = gate.feed(content)
                                if shown:      # held: shown in the calm layout once complete
                                    held.append(shown)
                        if in_reasoning:
                            yield "</think>"
                            in_reasoning = False
                        rest = gate.finish()
                        if rest:
                            held.append(rest)
                        if gate.dropped:
                            logging.getLogger("memento.query").info(
                                "gaps: dropped %d characters before the first heading", gate.dropped)
                        # one parser decides what is shown and stored; anything dropped keeps the answer from being
                        # stored, also after a clean revision
                        from memento.research_gaps.passages import parse_gap_answer, split_check_notes
                        parsed = parse_gap_answer("".join(held))
                        dropped = gate.dropped > 0 or parsed.rejected
                        if parsed.kind == "rejected":
                            logging.getLogger("memento.query").info(
                                "gaps: answer rejected (%d characters), one retry without thinking",
                                len("".join(held)))
                            retry, retry_dropped = await gap_retry(turn.valves, turn.variant, gap_messages, turn.budget)
                            parsed = parse_gap_answer(retry)
                            dropped = retry_dropped or parsed.rejected
                            if parsed.kind == "rejected":
                                logging.getLogger("memento.query").info("gaps: retry rejected too, not stored")
                                async for piece in presentation.type_out(texts.GAP_FAILED_MSG, *presentation.typing_speed(turn.valves, turn.metadata)):
                                    yield piece
                                await turn.status("Done", done=True)
                                turn.done = True
                                return
                        held = [parsed.text]
                        check_ok = False
                        if gap_verify and "".join(held).strip():
                            final, result, error, late = await checking.thorough_check(turn.valves,
                                answer="".join(held), hits=gap_chunks, question=conversation.gap_question(turn.original_query),
                                check_prompt=gap_prompt, think=gap_think, client=turn.client, dom=turn.dom,
                                collection=turn.collection,
                                api_key=turn.api_key, status=turn.status,
                                retrieve_filter=resolve.check_search_filter(gap_paper, turn.in_progress),
                                seconds=generation.check_seconds(turn.valves, turn.budget),
                                background=bool((turn.metadata or {}).get("warmup")))
                            if result is not None and error is None:
                                for i in range(len(gap_chunks), len(result.hits)):   # found by re-searching
                                    await turn.emit(presentation.gap_citation(i, result.hits[i], turn.valves.FILES_URL))
                            check_ok = generation.check_ok(result, error, late)
                            if result is not None and error is None:
                                # the revision is model output too: parsed again; unreadable: the first draft's
                                # accepted sections, said to be unchecked
                                body, notes = split_check_notes(final)
                                revised = parse_gap_answer(body)
                                if revised.kind == "rejected":
                                    logging.getLogger("memento.query").info("gaps: revision unreadable, first draft")
                                    final = f"{''.join(held).rstrip()}\n\n{texts.GAP_REVISION_UNREADABLE_NOTE}"
                                    check_ok, dropped = False, True
                                elif revised.rejected:
                                    final = revised.text.rstrip() + (f"\n\n{notes}" if notes else "")
                                    dropped = True
                            turn.rec.checked = check_ok
                            async for piece in presentation.type_out(calm_gap_answer(final, focus) if show_table else "",
                                                      *presentation.typing_speed(turn.valves, turn.metadata, at_once=True)):
                                yield piece
                        else:
                            async for piece in presentation.type_out(calm_gap_answer("".join(held), focus) if show_table else "",
                                                      *presentation.typing_speed(turn.valves, turn.metadata)):
                                yield piece
                        authors = final if gap_verify and "".join(held).strip() and check_ok else "".join(held)
                        if turn.suggest and authors.strip():
                            await turn.status("Suggesting extensions...")
                            try:
                                block = await gap_suggestions(turn.valves,
                                    source=gap_paper, docs=turn.docs, variant=turn.variant, authors=authors,
                                    chunks=gap_chunks, question=turn.original_query, think=gap_think,
                                    budget=gap_budget, db_engine=turn.engine, time_budget=turn.budget)
                            except Exception as e:
                                import logging
                                logging.getLogger("memento.errors").warning("E-gap-suggest: %r", e)
                                block = texts.SUGGEST_FAILED_NOTE
                            from memento.research_gaps.passages import unfold_suggestions
                            block = block if focus else unfold_suggestions(block)    # asked to extend: open
                            async for piece in presentation.type_out(block, *presentation.typing_speed(turn.valves, turn.metadata)):
                                yield piece
                        yield generation.answered_by(answered.label)
                        if cache.storable(keyed=bool(turn.rec.key), got_content=bool("".join(held).strip()) or not gap_verify,
                                     interrupted=False, note=note, by_chosen_model=answered.id == turn.variant.id,
                                     verify=gap_verify, check_ok=check_ok):
                            if cache.gap_storable(gate.storable, authors) and not dropped:
                                turn.rec.clean, turn.rec.generated_by = True, answered.label
                            elif dropped:
                                logging.getLogger("memento.query").info("gaps: parts dropped, not stored")
                            elif gate.storable:
                                logging.getLogger("memento.query").info(
                                    "gaps: text outside headings and points (leaked reasoning?), not stored")
                            else:
                                logging.getLogger("memento.query").info("gaps: no heading in a long answer, not stored")
                    except generation.AnswerTimeout:                                 # over the limit: model already stopped
                        yield ("</think>" if in_reasoning else "") + "".join(held) + "\n\n" + texts.TIMEOUT_NOTE
                    except Exception as e:
                        if answered is not None and answered.is_cloud:      # broke after output began
                            yield ("</think>" if in_reasoning else "") + "".join(held) + "\n\n" + texts.STOPPED_NOTE
                        else:
                            yield ("</think>" if in_reasoning else "") + "\n\n" + texts.user_error("gap", e)
                    await turn.status("Done", done=True)
                    turn.done = True
                    return
                # no gap-tagged passages: say so
                yield (f"No research gap annotations found for **{gap_paper}**. "
                       f"Gap tagging may not have been run for this document yet.\n\n"
                       f"Falling back to regular retrieval...\n\n")
    except ImportError:
        pass
    except Exception as e:
        # the normal answer path follows; logged, not silent
        import logging
        logging.getLogger("memento.errors").warning("E-gap-path: %r", e)
