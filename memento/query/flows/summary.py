"""A document summarised: the whole document when it fits the budget, else the budgeted selection of its passages,
in reading order, as the summary's numbered sources.
"""
import asyncio
import re

from memento.query import cache, checking, generation, library_state, presentation, prompts, request, texts
from memento.query.flows import figures
from memento.query.scope import resolve


# ── Whole-document summary ──
# The normal path would see only the top reranked passages. A document whose text fits the model's window is sent whole
# instead: every passage in reading order, grouped into one numbered source per page, so the summary covers the whole
# document and cites pages. Longer documents get a budgeted selection of their main parts (summary_hits). Keyword
# fragments and the reference list are left out.
SUMMARY_SKIP_SECTIONS = ("Keywords", "References")


# document + history must fit the model's window minus MAX_TOKENS and the instructions; the default caps (100,000 +
# 6,000) fit a 131,072-token window
SUMMARY_PROMPT_TOKENS = 106000


def _summary_sources(passages: list) -> list:
    """The summary's sources: the document's passages themselves, so each [N] marks its passage on the page. Text only:
    a figure is read by its caption and description, its image is not attached (a whole document's figures would
    overflow the prompt)."""
    return [{**h, "img_path": None} for h in passages]


def summary_hits(passages: list, history_tokens: float, max_tokens: float, prompt_tokens: float) -> tuple[list, str | None]:
    """(sources, mode) for a document summary: the whole document when it fits ("whole"), else the budgeted selection
    ("partial"); counts to the log only. Labels count toward the budget."""
    import logging
    if not passages:
        return [], None
    from memento.query.doc_budget import select_summary_passages
    src = passages[0].get("source", "")
    est = sum(len(p["text"]) + prompts.SUMMARY_LABEL_CHARS for p in passages) / request.CHARS_PER_TOKEN
    if est <= max_tokens and est + history_tokens <= prompt_tokens:
        logging.getLogger("memento.query").info("summary: %s passages=%d/%d est_tokens=%d history=%d partial=False",
                                                src, len(passages), len(passages), est, history_tokens)
        return _summary_sources(passages), "whole"
    budget = min(max_tokens, prompt_tokens - history_tokens)
    if budget <= 0:
        return [], None
    chosen, st = select_summary_passages(passages, budget, request.CHARS_PER_TOKEN, prompts.SUMMARY_LABEL_CHARS)
    logging.getLogger("memento.query").info("summary: %s passages=%d/%d tokens=%d history=%d partial=True",
                                            src, st["chosen"], st["total"], st["tokens"], history_tokens)
    return (_summary_sources(chosen), "partial") if chosen else ([], None)


def document_passages(client, collection: str, source: str) -> list:
    """The document in reading order without keyword fragments and the reference list (the summary's input)."""
    return [h for h in library_state.ordered_passages(client, collection, source) if h.get("section") not in SUMMARY_SKIP_SECTIONS]


async def prepare(turn):
    """A summary of one document: its stored answer, or the whole document (its budgeted parts when long) as the
    passages."""
    import logging
    try:
        passages = await asyncio.to_thread(document_passages, turn.client, turn.collection, turn.paper)
        hist_est = sum(len(m["content"]) for m in turn.history) / request.CHARS_PER_TOKEN
        # document + history within the prompt budget (window − answer tokens − instructions); a longer document
        # gets its budgeted main parts
        doc_hits, turn.summary_mode = summary_hits(passages, hist_est, turn.valves.SUMMARY_MAX_TOKENS,
                                               SUMMARY_PROMPT_TOKENS)
        if doc_hits:
            row = None if turn.doc_chat is not None else await cache.prepare(
                turn.valves, turn.rec, db_engine=turn.engine, picked=turn.picked, history=turn.history, docs=turn.docs, in_progress=turn.in_progress, source=turn.paper,
                action="summary", mode=turn.summary_mode,
                thorough=request.THOROUGH_FILTER_ID in ((turn.metadata or {}).get("filter_ids") or []),
                prompt_version=cache.prompt_version("summary", turn.summary_mode, turn.dom, turn.valves, think=turn.think),
                variant=turn.variant, metadata=turn.metadata)
            if row:
                async for c in cache.replay(row, turn.emit, *presentation.typing_speed(turn.valves, turn.metadata)):
                    yield c
                await turn.status("Done", done=True)
                turn.done = True
                return
            turn.hits, turn.whole_doc = doc_hits, True
            await turn.status("Reading the whole document...")
    except Exception as e:
        logging.getLogger("memento.errors").warning("E-summary: %r", e)


async def prepare_files(turn) -> None:
    """Document Chat with several files and none named: each file's passages for its own summary."""
    import logging
    try:
        hist_est = sum(len(m["content"]) for m in turn.history) / request.CHARS_PER_TOKEN
        all_hits: list = []
        for src in turn.file_order:
            passages = await asyncio.to_thread(document_passages, turn.client, turn.collection, src)
            part_hits, mode = summary_hits(passages, hist_est, turn.valves.SUMMARY_MAX_TOKENS,
                                            SUMMARY_PROMPT_TOKENS)
            if part_hits:
                turn.multi_parts.append((src, len(all_hits), len(all_hits) + len(part_hits), mode))
                all_hits += part_hits
        if len(turn.multi_parts) >= 2:
            turn.hits, turn.whole_doc = all_hits, True
            await turn.status("Reading the files...")
        else:
            turn.multi_parts = []
    except Exception as e:
        turn.multi_parts = []
        logging.getLogger("memento.errors").warning("E-summary: %r", e)


async def reply_files(turn):
    """Document Chat's summary of several files: one answer per file under its heading, checked per file with Thorough
    check; one footer and one source list for all."""
    import logging
    thorough = request.THOROUGH_FILTER_ID in ((turn.metadata or {}).get("filter_ids") or [])
    turn.verify = checking.verify_applies(turn.valves.VALIDATION_ENABLED, thorough=thorough, query_intent="summary")
    p1 = ""
    if turn.verify:
        try:
            from memento.agentic_validation.strategies import P1_ADDENDUM as p1
        except Exception:
            turn.verify = False
    answered, full, n_parts = turn.variant, [], len(turn.multi_parts)
    try:
        for k, (src, lo, hi, mode) in enumerate(turn.multi_parts, start=1):
            await turn.status(f"Summarizing file {k} of {n_parts}...")
            turn.check_prompt = prompts.answer_prompt(turn.dom) + (prompts.SUMMARY_PARTIAL_ADDENDUM if mode == "partial"
                                                  else prompts.SUMMARY_ADDENDUM)
            part_prompt = turn.check_prompt + (p1 if turn.verify else "")
            if turn.history:
                part_prompt += texts.HISTORY_RULE
            context = "\n\n---\n\n".join(f"[{i + 1}] {presentation.chunk_label(turn.hits[i], turn.whole_doc, turn.ctx_names)}\n{turn.hits[i]['text']}"
                                          for i in range(lo, hi))
            user_message = (f'Sources (all from the document "{presentation.shown(src)}"):\n{context}\n\n'
                            f"Question: {turn.original_query}")
            content: list | str = user_message
            images = []
            for i in range(lo, hi):
                if len(images) // 2 >= turn.valves.MAX_IMAGES_TO_LLM:
                    break
                h = turn.hits[i]
                if (h["type"] in ("image", "chart") and h.get("img_path") and not presentation.icon_image(h)
                        and not presentation.reference_image(h)):
                    url = prompts.verified_image_url(h["img_path"])
                    if url:
                        images += [{"type": "text", "text": presentation.image_label(i + 1)},
                                   {"type": "image_url", "image_url": {"url": url}}]
            if images:
                content = [{"type": "text", "text": user_message}] + images
            yield ("" if k == 1 else "\n\n") + f"## {presentation.shown(src)}\n\n"
            chunks, answered, note = await generation.open_answer(
                turn.valves, turn.variant, think=False, budget=turn.budget,   # as a single summary: no thinking first
                messages=[{"role": "system", "content": part_prompt}] + turn.answer_history
                         + [{"role": "user", "content": content}])
            if note:
                yield note
            if turn.verify:
                text = re.sub(r"(?s)<think>.*?</think>", "", await generation.drain_content(chunks, answered.is_cloud))
                final, result, error, late = await checking.thorough_check(turn.valves,
                    answer=text.strip(), hits=turn.hits, question=turn.original_query, check_prompt=turn.check_prompt,
                    think=turn.think, client=turn.client, dom=turn.dom, api_key=turn.api_key, status=turn.status,
                    collection=turn.collection, retrieve_filter=resolve.check_search_filter(src, turn.in_progress),
                    seconds=generation.check_seconds(turn.valves, turn.budget))
                if result is not None and error is None:
                    for i in range(len(turn.hits), len(result.hits)):      # sources added by the check
                        await turn.emit(presentation.citation_event(i, result.hits[i], turn.valves.FILES_URL))
                    turn.hits = result.hits
                text = final
                yield text
            else:
                got = []
                try:
                    async for ch in chunks:
                        if ch.choices and ch.choices[0].delta.content:
                            got.append(ch.choices[0].delta.content)
                            yield got[-1]
                except generation.AnswerTimeout:
                    got.append("\n\n" + texts.TIMEOUT_NOTE)
                    yield got[-1]
                text = "".join(got)
            full.append(text)
    except Exception as e:
        yield "\n\n" + texts.user_error("answer", e)
        await turn.status("Done", done=True)
        turn.done = True
        return
    logging.getLogger("memento.query").info("summary of %d files: %s", n_parts,
                                            " | ".join(f"{src} {hi - lo}" for src, lo, hi, _ in turn.multi_parts))
    cited = {int(n) - 1 for n in re.findall(r"\[(\d+)\]", "".join(full))}
    yield generation.answered_by(answered.label)
    yield presentation.sources_marker(turn.hits, cited)
    cap = turn.valves.MAX_FIGURES_SHOWN if turn.valves.MAX_FIGURES_SHOWN > 0 else None
    shown, more = figures.figures_to_show(turn.hits, cited, "".join(full), set(), cap)
    for i in shown:
        url = prompts.verified_image_url(turn.hits[i]["img_path"])
        if url:
            yield figures.figure_markdown(i, turn.hits[i], url)
    if more:
        from memento.common.text_language import question_language
        yield figures.more_figures_note(more, question_language(turn.original_query) == "de")
    await turn.status("Done", done=True)
    turn.done = True
