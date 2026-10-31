"""Rework of an earlier answer (shorter, simpler, in a table, another language): the answer and the passages it cited,
with the model asked to rewrite, not to search again.
"""
import re

from memento.query import generation, presentation, prompts, request, texts
from memento.query.scope import conversation


# A rework follow-up (summarize, reformat or translate the previous answer): no new search; the previous answer and its
# own sources go to the model, and the same citations are sent again, so every [N] still opens the same passage.
REWORK_ADDENDUM = ("\n\nThe user asks you to rework your previous answer, shown below with its numbered sources: "
                   "shorten, reformat, translate, simplify or summarize it as requested. Use only what the previous "
                   "answer and its sources state, and add no new facts. Keep each [N] citation on the statement it "
                   "supports, with the same numbers. When the previous answer covers several documents, keep what each "
                   "one says apart: do not link them or say that one relates to, uses or applies to another unless "
                   "the previous answer says so.")
# The language of a rework asked in English: with the German address rule as the prompt's only language line, a German
# answer was reworked in German. A request for another language ("translate it into German") still gets it.
REWORK_EN_RULE = ("\n\nThe request is in English: write the reworked answer in English, unless the request asks for "
                  "another language.")
_LANG = request.LANG                # rework_messages' parameter `request` shadows the module


REWORK_MAX_CHARS = 60000      # previous answer + its sources; longer (a verbatim chapter) → the normal path


def _cited_numbers(text: str, n: int) -> list:
    """The [N] the text cites that exist among n sources, in order (0-based)."""
    return sorted({int(k) - 1 for k in re.findall(r"\[(\d+)\]", text or "") if 0 < int(k) <= n})


def rework_size(previous: str, sources: list) -> int:
    """What a rework sends: the previous answer and the sources it cites, not all of its sources."""
    return len(previous) + sum(len(" ".join(sources[i].get("document") or []))
                               for i in _cited_numbers(previous, len(sources)))


def rework_messages(system_prompt: str, previous: str, sources: list, request: str) -> list:
    parts = []
    cited = _cited_numbers(previous, len(sources))
    if cited:                       # only the sources the answer cites, under their own numbers
        parts.append("Sources:\n" + "\n\n---\n\n".join(
            f"[{i + 1}] {(sources[i].get('source') or {}).get('name', '')}\n{' '.join(sources[i].get('document') or [])}"
            for i in cited))
    parts.append(f"Previous answer:\n{previous}")
    parts.append(f"Request: {request}")
    lang = "" if _LANG.get() == "de" else REWORK_EN_RULE
    return [{"role": "system", "content": system_prompt + REWORK_ADDENDUM + texts.DE_ADDRESS_RULE + lang},
            {"role": "user", "content": "\n\n".join(parts)}]


def previous_answer(chat_messages: list) -> str:
    """The last answer before this message as the reader saw it: [N] kept, thinking, figures, notes and the
    sources / related-documents blocks removed; "" when there is none."""
    for m in reversed(chat_messages[:-1]):
        if m.get("role") == "assistant":
            text = texts.sanitize_assistant_text(request.message_text(m.get("content")))
            text = presentation.RELATED_MARK_RE.sub("\n", presentation.SOURCES_MARK_RE.sub("\n", text))
            text = re.sub(r"!\[[^\]]*\]\([^)]*\)\s*(\*Figure \(\d+\):[^\n]*\*)?", "", texts.MODEL_NOTE_RE.sub("\n", text))
            return text.strip()
    return ""


async def reply(turn):
    """A rework of the previous answer ("in three bullets", "as a table"): its own sources sent again, no search;
    without them the normal path goes on (query_intent ranked_retrieval)."""
    import logging
    previous = previous_answer(turn.chat_messages)
    stored = await conversation.stored_sources(turn.metadata) if previous else None
    sources_ = stored or []
    size = rework_size(previous, sources_)
    usable = bool(previous) and size < REWORK_MAX_CHARS and (bool(sources_) or not re.search(r"\[\d+\]", previous))
    logging.getLogger("memento.query").info("rework: %s (previous %d chars, %s sources)",
                                            "yes" if usable else "no, normal path", len(previous),
                                            "unknown" if stored is None else len(sources_))
    if usable:
        await turn.status("Reworking the previous answer...")
        for src in sources_:
            await turn.emit({"type": "citation", "data": src})
        out = []
        try:
            chunks, answered, note = await generation.open_answer(
                turn.valves, turn.variant, think=False, budget=turn.budget,
                messages=rework_messages(prompts.answer_prompt(turn.dom), previous, sources_, turn.original_query))
            if note:
                yield note
            async for chunk in chunks:
                piece = chunk.choices[0].delta.content if chunk.choices else None
                if piece:
                    out.append(piece)
                    yield piece
        except generation.AnswerTimeout:
            yield "\n\n" + texts.TIMEOUT_NOTE
        except Exception as e:
            yield "\n\n" + texts.user_error("answer", e)
            await turn.status("Done", done=True)
            turn.done = True
            return
        cited = {int(n) - 1 for n in re.findall(r"\[(\d+)\]", "".join(out))}
        yield generation.answered_by(answered.label)
        yield presentation.names_marker([(x.get("source") or {}).get("name", "") for x in sources_], cited,
                            docs=presentation.stored_doc_ids(sources_))
        await turn.status("Done", done=True)
        turn.done = True
        return
    turn.query_intent = "ranked_retrieval"
