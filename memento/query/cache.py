"""Stored answers of the start-screen buttons (summaries, research gaps): what one answer recorded and whether it may be
stored, the prompt version and model key it is stored under, and a stored answer replayed.
"""
import asyncio
import re

from memento.query import generation, presentation, prompts, request, texts


# ── Stored answers for picker questions (memento/library/answers.py) ──
# A summary or research-gap question sent by the start-screen picker, in a new chat, for a document with a known
# fingerprint: answered from the stored answer when one exists for the same key (document, fingerprint, action,
# language, model, mode, prompt version), else answered normally and stored if it finished cleanly. Regenerate always
# answers afresh and replaces the stored answer.
ANSWER_FORMAT_VERSION = "2026-09-30"      # bump when code changes the shape of summary / gap answers


# the check failed (strategies.finalize_for_display, memento/agentic_validation/strategies.py): its note, prefix only
UNAVAILABLE_NOTE_PREFIX = "_⚠ Verification unavailable for this answer"


UNSTORED_NOTES = (texts.TIMEOUT_NOTE, texts.STOPPED_NOTE, texts.CHECK_TIMEOUT_NOTE, texts.QUOTA_NOTE.strip(), UNAVAILABLE_NOTE_PREFIX)


class Recorder:
    """What one answer yielded and emitted, and whether it may be stored."""

    def __init__(self):
        self.text, self.citations = [], []
        self.key, self.fields, self.generated_by = None, None, ""
        self.clean = self.hit = self.checked = False
        # for a warm-up's report: why the answer is not stored, and the key a replayed answer had
        self.skip_reason, self.hit_key = "", None


class PickerRequest:
    """One picker question's own choices: the model that answers, the Thorough-check switch, a Regenerate, the
    question's language. The live picker passes the request's; the pre-warm a standard local click."""

    def __init__(self, variant, thorough: bool = False, regenerated: bool = False, language: str = "en"):
        self.variant, self.thorough, self.regenerated, self.language = variant, thorough, regenerated, language


def replay_allowed(*, checked: bool, regenerated: bool, thorough: bool) -> bool:
    """May a stored summary / gap answer be replayed? Not after Regenerate; with "Thorough check" on only if it was
    checked (the warm-up writes checked answers; an unchecked one is written anew and checked)."""
    return not regenerated and (checked or not thorough)


def storable(*, keyed: bool, got_content: bool, interrupted: bool, note, by_chosen_model: bool, verify: bool,
             check_ok: bool) -> bool:
    """May this answer be stored (and replace the stored one)? A checked answer only if the check finished."""
    return keyed and got_content and not interrupted and not note and by_chosen_model and (not verify or check_ok)


def strip_think(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    return text.split("<think>")[0].strip()


def _regenerated(messages: dict, message_id) -> bool:
    """Is the answer being written a second answer to the same question (OpenWebUI's Regenerate)?"""
    m = (messages or {}).get(message_id) if message_id else None
    parent = (messages or {}).get((m or {}).get("parentId"))
    if not m or not parent:
        return False
    return any(c != message_id and ((messages.get(c) or {}).get("role") == "assistant")
               for c in parent.get("childrenIds") or [])


async def is_regeneration(metadata: dict | None) -> bool:
    chat_id = (metadata or {}).get("chat_id")
    if not chat_id or str(chat_id).startswith("local:"):
        return False
    try:
        from open_webui.models.chats import Chats
        chat = await Chats.get_chat_by_id(chat_id)
        hist = ((chat.chat if chat else {}) or {}).get("history") or {}
        return _regenerated(hist.get("messages") or {}, (metadata or {}).get("message_id"))
    except Exception:
        return False


def cache_eligible(picked, history, content_hash, in_progress) -> tuple[bool, str]:
    if not picked:
        return False, "not a picker question"
    if history:
        return False, "history"
    if not content_hash:
        return False, "no fingerprint"
    if in_progress:
        return False, "library changing"
    return True, ""


def model_key(variant, valves) -> str:
    """The dropdown entry and the actual model behind it: another model, or a swapped local model, is another key."""
    return f"{variant.id}:{variant.model if variant.is_cloud else valves.LLM_MODEL}"


def summary_prompt_version(system_prompt: str, mode: str, valves) -> str:
    from memento.library.answers import prompt_version
    return prompt_version(ANSWER_FORMAT_VERSION, "summary", system_prompt,
                          prompts.SUMMARY_PARTIAL_ADDENDUM if mode == "partial" else prompts.SUMMARY_ADDENDUM, request.CHARS_PER_TOKEN,
                          valves.SUMMARY_MAX_TOKENS, prompts.SUMMARY_LABEL_CHARS, valves.TEMPERATURE, valves.MAX_TOKENS)


def gap_prompt_version(budget, valves, thinking: bool = True, prompt: str | None = None) -> str:
    from memento.library.answers import prompt_version
    from memento.research_gaps.passages import GAP_SYSTEM_PROMPT
    # an answer written without thinking is stored under its own key; with thinking the key is unchanged. `prompt`: the
    # document's own gap prompt (gap_system_prompt), part of the key
    return prompt_version(ANSWER_FORMAT_VERSION, "gaps", prompt or GAP_SYSTEM_PROMPT, valves.GAP_BUDGET_TOKENS, budget,
                          valves.TEMPERATURE, valves.MAX_TOKENS, *(() if thinking else ("no-thinking",)))


def still_in_library(engine, collection: str, source: str) -> bool:
    """Is the document still in the library list? Checked before an answer is stored, so an answer written while its
    document was removed is not stored. A failed check stores."""
    try:
        from memento.library import documents as ldb
        return ldb.get_document(engine, collection, source) is not None
    except Exception:
        return True


def gap_answer_replayable(answer: str) -> bool:
    """May a stored research-gap answer be replayed (research_gaps.passages.stored_gap_ok)?"""
    from memento.research_gaps.passages import stored_gap_ok
    return stored_gap_ok(answer)


def gap_storable(gate_storable: bool, answer: str) -> bool:
    """Store this research-gap answer for everyone? Not when the heading gate says no, and never with leaked
    reasoning in it (gap_answer_clean)."""
    from memento.research_gaps.passages import gap_answer_clean
    return gate_storable and gap_answer_clean(answer)


async def replay(row: dict, emit, rate: float = 0, max_s: float = 0, typed_chars: int = 0):
    """A stored answer: its citation events first (so [N] map as they did), then its text, typed out (its sources list
    in the current layout)."""
    for ev in row["citations"]:
        await emit(ev)
    async for piece in presentation.type_out(presentation.replay_sources(row["answer"], row["citations"]), rate, max_s, typed_chars):
        yield piece


def cache_lookup(engine, fields: dict, regenerated: bool, count_hit: bool = True) -> dict | None:
    if regenerated:
        return None
    from memento.library import answers as ac
    return ac.get(engine, ac.cache_key(**fields), count_hit=count_hit)


def suggest_prompt_version(authors: str, *, german: bool) -> str:
    """The stored extensions' prompt version: German ones also by the address rule, so a change to it is not answered
    from an older stored text; English ones unchanged."""
    from memento.library import answers as ac
    from memento.research_gaps.passages import SUGGEST_SYSTEM_PROMPT
    return ac.prompt_version(SUGGEST_SYSTEM_PROMPT, authors, texts.DE_ADDRESS_RULE) if german else \
        ac.prompt_version(SUGGEST_SYSTEM_PROMPT, authors)


# ── picker answers: one key builder and one replay decision for the live picker and the pre-warm ──
def prompt_version(action: str, mode: str | None, dom: dict | None, valves, *, think: bool,
                   title: str | None = None) -> str:
    """The prompt version of a picker answer: a summary by its mode, research gaps by the thinking decision and the
    document's gap prompt (`title`: its library title; a talk's prompt differs)."""
    if action == "summary":
        return summary_prompt_version(prompts.answer_prompt(dom), mode, valves)
    from memento.research_gaps.passages import gap_system_prompt
    return gap_prompt_version(generation.gap_thinking_budget(False, valves.GAP_THINKING_BUDGET), valves,
                               thinking=generation.gap_thinks(think, False, valves.GAP_THINKING),
                               prompt=gap_system_prompt(title))


def picker_fields(valves, req: "PickerRequest", source: str, content_hash: str, action: str, mode: str | None,
                  prompt_version: str) -> dict:
    return dict(collection=valves.COLLECTION, source=source, content_hash=content_hash, action=action,
                language=req.language, model=model_key(req.variant, valves), mode=mode,
                prompt_version=prompt_version)


def picker_replay(engine, fields: dict, req: "PickerRequest", valid=None, *,
                  count_hit: bool = True) -> tuple:
    """(the stored answer this request may replay, or None; why not). Order: the lookup (none on a Regenerate), the
    request's Thorough check, the stored answer's own check."""
    try:
        row = cache_lookup(engine, fields, req.regenerated, count_hit=count_hit)
    except Exception as e:
        import logging
        logging.getLogger("memento.errors").warning("E-answer-cache: %r", e)
        return None, "cache not readable"
    if not row:
        return None, "regenerate" if req.regenerated else "not stored"
    if not replay_allowed(checked=row.get("effective_checked", bool(row.get("checked"))),
                           regenerated=req.regenerated, thorough=req.thorough):
        return None, "not checked"           # "Thorough check" on and the stored answer is not checked
    if valid is not None and not valid(row["answer"]):
        return None, "fails its check"       # a research-gap answer with a leaked line, say
    return dict(row, answer=generation.replay_text(row)), ""     # no old claim of a check


async def prepare(valves, rec, *, picked, history, docs, in_progress, source, action, mode, prompt_version,
                  variant, metadata, db_engine, thorough: bool = False, valid=None) -> dict | None:
    """The stored answer for this picker question, or None; on a miss `rec` is keyed so the answer gets stored.
    The request's own choices (its model, Thorough check, Regenerate) decide (PickerRequest)."""
    import logging
    log = logging.getLogger("memento.query")
    h = ((docs or {}).get(source) or {}).get("content_hash")
    ok, why = cache_eligible(picked if picked and picked[0] == source else None, history, h, in_progress)
    if not ok:
        if picked:
            log.info("cache: skip %s %s (%s)", action, source, why)
        # a warm-up's final verdict, except for passing states (an upload under way, a document not listed yet):
        # those report nothing, so the warm-up is tried again
        rec.skip_reason = "" if why in ("library changing", "no fingerprint") else why
        return None
    from memento.library import answers as ac
    req = PickerRequest(variant=variant, thorough=thorough, regenerated=await is_regeneration(metadata))
    fields = picker_fields(valves, req, source, h, action, mode, prompt_version)
    key = ac.cache_key(**fields)
    row, why = await asyncio.to_thread(picker_replay, db_engine(), fields, req, valid)
    if why == "fails its check":
        log.info("cache: stored %s %s fails its check, written anew", action, source)
    if row:
        rec.hit, rec.hit_key = True, key
        log.info("cache: hit %s %s %s %s", action, source, fields["model"], key[:12])
        return row
    rec.key, rec.fields = key, fields
    log.info("cache: miss %s %s %s %s%s", action, source, fields["model"], key[:12],
             " (thorough)" if thorough else " (regenerate)" if req.regenerated else "")
    return None
