"""The answer model's calls: the copy's model variants, the answer and thinking budgets, the guarded client with its
time limits, a cloud model's quota and its local fallback, the answer opened and streamed, and the notes under it.
"""
import asyncio
import os
import re
import threading
import time
from pathlib import Path

from memento.query import request, texts


# ── Model variants ──
# Users pick the model that writes the answer (config/query_models.yaml); the router, chat titles and Thorough check
# stay on the local model. A cloud model that is out of quota or fails before any output is replaced by the local one.
FIRST_OUTPUT_TIMEOUT = 90          # seconds until the first streamed output from a cloud model


# Time limits: an answer longer than the ANSWER_TIMEOUT Valve or silent for STALL_TIMEOUT s is stopped (the model
# connection is closed) and ends with TIMEOUT_NOTE; Thorough check has its own VERIFY_TIMEOUT and then shows the first
# answer with CHECK_TIMEOUT_NOTE.
STALL_TIMEOUT = 90


_CHECK_STATUS_NOTES = (texts.CHECKED_NOTE, texts.PARTLY_CHECKED_NOTE, texts.NO_CLAIMS_NOTE, texts.CHECKED_UNCONFIRMED_NOTE, texts.CHECKED_UNCONFIRMED_UNMARKED_NOTE,
                       texts.PARTLY_UNCONFIRMED_NOTE, texts.PARTLY_UNCONFIRMED_UNMARKED_NOTE)


def checked_text(final: str, result, error, late: bool) -> str:
    """A Thorough-checked answer with its note: CHECKED_NOTE (every statement judged, none still failing),
    PARTLY_CHECKED_NOTE (some could not be judged) or NO_CLAIMS_NOTE (none was extracted); statements still failing after the revisions (verdicts and
    scientific-check issues) are marked ⚠ after their sentence and the note says so. Unchanged when the check could not
    be done: that answer already carries its note (CHECK_TIMEOUT_NOTE, or finalize_for_display's on an error)."""
    from memento.agentic_validation.strategies import check_state, mark_unconfirmed, remaining_failures
    state = check_state(result, error, late)
    if state == "unavailable":
        return final
    if state == "no_claims":
        return f"{final.rstrip()}\n\n{texts.NO_CLAIMS_NOTE}"
    left = remaining_failures(result)
    if not left:
        note = texts.CHECKED_NOTE if state == "complete" else texts.PARTLY_CHECKED_NOTE
    else:
        marked = False                        # arm C removes sentences, so its spans no longer fit: it never marks
        if getattr(result, "arm", "A") != "C":
            final, marked = mark_unconfirmed(final, left)
        note = ((texts.CHECKED_UNCONFIRMED_NOTE if marked else texts.CHECKED_UNCONFIRMED_UNMARKED_NOTE) if state == "complete"
                else (texts.PARTLY_UNCONFIRMED_NOTE if marked else texts.PARTLY_UNCONFIRMED_UNMARKED_NOTE))
    return f"{final.rstrip()}\n\n{note}"


def report_checked(res):
    """A report's or proposal's prose check as its `checked`: True (every statement judged), "partial" (some could not
    be judged: kept and said), "no_claims" (none was extracted, so nothing was checked: kept and said), False (none
    judged: the prose is left out)."""
    from memento.agentic_validation.strategies import check_state
    state = check_state(res, None, False)
    return True if state == "complete" else state if state in ("partial", "no_claims") else False


def check_ok(result, error, late: bool) -> bool:
    """A check that judged every statement and left none failing: only then is an answer stored as checked."""
    from memento.agentic_validation.strategies import check_state, remaining_failures
    return check_state(result, error, late) == "complete" and not remaining_failures(result)


def _strip_check_status(text: str) -> str:
    """The text without its check-status notes (EN or DE): a stored answer not checked under the current contract
    replays without its old claim of a check. Warnings (⚠ …) stay; a text without such a note is returned unchanged."""
    out, found = text or "", False
    for note in sorted(_CHECK_STATUS_NOTES + tuple(texts.DE_FIXED[n] for n in _CHECK_STATUS_NOTES), key=len, reverse=True):
        if note in out:
            out, found = out.replace(note, ""), True
    return re.sub(r"\n{3,}", "\n\n", out).strip() if found else text


def replay_text(row: dict) -> str:
    """A stored answer's text as it is replayed: unchanged when it is checked under the current contract
    (answer_cache.CHECK_CONTRACT), else without an old check-status note."""
    effective = row.get("effective_checked", bool(row.get("checked")))
    return row["answer"] if effective else _strip_check_status(row["answer"])


def answered_by(label: str, checked_by: str | None = None) -> str:
    return f"\n\n_Answered by {label}{', checked by ' + checked_by if checked_by else ''}._\n"


def text_only(messages: list[dict]) -> list[dict]:
    """The messages for a model without vision: a multimodal user message keeps its first text part (the prompt with
    the figure captions); the "Image for Source [N]:" labels and images are dropped."""
    out = []
    for m in messages:
        if isinstance(m.get("content"), list):
            text = next((p["text"] for p in m["content"] if p.get("type") == "text"), "")
            m = {**m, "content": text}
        out.append(m)
    return out


def copy_variants(valves=None):
    """This copy's models (its VARIANTS_FILE Valve, a file name in config/); a broken or unsafe file leaves only the
    local model, never an empty list."""
    from memento.query.models import CONFIG, Variant, load_variants
    name = "query_models.yaml" if valves is None else (getattr(valves, "VARIANTS_FILE", "") or "").strip()
    try:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+\.yaml", name) or name.startswith("."):
            raise ValueError(f"not a file name in config/: {name!r}")
        return load_variants(CONFIG.parent / name)
    except Exception as e:
        import logging
        logging.getLogger("memento.errors").warning("E-models: config/%s unusable: %r", name, e)
        return [Variant(id="local", name="Library Chat", label="the local model", provider="local", model="",
                        description="")]


def _has_output(ch) -> bool:
    """A chunk with answer text or thinking (servers often send an empty role-only chunk at once, which is not output)."""
    if not getattr(ch, "choices", None):
        return False
    d = ch.choices[0].delta
    return bool(getattr(d, "content", None) or getattr(d, "reasoning_content", None) or getattr(d, "reasoning", None))


async def primed(stream, deadline: float):
    """Pull chunks until one carries real output, by `deadline` (event-loop time; asyncio.TimeoutError otherwise),
    then hand out everything pulled plus the rest: the fallback decision is made before the user sees anything."""
    it = stream.__aiter__()
    loop = asyncio.get_running_loop()
    pulled = []
    while True:
        try:
            ch = await asyncio.wait_for(it.__anext__(), max(0.0, deadline - loop.time()))
        except StopAsyncIteration:
            break
        pulled.append(ch)
        if _has_output(ch):
            break

    async def chunks():
        for ch in pulled:
            yield ch
        async for ch in it:
            yield ch
    return chunks()


class Budget:
    """A response's time: `end` (event-loop time), and the cut-off for thinking `reserve` s before it."""

    def __init__(self, seconds: float, reserve: float):
        self.end = asyncio.get_running_loop().time() + seconds
        self.reserve = reserve

    @property
    def think_end(self) -> float:
        return self.end - self.reserve

    def left(self) -> float:
        return self.end - asyncio.get_running_loop().time()


class Answering:
    """The variant that answers, which can change while the answer streams (thinking cut off → the local model)."""

    def __init__(self, v):
        self.v = v

    def __getattr__(self, name):
        return getattr(self.__dict__["v"], name)


def note_chunk(text: str):
    """A note to the reader as one more piece of the answer's text."""
    from types import SimpleNamespace
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text, reasoning_content=None))])


class AnswerTimeout(Exception):
    """The answer ran past its time limit or went silent; the model connection is already closed."""


class CheckStopped(BaseException):
    """Thorough check stopped (Stop pressed, or its time limit). A BaseException on purpose: the verifiers turn a
    failed call (any Exception) into "unjudged" and carry on, which would keep calling the model."""


def stopped() -> bool:
    """True inside a task OpenWebUI is cancelling (the user pressed Stop): nothing more is sent to the chat."""
    task = asyncio.current_task()
    return bool(task is not None and task.cancelling())


async def limited(chunks, stream, deadline: float):
    """`chunks` until `deadline` (event-loop time), each within STALL_TIMEOUT s; otherwise AnswerTimeout. However it
    ends (done, time limit, Stop), `stream` is closed so the model stops generating."""
    it = chunks.__aiter__()
    loop = asyncio.get_running_loop()
    try:
        while True:
            left = deadline - loop.time()
            try:
                ch = await asyncio.wait_for(it.__anext__(), max(0.0, min(STALL_TIMEOUT, left)))
            except StopAsyncIteration:
                return
            except asyncio.TimeoutError:
                raise AnswerTimeout("time limit" if left <= STALL_TIMEOUT else f"no output for {STALL_TIMEOUT} s")
            yield ch
    except asyncio.CancelledError:
        # Stop, or the thinking cut at the budget (think_capped): only pipe() knows which, and logs a Stop
        import logging
        logging.getLogger("memento.query").info("model stream cancelled; model connection closed")
        raise
    finally:
        close = getattr(stream, "close", None)
        if close is not None:
            try:
                await close()
            except Exception:
                pass


class GuardedLLM:
    """An OpenAI-shaped client for Thorough check: before each call, stop if Stop was pressed or the check's time is
    up; each call is capped at the time left. A call already running cannot be aborted (closing a sync client does
    not cancel it), so the check ends after at most that one call."""

    def __init__(self, llm, stop: threading.Event, deadline: float, slots: threading.Semaphore | None = None):
        # slots: at most this many calls at once (the background warm-up: one, so users keep the model's other slots)
        self._llm, self._stop, self._deadline, self._slots = llm, stop, deadline, slots
        self.chat = self
        self.completions = self

    def create(self, **kw):
        if self._slots is None:
            return self._call(**kw)
        with self._slots:
            return self._call(**kw)

    def _call(self, **kw):
        if self._stop.is_set():
            raise CheckStopped("stopped by the user")
        left = self._deadline - time.monotonic()
        if left <= 0:
            raise CheckStopped("time limit")
        kw["timeout"] = min(kw.get("timeout") or left, left)
        return self._llm.chat.completions.create(**kw)


async def drain_content(stream, cloud: bool) -> str:
    """A stream's answer text. A cloud stream that breaks keeps what came and ends with the stopped note; a local one
    raises. An answer over its time limit keeps what came and ends with the timeout note."""
    out = []
    try:
        async for ch in stream:
            if ch.choices and ch.choices[0].delta.content:
                out.append(ch.choices[0].delta.content)
    except AnswerTimeout:
        out.append("\n\n" + texts.TIMEOUT_NOTE)
    except Exception:
        if not cloud:
            raise
        out.append("\n\n" + texts.STOPPED_NOTE)
    return "".join(out)


def thorough_thinking_budget(valves) -> int:
    """The thinking cap of an answer with Thorough check on and of the check's revisions (THOROUGH_THINKING_BUDGET;
    0 = the normal THINKING_BUDGET)."""
    return valves.THOROUGH_THINKING_BUDGET or valves.THINKING_BUDGET


def gap_thinking_budget(thorough: bool, cap: int) -> int | None:
    """The local model's thinking cap for a research-gap answer: GAP_THINKING_BUDGET, also with "Thorough check"
    (uncapped thinking can use up the check's time); None (the normal THINKING_BUDGET) only when the cap is 0."""
    return None if cap <= 0 else cap


def response_budget(valves, *, thorough: bool, warm: bool = False) -> "Budget | None":
    """The time budget of a response: Thorough check (or a long task) THOROUGH_BUDGET_S, else ANSWER_BUDGET_S; none
    for the background warm-up."""
    if warm:
        return None
    return Budget(valves.THOROUGH_BUDGET_S if thorough else valves.ANSWER_BUDGET_S,
                   valves.ANSWER_RESERVE_S)


def check_seconds(valves, budget: "Budget | None") -> float:
    """The time a check after the answer may take: VERIFY_TIMEOUT, at most the time left (at least 10 s)."""
    if budget is None:
        return valves.VERIFY_TIMEOUT
    return max(10.0, min(valves.VERIFY_TIMEOUT, budget.left()))


def gen_extra(valves, think: bool, budget: int | None = None) -> dict:
    """extra_body for an answer call: Qwen3 thinking switch + the thinking-token cap when thinking (budget: this
    answer's cap, else THINKING_BUDGET)."""
    extra = {"chat_template_kwargs": {"enable_thinking": think}}
    cap = valves.THINKING_BUDGET if budget is None else budget
    if think and cap > 0:
        extra["thinking_token_budget"] = cap
    return extra


def client(base_url: str, api_key: str, cloud: bool):
    from openai import AsyncOpenAI
    if cloud:          # no silent SDK retries: a failure falls back to the local model at once
        # read timeout above STALL_TIMEOUT: our own limit decides, with its note
        return AsyncOpenAI(base_url=base_url, api_key=api_key, max_retries=0, timeout=FIRST_OUTPUT_TIMEOUT + 30)
    return AsyncOpenAI(base_url=base_url, api_key=api_key)


def record_quota(valves, headers) -> None:
    """Remember the shared key's remaining quota for the picker (best effort; never fails an answer)."""
    root = os.getenv("OUTPUT_ROOT")
    if not root:
        return
    try:
        import json
        from datetime import datetime, timezone
        from memento.common import cloud_quota as cq
        from memento.query.models import workspace_id
        state = cq.from_headers(headers or {}, datetime.now(timezone.utc))
        if state:
            # one key, one budget for every copy: its cloud models, and the other copies' from the model map
            own = [workspace_id(v, valves.QUERY_MODEL_ID) for v in copy_variants(valves) if v.is_cloud]
            try:
                shown = json.loads((Path(root) / "_runtime" / "query_models.json").read_text()).get("models", [])
            except (OSError, ValueError):
                shown = []
            others = [m["id"] for m in shown if isinstance(m, dict) and m.get("cloud") and m.get("id")]
            cq.write_state(cq.state_path(root), state, list(dict.fromkeys(own + others)))
    except Exception as e:
        import logging
        logging.getLogger("memento.errors").warning("E-quota: %r", e)


async def open_answer(valves, variant, *, messages: list, think: bool, max_tokens: int | None = None,
                      local_thinking_budget: int | None = None, budget: "Budget | None" = None,
                      held: bool = False):
    """(chunks, the variant that answers, a note for the reader or ""). A cloud variant that is out of quota,
    answers with an error, or sends no output within FIRST_OUTPUT_TIMEOUT s is replaced by the local model.
    local_thinking_budget: the local model's thinking cap for this answer (else THINKING_BUDGET); cloud unchanged.
    `budget`: the response's time; thinking without answer text by its cut-off is closed and the request goes again
    without thinking (a cloud model too slow for that → the local model); a `held` answer (not shown while it is
    written) stops at the budget's end. The variant returned follows a change of model. A German question's messages
    get DE_ADDRESS_RULE here, so every attempt (retry, fallback) has it."""
    messages = texts.address_messages(messages)
    chunks, answered, note = await open_answer_once(valves, variant, messages=messages, think=think,
                                                          max_tokens=max_tokens,
                                                          local_thinking_budget=local_thinking_budget,
                                                          budget=budget, held=held)
    if budget is None or not think:
        return chunks, answered, note
    holder = Answering(answered)

    async def retry():
        return await open_answer_once(valves, holder.v, messages=messages, think=False, max_tokens=max_tokens,
                                            budget=budget, held=held, slow=True)
    return think_capped(chunks, holder, budget, retry), holder, note


async def think_capped(chunks, holder: "Answering", budget: "Budget", retry):
    """`chunks` while the model thinks until the budget's cut-off; once answer text has started, everything.
    No answer text by the cut-off: this stream is closed and `retry()` (no thinking) answers instead."""
    import logging
    it, loop, started = chunks.__aiter__(), asyncio.get_running_loop(), False
    while True:
        if started:
            try:
                ch = await it.__anext__()
            except StopAsyncIteration:
                return
            yield ch
            continue
        try:
            ch = await asyncio.wait_for(it.__anext__(), max(0.0, budget.think_end - loop.time()))
        except StopAsyncIteration:
            return
        except asyncio.TimeoutError:
            break
        if ch.choices and getattr(ch.choices[0].delta, "content", None):
            started = True
        yield ch
    close = getattr(chunks, "aclose", None)
    if close is not None:
        try:
            await close()
        except Exception:
            pass
    logging.getLogger("memento.query").info("thinking cut off at the budget (%s, %.0f s left): asked again "
                                            "without thinking", holder.v.id, budget.left())
    chunks2, answered2, note2 = await retry()
    holder.v = answered2
    if note2:
        yield note_chunk(note2)
    async for ch in chunks2:
        yield ch


async def open_answer_once(valves, variant, *, messages: list, think: bool, max_tokens: int | None = None,
                           local_thinking_budget: int | None = None, budget: "Budget | None" = None,
                           held: bool = False, slow: bool = False):
    """One attempt of open_answer (see there). `slow`: a cloud model that sends nothing in time hands over with
    SLOW_NOTE (a retry within the budget)."""
    import logging
    import openai
    from datetime import datetime, timezone
    from memento.common import cloud_quota as cq
    log = logging.getLogger("memento.query")
    local = next(v for v in copy_variants(valves) if not v.is_cloud)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + valves.ANSWER_TIMEOUT      # the whole answer, any model
    if budget is not None and held:          # nothing the reader cannot see runs past the budget
        deadline = min(deadline, budget.end)
    first_timeout = FIRST_OUTPUT_TIMEOUT if budget is None else max(1.0, min(FIRST_OUTPUT_TIMEOUT, budget.left()))

    async def call(v):
        if v.is_cloud:
            base, key, model = os.getenv(v.base_url_env, ""), os.getenv(v.key_env, ""), v.model
        else:
            base, key, model = valves.LLM_BASE_URL, request.llm_key(valves), valves.LLM_MODEL
        kwargs = dict(model=model, messages=messages if v.vision else text_only(messages), stream=True,
                      temperature=valves.TEMPERATURE, max_tokens=max_tokens or valves.MAX_TOKENS)
        if v.thinking:
            kwargs["extra_body"] = gen_extra(valves, think, None if v.is_cloud else local_thinking_budget)
        return await client(base, key, v.is_cloud).chat.completions.create(**kwargs)

    if not variant.is_cloud:
        log.info("answer variant=%s model=%s", variant.id, valves.LLM_MODEL)
        stream = await call(variant)
        return limited(stream, stream, deadline), variant, ""
    root = os.getenv("OUTPUT_ROOT")
    state = cq.read_state(cq.state_path(root)) if root else None
    if not cq.availability(state, datetime.now(timezone.utc))["available"]:
        note = texts.QUOTA_NOTE
    else:
        stream = None
        first = asyncio.get_running_loop().time() + first_timeout   # one budget: request + first output
        try:
            stream = await asyncio.wait_for(call(variant), first_timeout)
            record_quota(valves, getattr(getattr(stream, "response", None), "headers", None))
            chunks = await primed(stream, first)
            log.info("answer variant=%s model=%s", variant.id, variant.model)
            return limited(chunks, stream, deadline), variant, ""
        except openai.RateLimitError as e:
            record_quota(valves, e.response.headers)
            note = texts.QUOTA_NOTE
        except asyncio.TimeoutError as e:
            log.warning("cloud %s sent nothing in time: %r", variant.id, e)
            note = texts.SLOW_NOTE if slow or budget is not None else texts.UNREACHABLE_NOTE
        except Exception as e:
            log.warning("cloud %s failed before output: %r", variant.id, e)
            note = texts.UNREACHABLE_NOTE
        if stream is not None:
            try:
                await stream.close()
            except Exception:
                pass
    log.info("answer variant=%s fell back to local (%s)", variant.id, "quota" if note == texts.QUOTA_NOTE else "error")
    stream = await call(local)
    return limited(stream, stream, deadline), local, note


def list_thinks(think: bool, query_intent: str, is_cloud: bool, enabled: bool) -> bool:
    """Thinking for this answer: a list or count question on the local model does not think unless LIST_THINKING is on
    (without thinking it finds the same documents, several times faster). Cloud models keep theirs."""
    if query_intent == "set_query" and not is_cloud and not enabled:
        return False
    return think


def live_thinks(think: bool, *, thorough: bool, warm: bool, enabled: bool) -> bool:
    """Thinking for a live answer: only with Thorough check on, unless LIVE_THINKING is (thinking made ordinary answers
    far slower for little gain). Warm-ups keep the router's choice, and so does Thorough check: it never starts thinking
    the router turned off for a plain lookup."""
    return think and (thorough or warm or enabled)


def gap_thinks(think: bool, thorough: bool, enabled: bool) -> bool:
    """Thinking for a research-gap answer, while GAP_THINKING is on (the default). With "Thorough check" too, capped
    like any gap answer."""
    return (think or thorough) and enabled
