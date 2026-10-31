"""Thorough check around one answer: its claims checked against the passages (and the library's claim rules), a revision
where a claim fails, within the time the answer has left.
"""
import asyncio
import os
import threading
import time

from memento.query import generation, request, texts


# a warm-up's check runs one model call at a time (users keep the other slots), so it may take this long (s)
WARMUP_VERIFY_TIMEOUT = float(os.getenv("WARMUP_VERIFY_TIMEOUT", "1800"))


async def thorough_check(valves, *, answer: str, hits: list, question: str, check_prompt: str, think: bool,
                         client, dom: dict, retrieve_filter, api_key: str, status, background: bool = False,
                         seconds: float | None = None,
                         collection: str | None = None) -> tuple:
    """"Thorough check": the RQ3 loop (split into statements, judge each against the sources it cites, revise; the
    second round with thinking) on a held answer. Shared by normal answers, summaries and research-gap answers.
    Returns (final text, result, error, late)."""
    import logging
    check_prompt = texts.address_prompt(check_prompt)      # its revisions are written without generation.open_answer
    # neutral wording: every answer is source-grounded, so the status must not make normal answers sound unchecked
    await status("Thorough check: refining the answer...")
    result, error, late = None, None, False
    # Stop pressed or VERIFY_TIMEOUT reached: the check ends before its next model call
    check_stop = threading.Event()
    # the background warm-up checks one call at a time, so it gets a longer limit; `seconds`: at most the response's
    # time left (generation.check_seconds)
    check_deadline = time.monotonic() + (WARMUP_VERIFY_TIMEOUT if background else
                                         (seconds if seconds is not None else valves.VERIFY_TIMEOUT))
    slots = threading.Semaphore(1) if background else None
    try:
        from openai import OpenAI as SyncOpenAI
        from memento.agentic_validation.strategies import ArmContext, run_strategy
        from memento.agentic_validation.verifiers import LLMJudgeVerifier, VerdictCache, VLMFigureVerifier
        gen_llm = generation.GuardedLLM(SyncOpenAI(base_url=valves.LLM_BASE_URL, api_key=api_key), check_stop,
                              check_deadline, slots)
        ver_llm = (generation.GuardedLLM(SyncOpenAI(base_url=valves.VERIFIER_URL, api_key=api_key), check_stop,
                               check_deadline, slots)
                   if valves.VERIFIER_URL else gen_llm)
        ver_model = valves.VERIFIER_MODEL or valves.LLM_MODEL

        def _retrieve(q: str) -> list:
            from memento.retrieval.search import search as _search
            found, _ = _search(
                client, collection or valves.COLLECTION, q, embedder_url=valves.EMBEDDER_URL,
                query_filter=retrieve_filter,
                query_instruction=dom.get("instruction") or request.FALLBACK_INSTRUCTION,
                retrieve_k=valves.RETRIEVE_K, hybrid=valves.HYBRID_ENABLED,
                hybrid_prefetch_k=valves.HYBRID_PREFETCH_K, rerank_enabled=valves.RERANK_ENABLED,
                reranker_url=valves.RERANKER_URL, rerank_top=valves.RERANK_TOP,
                rerank_min_score=valves.RERANK_MIN_SCORE,
                image_rerank_ratio=valves.IMAGE_RERANK_RATIO,
                rerank_instruction=request.rerank_instruction(dom))
            return found

        # revision round 2 with thinking (the configuration evaluated in RQ3)
        ctx = ArmContext(gen_llm, valves.LLM_MODEL, check_prompt, question, list(hits),
                         LLMJudgeVerifier(ver_llm, ver_model), VLMFigureVerifier(ver_llm, ver_model), VerdictCache(),
                         retrieve_fn=_retrieve, enable_thinking=think, revise_thinking="escalate",
                         thinking_budget=generation.thorough_thinking_budget(valves), claim_rules=request.domain_claim_rules(dom))
        try:
            result = await asyncio.to_thread(run_strategy, valves.VALIDATION_STRATEGY.strip().upper(), ctx,
                                             p1_answer=answer, max_rounds=valves.VALIDATION_MAX_ROUNDS)
        except asyncio.CancelledError:
            check_stop.set()                # Stop: the check's thread ends after its current call
            raise
    except generation.CheckStopped as e:
        error, late = e, True
        logging.getLogger("memento.errors").warning("E-verify-timeout: %r", e)
    except Exception as e:
        error = e
        logging.getLogger("memento.errors").warning("E-verify: %r", e)
    from memento.agentic_validation.strategies import finalize_for_display
    final = f"{answer}\n\n{texts.CHECK_TIMEOUT_NOTE}" if late else finalize_for_display(answer, result, error)
    final = generation.checked_text(final, result, error, late)
    if result is not None and error is None:
        # A: statements flagged by the first check and sent for revision; still failing after the last round (each
        # statement counted once, not once per round)
        if result.arm == "C":
            changed, left = len(result.removed), 0
        else:
            changed = len(result.trace[0]["failing"]) if result.rounds and result.trace else 0
            left = len(result.trace[-1]["failing"]) if result.rounds and result.trace else 0
        # the outcome is for the log only; readers just get the (checked) answer
        logging.getLogger("memento.verify").info(
            "strategy=%s claims=%d revised=%d unconfirmed=%d rounds=%d unjudged=%d calls=%d thorough=True "
            "latency=%.1fs steps=%s", result.arm, len(result.claims), changed, left, result.rounds,
            result.unjudged, result.llm_calls, result.latency_s,
            " ".join(f"{k}:{v}s" for k, v in getattr(result, "timings", [])))
    return final, result, error, late


def verify_applies(enabled: bool, *, thorough: bool, query_intent: str) -> bool:
    """"Thorough check" runs the statement check on normal answers, summaries and research-gap answers."""
    return enabled and thorough and query_intent in ("ranked_retrieval", "summary", "gap_query")
