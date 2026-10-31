#!/usr/bin/env python3
"""Research-gap tagging for document chunks: at ingestion time (or as a backfill), each text chunk is classified as
gap-relevant or not. A gap-relevant chunk is one the authors use to state a limitation, future-work direction, untested
condition or uncertainty. The tag and a one-line signal are written into the Qdrant payload, so the query-time gap flow
retrieves them with a metadata filter (no embedding or reranking).

This stage identifies, it does not extract: the reasoning model structures the answer at query time. It favours recall:
a false positive is cheap (the reasoning model ignores it), a false negative hides that gap from the reader."""

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed


from openai import OpenAI
from pydantic import BaseModel


class GapTag(BaseModel):
    index: int
    gap_type: str = "unknown"
    signal: str = ""


class GapClassificationResponse(BaseModel):
    tagged: list[GapTag] = []

CLASSIFY_PROMPT = """\
You are classifying passages from a scientific paper. For each passage,
decide whether it states or implies a research gap, limitation, untested
condition, future work direction, or uncertainty.

Rules:
- Only flag passages where the AUTHORS state or clearly imply the gap.
- Do NOT flag passages that merely report results or methods.
- Bias toward inclusion — if uncertain, flag it.
- For each flagged passage, provide a one-line signal (what gap it contains).

Passages (numbered by index):
{chunks}

Respond with ONLY valid JSON:
{{"tagged": [{{"index": 0, "gap_type": "limitation|future_work|untested_condition|uncertainty", "signal": "one-line description"}}]}}

If no passages contain research gaps, respond: {{"tagged": []}}"""


def _estimate_tokens(text: str) -> int:
    return len(text) // 4


def batch_chunks(chunks: list[dict], max_tokens: int = 28000) -> list[list[tuple[int, dict]]]:
    """Split (global_index, chunk) pairs into batches that fit context."""
    batches: list[list[tuple[int, dict]]] = []
    current: list[tuple[int, dict]] = []
    current_tokens = 0

    for idx, chunk in enumerate(chunks):
        tok = _estimate_tokens(chunk["text"]) + 20
        if current and current_tokens + tok > max_tokens:
            batches.append(current)
            current = []
            current_tokens = 0
        current.append((idx, chunk))
        current_tokens += tok

    if current:
        batches.append(current)
    return batches


def _format_batch(batch: list[tuple[int, dict]], all_chunks: list[dict] | None = None) -> str:
    """Format a batch for the classification prompt.

    When `all_chunks` is provided, prepends a one-sentence preceding
    context for each chunk so the LLM can catch cross-boundary implicit
    gaps (e.g. a result that's only a limitation relative to a goal
    stated in the previous chunk).
    """
    lines = []
    for idx, c in batch:
        section = f" [{c.get('section', '')}]" if c.get("section") else ""
        # Preceding context: last sentence of the previous chunk
        preceding = ""
        if all_chunks and idx > 0:
            prev_text = (all_chunks[idx - 1].get("text") or "").strip()
            if prev_text:
                sentences = prev_text.rsplit(".", 2)
                last_sentence = (sentences[-2] + ".").strip() if len(sentences) > 1 else prev_text[-200:]
                preceding = f"\n  [preceding context: ...{last_sentence}]"
        lines.append(f"[{idx}]{section}{preceding}\n{c['text']}")
    return "\n\n".join(lines)


def classify_batch(
    client: OpenAI,
    model: str,
    batch: list[tuple[int, dict]],
    all_chunks: list[dict] | None = None,
) -> list[dict]:
    """Classify one batch.  Returns list of {index, gap_type, signal}."""
    prompt = CLASSIFY_PROMPT.format(chunks=_format_batch(batch, all_chunks))

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
            temperature=0.0,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        raw = (resp.choices[0].message.content or "").strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
        parsed = GapClassificationResponse.model_validate(json.loads(raw))
        return [{"index": t.index, "gap_type": t.gap_type, "signal": t.signal} for t in parsed.tagged]
    except Exception as e:
        print(f"    WARNING: classify_batch failed: {e}")
        return []


# Chunk types sent to the gap tagger by both ingestion paths (run_pipeline.py, the chat ingestion pipe): prose and
# bullet lists. Not equations: their chunk text is the LaTeX, and the sentences around them are tagged as their own text
# chunks.
GAP_TAG_TYPES = {"text", "list"}


def tag_paper_chunks(
    chunks: list[dict],
    reasoning_url: str | None = None,
    model: str | None = None,
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    workers: int | None = None,
) -> dict[int, dict]:
    """Tag all text chunks for one paper.

    Parameters
    ----------
    chunks : list[dict]
        Each dict must have at least "text"; "section" is optional.
    reasoning_url : str, optional
        The answer model's server (REASONING_URL), with or without /v1.
    model : str, optional
        Model name served by the endpoint.
    base_url : str, optional
        Full OpenAI-compatible base URL (preferred over reasoning_url).
    api_key : str, optional
        API key for the endpoint.

    Returns
    -------
    dict[int, dict]
        Maps chunk index → {"gap_type": str, "gap_signal": str} for
        every chunk flagged as gap-relevant.  Chunks not in the dict
        are gap_relevant=False.
    """
    if base_url:
        # a given key is used as it is, an empty one too: never another copy's key
        client = OpenAI(base_url=base_url, api_key=api_key if api_key is not None else os.getenv("OPENAI_API_KEY", ""))
        model = model or os.getenv("CLOUD_REASONING_MODEL", "qwen3.8-27b")
    elif reasoning_url:
        # the answer model of config/.env (REASONING_URL, REASONING_MODEL_NAME, REASONING_API_KEY)
        client = OpenAI(base_url=f"{reasoning_url.rstrip('/').removesuffix('/v1')}/v1",
                        api_key=api_key or os.getenv("REASONING_API_KEY", "none"))
        model = model or os.getenv("REASONING_MODEL_NAME", "qwen-flash-next")
    else:
        client = OpenAI(
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            api_key=os.getenv("OPENAI_API_KEY", ""),
        )
        model = model or os.getenv("CLOUD_REASONING_MODEL", "qwen3.8-27b")
    text_chunks = [(i, c) for i, c in enumerate(chunks) if (c.get("text") or "").strip()]

    if not text_chunks:
        return {}

    from memento.config.settings import GAP_TAG_WORKERS
    if workers is None:
        workers = GAP_TAG_WORKERS

    batches = batch_chunks(
        [c for _, c in text_chunks],
        max_tokens=28000,
    )

    tagged_map: dict[int, dict] = {}

    def _process_batch(batch):
        return classify_batch(client, model, batch, all_chunks=chunks)

    n_workers = max(1, min(workers, len(batches)))
    if n_workers == 1:
        all_results = [_process_batch(b) for b in batches]
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = [pool.submit(_process_batch, b) for b in batches]
            all_results = [f.result() for f in as_completed(futures)]

    for results in all_results:
        for r in results:
            local_idx = r.get("index")
            if local_idx is not None and 0 <= local_idx < len(text_chunks):
                orig_idx = text_chunks[local_idx][0]
                tagged_map[orig_idx] = {
                    "gap_type": r.get("gap_type", "unknown"),
                    "gap_signal": r.get("signal", ""),
                }

    return tagged_map
