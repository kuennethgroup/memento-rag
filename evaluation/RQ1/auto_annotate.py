#!/usr/bin/env python3
"""
evaluation/RQ1/auto_annotate.py — LLM-automated, independently-verified ground
truth generation for RQ1 (Step 2 of the evaluation pipeline).

Two deliberately decoupled phases:

  1. GENERATE — for each paper, sample its chunks (text/table/image/chart,
     image/chart sampled WITH actual pixels attached) and ask the cloud LLM
     to write one realistic query per proposal category (factual,
     methodological, visual, comparative), each grounded in a specific
     seed chunk.

  2. LABEL — for each generated query, retrieve via the PRODUCTION path
     (memento/retrieval/search.py — the exact module query_pipe.py imports) against
     the FULL corpus (no paper restriction), then have a SEPARATE LLM call
     judge which of the retrieved chunks actually answer the query. This is
     the independent-verification step: relevance is never just "the chunk
     the generator claimed" — it's re-derived from what retrieval actually
     surfaces, judged by a prompt that has no knowledge of the seed chunk.

Relevance key is (source, page) — parser-independent (chunk_id is not
portable between the VLM and OCR parses; `type` is recorded for per-target-
type bucketing only, never as a match requirement). Output schema matches
evaluation/ground_truth_input_papers_fixed.json.

Usage:
    python evaluation/RQ1/auto_annotate.py --papers SRC1 SRC2 --out evaluation/pilot.json
    python evaluation/RQ1/auto_annotate.py --n-papers 3 --out evaluation/pilot.json
"""
import argparse
import base64
import threading
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from memento.config.cli_domain import apply_domain_arg, domain_banner  # noqa: E402
apply_domain_arg()
from memento.config.settings import DOMAIN, COLLECTION, QDRANT_URL, EMBEDDER_URL, RERANKER_URL

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchAny

from openai import OpenAI, RateLimitError, APIStatusError, APITimeoutError, APIConnectionError

from memento.retrieval.search import search as production_search

MAX_LLM_RETRIES = 6
RETRY_BASE_DELAY = 8  # seconds; exponential backoff: 8, 16, 32, 64, 128, 256

# Optional audit trail: every generation/judge call's exact request+response,
# for reproducibility and for handing the same prompts to a different model/
# server without re-deriving them from Qdrant. Set via --log-calls; off by
# default (each entry embeds image bytes as base64, gets large fast).
CALL_LOG_PATH: Path | None = None
_LOG_LOCK = threading.Lock()  # parallel pooling workers append to the same log


def _log_call(kind: str, meta: dict, messages: list, raw_response: str) -> None:
    if CALL_LOG_PATH is None:
        return
    logged_messages = []
    for m in messages:
        content = m["content"]
        if isinstance(content, str):
            logged_messages.append(m)
            continue
        parts = []
        for part in content:
            if part.get("type") == "image_url":
                url = part["image_url"]["url"]
                parts.append({"type": "image_url", "image_b64_len": len(url)})
            else:
                parts.append(part)
        logged_messages.append({"role": m["role"], "content": parts})
    entry = {"kind": kind, **meta, "messages": logged_messages, "raw_response": raw_response}
    with _LOG_LOCK, open(CALL_LOG_PATH, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _chat_with_retry(llm: OpenAI, **kwargs):
    """Cloud LLM rate limits 429 hard under this run's call volume (1
    generation + 1 judge call per query, both multimodal). Retry with
    exponential backoff rather than silently dropping the query — a dropped
    query costs a paper slot against --max-papers for nothing."""
    for attempt in range(MAX_LLM_RETRIES):
        try:
            return llm.chat.completions.create(**kwargs)
        except (APITimeoutError, APIConnectionError) as e:
            # Transient network/timeout — retry rather than abort the whole run.
            if attempt == MAX_LLM_RETRIES - 1:
                raise
            delay = RETRY_BASE_DELAY * (2 ** attempt)
            print(f"    [{type(e).__name__}] retry {attempt + 1}/{MAX_LLM_RETRIES} in {delay}s...")
            time.sleep(delay)
        except (RateLimitError, APIStatusError) as e:
            status = getattr(e, "status_code", None)
            # 429 (rate limit) and 5xx (server overload/500-503) are transient under
            # this run's load on a congested endpoint — retry; other 4xx are real.
            if status is not None and status != 429 and status < 500:
                raise
            if attempt == MAX_LLM_RETRIES - 1:
                raise
            delay = RETRY_BASE_DELAY * (2 ** attempt)
            print(f"    [rate limit] retry {attempt + 1}/{MAX_LLM_RETRIES} in {delay}s...")
            time.sleep(delay)

QUERY_INSTRUCTION = DOMAIN["instruction"]

REASONING_PROVIDER = os.getenv("REASONING_PROVIDER", "local")
CLOUD_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
CLOUD_API_KEY = os.getenv("CLOUD_API_KEY", os.getenv("OPENAI_API_KEY", ""))
CLOUD_MODEL = os.getenv("CLOUD_REASONING_MODEL", "qwen3.5-397b-a17b")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "glm-5.3-flash")

CATEGORIES = ["factual", "methodological", "visual", "comparative"]

MAX_CHUNKS_PER_TYPE = 8
MAX_IMAGES_IN_PROMPT = 4
RETRIEVE_K = 80
RERANK_TOP = 20
RERANK_MIN_SCORE = 0.03
IMAGE_RERANK_RATIO = 0.2
HYBRID_ENABLED = os.getenv("HYBRID_ENABLED", "false").strip().lower() in ("1", "true", "yes", "on")
HYBRID_PREFETCH_K = int(os.getenv("HYBRID_PREFETCH_K", "80"))


def get_llm() -> tuple[OpenAI, str]:
    """Generator LLM — local by default."""
    if REASONING_PROVIDER == "cloud":
        return OpenAI(base_url=CLOUD_BASE_URL, api_key=CLOUD_API_KEY), CLOUD_MODEL
    reasoning_url = os.getenv("REASONING_URL", "http://localhost:8023")
    api_key = os.getenv("REASONING_API_KEY", "none")
    return OpenAI(base_url=f"{reasoning_url}/v1", api_key=api_key), os.getenv("REASONING_MODEL_NAME", "/models/reasoning")


def get_judge_llm() -> tuple[OpenAI, str]:
    """Judge LLM — cloud by default (different model family to avoid same-model bias)."""
    # Short timeout, no client-side retries: the cloud endpoint intermittently stalls
    # (esp. image prompts); _chat_with_retry is the single retry loop, with backoff.
    return OpenAI(base_url=CLOUD_BASE_URL, api_key=CLOUD_API_KEY, timeout=60.0, max_retries=0), JUDGE_MODEL


_INVALID_ESCAPE_RE = re.compile(r'\\(?!["\\/bfnrtu])')


def _repair_backslashes(text: str) -> str:
    """LLMs asked for queries about scientific content routinely emit raw
    LaTeX (\\Delta, \\circ, ...) inside JSON string values — a lone
    backslash not followed by a valid JSON escape char. Double any such
    backslash so the string parses; this changes the in-JSON representation
    (\\Delta -> \\\\Delta, i.e. a literal backslash in the decoded string)
    but never changes already-valid escapes."""
    return _INVALID_ESCAPE_RE.sub(r"\\\\", text)


def _extract_json(text: str):
    """LLMs wrap JSON in prose/code-fences sometimes — pull the first
    top-level {...} or [...] block out defensively, repairing common
    invalid-backslash escapes (raw LaTeX) before giving up."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()

    candidates = [text]
    for open_c, close_c in (("{", "}"), ("[", "]")):
        start = text.find(open_c)
        end = text.rfind(close_c)
        if start != -1 and end != -1 and end > start:
            candidates.append(text[start:end + 1])

    for cand in candidates:
        for variant in (cand, _repair_backslashes(cand)):
            try:
                return json.loads(variant)
            except json.JSONDecodeError:
                continue
    raise ValueError(f"Could not parse JSON from LLM output: {text[:300]!r}")


# ── Sampling a paper's chunks ────────────────────────────────────────────────
def sample_chunks_for_paper(client: QdrantClient, source: str) -> dict[str, list[dict]]:
    by_type: dict[str, list[dict]] = {}
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION,
            scroll_filter=Filter(must=[FieldCondition(key="source", match=MatchAny(any=[source]))]),
            with_payload=True,
            with_vectors=False,
            limit=500,
            offset=offset,
        )
        for p in points:
            pl = p.payload or {}
            t = pl.get("type", "text")
            if not (pl.get("text") or "").strip():
                continue
            by_type.setdefault(t, []).append({
                "page": pl.get("page", 0),
                "type": t,
                "section": pl.get("section"),
                "text": pl.get("text", ""),
                "img_path": pl.get("img_path"),
            })
        if offset is None:
            break
    for t in by_type:
        by_type[t].sort(key=lambda c: c["page"])
        by_type[t] = _spread_sample(by_type[t], MAX_CHUNKS_PER_TYPE)
    return by_type


def _spread_sample(items: list, n: int) -> list:
    """Evenly-spaced sample across the FULL page-sorted list, not a prefix
    slice. A plain `items[:n]` (the original implementation) systematically
    seeds every generated query from a paper's first N chunks of each type —
    almost always its opening pages — biasing the whole ground truth toward
    document openings and away from methods/results/discussion content. 
    Deterministic (no RNG needed):
    picks n indices spread across [0, len(items)) inclusive of both ends."""
    if len(items) <= n:
        return items
    if n <= 1:
        return items[:1]
    step = (len(items) - 1) / (n - 1)
    indices = sorted({round(i * step) for i in range(n)})
    return [items[i] for i in indices]


# ── Phase 1: query generation ────────────────────────────────────────────────
# Generation is keyed on `target_type` (text/table/image/chart/equation), not
# `category` — RQ1's per-modality breakdown (the claim VLM specifically wins
# on table/figure/equation queries) needs a guaranteed minimum N per TYPE
# bucket; `category` (factual/methodological/visual/comparative) is just a
# descriptive label the LLM assigns, kept for analysis, not for sizing.
GENERATION_SYSTEM_PROMPT = """\
You write ONE evaluation query for a scientific-document retrieval system, \
grounded in EXACT excerpts from one paper, all of the same content type \
(shown below). Figures/charts are attached as actual images.
Write ONE realistic query a real researcher might type, whose answer is \
found in ONE specific excerpt you were shown. The query must NOT name the \
paper or quote the excerpt verbatim — it should read like an independent \
search query on the topic.
If the excerpts are figures/charts, the query must ask about something that \
can ONLY be answered by looking at the image's visual content (trends, \
morphology, axis values, ratios) — not just paraphrasable from a caption.
Also assign the query a category — whichever best fits:
- factual: asks for a specific fact, value, or definition.
- methodological: asks about a procedure, method, or experimental setup.
- visual: asks about visual content only answerable by looking at an image.
- comparative: asks to compare two or more quantities/findings/samples.
Output ONLY one JSON object: {"category": "...", "query": "...", \
"seed_index": <int>, "rationale": "..."}. `seed_index` is the [N] number of \
the excerpt the answer comes from. Keep rationale under 25 words. If you \
cannot write a genuine query from these excerpts, output {"query": null}."""


def build_generation_messages(source: str, chunks: list[dict], target_type: str):
    index = []
    content_parts = [{
        "type": "text",
        "text": f"Paper: {source}\nContent type: {target_type}\n\nExcerpts:\n",
    }]
    n_images = 0
    for c in chunks:
        i = len(index)
        index.append({"page": c["page"], "type": c["type"], "text": c["text"]})
        label = f"[{i}] (page {c['page'] + 1}, {c['type']}{', ' + c['section'] if c.get('section') else ''})"
        content_parts.append({"type": "text", "text": f"\n{label}\n{c['text'][:1200]}\n"})
        if c["type"] in ("image", "chart") and c.get("img_path") and n_images < MAX_IMAGES_IN_PROMPT:
            try:
                data = base64.b64encode(Path(c["img_path"]).read_bytes()).decode()
                content_parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}})
                n_images += 1
            except Exception:
                pass
    messages = [
        {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
        {"role": "user", "content": content_parts},
    ]
    return messages, index


def generate_query_for_type(llm: OpenAI, model: str, source: str,
                             chunks: list[dict], target_type: str) -> dict | None:
    if not chunks:
        return None
    messages, index = build_generation_messages(source, chunks, target_type)
    resp = _chat_with_retry(
        llm, model=model, messages=messages, temperature=0.3, max_tokens=2048,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    raw = resp.choices[0].message.content or "{}"
    _log_call("generation", {"model": model, "source": source, "target_type": target_type}, messages, raw)
    it = _extract_json(raw)
    if isinstance(it, list):
        it = it[0] if it else {}
    if not it.get("query"):
        return None
    cat = it.get("category")
    seed_i = it.get("seed_index")
    if cat not in CATEGORIES or not isinstance(seed_i, int) or not (0 <= seed_i < len(index)):
        return None
    seed = index[seed_i]
    return {
        "category": cat,
        "query": it["query"].strip(),
        "target_paper": source,
        "target_type": seed["type"],
        "seed_page": seed["page"],
        "rationale": it.get("rationale", ""),
    }


# ── Phase 2: independent relevance labeling ──────────────────────────────────
JUDGE_SYSTEM_PROMPT = """\
You judge search-result relevance for a scientific-document retrieval \
evaluation. You will see a query and a numbered list of retrieved excerpts \
(figures/charts attached as actual images where applicable). Mark ONLY the \
excerpts that genuinely contain the answer to the query — not excerpts that \
are merely topically related. Be strict: an excerpt about the same general \
subject but lacking the specific fact/value/method/visual content asked for \
is NOT relevant.
Output ONLY a JSON array of objects for relevant excerpts:
[{"index": <int>, "rationale": "..."}, ...]
Keep each rationale under 15 words. Output [] if none of the excerpts answer the query."""


def build_judge_messages(query: str, hits: list[dict]):
    content_parts = [{"type": "text", "text": f"Query: {query}\n\nRetrieved excerpts:\n"}]
    n_images = 0
    for i, h in enumerate(hits):
        label = f"\n[{i}] {h['source']} (page {h['page'] + 1}, {h['type']}{', ' + h['section'] if h.get('section') else ''})"
        content_parts.append({"type": "text", "text": f"{label}\n{(h['text'] or '')[:1200]}\n"})
        if h["type"] in ("image", "chart") and h.get("img_path") and n_images < MAX_IMAGES_IN_PROMPT:
            try:
                data = base64.b64encode(Path(h["img_path"]).read_bytes()).decode()
                content_parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}})
                n_images += 1
            except Exception:
                pass
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": content_parts},
    ]


def judge_hits(llm: OpenAI, model: str, query: str, hits: list[dict],
               source_collection: str, annotator: str = "llm-auto") -> list[dict]:
    """Independent-judge step, factored out of label_query so callers with
    their own already-retrieved `hits` (the TREC-style pooling pass in
    pool_relevance_all.py, which retrieves from other collections) can reuse
    the exact same judge prompt/parsing without re-implementing it."""
    if not hits:
        return []
    messages = build_judge_messages(query, hits)
    kwargs = dict(model=model, messages=messages, temperature=0.0, max_tokens=2048)
    if "localhost" in (llm.base_url.host or ""):
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    resp = _chat_with_retry(llm, **kwargs)
    raw = resp.choices[0].message.content or "[]"
    _log_call("judge", {"model": model, "query": query, "source_collection": source_collection}, messages, raw)
    judged = _extract_json(raw)
    if not isinstance(judged, list):
        judged = [judged]

    relevant = []
    for j in judged:
        # Expected shape is {"index": N, "rationale": "..."} but the judge
        # model occasionally shortcuts to a bare index (int) instead of the
        # full object, or the response fails to parse into anything
        # index-shaped at all (str/float/etc) — skip those rather than
        # crashing the whole run on one malformed judgment.
        if isinstance(j, dict):
            i = j.get("index")
            rationale = j.get("rationale", "")
        elif isinstance(j, int):
            i = j
            rationale = ""
        else:
            continue
        if not isinstance(i, int) or not (0 <= i < len(hits)):
            continue
        h = hits[i]
        relevant.append({
            "paper_id": h["source"],
            "page": h["page"],
            "type": h["type"],
            "rationale": rationale,
            "annotated_from_collection": source_collection,
            "annotator": annotator,
        })
    return relevant


def label_query(llm: OpenAI, model: str, qdrant_client: QdrantClient, query: str) -> tuple[list[dict], list[dict]]:
    """Returns (relevant_chunks, all_retrieved_hits) — the latter kept for
    pilot inspection (did the seed chunk even get retrieved?)."""
    hits, _ = production_search(
        qdrant_client, COLLECTION, query,
        embedder_url=EMBEDDER_URL, query_instruction=QUERY_INSTRUCTION,
        retrieve_k=RETRIEVE_K, hybrid=HYBRID_ENABLED, hybrid_prefetch_k=HYBRID_PREFETCH_K,
        rerank_enabled=True, reranker_url=RERANKER_URL, rerank_top=RERANK_TOP,
        rerank_min_score=RERANK_MIN_SCORE, image_rerank_ratio=IMAGE_RERANK_RATIO,
    )
    if not hits:
        return [], []
    relevant = judge_hits(llm, model, query, hits, COLLECTION, annotator="llm-auto")
    return relevant, hits


TARGET_TYPES = ["equation", "chart", "table", "image", "text"]  # sparsest first


def list_all_sources(client: QdrantClient) -> list[str]:
    offset = None
    seen = set()
    while True:
        pts, offset = client.scroll(COLLECTION, limit=2000, offset=offset, with_payload=["source"])
        for p in pts:
            s = (p.payload or {}).get("source")
            if s:
                seen.add(s)
        if offset is None:
            break
    return sorted(seen)


def main():
    # A failed rerank must abort, never degrade silently to vector order.
    os.environ["MEMENTO_STRICT_RERANK"] = "1"
    ap = argparse.ArgumentParser()
    ap.add_argument('--domain', required=True, help='Corpus domain profile (config/domains/<name>.yaml), e.g. ceramics | polymer-engineering')
    ap.add_argument("--papers", nargs="*", help="Explicit source ids to use (skips bucket-filling)")
    ap.add_argument("--per-bucket-target", type=int, default=25,
                     help="Stop once each target_type bucket has at least this many queries")
    ap.add_argument("--max-papers", type=int, default=80,
                     help="Safety cap on papers processed while filling buckets")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(Path(__file__).parent / "ground_truth_auto.json"))
    ap.add_argument("--resume", action="store_true",
                     help="Load --out if it exists and continue filling buckets from there "
                          "(skips papers already processed) instead of starting over")
    ap.add_argument("--log-calls", default=None,
                     help="Append every generation/judge call's exact request+response "
                          "(JSONL) to this path — audit trail / lets another server "
                          "reproduce the same prompts without re-deriving them.")
    args = ap.parse_args()
    domain_banner()

    global CALL_LOG_PATH
    if args.log_calls:
        CALL_LOG_PATH = Path(args.log_calls)
        CALL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"Logging every LLM call to {CALL_LOG_PATH}")

    client = QdrantClient(url=QDRANT_URL, timeout=30)
    llm, model = get_llm()
    judge_llm, judge_model = get_judge_llm()
    print(f"Generator: {model} ({'cloud' if REASONING_PROVIDER == 'cloud' else 'local'})")
    print(f"Judge:     {judge_model} (cloud)")

    if args.papers:
        sources = args.papers
    else:
        import random
        sources = list_all_sources(client)
        random.Random(args.seed).shuffle(sources)

    bucket_counts = {t: 0 for t in TARGET_TYPES}
    queries_out = []
    qid = 0
    papers_used = []

    if args.resume and Path(args.out).exists():
        prior = json.loads(Path(args.out).read_text())
        queries_out = prior.get("queries", [])
        papers_used = prior.get("papers", [])
        qid = (max((q["query_id"] for q in queries_out), default=-1)) + 1
        for t in TARGET_TYPES:
            bucket_counts[t] = sum(1 for q in queries_out if q["target_type"] == t)
        print(f"Resumed from {args.out}: {len(queries_out)} queries, "
              f"{len(papers_used)} papers, bucket_counts={bucket_counts}")

    already_done = set(papers_used)

    for source in sources:
        if not args.papers and all(bucket_counts[t] >= args.per_bucket_target for t in TARGET_TYPES):
            break
        if not args.papers and len(papers_used) >= args.max_papers:
            print(f"Hit --max-papers cap ({args.max_papers}) before filling all buckets: {bucket_counts}")
            break
        if source in already_done:
            continue

        print(f"\n=== {source} ===  bucket_counts so far: {bucket_counts}")
        by_type = sample_chunks_for_paper(client, source)
        print(f"  available types: {[(t, len(c)) for t, c in by_type.items()]}")
        papers_used.append(source)
        paper_had_query = False

        for target_type in TARGET_TYPES:
            if not args.papers and bucket_counts[target_type] >= args.per_bucket_target:
                continue
            chunks = by_type.get(target_type)
            if not chunks:
                continue
            try:
                g = generate_query_for_type(llm, model, source, chunks, target_type)
            except Exception as e:
                print(f"  [{target_type}] generation failed: {e}")
                continue
            if not g:
                continue
            paper_had_query = True
            print(f"  [{g['category']}/{g['target_type']}] {g['query']!r}  (seed page {g['seed_page']})")
            try:
                relevant, _hits = label_query(judge_llm, judge_model, client, g["query"])
            except Exception as e:
                print(f"    labeling failed: {e}")
                continue
            seed_found = any(
                r["paper_id"] == source and r["page"] == g["seed_page"] for r in relevant
            )
            print(f"    -> {len(relevant)} judged relevant; seed chunk recovered: {seed_found}")
            queries_out.append({
                "query_id": qid,
                "query": g["query"],
                "query_en": g["query"],
                "category": g["category"],
                "target_paper": source,
                "target_type": g["target_type"],
                "generation_rationale": g["rationale"],
                "seed_page": g["seed_page"],
                "seed_chunk_recovered": seed_found,
                "relevant_chunks": relevant,
            })
            bucket_counts[g["target_type"]] += 1
            qid += 1

        # checkpoint after every paper — a long run (40-80 papers x up to 5
        # type-queries each, 2 LLM calls per query) shouldn't lose progress
        # to a crash/timeout partway through.
        out = {
            "version": 1,
            "description": f"LLM auto-annotated ground truth ({len(papers_used)} papers so far, "
                            f"auto_annotate.py) — relevance independently re-derived via "
                            f"memento/retrieval/search.py + judge LLM, NOT just the generator's seed claim. "
                            f"Sized to >= {args.per_bucket_target} queries per target_type bucket.",
            "generator_model": model,
            "judge_model": judge_model,
            "annotator": "llm-auto",
            "papers": papers_used,
            "bucket_counts": bucket_counts,
            "queries": queries_out,
        }
        Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
        if not paper_had_query:
            print("  (no usable query generated for this paper)")

    print(f"\nWrote {len(queries_out)} queries from {len(papers_used)} papers to {args.out}")
    print(f"Final bucket counts: {bucket_counts}")
    n_recovered = sum(1 for q in queries_out if q["seed_chunk_recovered"])
    print(f"Seed-chunk recovery rate: {n_recovered}/{len(queries_out)} "
          f"({100 * n_recovered / max(1, len(queries_out)):.0f}%)")
    short = [t for t, n in bucket_counts.items() if n < args.per_bucket_target]
    if short and not args.papers:
        print(f"WARNING: buckets below target after exhausting candidates/cap: {short}")


if __name__ == "__main__":
    main()
