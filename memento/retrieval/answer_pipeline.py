"""The production answer step as a library: builds exactly the messages Library Chat sends to the answer model (source
labels, sources and question, figure images, history note), so the RQ3 evaluation answers the way production does.
tests/retrieval/test_rq3_answer_pipeline.py runs the real pipe with fake search and a recording model and checks the messages are
identical.

The live pipe imports `source_text` and `linked_text_addendum` from it (the authors' text under a table or an
equation). Context hygiene (dropping table-of-contents lines and near-duplicate chunks) is optional here and off in the
pipe."""
from __future__ import annotations

import base64
import io
import re
import time

HISTORY_NOTE = ("\nEarlier turns of this conversation are context only, not evidence: support every claim in this "
                "answer with the numbered sources below. Cite them only as [N], never by document name, page or "
                "section.")


# ── Labels, context, prompt (query_pipe step 8) ───────────────────────────────
def chunk_label(h: dict) -> str:
    """'<source>, page <n>' + ', Section: <sec>' or ', Section: <sec> › <raw heading>' — as the pipe labels sources."""
    sec, raw = h.get("section"), h.get("section_raw")
    label = f"{h['source']}, page {h['page'] + 1}"
    if sec:
        if raw and raw.lower() != sec.lower():
            label += f", Section: {sec} › {raw}"
        else:
            label += f", Section: {sec}"
    return label


# the authors' linked text the answer model reads under a table or an equation: (payload field, label)
LINKED_TEXT = {"table": ("table_context", "Authors' text about this table (not part of the table)"),
               "equation": ("equation_context", "Authors' text about this equation (not part of the formula)")}

# added to the answer's instructions only when a source carries that text, so other answers keep the prompt RQ3
# evaluated; outside the stored answers' key. It keeps the authors' statements apart from the element's own values.
LINKED_TEXT_ADDENDUM = """

Some tables and formulas among the sources carry the authors' text about them, marked as not part of the table or \
formula. Report a value or entry as the table's or formula's only if it is in the table or formula itself; give what \
the authors' text says as the authors' statement, citing the same source."""


def _linked(h: dict) -> str:
    field = LINKED_TEXT.get(h.get("type"), (None, None))[0]
    return (h.get(field) or "").strip() if field else ""


def source_text(h: dict) -> str:
    """A source as the answer model, the claim checker and the rules read it: its text, and under a table or an
    equation the authors' linked text, labelled. Hits without that field (stored RQ3 hits) read as before."""
    text, linked = h.get("text", ""), _linked(h)
    return f"{text}\n\n{LINKED_TEXT[h['type']][1]}:\n{linked}" if linked else text


def linked_text_addendum(hits: list[dict]) -> str:
    """LINKED_TEXT_ADDENDUM when a source carries the authors' text under a table or a formula, else ""."""
    return LINKED_TEXT_ADDENDUM if any(_linked(h) for h in hits) else ""


def build_context(hits: list[dict]) -> str:
    return "\n\n---\n\n".join(f"[{i + 1}] {chunk_label(h)}\n{source_text(h)}" for i, h in enumerate(hits))


def system_prompt_for(domain_prompt: str, has_history: bool) -> str:
    return domain_prompt + (HISTORY_NOTE if has_history else "")


def image_data_url(path: str) -> str | None:
    """A figure as a data URL the model can read, or None (caption only): memento/common/model_servers.image_data_url
    (SVG rasterized, image type set), then a decode check."""
    try:
        from PIL import Image
        from memento.common.model_servers import image_data_url as _url
        url = _url(path)
        with Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))) as im:
            im.verify()
        return url
    except Exception:
        return None


def image_label(n: int) -> str:
    """The text part before a source's image. The model's chat template joins text parts without a separator, so the
    label starts on its own line (else it runs into the question's last word)."""
    return f"\n\nImage for Source [{n}]:"


def build_messages(system_prompt: str, hits: list[dict], question: str, *,
                   history: list[dict] | None = None, max_images: int = 6) -> list[dict]:
    """system + history + user. The user turn is the text "Sources: <context> Question: <question>" (on separate lines),
    plus, for the first `max_images` image/chart hits with a readable file, "Image for Source [i]:" and the image, in
    hit order. `question` is the original wording (the pipe answers what was typed, not the rewritten search query)."""
    user_message = f"Sources:\n{build_context(hits)}\n\nQuestion: {question}"
    parts = []
    for i, h in enumerate(hits):
        if len(parts) // 2 >= max_images:      # figures, not parts (label + image each), as the pipe counts
            break
        if h.get("type") in ("image", "chart") and h.get("img_path"):
            url = image_data_url(h["img_path"])
            if url:
                parts.append({"type": "text", "text": image_label(i + 1)})
                parts.append({"type": "image_url", "image_url": {"url": url}})
    content: list | str = ([{"type": "text", "text": user_message}] + parts) if parts else user_message
    return ([{"role": "system", "content": system_prompt}] + list(history or [])
            + [{"role": "user", "content": content}])




# ── Generation ────────────────────────────────────────────────────────────────
def generate_answer(llm, model: str, messages: list[dict], *, temperature: float = 0.15, max_tokens: int = 16384,
                    enable_thinking: bool = True, thinking_budget: int = 12288) -> dict:
    """One non-streamed call with the pipe's parameters (incl. its THINKING_BUDGET cap). Returns {"answer",
    "thinking", "usage", "latency_s"}; the reasoning (reasoning_content / reasoning, or an inline <think> block) is
    kept apart from the answer."""
    t0 = time.perf_counter()
    resp = llm.chat.completions.create(
        model=model, messages=messages, temperature=temperature, max_tokens=max_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking},
                    **({"thinking_token_budget": thinking_budget} if enable_thinking and thinking_budget > 0 else {})},
    )
    msg = resp.choices[0].message
    answer = msg.content or ""
    thinking = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)
    m = re.match(r"\s*<think>(.*?)</think>\s*", answer, re.S)
    if m:
        thinking = (thinking or "") + m.group(1)
        answer = answer[m.end():]
    usage = getattr(resp, "usage", None)
    return {"answer": answer.strip(), "thinking": thinking,
            "usage": usage.model_dump() if hasattr(usage, "model_dump") else (usage or {}),
            "latency_s": round(time.perf_counter() - t0, 2)}


# ── Context hygiene (RQ3 option; off in production) ───────────────────────────
_TOC_LEADER = re.compile(r"(\.{4,}|…{2,}|(\s?\.){4,})\s*\d{1,4}\s*$")
_NUMBERED_HEADING_ONLY = re.compile(r"^\*{0,2}\d+(\.\d+)+\.?\s+\S.{0,120}?\*{0,2}$")


def _norm_words(t: str) -> list[str]:
    return re.sub(r"\s+", " ", re.sub(r"[*_]", "", t or "")).strip().lower().split(" ")


def _shingles(words: list[str], n: int = 5) -> set:
    return {tuple(words[i:i + n]) for i in range(max(1, len(words) - n + 1))} if words else set()


def is_toc_line(text: str) -> bool:
    t = (text or "").strip()
    return bool(_TOC_LEADER.search(t)) or bool(_NUMBERED_HEADING_ONLY.match(t))


def apply_context_hygiene(hits: list[dict], *, near_dup_jaccard: float = 0.9) -> tuple[list[dict], dict]:
    """Drops table-of-contents / list-of-figures lines and near-duplicate texts (5-word-shingle Jaccard ≥ threshold;
    the better-ranked copy is kept). Image/chart hits are never dropped as duplicates (their text is a caption).
    Returns (kept hits in order, {"dropped_toc": n, "dropped_dup": n})."""
    kept, seen, stats = [], [], {"dropped_toc": 0, "dropped_dup": 0}
    for h in hits:
        if h.get("type") not in ("image", "chart") and is_toc_line(h.get("text", "")):
            stats["dropped_toc"] += 1
            continue
        sh = _shingles(_norm_words(h.get("text", "")))
        if h.get("type") not in ("image", "chart") and sh and any(
                len(sh & o) / len(sh | o) >= near_dup_jaccard for o in seen):
            stats["dropped_dup"] += 1
            continue
        kept.append(h)
        if h.get("type") not in ("image", "chart") and sh:
            seen.append(sh)
    return kept, stats
