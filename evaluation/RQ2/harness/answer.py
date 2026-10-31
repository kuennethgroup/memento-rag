"""Answer generation for every arm: the query pipe's ranked-retrieval prompt on the local model
(the same generator, prompt template and temperature for all arms)."""
from __future__ import annotations

import base64
import re
import time
from pathlib import Path

TEMPERATURE, MAX_TOKENS, ENABLE_THINKING, MAX_IMAGES_TO_LLM = 0.15, 16384, True, 6
_THINK = re.compile(r"^.*?</think>", re.S)


def _label(h):
    sec, raw = h.get("section"), h.get("section_raw")
    label = f"{h['source']}, page {h['page'] + 1}"
    if sec:
        label += f", Section: {sec} › {raw}" if raw and raw.lower() != sec.lower() else f", Section: {sec}"
    return label


def build_messages(question, out):
    from memento.config.settings import DOMAIN
    context = "\n\n---\n\n".join(f"[{i + 1}] {_label(h)}\n{h['text']}" for i, h in enumerate(out.context))
    user = f"Sources:\n{context}" + (f"\n\n{out.block}" if out.block else "") + f"\n\nQuestion: {question}"
    images = []
    for i, h in enumerate(out.context):
        if len(images) >= 2 * MAX_IMAGES_TO_LLM:
            break
        if h.get("type") in ("image", "chart") and h.get("img_path") and Path(h["img_path"]).is_file():
            data = base64.b64encode(Path(h["img_path"]).read_bytes()).decode()
            images += [{"type": "text", "text": f"Image for Source [{i + 1}]:"},
                       {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}}]
    content = [{"type": "text", "text": user}] + images if images else user
    return [{"role": "system", "content": DOMAIN["system_prompt"]}, {"role": "user", "content": content}]


def generate_answer(llm, model, question, out):
    t = time.time()
    r = llm.chat.completions.create(model=model, messages=build_messages(question, out), temperature=TEMPERATURE,
                                    max_tokens=MAX_TOKENS,
                                    extra_body={"chat_template_kwargs": {"enable_thinking": ENABLE_THINKING}})
    msg = r.choices[0].message
    text = _THINK.sub("", msg.content or "").strip()
    reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None) or ""
    return {"answer": text, "reasoning_chars": len(reasoning), "latency_s": round(time.time() - t, 2)}


def answers_for(table, outs, gen):
    done = {}
    for row in table:
        if row["key"] not in done:
            done[row["key"]] = gen(outs[row["key"]])
    return done
