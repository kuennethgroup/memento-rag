#!/usr/bin/env python3
"""FastAPI server for a vision-capable reasoning model (vLLM): the model is loaded once at startup (AsyncLLMEngine) and
stays resident.

Start (inside the container, CMD in docker/Dockerfile.reasoning): python /app/reasoning_api.py

Endpoints: GET /health (liveness, GPU memory, model status); GET /v1/models; POST /v1/chat/completions
(OpenAI-compatible, with thinking mode).

Thinking mode: {"reasoning_effort": "high"} in extra_body turns it on; the response then carries "reasoning_content"
beside "content", as vLLM's --reasoning-parser qwen3 does."""

import base64
import os
import re
import sys
import time
import uuid
import logging
from contextlib import asynccontextmanager
from io import BytesIO
from typing import Any, Dict, List, Optional, Union

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, Field

# ── Config from env ────────────────────────────────────────────────────────────
MODEL_PATH    = os.getenv("MODEL_PATH",     "/models/reasoning")
GPU_DEVICE    = int(os.getenv("GPU_DEVICE", "0"))
PORT          = int(os.getenv("REASONING_PORT", "8023"))
MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", "32768"))
MAX_NUM_SEQS  = int(os.getenv("MAX_NUM_SEQS", "4"))
# vLLM's own default (0.92) sizes the KV cache against the *whole* card, so the
# engine refuses to start on a GPU that already hosts the embedder or reranker.
# Lower this to co-locate; leave it alone when the model has the card to itself.
GPU_MEM_UTIL  = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.92"))
LOG_LEVEL     = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)-8s] reasoning_api — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("reasoning_api")

# ── Model state ────────────────────────────────────────────────────────────────
_state: dict = {}


# ── Thinking parser ────────────────────────────────────────────────────────────
_THINK_RE = re.compile(r"<think>(.*?)</think>(.*)", re.DOTALL)

def split_thinking(text: str) -> tuple[Optional[str], str]:
    """Separate <think>…</think> from the final answer."""
    m = _THINK_RE.match(text.strip())
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return None, text.strip()


# ── Multimodal content handling ─────────────────────────────────────────────────
def _decode_data_url(url: str) -> Image.Image:
    """`data:image/...;base64,<data>` → PIL Image."""
    b64 = url.split(",", 1)[1] if "," in url else url
    return Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")


def _prepare_multimodal_messages(messages: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], List[Image.Image]]:
    """
    Convert OpenAI-style message content (str, or a list of
    {"type": "text"/"image_url", ...} parts) into the form Qwen-VL's chat
    template expects (content parts typed "text"/"image", images supplied
    separately via multi_modal_data — the template only inserts a
    placeholder). Returns (template_messages, images_in_order).
    """
    template_messages: List[Dict[str, Any]] = []
    images: List[Image.Image] = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str) or content is None:
            template_messages.append({"role": msg["role"], "content": content or ""})
            continue
        parts: List[Dict[str, Any]] = []
        for part in content:
            ptype = part.get("type")
            if ptype == "text":
                parts.append({"type": "text", "text": part.get("text", "")})
            elif ptype == "image_url":
                url = (part.get("image_url") or {}).get("url", "")
                try:
                    images.append(_decode_data_url(url))
                    parts.append({"type": "image"})
                except Exception:
                    logger.warning("Failed to decode image_url content part — skipping image")
        template_messages.append({"role": msg["role"], "content": parts})
    return template_messages, images


# ── Lifespan — load / unload ───────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load vLLM engine and tokenizer at startup; release at shutdown."""
    from vllm import AsyncLLMEngine
    from vllm.engine.arg_utils import AsyncEngineArgs
    from transformers import AutoProcessor

    # AutoProcessor, not only AutoTokenizer: vision models need the paired image processor so apply_chat_template
    # expands the image placeholder; it works for text-only templates too.
    logger.info(f"Loading processor from {MODEL_PATH} …")
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer = processor.tokenizer

    logger.info(
        f"Loading {MODEL_PATH} via AsyncLLMEngine  "
        f"(GPU {GPU_DEVICE}, ctx {MAX_MODEL_LEN}) …"
    )
    t0 = time.perf_counter()

    engine_args = AsyncEngineArgs(
        model                = MODEL_PATH,
        max_model_len        = MAX_MODEL_LEN,
        tensor_parallel_size = 1,
        dtype                = "bfloat16",     # compute dtype
        trust_remote_code    = True,
        max_num_seqs         = MAX_NUM_SEQS,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    load_time = time.perf_counter() - t0
    alloc_gb  = torch.cuda.memory_allocated(GPU_DEVICE) / 1e9
    logger.info(
        f"Engine ready in {load_time:.1f}s  |  "
        f"{alloc_gb:.1f} GB allocated on GPU {GPU_DEVICE}"
    )

    _state["engine"]    = engine
    _state["tokenizer"] = tokenizer
    _state["processor"] = processor
    _state["load_time"] = load_time

    yield  # ← serve

    logger.info("Shutdown — releasing engine")
    del _state["engine"]
    del _state["tokenizer"]
    torch.cuda.empty_cache()


app = FastAPI(
    title       = "Reasoning API",
    version     = "1.0",
    lifespan    = lifespan,
)


# ── Schemas ────────────────────────────────────────────────────────────────────
# `content` accepts either a plain string or OpenAI's multimodal content-part
# list (e.g. [{"type": "text", "text": ...}, {"type": "image_url", "image_url":
# {"url": "data:image/jpeg;base64,..."}}]) — the shape memento/graph/build/llm_enrich.py
# sends for image/chart chunk extraction.
class ChatMessage(BaseModel):
    role:    str
    content: Union[str, List[Dict[str, Any]]]


class ChatCompletionRequest(BaseModel):
    model:            str              = "reasoning"
    messages:         List[ChatMessage]
    temperature:      float            = 0.15
    top_p:            float            = 0.9
    top_k:            int              = 20
    max_tokens:       int              = 4096
    # Qwen thinking mode — passed via extra_body in the openai client:
    #   {"reasoning_effort": "high"}   → enable thinking
    #   {"reasoning_effort": "low"}    → skip thinking (default)
    reasoning_effort: Optional[str]    = "low"

    class Config:
        extra = "allow"   # absorb any unknown extra_body fields gracefully


class AssistantMessage(BaseModel):
    role:              str = "assistant"
    content:           str
    reasoning_content: Optional[str] = None


class ChatChoice(BaseModel):
    index:         int = 0
    message:       AssistantMessage
    finish_reason: Optional[str] = None


class UsageInfo(BaseModel):
    prompt_tokens:     int
    completion_tokens: int
    total_tokens:      int


class ChatCompletionResponse(BaseModel):
    id:      str
    object:  str = "chat.completion"
    model:   str
    choices: List[ChatChoice]
    usage:   UsageInfo


# ── /health ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    loaded = "engine" in _state
    mem    = {}
    if torch.cuda.is_available():
        alloc    = torch.cuda.memory_allocated(GPU_DEVICE) / 1e9
        reserved = torch.cuda.memory_reserved(GPU_DEVICE) / 1e9
        mem = {"allocated_gb": round(alloc, 2), "reserved_gb": round(reserved, 2)}
    return {
        "status":       "ok",
        "model_loaded": loaded,
        "model":        MODEL_PATH,
        "load_time_s":  _state.get("load_time"),
        "gpu":          GPU_DEVICE,
        "max_model_len": MAX_MODEL_LEN,
        "memory":        mem,
    }


# ── /v1/models ─────────────────────────────────────────────────────────────────
@app.get("/v1/models")
def list_models():
    """OpenAI-compatible model list — required by the openai Python client."""
    return {
        "object": "list",
        "data": [{
            "id":       "reasoning",
            "object":   "model",
            "owned_by": "local",
        }],
    }


# ── /v1/chat/completions ───────────────────────────────────────────────────────
@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(req: ChatCompletionRequest):
    if "engine" not in _state:
        raise HTTPException(503, detail="Model not yet loaded — try again shortly")

    from vllm.sampling_params import SamplingParams

    engine    = _state["engine"]
    processor = _state["processor"]

    # Determine thinking mode
    enable_thinking = req.reasoning_effort in ("high", "auto")

    # Build prompt using Qwen3's chat template. Multimodal content parts
    # (image_url) get rewritten to the template's "image" placeholder form;
    # the actual pixels are supplied to vLLM separately via multi_modal_data.
    messages = [m.model_dump() for m in req.messages]
    template_messages, images = _prepare_multimodal_messages(messages)
    try:
        prompt = processor.apply_chat_template(
            template_messages,
            tokenize            = False,
            add_generation_prompt = True,
            enable_thinking     = enable_thinking,
        )
    except TypeError:
        # Fallback: processor doesn't support enable_thinking yet
        prompt = processor.apply_chat_template(
            template_messages,
            tokenize            = False,
            add_generation_prompt = True,
        )

    sampling_params = SamplingParams(
        temperature = max(req.temperature, 1e-5),   # vLLM rejects exactly 0
        top_p       = req.top_p,
        top_k       = req.top_k,
        max_tokens  = req.max_tokens,
    )

    request_id = f"chatcmpl-{uuid.uuid4().hex}"

    engine_input = (
        {"prompt": prompt, "multi_modal_data": {"image": images if len(images) > 1 else images[0]}}
        if images else prompt
    )

    # Collect final output (non-streaming)
    final_output = None
    async for output in engine.generate(engine_input, sampling_params, request_id=request_id):
        final_output = output

    if final_output is None or not final_output.outputs:
        raise HTTPException(500, detail="vLLM returned no output")

    generated_text  = final_output.outputs[0].text
    finish_reason   = final_output.outputs[0].finish_reason

    # Split <think>…</think> from answer
    thinking, answer = split_thinking(generated_text) if enable_thinking \
                       else (None, generated_text.strip())

    prompt_tokens     = len(final_output.prompt_token_ids)
    completion_tokens = len(final_output.outputs[0].token_ids)

    logger.debug(
        f"{request_id}  thinking={enable_thinking}  "
        f"prompt={prompt_tokens} + completion={completion_tokens} tokens"
    )

    return ChatCompletionResponse(
        id      = request_id,
        model   = "reasoning",
        choices = [ChatChoice(
            message       = AssistantMessage(
                content           = answer,
                reasoning_content = thinking,
            ),
            finish_reason = finish_reason,
        )],
        usage   = UsageInfo(
            prompt_tokens     = prompt_tokens,
            completion_tokens = completion_tokens,
            total_tokens      = prompt_tokens + completion_tokens,
        ),
    )


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "reasoning_api:app",
        host      = "0.0.0.0",
        port      = PORT,
        log_level = LOG_LEVEL.lower(),
    )
