"""
caption_image.py — VLM fallback captioning for standalone images.

When MinerU parses a document image that has no author text (no caption, no
enriched_context), this module sends the image pixels to the configured
reasoning model and returns a generated description so the chunk is
semantically retrievable via text search.

Used by both ingestion paths (the command line and Library Upload).
"""
import base64
import io
import logging
import os
from pathlib import Path

import memento.config.settings  # noqa: F401 — triggers load_dotenv so os.getenv reads .env
from openai import OpenAI
from PIL import Image

log = logging.getLogger(__name__)

_CAPTION_PROMPT = (
    "Describe this scientific/technical image in 2-3 sentences. "
    "Focus on: what type of figure it is, what it shows, and any "
    "visible labels/scales/annotations."
)

_MAX_PX = 1024
_JPEG_QUALITY = 85
_MAX_BYTES = 1 * 1024 * 1024

_MIME_BY_SUFFIX = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".tiff": "image/tiff", ".tif": "image/tiff",
}


def _encode_image(img_path: Path) -> tuple[str, str]:
    """Return (base64_data, mime_type). Resizes to _MAX_PX on the longest
    side if the file exceeds _MAX_BYTES or is oversized."""
    img = Image.open(img_path)
    raw_size = img_path.stat().st_size
    needs_resize = raw_size > _MAX_BYTES or max(img.size) > _MAX_PX
    if needs_resize:
        ratio = _MAX_PX / max(img.size)
        if ratio < 1:
            img = img.resize(
                (int(img.width * ratio), int(img.height * ratio)),
                Image.LANCZOS,
            )
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=_JPEG_QUALITY)
        return base64.b64encode(buf.getvalue()).decode(), "image/jpeg"
    mime = _MIME_BY_SUFFIX.get(img_path.suffix.lower(), "image/jpeg")
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode(), mime


def _get_client_and_model() -> tuple[OpenAI, str]:
    if os.getenv("REASONING_PROVIDER", "local") == "cloud":
        return (
            OpenAI(
                base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                api_key=os.getenv("OPENAI_API_KEY", ""),
            ),
            os.getenv("CLOUD_REASONING_MODEL", "qwen3.8-27b"),
        )
    reasoning_url = os.getenv("REASONING_URL", "http://localhost:8023")
    return (
        OpenAI(base_url=f"{reasoning_url}/v1",
               api_key=os.getenv("REASONING_API_KEY", os.getenv("OPENAI_API_KEY", "none"))),
        os.getenv("REASONING_MODEL_NAME", "/models/reasoning"),
    )


def caption_image(
    img_path: Path,
    base_url: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> str | None:
    """Send the image to the configured reasoning model and return a caption; None on any error, so ingestion continues.
    base_url/model override the environment defaults (the pipe's Valves); api_key: the copy's own key (None:
    OPENAI_API_KEY)."""
    if not img_path.exists():
        return None
    try:
        b64, mime = _encode_image(img_path)
        if base_url and model:
            client = OpenAI(base_url=base_url,
                            api_key=api_key if api_key is not None else os.getenv("OPENAI_API_KEY", ""))
        else:
            client, model = _get_client_and_model()
        resp = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    {"type": "text", "text": _CAPTION_PROMPT},
                ],
            }],
            temperature=0.1,
            max_tokens=512,
            extra_body={
                "enable_thinking": False,
                "reasoning_effort": "low",
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        content = resp.choices[0].message.content
        if not content or not content.strip():
            log.warning(f"[caption] model returned empty content for {img_path.name}")
            return None
        log.info(f"[caption] generated caption for {img_path.name} ({len(content)} chars)")
        return content.strip()
    except Exception as e:
        log.warning(f"[caption] failed for {img_path.name}: {e}")
        return None


def cached_caption(paper_dir: Path, rel_img_path: str, generate: bool = True,
                   base_url: str | None = None, model: str | None = None, api_key: str | None = None) -> str | None:
    """Caption for a figure without author text, saved per document in <paper_dir>/vlm_captions.json. LLM captions are
    nondeterministic, so a re-embed reuses the saved one; both ingestion paths use this. Keyed by the item's img_path
    ("images/abc.jpg"). Locked and atomically replaced: runs for different backends can share an output folder (office/)
    and must neither corrupt nor drop entries. base_url/model: the chat pipe's Valves; the command line uses the
    environment defaults."""
    import fcntl
    import json
    cache_path = paper_dir / "vlm_captions.json"
    with open(paper_dir / ".vlm_captions.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        cache = json.load(open(cache_path)) if cache_path.exists() else {}
        if rel_img_path in cache:
            return cache[rel_img_path]
        if not generate:
            return None
        generated = caption_image(paper_dir / rel_img_path, base_url=base_url, model=model, api_key=api_key)
        if generated:
            cache[rel_img_path] = generated
            tmp_path = cache_path.with_name(f"{cache_path.name}.{os.getpid()}.tmp")
            with open(tmp_path, "w") as f:
                json.dump(cache, f, ensure_ascii=False, indent=1)
            os.replace(tmp_path, cache_path)
        return generated
