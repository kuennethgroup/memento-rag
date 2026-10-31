#!/usr/bin/env python3
"""
file_server.py — minimal static file server for cited PDFs and figures

Lets OpenWebUI citation links open the actual source content (PDF page,
figure) instead of just showing the cited excerpt text. Independent of the
embedder/reranker/mineru lifecycle.

Start:
    INPUT_DIR=/data/input OUTPUT_DIR=/data/output python file_server.py      (127.0.0.1:8024; FILES_HOST, FILES_PORT)

Endpoints:
    GET /health                  liveness check
    GET /view/<stem>/<page>, /content/<stem>/<page>  a cited page (Document Chat files: their owner only)
    GET /figures/<path>           figure images of the parsed output (data/output/), nothing else
    GET /quota, /library, /models  runtime JSON for loader.js (cloud quota, document picker, model colours)
"""

import asyncio
import base64
import hashlib
import hmac
import html
import json
import os
import re
import secrets
import time
import urllib.request
from pathlib import Path
from urllib.parse import quote

import fitz  # PyMuPDF
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

INPUT_DIR  = Path(os.getenv("INPUT_DIR",  "/data/input"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/data/output"))
PORT       = int(os.getenv("FILES_PORT", "8024"))
HOST       = os.getenv("FILES_HOST", "127.0.0.1")
OPENWEBUI_URL = os.getenv("OPENWEBUI_URL", "http://127.0.0.1:8086").rstrip("/")

PAGE_RENDER_DPI = 150

HIGHLIGHT_COLOR = (1, 0.65, 0)  # orange outline
HIGHLIGHT_FILL  = (1, 1, 0)     # yellow fill

app = FastAPI(title="Memento File Server")

# Behind the remote-access proxy the server is reached under a path prefix (/files); the proxy says so in
# X-Forwarded-Prefix, and every link a page builds starts with it. Direct access (no header) is unchanged.
_PREFIX_OK = re.compile(r"(/[A-Za-z0-9._-]+)*")


def _prefix(request: Request) -> str:
    p = request.headers.get("x-forwarded-prefix", "").rstrip("/")
    return p if _PREFIX_OK.fullmatch(p) else ""


# Signed download links: Open WebUI's side panel is a sandboxed frame that sends no login cookie, so the proxy would
# refuse a download clicked there. A /view page (reached with the login) signs its two download links; the proxy passes
# a request carrying "t=" without the login (docker/Caddyfile) and the download checks it here. The key is new at every
# start (links are made per page view and live 4 h), so nothing is shared with the proxy.
_LINK_KEY = secrets.token_bytes(32)
_LINK_TTL = 4 * 3600


def _sign(path: str, now: float | None = None) -> str:
    exp = int((time.time() if now is None else now) + _LINK_TTL)
    mac = hmac.new(_LINK_KEY, f"{path}|{exp}".encode(), hashlib.sha256).hexdigest()[:32]
    return f"{exp}.{mac}"


def _check_signature(path: str, t: str | None) -> None:
    """No token: the proxy's login check applied. A token: it must be this path's and not expired, else 403."""
    if t is None:
        return
    exp, _, mac = t.partition(".")
    good = exp.isdigit() and hmac.compare_digest(
        mac, hmac.new(_LINK_KEY, f"{path}|{exp}".encode(), hashlib.sha256).hexdigest()[:32])
    if not good or int(exp) < time.time():
        raise HTTPException(403, "Download link expired: open the page again")


# Document Chat files (INPUT_ROOT/_temp/<chat id>/<source>.*, OUTPUT_ROOT/_temp/<source>/) open only for the account
# that attached them, recorded in the chat's folder (memento/document_chat/intake.py, same name). Whose login a request
# carries, Open WebUI says: the request's own cookie (or API key) goes to its session endpoint, the answer is kept a
# minute. Library documents are shared and need no lookup.
OWNER_FILE = ".owner"
_WHO_TTL = 60
_who: dict[str, tuple[float, str | None]] = {}


def _session_user(cookie: str, auth: str) -> str | None:
    """The id of the approved Open WebUI account this login belongs to, or None."""
    headers = {k: v for k, v in (("Cookie", cookie), ("Authorization", auth)) if v}
    try:
        req = urllib.request.Request(f"{OPENWEBUI_URL}/api/v1/auths/", headers=headers)
        with urllib.request.urlopen(req, timeout=5) as r:
            user = json.load(r)
    except Exception:
        return None
    return user.get("id") if user.get("role") in ("user", "admin") else None


async def _caller(request: Request) -> str | None:
    cookie, auth = request.headers.get("cookie", ""), request.headers.get("authorization", "")
    if not (cookie or auth):
        return None
    key = hashlib.sha256(f"{cookie}\n{auth}".encode()).hexdigest()
    hit = _who.get(key)
    if hit and hit[0] > time.monotonic():
        return hit[1]
    uid = await asyncio.to_thread(_session_user, cookie, auth)
    if len(_who) > 1000:
        _who.clear()
    _who[key] = (time.monotonic() + _WHO_TTL, uid)
    return uid


def _chat_dir(stem: str) -> Path | None:
    """The Document Chat folder holding this source, or None (a library document)."""
    for p in (INPUT_DIR / "_temp").glob(f"*/{glob_escape(stem)}.*"):
        if p.stem == stem:
            return p.parent
    return None


async def _require_owner(request: Request, stem: str) -> None:
    """A Document Chat file: its owner only, anyone else gets 404 (as for a missing file). A library document: no
    check."""
    if not stem or "/" in stem or stem in (".", ".."):
        return
    d = _chat_dir(stem)
    if d is None and not (OUTPUT_DIR / "_temp" / stem).is_dir():
        return
    try:
        owner = (d / OWNER_FILE).read_text().strip() if d is not None else ""
    except OSError:
        owner = ""
    if not owner or owner != await _caller(request):
        raise HTTPException(404, "Not found")


def _draw_highlights(page: fitz.Page, bboxes: list[str]) -> None:
    """Draw highlight rectangles for `bboxes` onto `page` before rendering.

    Each bbox is "x0,y0,x1,y1" in MinerU's 0-1000-per-axis page-normalized
    coordinates (bbox = coord * 1000 / page_dim, see
    mineru/backend/vlm/vlm_middle_json_mkcontent.py), independently scaled
    back to this page's point dimensions."""
    pw, ph = page.rect.width, page.rect.height
    for raw in bboxes:
        try:
            x0, y0, x1, y1 = (float(v) for v in raw.split(","))
        except ValueError:
            continue
        rect = fitz.Rect(x0 / 1000 * pw, y0 / 1000 * ph, x1 / 1000 * pw, y1 / 1000 * ph)
        page.draw_rect(
            rect, color=HIGHLIGHT_COLOR, fill=HIGHLIGHT_FILL,
            fill_opacity=0.25, width=1.5, overlay=True,
        )


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/quota")
async def quota():
    """The shared cloud quota as the Query pipe last saw it (counts and times only), for the model picker. Missing or
    unreadable file → {} (the picker then shows nothing)."""
    try:
        data = json.loads((OUTPUT_DIR / "_runtime" / "cloud_quota.json").read_text())
    except (OSError, ValueError):
        data = {}
    return JSONResponse(data, headers={"Cache-Control": "no-store"})


@app.get("/models")
async def models():
    """The Library Chat models' colours ({"models": [{"id", "name", "cloud", "accent", "accent_dark"}]}), written by the
    Query pipe from config/query_models.yaml, for the chat box and the model dropdown. Missing or unreadable file → {}
    (loader.js then tells cloud models by their names and uses its default colours)."""
    try:
        data = json.loads((OUTPUT_DIR / "_runtime" / "query_models.json").read_text())
    except (OSError, ValueError):
        data = {}
    return JSONResponse(data, headers={"Cache-Control": "no-store"})


_COLLECTION_RE = re.compile(r"[a-z0-9-]{1,80}")
# the default library (MEMENTO_DEFAULT_COLLECTION, as memento/library/documents.LEGACY_COLLECTION; this image has no
# memento package): served from library.json until its own file exists; not set: no such fallback
LEGACY_COLLECTION = os.getenv("MEMENTO_DEFAULT_COLLECTION", "").strip()


@app.get("/library")
async def library(collection: str | None = None):
    """The library's documents as the Query pipe last listed them ({"updated", "documents": [{"source", "title"}]}), for
    the start-screen document picker. Without `collection`: the single-library file; missing or unreadable → {} (the
    picker then says the library is empty). With `collection`: that library's own list,
    _runtime/library/<collection>.json; a name that is not a plain collection name, or a library without a list: 404."""
    if collection is None:
        path = OUTPUT_DIR / "_runtime" / "library.json"
    elif not _COLLECTION_RE.fullmatch(collection):
        raise HTTPException(404, "no such library")
    else:
        path = OUTPUT_DIR / "_runtime" / "library" / f"{collection}.json"
        if not path.is_file() and collection == LEGACY_COLLECTION:
            path = OUTPUT_DIR / "_runtime" / "library.json"
        if not path.is_file():
            raise HTTPException(404, "no such library")
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        data = {}
    return JSONResponse(data, headers={"Cache-Control": "no-store"})


# Draft research proposals (memento/research_gaps/proposal.py): the Query pipe writes OUTPUT_ROOT/_runtime/reports/<24 hex>.pdf
# and links /report/<token>.pdf in the answer; the random token is the only way to it (behind the login).
_REPORT_RE = re.compile(r"[0-9a-f]{24}\.pdf")


@app.get("/report/{name}")
async def report_pdf(name: str):
    if not _REPORT_RE.fullmatch(name):
        raise HTTPException(404, "Not found")
    path = OUTPUT_DIR / "_runtime" / "reports" / name
    if not path.is_file():
        raise HTTPException(404, "This draft is no longer available: ask for it again")
    return FileResponse(path, media_type="application/pdf", filename="research_proposal_draft.pdf")


# Paper cards: an outside paper cited by a research-gap report or proposal, shown in Open WebUI's side panel like a
# library page. The Query pipe writes OUTPUT_ROOT/_runtime/papers/<24 hex>.json (title, year, journal, DOI, the services
# that found it, per question the verdict, the checker's reason and the abstract sentences the reranker found most
# related) and cites /paper/<token>; the random token is the only way to it (behind the login).
_PAPER_RE = re.compile(r"[0-9a-f]{24}")
_SAFE_LINK = re.compile(r"https://(?:doi\.org/|www\.semanticscholar\.org/paper/[0-9a-f]{40}$|openalex\.org/W\d+$)\S*")
_PAPER_CSS = (
    ".card{padding:18px 20px 22px}.card h1{margin:0 0 6px;font-size:18px;line-height:1.35}"
    ".meta{margin:0 0 12px;color:var(--muted);font-size:13.5px}"
    ".links{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:0 0 16px}"
    ".links .btn{display:inline-flex;align-items:center;gap:6px;padding:6px 12px;border-radius:8px;"
    "border:1px solid var(--accent);color:var(--accent);text-decoration:none;font-weight:600;font-size:13.5px}"
    ".links .btn.primary{background:var(--accent);color:var(--on-accent)}.links .found{color:var(--muted);font-size:13px}"
    ".links .found a{color:var(--accent)}"
    "h2{margin:16px 0 8px;font-size:14px;letter-spacing:.01em;color:var(--muted);text-transform:uppercase}"
    ".q{display:block;width:100%;text-align:left;margin:0 0 8px;padding:9px 12px;border-radius:9px;cursor:pointer;"
    "border:1px solid var(--line);background:transparent;color:var(--ink);font:inherit}"
    ".q[aria-pressed='true']{border-color:var(--accent);background:var(--accent-soft)}"
    ".q .lab{font-weight:650}.q .v{color:var(--accent);font-weight:600}.q .st{display:block;margin-top:2px}"
    ".q .why{display:block;margin-top:4px;color:var(--muted);font-size:13.5px}"
    ".abs{margin:0;line-height:1.7}.s.mark{background:var(--mark);border-radius:3px;box-shadow:0 0 0 2px var(--mark)}"
    ".note{margin:6px 0 10px;color:var(--muted);font-size:12.5px}"
)
# selecting a question marks its sentences (data-marks on the question, data-i on each sentence)
_PAPER_JS = ("<script>document.addEventListener('click',e=>{const q=e.target.closest('.q');if(!q)return;"
             "document.querySelectorAll('.q').forEach(x=>x.setAttribute('aria-pressed',x===q?'true':'false'));"
             "const m=(q.dataset.marks||'').split(',').filter(Boolean);"
             "document.querySelectorAll('.s').forEach(s=>s.classList.toggle('mark',m.includes(s.dataset.i)));"
             "const n=document.getElementById('none');if(n)n.hidden=m.length>0;});</script>")


def _paper_gone() -> str:
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,"
        f"initial-scale=1'><title>Paper</title><style>{_VIEW_CSS}</style>{_THEME_JS}</head><body>"
        f"<main class='desk'><div class='sheet gone'>{_ICON_GONE}"
        "<h2>This paper card is no longer available</h2>"
        "<p>Ask for the report again to see the paper and its abstract.</p></div></main></body></html>")


def _paper_page(card: dict) -> str:
    e = lambda v: html.escape(str(v or ""), quote=True)         # noqa: E731
    link = lambda u: u if isinstance(u, str) and _SAFE_LINK.fullmatch(u) else ""   # noqa: E731
    doi = str(card.get("doi") or "")
    open_url = link(card.get("url")) or link(f"https://doi.org/{doi}" if doi.startswith("10.") else "")
    meta = " · ".join(e(x) for x in (card.get("year"), card.get("journal")) if x)
    found = ", ".join((f"<a href='{e(link(sv.get('url')))}' target='_blank' rel='noopener'>{e(sv.get('name'))}</a>"
                       if link(sv.get("url")) else e(sv.get("name"))) for sv in card.get("services") or [])
    qs = card.get("questions") or []
    first = [int(i) for i in (qs[0].get("marks") or [])] if qs else []
    buttons = "".join(
        f"<button type='button' class='q' aria-pressed='{'true' if k == 0 else 'false'}' "
        f"data-marks='{','.join(str(int(i)) for i in q.get('marks') or [])}'>"
        f"<span class='lab'>{e(q.get('label'))}</span> · <span class='v'>{e(q.get('verdict'))}</span>"
        f"<span class='st'>{e(q.get('statement'))}</span><span class='why'>{e(q.get('reason'))}</span></button>"
        for k, q in enumerate(qs))
    sentences = " ".join(f"<span class='s{' mark' if i in first else ''}' data-i='{i}'>{e(t)}</span>"
                         for i, t in enumerate(card.get("sentences") or []))
    none = ("<p class='note' id='none'" + ("" if not first else " hidden") + ">No sentence stands out for this "
            "question.</p>") if qs else ""
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,"
        f"initial-scale=1'><title>{e(card.get('title'))}</title><style>{_VIEW_CSS}{_PAPER_CSS}</style>{_THEME_JS}"
        "</head><body><header class='bar'><span class='kind'>Paper</span>"
        f"<span class='loc'>{meta or 'Published paper'}</span></header>"
        f"<main class='desk'><div class='sheet text card'><h1>{e(card.get('title'))}</h1>"
        + (f"<p class='meta'>DOI {e(doi)}</p>" if doi else "")
        + "<div class='links'>"
        + (f"<a class='btn primary' href='{e(open_url)}' target='_blank' rel='noopener'>Open the paper</a>"
           if open_url else "")
        + (f"<span class='found'>Found in: {found}</span>" if found else "") + "</div>"
        + (f"<h2>For the questions that cite it</h2>{buttons}" if qs else "")
        + ("<h2>What the check read: the abstract and passages of the full text, not shown here</h2>"
           if card.get("full_text") else "<h2>What the check read: the abstract</h2>")
        + "<p class='note'>Highlighted: the sentences most related to the selected question (selected by the reranker, "
        "not by the check itself).</p>" + none
        + f"<p class='abs'>{sentences}</p></div></main>{_PAPER_JS}</body></html>")


@app.get("/paper/{token}", response_class=HTMLResponse)
async def paper_card(token: str):
    if not _PAPER_RE.fullmatch(token):
        raise HTTPException(404, "Not found")
    path = OUTPUT_DIR / "_runtime" / "papers" / f"{token}.json"
    try:
        card = json.loads(path.read_text())
    except (OSError, ValueError):
        return HTMLResponse(_paper_gone(), status_code=404)
    return HTMLResponse(_paper_page(card))


@app.get("/download-parsed/{stem}")
async def download_parsed(request: Request, stem: str, t: str | None = None):
    """Serve the original content_list.json + images as a ZIP."""
    _check_signature(f"/download-parsed/{stem}", t)
    if t is None:                     # a signed link was issued by a page view that passed the owner check
        await _require_owner(request, stem)
    import io
    import zipfile

    cl_path, backend_subdir = _find_content_list(stem)
    if cl_path is None:
        raise HTTPException(404, "No parsed content found")
    backend_dir = cl_path.parent

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(cl_path, f"{stem}/content_list.json")
        md = backend_dir / f"{stem}.md"
        if md.is_file():
            zf.write(md, f"{stem}/{stem}.md")
        images_dir = backend_dir / "images"
        if images_dir.is_dir():
            for img in sorted(images_dir.iterdir()):
                if img.is_file():
                    zf.write(img, f"{stem}/images/{img.name}")
    buf.seek(0)
    return Response(
        content=buf.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{stem}_parsed.zip"'},
    )


def _find_content_list(stem: str) -> tuple[Path, str] | tuple[None, None]:
    """Find content_list.json across all backend subdirectories.

    Returns (path, backend_subdir) so callers can build correct image URLs.
    Prefers enriched over plain, checks vlm first (best quality)."""
    for base in _stem_dirs(stem):
        for subdir in ("vlm", "auto", "hybrid_auto", "office"):
            enriched = base / subdir / f"{stem}_content_list_enriched.json"
            if enriched.is_file():
                return enriched, subdir
            cl = base / subdir / f"{stem}_content_list.json"
            if cl.is_file():
                return cl, subdir
    return None, None


def _stem_dirs(stem: str) -> list[Path]:
    """Parsed-output folders for a document: OUTPUT_ROOT/<corpus>/<stem>/ (current layout) or the legacy
    OUTPUT_ROOT/<stem>/."""
    if not stem or "/" in stem or stem in (".", ".."):
        return []
    out = [d for d in OUTPUT_DIR.glob(f"*/{glob_escape(stem)}") if d.is_dir()]
    legacy = OUTPUT_DIR / stem
    return out + ([legacy] if legacy.is_dir() else [])


def glob_escape(s: str) -> str:
    import glob as _g
    return _g.escape(s)


CONTEXT_WINDOW = 5  # entries before/after the match to show


@app.get("/content/{stem}/{page}", response_class=HTMLResponse)
async def content_viewer(request: Request, stem: str, page: int, highlight: str | None = None):
    """Render content_list.json as HTML with optional text-match highlighting.

    Used for non-PDF sources (docx, pptx, xlsx) where PyMuPDF page rendering
    is not available. When highlight text is given, shows only a focused window
    around the matching entry (±CONTEXT_WINDOW entries) instead of the full
    document. Images are inlined from the figures endpoint."""
    import json as _json

    await _require_owner(request, stem)
    pre = _prefix(request)
    cl_path, backend_subdir = _find_content_list(stem)
    if cl_path is None:
        raise HTTPException(404, "No parsed content found")

    entries = _json.loads(cl_path.read_text())
    page_entries = [e for e in entries if e.get("page_idx", 0) == page]
    if not page_entries:
        page_entries = entries

    highlight_lower = highlight.lower() if highlight else None

    # Find the matching entry index and focus the window around it
    match_idx = None
    if highlight_lower:
        snippet = highlight_lower[:60]
        for i, e in enumerate(page_entries):
            if e.get("type", "text") != "image" and snippet in e.get("text", "").lower():
                match_idx = i
                break

    if match_idx is not None:
        start = max(0, match_idx - CONTEXT_WINDOW)
        end = min(len(page_entries), match_idx + CONTEXT_WINDOW + 1)
        visible = page_entries[start:end]
        has_before = start > 0
        has_after = end < len(page_entries)
    else:
        visible = page_entries
        has_before = False
        has_after = False

    blocks = []
    if has_before:
        blocks.append("<p style='color:var(--muted);font-size:0.85em;text-align:center'>⋯ earlier content ⋯</p>")

    for e in visible:
        etype = e.get("type", "text")

        if etype == "image":
            img_path = e.get("img_path", "")
            caption = e.get("image_caption", "")
            if isinstance(caption, list):
                caption = " ".join(str(c) for c in caption)
            img_url = f"{pre}/figures/" + quote(str((cl_path.parent / img_path).relative_to(OUTPUT_DIR)))
            blocks.append(
                f"<figure><img src='{img_url}' style='max-width:100%'>"
                f"<figcaption>{html.escape(caption)}</figcaption></figure>"
            )
        else:
            text = e.get("text", "")
            level = e.get("text_level")
            is_match = (highlight_lower and highlight_lower[:60] in text.lower()) if text else False
            hl_style = "background:var(--hl-bg);border-left:4px solid var(--hl-border);padding:8px;margin:4px 0;" if is_match else ""

            if level is not None and (isinstance(level, int) or "title" in str(level).lower()):
                tag = f"h{min(level, 4)}" if isinstance(level, int) and 1 <= level <= 6 else "h3"
                blocks.append(f"<{tag} style='{hl_style}'>{html.escape(text)}</{tag}>")
            elif text.startswith("**") and text.endswith("**"):
                blocks.append(f"<h4 style='{hl_style}'>{html.escape(text.strip('*'))}</h4>")
            elif text.startswith("*") and text.endswith("*"):
                blocks.append(f"<p style='font-style:italic;{hl_style}'>{html.escape(text.strip('*'))}</p>")
            else:
                blocks.append(f"<p style='{hl_style}'>{html.escape(text)}</p>")

    if has_after:
        blocks.append("<p style='color:var(--muted);font-size:0.85em;text-align:center'>⋯ more content ⋯</p>")

    content_html = "\n".join(blocks)

    return (
        "<!DOCTYPE html><html><head><style>"
        ":root{color-scheme:light dark;--bg:#fff;--fg:#222;--hl-bg:#fff3b0;--hl-border:#f0a000;"
        "--fig-border:#ddd;--cap-color:#555;--muted:#888;--toolbar-border:#ddd}"
        "@media(prefers-color-scheme:dark){:root:not([data-theme='light']){"
        "--bg:#1a1a1a;--fg:#e0e0e0;--hl-bg:#4a3d00;--hl-border:#c89000;"
        "--fig-border:#444;--cap-color:#999;--muted:#777;--toolbar-border:#444}}"
        ":root[data-theme='dark']{--bg:#1a1a1a;--fg:#e0e0e0;--hl-bg:#4a3d00;"
        "--hl-border:#c89000;--fig-border:#444;--cap-color:#999;--muted:#777;--toolbar-border:#444}"
        "body{margin:0;font-family:Georgia,serif;max-width:800px;padding:20px 40px;"
        "line-height:1.6;background:var(--bg);color:var(--fg)}"
        "figure{margin:16px 0;text-align:center}"
        "figcaption{font-size:0.9em;color:var(--cap-color);margin-top:4px}"
        "img{border:1px solid var(--fig-border);border-radius:4px}"
        ".toolbar{padding:12px 0;border-top:1px solid var(--toolbar-border);margin-top:20px}"
        ".download-btn{display:inline-block;padding:6px 14px;border-radius:6px;"
        "border:1px solid rgba(128,128,128,.4);background:rgba(128,128,128,.12);"
        "color:inherit;text-decoration:none;font-size:13px}"
        ".download-btn:hover{background:rgba(128,128,128,.25)}"
        "</style></head><body>"
        f"{content_html}"
        "<div class='toolbar'>"
        f"<a class='download-btn' href='{pre}/download-original/{quote(stem)}?t={_sign(f'/download-original/{stem}')}'"
        " download>Download original</a> "
        f"<a class='download-btn' href='{pre}/download-parsed/{quote(stem)}?t={_sign(f'/download-parsed/{stem}')}'"
        " download>Download parsed (ZIP)</a>"
        "</div></body></html>"
    )


# ── Unified citation view ──
# One link format for every file type: /view/<stem>/<page_idx>?bbox=...&highlight=...&section=... The server decides:
# original PDF (incl. collision-renamed copies in OUTPUT_ROOT/_staged) → rendered page with the bbox highlight; MinerU's
# <stem>_origin.pdf (images, TIFF, INDD) → that page; otherwise (Office) → a reading view of the slide, section or table
# as Memento parsed it. Same header and the same two downloads everywhere.

_FIG_MEDIA = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp",
              ".gif": "image/gif"}
MAX_RENDER_PX = 2000   # longest side of a rendered page (photos parsed as images can be huge)
_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
OFFICE_EXT = {".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx", ".odt", ".odp", ".ods", ".rtf"}
TYPE_LABEL = {".pdf": "PDF", ".doc": "Word", ".docx": "Word", ".odt": "Word", ".rtf": "Word",
              ".ppt": "PowerPoint", ".pptx": "PowerPoint", ".odp": "PowerPoint",
              ".xls": "Excel", ".xlsx": "Excel", ".ods": "Excel"}


# what ingestion parses (memento/ingestion/convert.py ALL_SUPPORTED); other files sharing a stem (a .DSD next to
# a .TIF) were never parsed, so they are never the "original"
_RANK = {".pdf": 0, ".docx": 1, ".pptx": 1, ".xlsx": 1, ".doc": 2, ".ppt": 2, ".xls": 2, ".odt": 2, ".odp": 2,
         ".ods": 2, ".rtf": 2, ".png": 3, ".jpg": 3, ".jpeg": 3, ".tif": 3, ".tiff": 3, ".bmp": 3, ".webp": 3}
# collision-renamed files are symlinks in OUTPUT_ROOT/_staged pointing at HOST paths; map them into the container
HOST_INPUT_ROOT = os.getenv("HOST_INPUT_ROOT", "")


def _staged_target(link: Path) -> Path | None:
    if not link.is_symlink():
        return link if link.is_file() else None
    target = os.readlink(link)
    if HOST_INPUT_ROOT and target.startswith(HOST_INPUT_ROOT.rstrip("/") + "/"):
        target = str(INPUT_DIR / target[len(HOST_INPUT_ROOT.rstrip("/")) + 1:])
    t = Path(target)
    return t if t.is_file() else None


def _original(stem: str) -> Path | None:
    """The original file of a document: the parseable input with this stem (PDF first), or the target of a
    collision-renamed copy in OUTPUT_ROOT/_staged."""
    if not stem or "/" in stem or stem in (".", ".."):
        return None
    cands = [p for p in INPUT_DIR.rglob(f"{glob_escape(stem)}.*") if p.is_file() and p.stem == stem]
    for link in (OUTPUT_DIR / "_staged").glob(f"{glob_escape(stem)}.*"):
        t = _staged_target(link) if link.stem == stem else None
        if t is not None:
            cands.append(t)
    cands = [p for p in cands if p.suffix.lower() in _RANK and not p.name.startswith("._")]
    if not cands:
        return None
    # match how the document was parsed: Office parser output → an Office file, else a PDF or image
    _cl, sub = _find_content_list(stem)
    office_first = sub == "office"
    return sorted(cands, key=lambda p: ((_RANK[p.suffix.lower()] in (1, 2)) != office_first,
                                        _RANK[p.suffix.lower()], str(p)))[0]


def _origin_pdf(stem: str) -> Path | None:
    for base in _stem_dirs(stem):
        for p in sorted(base.glob(f"*/{glob_escape(stem)}_origin.pdf")):
            return p
    return None


def _page_pdf(stem: str) -> Path | None:
    """What a page is rendered from: the PDF itself, MinerU's <stem>_origin.pdf (images), or, for an image parsed
    through MinerU's web API (chat uploads have no _origin.pdf), the image file itself, which PyMuPDF opens as a
    one-page document (bboxes are page-normalized, so highlights still fit)."""
    orig = _original(stem)
    if orig is not None and orig.suffix.lower() == ".pdf":
        return orig
    origin = _origin_pdf(stem)
    if origin is not None or orig is None or orig.suffix.lower() not in _IMAGE_EXT:
        return origin
    try:
        with fitz.open(orig) as doc:     # opens lazily: loading the page is what decodes the image
            return orig if doc.page_count and doc[0].rect.width > 0 else None
    except Exception:
        return None              # a format PyMuPDF cannot open: the text view is shown instead


def _norm(t: str) -> list[str]:
    import re as _re
    return _re.findall(r"[a-z0-9äöüß]+", (t or "").lower())


def _best_entry(entries: list[dict], snippet: str) -> int | None:
    """Index of the parsed entry that best contains the cited passage (token overlap, robust to whitespace,
    hyphenation and merged paragraphs); None below 50 % of the snippet's tokens."""
    want = _norm(snippet)[:40]
    if not want:
        return None
    best, best_i = 0.0, None
    for i, e in enumerate(entries):
        text = " ".join(str(e.get(k) or "") for k in ("text", "table_body", "table_caption", "image_caption"))
        if isinstance(e.get("list_items"), list):
            text += " " + " ".join(map(str, e["list_items"]))
        have = set(_norm(text))
        if not have:
            continue
        score = sum(w in have for w in want) / len(want)
        if score > best:
            best, best_i = score, i
    return best_i if best >= 0.5 else None


_ALLOWED_TABLE_TAGS = {"table", "thead", "tbody", "tr", "td", "th", "br", "sub", "sup", "b", "i"}


def _safe_table(html_src: str) -> str:
    """Keep only plain table markup (no attributes except colspan/rowspan): the table comes from a parsed, uploaded
    document."""
    import re as _re

    def tag(m):
        closing, name, attrs = m.group(1), m.group(2).lower(), m.group(3) or ""
        if name not in _ALLOWED_TABLE_TAGS:
            return ""
        keep = " ".join(a for a in _re.findall(r'(?:colspan|rowspan)="\d+"', attrs))
        return f"<{closing}{name}{(' ' + keep) if keep and not closing else ''}>"
    return _re.sub(r"<(/?)([a-zA-Z0-9]+)([^>]*)>", tag, html_src or "")


def _inline(text: str) -> str:
    """Escape parsed text; render MinerU's Markdown emphasis (**bold**, *italic*) and drop list dashes."""
    import re as _re
    t = html.escape(_re.sub(r"^\s*[-•]\s+", "", str(text)))
    t = _re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    return _re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", t)


def _entry_html(e: dict, cl_path: Path, is_match: bool, pre: str = "") -> str:
    etype = e.get("type", "text")
    cls = " class='hl'" if is_match else ""
    if etype == "image":
        img = e.get("img_path", "")
        cap = e.get("image_caption", "")
        cap = " ".join(map(str, cap)) if isinstance(cap, list) else str(cap or "")
        f = (cl_path.parent / img) if img else None
        src = (_data_uri(f.read_bytes(), _FIG_MEDIA.get(f.suffix.lower(), "image/jpeg"))   # embedded, see view()
               if f is not None and f.is_file() and f.resolve().is_relative_to(OUTPUT_DIR.resolve()) else "")
        return (f"<figure{cls}>" + (f"<img src='{src}' style='max-width:100%'>" if src else "")
                + f"<figcaption>{html.escape(cap)}</figcaption></figure>")
    if etype == "table":
        cap = e.get("table_caption", "")
        cap = " ".join(map(str, cap)) if isinstance(cap, list) else str(cap or "")
        return (f"<div{cls}><div class='cap'>{html.escape(cap)}</div>"
                f"<div class='tbl'>{_safe_table(e.get('table_body', ''))}</div></div>")
    if etype == "list" and isinstance(e.get("list_items"), list):
        items = "".join(f"<li>{_inline(x)}</li>" for x in e["list_items"])
        return f"<ul{cls}>{items}</ul>"
    if etype == "equation":
        return f"<pre{cls}>{html.escape(str(e.get('text', '')))}</pre>"
    text = str(e.get("text", ""))
    if not text.strip():
        return ""
    level = e.get("text_level")
    if isinstance(level, int) and 1 <= level <= 6:
        h = min(level + 1, 4)
        return f"<h{h}{cls}>{_inline(text).replace('<b>', '').replace('</b>', '')}</h{h}>"
    return f"<p{cls}>{_inline(text)}</p>"


# The page as a sheet on a reading desk: one quiet toolbar (type + page), the sheet, a footer with the two downloads
# that stays in view. Accent: the theme green (#009260). System fonts only: no web-font requests from readers' browsers.
# The desk takes the sidebar's greys: Open WebUI's gray-50 in light mode, the charcoal of Memento's dark sidebar
# (openwebui/custom.css) in dark mode; neutral lines and muted text.
_VIEW_CSS = (
    ":root{color-scheme:light dark;--desk:oklch(98% 0 0);--sheet:#fff;--ink:#1d2428;--muted:#5b666c;--line:#e3e3e3;"
    "--accent:#009260;--accent-hover:#007a50;--accent-soft:#e2f2ea;--on-accent:#fff;--mark:#fdf1b5;"
    "--shadow:0 1px 2px rgba(22,34,30,.07),0 10px 28px rgba(22,34,30,.10)}"
    "@media(prefers-color-scheme:dark){:root:not([data-theme='light']){--desk:oklch(20% 0 0);--sheet:oklch(25% 0 0);--ink:#e8e8e8;"
    "--muted:#a3a3a3;--line:oklch(31% 0 0);--accent:#2bb383;--accent-hover:#3cc793;--accent-soft:#12372a;--on-accent:#0c1a15;"
    "--mark:rgba(234,179,8,.17);--shadow:0 1px 2px rgba(0,0,0,.4),0 10px 28px rgba(0,0,0,.35);--page-filter:brightness(.9) contrast(1.04)}}"
    ":root[data-theme='dark']{--desk:oklch(20% 0 0);--sheet:oklch(25% 0 0);--ink:#e8e8e8;--muted:#a3a3a3;--line:oklch(31% 0 0);"
    "--accent:#2bb383;--accent-hover:#3cc793;--accent-soft:#12372a;--on-accent:#0c1a15;"
    "--mark:rgba(234,179,8,.17);--shadow:0 1px 2px rgba(0,0,0,.4),0 10px 28px rgba(0,0,0,.35);"
    "--page-filter:brightness(.9) contrast(1.04)}"
    "*{box-sizing:border-box}"
    "html,body{height:100%}"
    "body{margin:0;display:flex;flex-direction:column;background:var(--desk);color:var(--ink);"
    "font:15px/1.6 system-ui,-apple-system,'Segoe UI',Roboto,'Helvetica Neue',sans-serif;"
    "-webkit-font-smoothing:antialiased}"
    # toolbar and footer take the desk colour (the sidebar's grey), like Open WebUI's own panel header above them; only
    # the page or text itself is a sheet
    ".bar{display:flex;align-items:center;gap:10px;flex-wrap:wrap;padding:12px 16px;background:var(--desk);"
    "border-bottom:1px solid var(--line)}"
    ".bar .doc{margin:0;flex-basis:100%;font-size:15px;font-weight:600;line-height:1.35}"
    ".kind{padding:2px 9px;border-radius:999px;background:var(--accent-soft);color:var(--accent);font-size:12px;"
    "font-weight:600;letter-spacing:.01em}"
    ".loc{font-size:14px;color:var(--muted);font-variant-numeric:tabular-nums}.loc b{color:var(--ink);font-weight:650}"
    # desk + sheet
    ".desk{flex:1;padding:14px 16px 20px}"
    ".hint{display:flex;gap:8px;align-items:flex-start;margin:0 2px 12px;color:var(--muted);font-size:12.5px;"
    "line-height:1.5}.hint svg{flex:none;margin-top:2px;color:var(--accent)}"
    ".sheet{background:var(--sheet);border-radius:8px;box-shadow:var(--shadow);overflow:hidden}"
    ".sheet.page{background:#fff}"
    ".sheet img.page{width:100%;display:block;filter:var(--page-filter,none)}"
    ".sheet.text{padding:18px 20px;max-width:72ch;margin:0 auto}"
    ".sheet.text h2,.sheet.text h3,.sheet.text h4{line-height:1.3;margin:1.1em 0 .5em}"
    ".sheet.text p{margin:0 0 .9em}"
    ".zoom{position:relative;overflow:hidden;cursor:zoom-in;touch-action:manipulation}"
    # between the pages of a passage continued on a later page
    ".cont{margin:0;padding:6px 12px;font-size:12.5px;color:var(--muted);background:var(--desk);"
    "border-block:1px solid var(--line)}"
    ".zlabel{position:absolute;right:10px;top:10px;padding:3px 9px;border-radius:999px;font-size:12px;font-weight:600;"
    "color:var(--on-accent);background:var(--accent);opacity:0;transition:opacity .15s;pointer-events:none;"
    "font-variant-numeric:tabular-nums}"
    ".zoom img.page{transition:transform .08s ease-out;will-change:transform}"
    "@media(prefers-reduced-motion:reduce){.zoom img.page,.zlabel{transition:none}}"
    ".hl{background:var(--mark);box-shadow:inset 3px 0 0 var(--accent);border-radius:4px;padding:6px 10px 6px 12px}"
    ".tbl{overflow-x:auto}.tbl table{border-collapse:collapse;font-size:13px}"
    ".tbl td,.tbl th{border:1px solid var(--line);padding:4px 7px}.cap{color:var(--muted);font-size:13px}"
    "figure{margin:14px 0;text-align:center}figure img{max-width:100%;border-radius:4px}"
    "figcaption{color:var(--muted);font-size:13px;margin-top:6px}"
    ".more{color:var(--muted);text-align:center;font-size:12px;letter-spacing:.2em;margin:10px 0}"
    # a cited document that is no longer in the library
    ".gone{max-width:46ch;margin:12vh auto 0;padding:22px 24px;text-align:center}"
    ".gone svg{color:var(--muted);margin-bottom:10px}"
    ".gone h2{margin:0 0 8px;font-size:17px;line-height:1.35}"
    ".gone p{margin:0;color:var(--muted);font-size:14px;line-height:1.55}"
    ".gone .name{margin-top:14px;font-size:12.5px;overflow-wrap:anywhere}"
    # downloads
    ".dock{position:sticky;bottom:0;display:flex;gap:10px;flex-wrap:wrap;padding:12px 16px;"
    "background:color-mix(in srgb,var(--desk) 88%,transparent);backdrop-filter:blur(8px);"
    "-webkit-backdrop-filter:blur(8px);border-top:1px solid var(--line)}"
    ".btn{display:inline-flex;align-items:center;gap:8px;min-height:40px;padding:8px 14px;border-radius:10px;"
    "border:1px solid var(--line);background:var(--sheet);color:var(--ink);text-decoration:none;font-size:14px;"
    "font-weight:600;line-height:1.2;transition:background .12s,border-color .12s}"
    ".btn:hover{border-color:var(--accent);background:var(--accent-soft)}"
    ".btn .sub{font-weight:400;color:var(--muted);font-size:12.5px}"
    ".btn.primary{background:var(--accent);border-color:var(--accent);color:var(--on-accent)}"
    ".btn.primary:hover{background:var(--accent-hover);border-color:var(--accent-hover)}"
    ".btn.primary .sub{color:inherit;opacity:.85}"
    ".btn:focus-visible{outline:2px solid var(--accent);outline-offset:2px}"
    ".btn svg{flex:none}"
    "@media(max-width:420px){.dock .btn{flex:1 1 100%;justify-content:center}}"
)


# Hover zoom for each page image in Open WebUI's side panel: the panel is narrow and its frame may not open new tabs,
# so the part under the pointer is magnified in place. The scroll wheel sets the zoom (1.25–6×, remembered while the
# page is open), moving the pointer pans; a label shows the level. On touch screens a tap toggles the zoom. The page
# image is rendered at ~150 dpi, so the zoomed view stays sharp.
_ZOOM, _ZOOM_MIN, _ZOOM_MAX = 2.5, 1.25, 6.0
_ZOOM_JS = (
    f"<script>(()=>{{let z={_ZOOM};document.querySelectorAll('.zoom').forEach(w=>{{const im=w.querySelector('img');"
    "const lbl=document.createElement('div');lbl.className='zlabel';w.appendChild(lbl);"
    "let pinned=false;"
    "const at=(x,y)=>{const r=w.getBoundingClientRect();"
    "im.style.transformOrigin=((x-r.left)/r.width*100)+'% '+((y-r.top)/r.height*100)+'%';"
    "im.style.transform='scale('+z+')';lbl.textContent=z.toFixed(1)+'×';lbl.style.opacity=1;};"
    "const off=()=>{im.style.transform='';lbl.style.opacity=0;};"
    "w.addEventListener('mousemove',e=>at(e.clientX,e.clientY));"
    "w.addEventListener('mouseleave',()=>{if(!pinned)off();});"
    "w.addEventListener('wheel',e=>{e.preventDefault();"
    f"z=Math.min({_ZOOM_MAX},Math.max({_ZOOM_MIN},z*(e.deltaY<0?1.2:1/1.2)));at(e.clientX,e.clientY);}},"
    "{passive:false});"
    "w.addEventListener('touchend',e=>{const t=e.changedTouches[0];pinned=!pinned;"
    "if(pinned)at(t.clientX,t.clientY);else off();e.preventDefault();},{passive:false});});})();</script>")


# OpenWebUI's own theme (dark/light switch) reaches the side panel as a message from openwebui/loader.js; without one
# (a new tab, another host) the page follows the device setting. Only the two values are accepted.
_THEME_JS = ("<script>addEventListener('message',e=>{const t=e.data&&e.data.memento_theme;"
             "if(t==='dark'||t==='light')document.documentElement.dataset.theme=t;});</script>")
_ICON_DOWNLOAD = ("<svg width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' "
                  "stroke-linecap='round' stroke-linejoin='round' aria-hidden='true'><path d='M12 4v11'/>"
                  "<path d='m7 11 5 5 5-5'/><path d='M5 20h14'/></svg>")
_ICON_INFO = ("<svg width='14' height='14' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' "
              "stroke-linecap='round' stroke-linejoin='round' aria-hidden='true'><circle cx='12' cy='12' r='9'/>"
              "<path d='M12 11v5'/><path d='M12 8h.01'/></svg>")


def _view_page(stem: str, location: str, type_label: str, note: str, body: str, pre: str = "",
               embedded: bool = False) -> str:
    # in OpenWebUI's side panel its header already shows the document name (as a link): no second title there
    title = "" if embedded else f"<h1 class='doc'>{html.escape(stem)}</h1>"
    loc = html.escape(location[:1].upper() + location[1:])
    loc = re.sub(r"^(Pages?|Slide) (\d+(?:(?:–|, )\d+)*)", r"\1 <b>\2</b>", loc)   # "Page <b>41</b> of 325"
    sheet = "page" if "<img class='page'" in body else "text"
    hint = (f"<p class='hint'>{_ICON_INFO}<span>{html.escape(note).replace(chr(10), '<br>')}</span></p>"
            if note else "")
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,"
        f"initial-scale=1'><title>{html.escape(stem)}</title><style>{_VIEW_CSS}</style>{_THEME_JS}</head><body>"
        f"<header class='bar'>{title}<span class='kind'>{html.escape(type_label)}</span>"
        f"<span class='loc'>{loc}</span></header>"
        f"<main class='desk'>{hint}<div class='sheet {sheet}'>{body}</div></main>"
        "<footer class='dock'>"
        f"<a class='btn primary' href='{pre}/download-original/{quote(stem)}?t={_sign(f'/download-original/{stem}')}'>"
        f"{_ICON_DOWNLOAD}Download Original<span class='sub'>{html.escape(type_label)}</span></a>"
        f"<a class='btn' href='{pre}/download-parsed/{quote(stem)}?t={_sign(f'/download-parsed/{stem}')}' download>"
        f"{_ICON_DOWNLOAD}Parsed Data"
        "<span class='sub'>Markdown, JSON, images</span></a>"
        "</footer></body></html>"
    )


_ICON_GONE = ("<svg width='30' height='30' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='1.6' "
              "stroke-linecap='round' stroke-linejoin='round' aria-hidden='true'><path d='M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 "
              "2 2h10a2 2 0 0 0 2-2V8z'/><path d='M14 3v5h5'/><path d='m9.5 12.5 5 5'/><path d='m14.5 12.5-5 5'/></svg>")


def _gone_page(stem: str) -> str:
    """The cited document is no longer in the library (removed, or the library was reset after the answer was written):
    a themed page instead of FastAPI's bare JSON."""
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,"
        f"initial-scale=1'><title>{html.escape(stem)}</title><style>{_VIEW_CSS}</style>{_THEME_JS}</head><body>"
        f"<main class='desk'><div class='sheet gone'>{_ICON_GONE}"
        "<h2>This document is no longer in the library</h2>"
        "<p>It was removed after this answer was written, so the cited page cannot be shown.</p>"
        f"<p class='name'>{html.escape(stem)}</p></div></main></body></html>"
    )


@app.get("/view/{stem}/{page}", response_class=HTMLResponse)
async def view(request: Request, stem: str, page: int, bbox: list[str] | None = Query(None),
               highlight: str | None = None,
               section: str | None = None):
    await _require_owner(request, stem)
    pre = _prefix(request)
    # OpenWebUI adds chat_id / message_id / source_id when it loads the page into its side panel (Embeds.svelte)
    embedded = "source_id" in request.query_params or "chat_id" in request.query_params
    orig = _original(stem)
    ext = orig.suffix.lower() if orig else ""
    pdf = _page_pdf(stem)
    if pdf is not None:
        with fitz.open(pdf) as doc:
            n = doc.page_count
        p = min(max(page, 0), max(n - 1, 0))
        boxes = {q: b for q, b in _passage_boxes(stem, p, bbox).items() if 0 <= q < n}
        pages = sorted(boxes)
        # the page images are embedded (data URI): Open WebUI's side panel sends no login cookie with sub-requests, so a
        # separate image request would be refused by the proxy
        imgs = [f"<img class='page' src='{_data_uri(*_render_page(stem, q, boxes[q] or None))}' alt='page {q + 1}'>"
                for q in pages]
        label = "PDF" if ext == ".pdf" else ("Image" if ext and ext not in OFFICE_EXT else TYPE_LABEL.get(ext, "Document"))
        if len(pages) > 1:                            # a passage continued on a later page: its pages, one below another
            span = (f"{pages[0] + 1}–{pages[-1] + 1}" if pages == list(range(pages[0], pages[-1] + 1))
                    else ", ".join(str(q + 1) for q in pages))
            loc = f"pages {span} of {n}"
        else:
            loc = f"page {p + 1} of {n}" if n > 1 else "page 1"
        wrap = (lambda im: f"<div class='zoom'>{im}</div>") if embedded else (lambda im: im)
        body = "".join((f"<p class='cont'>Continues on page {q + 1}</p>" if k else "") + wrap(im)
                       for k, (q, im) in enumerate(zip(pages, imgs)))
        if embedded:                                  # the side panel: hover zoom + how to get the full page
            return _view_page(stem, loc, label,
                              "Hover the page to zoom in, scroll to change the zoom.\n"
                              "Click the document name above to open it in a new tab.",
                              body + _ZOOM_JS, pre, embedded)
        return _view_page(stem, loc, label, "", body, pre, embedded)

    cl_path, _sub = _find_content_list(stem)
    if cl_path is None:
        return HTMLResponse(_gone_page(stem), status_code=404)
    import json as _json
    entries = _json.loads(cl_path.read_text())
    label = TYPE_LABEL.get(ext, "Document")
    pages = {e.get("page_idx", 0) for e in entries}
    if label == "PowerPoint" and len(pages) > 1:
        scope = [e for e in entries if e.get("page_idx", 0) == page] or entries
        loc = f"slide {page + 1} of {len(pages)}"
    else:
        scope = entries
        loc = f"section: {section}" if section else ("table" if label == "Excel" else "text")
    i = _best_entry(scope, highlight or "")
    if i is not None and len(scope) > 2 * CONTEXT_WINDOW + 1 and label != "PowerPoint":
        start, end = max(0, i - CONTEXT_WINDOW), min(len(scope), i + CONTEXT_WINDOW + 1)
    else:
        start, end = 0, len(scope)
    parts = (["<p class='more'>⋯ earlier content ⋯</p>"] if start > 0 else [])
    parts += [_entry_html(e, cl_path, (start + k) == i, pre) for k, e in enumerate(scope[start:end])]
    parts += (["<p class='more'>⋯ more content ⋯</p>"] if end < len(scope) else [])
    note = (f"Text view of this {label} file as Memento read it. Download the original for its layout."
            if label in ("Word", "PowerPoint", "Excel") else "Text view as Memento read it.")
    return _view_page(stem, loc, label, note, "\n".join(p for p in parts if p), pre, embedded)


# A paragraph MinerU split into blocks (into the other column, onto the next page) is one passage whose bbox is its
# first block's (ingestion merge_page_boundary_paragraphs); the later blocks follow it in the enriched list, marked
# merged_continuation (other kinds of items in between do not count); every block is marked. At most
# MAX_CONTINUATION_PAGES further pages are shown.
MAX_CONTINUATION_PAGES = 2


def _box_key(box) -> tuple | None:
    """A box ("x0,y0,x1,y1" or a list) as four rounded numbers, for comparing a link's box with the parse list's."""
    try:
        v = tuple(round(float(x)) for x in (box.split(",") if isinstance(box, str) else box))
    except (TypeError, ValueError):
        return None
    return v if len(v) == 4 else None


def _passage_boxes(stem: str, page: int, bbox: list[str] | None) -> dict[int, list[str]]:
    """{page: boxes} to mark for a cited passage: the link's own boxes on `page`, plus the blocks its paragraph
    continues in when its first box is a merged paragraph's first block."""
    boxes = {page: list(bbox or [])}
    key = _box_key(bbox[0]) if bbox else None
    cl_path = _find_content_list(stem)[0] if key else None
    try:
        entries = json.loads(cl_path.read_text()) if cl_path else []
    except (OSError, ValueError):
        entries = []
    head = next((i for i, e in enumerate(entries) if e.get("type") == "text" and not e.get("text_level")
                 and not e.get("merged_continuation") and e.get("page_idx", 0) == page
                 and _box_key(e.get("bbox") or ()) == key), None)
    for e in entries[head + 1:] if head is not None else []:
        if e.get("type") != "text":
            continue
        if not e.get("merged_continuation"):
            break
        q, b = e.get("page_idx", 0), _box_key(e.get("bbox") or ())
        if b is None or (q not in boxes and len(boxes) > MAX_CONTINUATION_PAGES):
            continue
        if b not in {_box_key(x) for x in boxes.get(q, [])}:
            boxes.setdefault(q, []).append(",".join(map(str, b)))
    return boxes


def _render_page(stem: str, page: int, bbox: list[str] | None) -> tuple[bytes, str]:
    """The page as an image, highlights drawn: (bytes, media type). Pages of image files are JPEG, PDF pages PNG."""
    pdf = _page_pdf(stem)
    if pdf is None:
        raise HTTPException(404, "No page rendering for this document")
    doc = fitz.open(pdf)
    if not doc.is_pdf:                   # an image file (see _page_pdf): highlights need a PDF page
        img_doc, doc = doc, fitz.open("pdf", doc.convert_to_pdf())
        img_doc.close()
    try:
        if not (0 <= page < doc.page_count):
            raise HTTPException(404, "Page out of range")
        pg = doc[page]
        if bbox:
            _draw_highlights(pg, bbox)
        zoom = min(PAGE_RENDER_DPI / 72, MAX_RENDER_PX / max(pg.rect.width, pg.rect.height, 1))
        pix = pg.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    finally:
        doc.close()
    if pdf.suffix.lower() in _IMAGE_EXT or (pdf.name.endswith("_origin.pdf")
                                            and (_original(stem) or pdf).suffix.lower() in _IMAGE_EXT):
        return pix.tobytes("jpg", jpg_quality=85), "image/jpeg"
    return pix.tobytes("png"), "image/png"


def _data_uri(content: bytes, media_type: str) -> str:
    return f"data:{media_type};base64," + base64.b64encode(content).decode()


@app.get("/view/{stem}/{page}/image")
async def view_image(request: Request, stem: str, page: int, bbox: list[str] | None = Query(None)):
    await _require_owner(request, stem)
    content, media_type = _render_page(stem, page, _passage_boxes(stem, page, bbox)[page] or None)
    return Response(content=content, media_type=media_type)


@app.get("/download-original/{stem}")
async def download_original(request: Request, stem: str, t: str | None = None):
    _check_signature(f"/download-original/{stem}", t)
    if t is None:
        await _require_owner(request, stem)
    orig = _original(stem)
    if orig is None:
        raise HTTPException(404, "Original file not found")
    return FileResponse(orig, filename=orig.name.split("__")[-1] if orig.parent.name == "_staged" else orig.name)


# Figure images for the pages above, and nothing else: parsed data goes out through /download-parsed, originals through
# signed /download-original links, runtime state (library list, reports, paper cards) through its own routes.
FIGURE_TYPES = {".png", ".jpg", ".jpeg", ".webp", ".gif"}


@app.get("/figures/{path:path}")
async def figure(request: Request, path: str):
    root = OUTPUT_DIR.resolve()
    p = (root / path).resolve()
    if p.suffix.lower() not in FIGURE_TYPES or not p.is_file() or root not in p.parents \
            or root / "_runtime" in (p, *p.parents):
        raise HTTPException(404, "Not found")
    rel = p.relative_to(root).parts
    if rel[0] == "_temp":                     # a Document Chat figure (checked on the resolved path)
        if len(rel) < 3:
            raise HTTPException(404, "Not found")
        await _require_owner(request, rel[1])
    return FileResponse(p)


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
