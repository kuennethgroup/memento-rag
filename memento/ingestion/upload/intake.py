"""Which files a Library Upload message brings: the files of this message (not the chat's earlier ones), the images
Open WebUI moved into it, only the uploader's own files, the file types read, and who may upload.
"""
from memento.ingestion.convert import ALL_SUPPORTED


def current_message_files(metadata: dict | None, files: list) -> tuple[list, list]:
    """(files attached to the message being sent, files from earlier messages of the chat). Open WebUI sends every file
    attached anywhere in the chat with each request, so a file attached earlier for Library Chat or Document Chat would
    reach Library Upload after a model switch. The message being sent comes as metadata["user_message"]; its "files" key
    is missing when it has no attachment. Without that message (API calls) every file counts."""
    msg = (metadata or {}).get("user_message")
    if not isinstance(msg, dict) or not msg:
        return list(files or []), []
    ids = {f.get("id") for f in msg.get("files") or [] if f.get("id")}
    now = [f for f in files or [] if f.get("id") is None or f.get("id") in ids]
    return now, [f for f in files or [] if f not in now]


async def chat_file_names(chat_id) -> list:
    """Names of the files uploaded in a stored chat, oldest message first; [] when the chat cannot be read."""
    if not chat_id or str(chat_id).startswith("local:"):
        return []
    try:
        from open_webui.models.chats import Chats
        chat = await Chats.get_chat_by_id(chat_id)
        msgs = ((((chat.chat if chat else {}) or {}).get("history") or {}).get("messages") or {}).values()
        names = []
        for m in sorted(msgs, key=lambda m: m.get("timestamp") or 0):
            if m.get("role") == "user":
                for f in m.get("files") or []:
                    info = f.get("file") or {}
                    names.append(f.get("name") or (info.get("meta") or {}).get("name") or info.get("filename") or "")
        return names
    except Exception:
        return []


async def message_images(metadata: dict | None, files: list, user: dict | None, lookup=None) -> list:
    """Images attached to the message that are missing from the pipe's file list. The upload model has vision on, which
    the chat box needs to accept images; Open WebUI then moves raster images out of `__files__` into the message as
    pictures. The upload is still in the user message's files; its path comes from Open WebUI's file table. Only the
    uploader's own files: the message comes from the browser."""
    have = {f.get("id") for f in files or [] if f.get("id")}
    if lookup is None:
        from open_webui.models.files import Files
        lookup = Files.get_file_by_id
    out = []
    for f in (((metadata or {}).get("user_message") or {}).get("files") or []):
        fid, ctype = f.get("id"), (f.get("content_type") or "").split(";")[0].strip().lower()
        if f.get("type") != "file" or not fid or fid in have or not ctype.startswith("image/") or ctype == "image/svg+xml":
            continue
        try:
            rec = await lookup(fid)
        except Exception:
            rec = None
        if not rec or not rec.path or rec.user_id != (user or {}).get("id"):
            continue
        name = f.get("name") or (rec.meta or {}).get("name") or rec.filename
        out.append({"id": fid, "name": name, "type": "file",
                    "file": {"id": fid, "filename": rec.filename, "path": rec.path, "meta": rec.meta or {}}})
        have.add(fid)
    return out


async def owned_uploads(files: list, user: dict | None, lookup=None) -> tuple[list, list]:
    """(files to read, names refused). The pipe's file list comes from the browser, so a forged entry could name any
    server-readable path: only files whose Open WebUI record belongs to this user are kept, each with the path Open
    WebUI stored; anything else is refused by name."""
    if lookup is None:
        from open_webui.models.files import Files
        lookup = Files.get_file_by_id
    uid = (user or {}).get("id")
    kept, refused = [], []
    for f in files or []:
        fid = f.get("id")
        name = f.get("name") or (f.get("file") or {}).get("filename") or "file"
        try:
            rec = await lookup(fid) if fid else None
        except Exception:
            rec = None
        if not uid or rec is None or not getattr(rec, "path", None) or getattr(rec, "user_id", None) != uid:
            refused.append(name)
            continue
        kept.append({**f, "file": {**(f.get("file") or {}), "id": fid, "path": rec.path,
                                   "filename": getattr(rec, "filename", None) or name}})
    return kept, refused


# Documents, older Office formats (converted first) and standalone images: the same set as run_pipeline.py.
SUPPORTED_EXTENSIONS = ALL_SUPPORTED


MIME_TYPES = {
    ".pdf":  "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".png":  "image/png",
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".tif":  "image/tiff",
    ".tiff": "image/tiff",
    ".bmp":  "image/bmp",
    ".webp": "image/webp",
}


async def may_ingest(valves, user: dict) -> bool:
    """Admins, or members of one of the CONTRIBUTOR_GROUP OpenWebUI groups. The model's access control is the first
    gate (readers never see Memento Ingestion); this check stops a reader who reaches the pipe anyway."""
    if (user or {}).get("role") == "admin":
        return True
    groups = {g.strip().casefold() for g in valves.CONTRIBUTOR_GROUP.split(",") if g.strip()}
    if not groups or not (user or {}).get("id"):
        return False
    try:
        from open_webui.models.groups import Groups
        return any(g.name.strip().casefold() in groups for g in await Groups.get_groups_by_member_id(user["id"]))
    except Exception:
        return False
