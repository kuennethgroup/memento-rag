"""Atomic text files shared by copies of the pipes. A file is written to a temporary name next to it, then renamed over
it, so a reader never sees half a file. The temporary name is unique per process and thread: copies run in one Open
WebUI process, and two threads sharing one temporary name could rename each other's file away or publish a half-written
one."""
from __future__ import annotations

import os
import secrets
import threading
from pathlib import Path


def write_text_atomic(path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.{secrets.token_hex(4)}.tmp")
    try:
        tmp.write_text(text)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
