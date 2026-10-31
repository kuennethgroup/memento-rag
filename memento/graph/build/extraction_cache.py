"""Raw LLM-response cache for graph extraction, one JSON file per paper.

The raw response is cached (not the fused result), keyed by everything that
determines it: model, system prompt, user prompt (overview + snippets +
schema) and the image hash. Registry edits, parser fixes and merge thresholds
are then re-applied with zero LLM calls; a schema or prompt change misses the
cache by construction."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


class RawResponseCache:
    def __init__(self, path: Path | None):
        self.path = Path(path) if path else None
        self.entries: dict[str, str] = {}
        self.hits = self.misses = 0
        self._dirty = False
        if self.path and self.path.exists():
            data = json.loads(self.path.read_text())
            self.entries = dict(data.get("entries", {}))

    @staticmethod
    def key(model: str, system: str, user: str, image_sha1: str = "") -> str:
        h = hashlib.sha256()
        for part in (model, system, user, image_sha1):
            h.update(part.encode("utf-8"))
            h.update(b"\x00")
        return h.hexdigest()

    def get(self, key: str) -> str | None:
        raw = self.entries.get(key)
        if raw is None:
            self.misses += 1
        else:
            self.hits += 1
        return raw

    def put(self, key: str, raw: str) -> None:
        self.entries[key] = raw
        self._dirty = True

    def save(self) -> None:
        if not self.path or not self._dirty:
            return
        tmp = self.path.with_name(f"{self.path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps({"version": 1, "entries": self.entries}, ensure_ascii=False))
        os.replace(tmp, self.path)
        self._dirty = False
