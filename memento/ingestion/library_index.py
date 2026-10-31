"""Which files go into a library, and under which name: one decision for both ingestion paths (the command line and
Library Upload). Per file, within one collection:
1. identical content already in the library under another name → "duplicate": not added again
2. the same name already in the library: same content → "unchanged"; different content from the same person → "changed"
(the new version replaces the old one); from someone else → "renamed" (added as <name>__<person>)
3. otherwise → "new"

"In the library" means the collection has vectors for that name (in_library), not merely a manifest entry, so a failed
parse or a deleted document never blocks a later upload. Content is the SHA-256 of the file as given, before any
LibreOffice conversion (converted files are not byte-stable). Documents without a recorded hash count as unchanged and
get their hash recorded, so nothing is re-indexed.

"Person": a chat upload's Open WebUI user id. The command line and folder paths typed in the chat are server-side
library maintenance and share one identity, SERVER_UPLOADER, which also owns every document without a recorded uploader."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import unicodedata
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable
from memento.config.paths import PROJECT_ROOT

MANIFEST_FILE = PROJECT_ROOT / "data" / "manifest.json"
SERVER_UPLOADER = "cli"


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def person_tag(name: str) -> str:
    """Short, readable, filesystem-safe tag for renamed documents: 'Jana Müller' → 'jana-muller'."""
    s = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", "-", s).strip("-")[:24] or "user"


@contextmanager
def locked_manifest(path: Path = MANIFEST_FILE):
    """Read-modify-write the shared manifest under a file lock: both ingestion paths may write at the same time.
    A manifest that exists but cannot be parsed raises instead of being overwritten with an empty one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_name(".manifest.lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            manifest = json.loads(path.read_text())
        except FileNotFoundError:
            manifest = {}
        manifest.setdefault("seen_file_ids", [])
        manifest.setdefault("documents", {})
        yield manifest
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(manifest, indent=2))
        os.replace(tmp, path)


# ── processed chat attachments per library ──
# Library Upload remembers the Open WebUI attachment ids it already added (a file is resent with every message). The
# default library keeps manifest["seen_file_ids"]; another library has manifest["seen_file_ids_by_collection"][<c>], so
# one attachment can be added to each library once.

def _seen_default(collection: str) -> bool:
    from memento.library.documents import LEGACY_COLLECTION
    return collection == LEGACY_COLLECTION


def seen_ids(manifest: dict, collection: str) -> set:
    if _seen_default(collection):
        return set(manifest.get("seen_file_ids") or [])
    return set(((manifest.get("seen_file_ids_by_collection") or {}).get(collection)) or [])


def add_seen_ids(manifest: dict, collection: str, ids) -> None:
    """Add `ids` to this library's list, in place (callers hold locked_manifest)."""
    if _seen_default(collection):
        manifest["seen_file_ids"] = sorted(set(manifest.get("seen_file_ids") or []) | set(ids))
        return
    by = manifest.setdefault("seen_file_ids_by_collection", {})
    by[collection] = sorted(set(by.get(collection) or []) | set(ids))


def clear_seen_ids(manifest: dict, collection: str) -> None:
    """Forget this library's list (a reset), in place; other libraries' lists stay."""
    if _seen_default(collection):
        manifest["seen_file_ids"] = []
    else:
        (manifest.get("seen_file_ids_by_collection") or {}).pop(collection, None)


@dataclass
class Decision:
    name: str            # the file name as given (upload name, or the file's name on disk)
    path: Path           # where its bytes are
    action: str          # new | unchanged | changed | renamed | duplicate
    stem: str            # the document name it is stored under; for "duplicate", the existing document
    content_hash: str

    @property
    def filename(self) -> str:
        return self.stem + Path(self.name).suffix


def namespace(input_root, output_root, exclude=()) -> set[str]:
    """Every document name the file server can find: the stems of all files under INPUT_ROOT (every corpus) and of the
    staged links in OUTPUT_ROOT/_staged, except the files in `exclude` (the batch's own files, already on disk for a
    command-line run). Built once per upload batch, so a new name never collides with another library's file."""
    skip = {Path(p).resolve() for p in exclude}
    names = set()
    for root in (Path(input_root), Path(output_root) / "_staged"):
        if root.is_dir():
            names |= {p.stem for p in root.rglob("*")
                      if (p.is_file() or p.is_symlink()) and p.resolve() not in skip}
    return names


def decide(items: Iterable[tuple[str, Path]], *, collection: str, uploader: str, tag: str, manifest: dict,
           in_library: Callable[[str], bool], taken: Callable[[str], bool] | None = None) -> list[Decision]:
    """`items`: (file name, path to its bytes). `uploader`: who adds them (user id, or SERVER_UPLOADER); `tag`: the
    readable form used when a file has to be renamed. `in_library(stem)`: does the collection hold that document?
    `taken(stem)`: is the name in the file server's namespace (namespace(), any corpus)?

    One name per document across libraries: the candidates are the file's own name, <name>__<tag>,
    <name>__<tag>-<uploader>, then <name>-2, -3 …; at each one, a document of this library under that name with the same
    content (or none recorded) is "unchanged", this uploader's own one with other content a new version ("changed"); a
    name held by another person's document, another library, an earlier file of this batch or any other file the file
    server can find is skipped. This library's own manifest entries (a failed parse to retry) never count as taken."""
    docs = manifest.get("documents", {})

    def recorded(stem: str) -> str | None:
        meta = docs.get(stem) or {}
        if meta.get("collection") not in (None, collection):
            return None
        return meta.get("content_hash") or (meta.get("parsed_hash") if meta.get("collection") == collection else None)

    def owner(stem: str) -> str:
        return (docs.get(stem) or {}).get("uploaded_by") or SERVER_UPLOADER

    known = {}                                   # content hash → a document of this collection with that content
    for stem, meta in docs.items():
        if meta.get("collection") == collection and meta.get("content_hash"):
            known.setdefault(meta["content_hash"], stem)
    batch: dict[str, str] = {}                   # content hash → name chosen earlier in this batch
    batch_names: dict[str, str] = {}             # name chosen earlier in this batch → its content hash
    library: dict[str, bool] = {}

    def present(stem: str) -> bool:
        if stem not in library:
            library[stem] = bool(in_library(stem))
        return library[stem]

    def elsewhere(stem: str) -> bool:
        """Held by something other than this library's own document of that name."""
        meta = docs.get(stem)
        if meta is not None:
            return (meta or {}).get("collection") not in (None, collection)     # this library's entry: ours
        return bool(taken and taken(stem))

    def candidates(stem: str, in_batch: bool):
        if not in_batch:                         # two different files with one name in one batch: numbers only
            yield stem, "new"
            yield f"{stem}__{person_tag(tag)}", "renamed"
            yield f"{stem}__{person_tag(tag)}-{re.sub(r'[^a-z0-9]', '', uploader.lower())[:6] or 'x'}", "renamed"
        n = 2
        while True:
            yield f"{stem}-{n}", "new" if in_batch else "renamed"
            n += 1

    out = []
    for name, path in items:
        h, stem = file_hash(path), Path(name).stem
        if h in batch:                           # the same content earlier in this upload, whatever its name (a file
                                                 # attached twice)
            out.append(Decision(name, path, "duplicate", batch[h], h))
            continue
        other = known.get(h)
        if other and other != stem and (other in batch_names or present(other)):
            out.append(Decision(name, path, "duplicate", other, h))
            continue
        for cand, fresh in candidates(stem, stem in batch_names):
            if cand in batch_names:
                continue
            if present(cand):
                old = recorded(cand)
                if old is None or old == h:      # the same document (no recorded hash: unchanged)
                    stem, action = cand, "unchanged"
                    break
                if owner(cand) == uploader:      # this person's own document: a new version
                    stem, action = cand, "changed"
                    break
                continue                         # another person's document
            if elsewhere(cand):
                continue
            stem, action = cand, fresh
            break
        batch[h] = stem
        batch_names[stem] = h
        out.append(Decision(name, path, action, stem, h))
    return out


def record(manifest: dict, decisions: Iterable[Decision], *, collection: str, uploader: str) -> None:
    """Note content hash, collection and (for new names) the uploader. Called when the decision is made; a later
    failure is harmless because "in the library" is checked against the collection, not the manifest."""
    docs = manifest.setdefault("documents", {})
    for d in decisions:
        if d.action == "duplicate":
            continue
        meta = docs.setdefault(d.stem, {})
        meta.update({"content_hash": d.content_hash, "collection": collection})
        if d.action in ("new", "renamed") or (d.action == "changed" and "uploaded_by" not in meta):
            meta["uploaded_by"] = uploader
