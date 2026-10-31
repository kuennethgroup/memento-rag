"""Per-corpus document registry (evaluation/RQ2/build/build_document_registry.py output)
and the graph's document policy: which documents may create entities and
which work a document belongs to."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from memento.config.paths import PROJECT_ROOT

log = logging.getLogger(__name__)
_DATA = PROJECT_ROOT / "data"
NON_RESEARCH_TYPES = frozenset({"image_artifact", "admin", "misc"})


def registry_path(domain: str) -> Path:
    return _DATA / f"document_registry_{domain}.json"


def load_document_registry(domain: str, path: Path | None = None) -> dict[str, dict]:
    p = Path(path) if path else registry_path(domain)
    if not p.exists():
        log.warning("document registry %s missing — every document treated as research", p)
        return {}
    return json.loads(p.read_text())


def is_person_folder(name: str | None) -> bool:
    """Doctoral folders under "8 Doktoranden" are named after the person
    ("Garcia, Pablo", "Schulz Michael"); archive folders start with a digit
    ("1_Promotionsordnung")."""
    return bool(name) and bool(name.strip()) and not name.strip()[0].isdigit()


def extracts_entities(info: dict | None) -> bool:
    """Non-research files, duplicate copies and chapter files that are copies
    of a dissertation in the corpus (`copy_of`, set by the registry builder
    from text containment) create no entities: copies would double document
    frequencies and create false cross-document bridges."""
    if not info:
        return True
    return (info.get("type") not in NON_RESEARCH_TYPES
            and not info.get("duplicate_of") and not info.get("copy_of"))


def work_id(source: str, info: dict | None) -> str:
    info = info or {}
    return info.get("duplicate_of") or info.get("parent") or info.get("parent_diss") or source


COPY_CONTAINMENT = 0.5   # share of a chapter's 8-grams found in its parent → treat as a copy


def containment(a_texts: list[str], b_texts: list[str], n: int = 8) -> float:
    """Share of document A's word n-grams that also occur in document B."""
    import re

    def grams(texts):
        out = set()
        for t in texts:
            w = re.findall(r"\w+", (t or "").lower())
            out.update(" ".join(w[i:i + n]) for i in range(len(w) - n + 1))
        return out
    a = grams(a_texts)
    if not a:
        return 0.0
    return len(a & grams(b_texts)) / len(a)
