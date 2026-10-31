#!/usr/bin/env python3
"""
Build a document registry that maps Qdrant source names to logical documents.

The corpus has 732 "sources" but far fewer real documents — dissertations
split into chapter files, defense slides, duplicate versions, images, CVs,
and admin docs all got separate source entries.

This script:
1. Classifies each source by type (dissertation, paper, defense, chapter, image, admin)
2. Groups related sources into logical documents (chapter fragments → parent dissertation)
3. Identifies duplicates (same dissertation, different version)
4. Writes a registry JSON for Stage 1 gap synthesis to count per-document, not per-source

Usage:
    source .venv/bin/activate
    python evaluation/RQ2/build/build_document_registry.py --domain polymer-engineering   # -> data/document_registry_<domain>.json
"""

import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
if __name__ == "__main__":
    from memento.config.cli_domain import apply_domain_arg
    apply_domain_arg()
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[3] / "config" / ".env")

from memento.config.settings import INPUT_DIR   # noqa: E402  the domain's INPUT_ROOT/<corpus_dir>


# ── Classification patterns ──────────────────────────────────────────────────

def classify_source(name: str, chunk_count: int) -> str:
    """Classify a source name into a document type."""
    nl = name.lower().strip()

    # Published papers (external)
    if nl.startswith("openalex_w"):
        return "paper"
    if nl.startswith("s2_"):
        return "paper"

    # Student theses from PolymerEngineering (numbered IDs)
    if re.match(r"^21\s+\d{7}\s+", name):
        return "student_thesis"

    # Image/artifact files (very few chunks, image-like names)
    if chunk_count <= 2:
        return "image_artifact"

    # Admin documents
    admin_patterns = [
        r"^cv[_\s]", r"lebenslauf", r"publikationsliste", r"antrag",
        r"freigabe", r"deckblatt", r"deckbl", r"umschlag", r"impressum",
        r"^eidess", r"^briefu", r"promotionsordnung", r"wegweiser",
        r"bekanntmachung", r"^muster", r"^absatz",
    ]
    if any(re.search(p, nl) for p in admin_patterns):
        return "admin"

    # Abbreviation lists, tables of contents, etc.
    if any(x in nl for x in ["abkürzung", "inhaltsverzeichnis", "inhalt~",
                              "verzeichnis", "literaturverzeichnis"]):
        return "admin"

    # Defense presentations / colloquia
    defense_patterns = [
        r"vortrag", r"verteidigung", r"kolloquium", r"promotionsvortrag",
        r"promotionskolloquium", r"abschlussvortrag", r"phd.*(colloquium|defense)",
        r"präsentation", r"presentation", r"diss\s+pr.*sentation",
    ]
    if any(re.search(p, nl) for p in defense_patterns):
        return "defense"

    # Dissertations (Diss_Klein, Diss_Lindemann, etc. — underscore after Diss is a word char)
    diss_patterns = [
        r"\bdiss[_\s\b]", r"^diss$", r"dissertation", r"phd.thesis",
        r"habilitationsschrift", r"printmaster\s+final",
    ]
    if any(re.search(p, nl) for p in diss_patterns):
        return "dissertation"

    # Chapter fragments (numbered chapters)
    # Large sources starting with digits are likely full dissertations with date prefixes
    # (a date or number before Diss_…), not chapters
    chapter_patterns = [
        r"^\d{1,2}[_\.\s]",            # 01_Zielsetzung, 04_3_Schaumextrusion, 5. Nano...
        r"^[a-p]_kap\d",               # a_Kap0_Deckbl_ok
        r"^k\d{1,2}_",                  # K1_Einleitung
    ]
    if any(re.match(p, nl) for p in chapter_patterns):
        if chunk_count >= 300:
            return "dissertation"
        if chunk_count < 10 and any(x in nl for x in ["lebenslauf", "literatur",
                                                        "anhang", "summary",
                                                        "zusammenfassung"]):
            return "admin"
        return "chapter"

    # Short-name truncated files (Windows 8.3 artifacts like GLIEDE~1)
    if re.match(r"^[A-Z]{6}~\d$", name):
        return "chapter"

    # Standalone scientific documents (books, theses, reports with substantial content)
    if chunk_count >= 50:
        return "document"

    # Small misc files
    if chunk_count < 10:
        return "misc"

    return "document"


# ── Filesystem-based grouping ────────────────────────────────────────────────

def build_stem_to_student(root: Path | None = None) -> dict[str, str]:
    """Ingested stem → person folder, from the corpus folder tree (walked recursively, any depth).

    The person is the first folder in a file's path that looks like a name
    (memento.graph.documents.is_person_folder: "8 Doktoranden" starts with a digit, so under it
    "Garcia, Pablo" is the person). Keys are the stems the pipeline actually ingests: files whose
    names collide get the folder-prefixed name from memento.ingestion.convert.resolve_stem_collisions.
    """
    from memento.graph.documents import is_person_folder
    from memento.ingestion.convert import discover_files, resolve_stem_collisions
    root = root or INPUT_DIR
    if not root.exists():
        return {}
    # the same file list and collision renaming the ingestion pipeline uses, so keys match Qdrant sources
    _, name_map = resolve_stem_collisions(discover_files(root), root)
    stem_student = {}
    for f, flat_name in name_map.items():
        person = next((part for part in f.relative_to(root).parent.parts if is_person_folder(part)), None)
        if person:
            stem_student[Path(flat_name).stem] = person
    return stem_student


def build_stem_to_path() -> dict[str, list[Path]]:
    """Map file stems to all their filesystem locations."""
    stem_paths: dict[str, list[Path]] = defaultdict(list)
    for f in INPUT_DIR.rglob("*"):
        if f.is_file() and "__MACOSX" not in str(f) and not f.name.startswith("._"):
            stem_paths[f.stem].append(f)
    return dict(stem_paths)


# ── Grouping heuristics ──────────────────────────────────────────────────────

def group_chapters(sources: dict[str, dict], stem_student: dict[str, str]) -> None:
    """Group chapter-type sources under their parent document."""

    # Group by student (filesystem-based)
    student_sources: dict[str, list[str]] = defaultdict(list)
    for src, info in sources.items():
        student = stem_student.get(src, stem_student.get(src.strip()))
        if student:
            from memento.graph.documents import is_person_folder
            info["folder"] = student
            # only a doctoral person folder ("Surname, First") names an author;
            # "1_Promotionsordnung" and similar are just archive folders
            if is_person_folder(student):
                info["student"] = student
            student_sources[student].append(src)

    # For each student, find the main dissertation and mark chapters
    for student, srcs in student_sources.items():
        dissertations = [s for s in srcs if sources[s]["type"] == "dissertation"]
        chapters = [s for s in srcs if sources[s]["type"] == "chapter"]

        if dissertations:
            # Pick the largest dissertation as the main one
            main_diss = max(dissertations, key=lambda s: sources[s]["chunks"])
            for ch in chapters:
                sources[ch]["parent"] = main_diss
                sources[ch]["document_id"] = main_diss
            # Mark non-main dissertations as duplicates
            for d in dissertations:
                if d != main_diss:
                    sources[d]["duplicate_of"] = main_diss
                    sources[d]["document_id"] = main_diss
            # Mark defense slides
            defenses = [s for s in srcs if sources[s]["type"] == "defense"]
            for d in defenses:
                sources[d]["parent_diss"] = main_diss
        elif chapters:
            # No full dissertation found — group chapters under a synthetic ID
            synthetic_id = f"_chapters_{student}"
            for ch in chapters:
                sources[ch]["parent"] = synthetic_id
                sources[ch]["document_id"] = synthetic_id


def detect_duplicates(sources: dict[str, dict]) -> None:
    """Detect duplicate documents (Print vs Print_SW, X vs X Kopie, etc.).

    Only matches within the same type to avoid merging a defense presentation
    with a dissertation that shares a normalized stem.
    """
    for doc_type in ("dissertation", "document", "defense"):
        typed_sources = {s: info for s, info in sources.items()
                        if info["type"] == doc_type and "duplicate_of" not in info}

        normalized: dict[str, list[str]] = defaultdict(list)
        for src in typed_sources:
            norm = src.lower().strip()
            for suffix in ["_sw", " kopie", "_kopie", " copy", "_print",
                           "_druck", "_100", "_final", "_druckversion",
                           "_kompr", "_compr", "_mit markierungen",
                           "_drucklegung", "_abgabe_final", "_kl"]:
                norm = norm.removesuffix(suffix)
            normalized[norm].append(src)

        for norm, group in normalized.items():
            if len(group) > 1:
                main = max(group, key=lambda s: sources[s]["chunks"])
                for s in group:
                    if s != main:
                        sources[s]["duplicate_of"] = main
                        sources[s]["document_id"] = main


DUPLICATE_CONTAINMENT = 0.5   # a duplicate must share most of its text with its original


def confirm_duplicates(sources: dict[str, dict], texts) -> None:
    """Keep a folder-heuristic `duplicate_of` only if one document's text is
    mostly contained in the other (max of both directions: which copy is
    "main" is chosen by size, so the direction is arbitrary). Measured
    on the polymer corpus: real copies 0.98–1.00, an earlier draft
    0.67, misfiled defence talks/presentations 0.00–0.07 — any threshold in
    0.1–0.65 separates them; 0.5 = "most of the text" is the domain-agnostic
    definition. Rejected pairs become ordinary research documents."""
    from memento.graph.documents import containment
    for src, info in sources.items():
        orig = info.get("duplicate_of")
        if not orig or orig not in sources:
            continue
        a, b = texts(src), texts(orig)
        share = max(containment(a, b), containment(b, a))
        info["duplicate_overlap"] = round(share, 3)
        if share < DUPLICATE_CONTAINMENT:
            del info["duplicate_of"]
            info["duplicate_rejected_overlap"] = round(share, 3)
            info["document_id"] = src


def mark_chapter_copies(sources: dict[str, dict], client, collection: str) -> None:
    """Chapter files are mostly drafts of their dissertation (60-90% shared
    8-grams, measured). A chapter whose text is largely contained
    in a parent dissertation present in the corpus is marked `copy_of` it —
    graph/documents.extracts_entities then gives it no entities, like a
    duplicate. The measured share is kept for the same-work audit."""
    from qdrant_client.models import FieldCondition, Filter, MatchValue
    from memento.graph.documents import COPY_CONTAINMENT, containment

    def texts(src):
        out, off = [], None
        while True:
            pts, off = client.scroll(collection, scroll_filter=Filter(must=[FieldCondition(
                key="source", match=MatchValue(value=src))]), limit=2000, offset=off, with_payload=["text"])
            out += [pt.payload.get("text") or "" for pt in pts]
            if off is None:
                return out

    cache: dict[str, list[str]] = {}
    for src, info in sources.items():
        parent = info.get("parent")
        if info["type"] != "chapter" or parent not in sources:
            continue
        if parent not in cache:
            cache[parent] = texts(parent)
        share = containment(texts(src), cache[parent])
        info["overlap_with_parent"] = round(share, 3)
        if share >= COPY_CONTAINMENT:
            info["copy_of"] = parent


# ── Main ─────────────────────────────────────────────────────────────────────

def build_registry(collection: str, out_path: Path):
    from qdrant_client import QdrantClient
    from memento.config.settings import QDRANT_URL

    client = QdrantClient(url=QDRANT_URL, timeout=120)

    # Count chunks per source
    chunk_counts: Counter = Counter()
    offset = None
    while True:
        results = client.scroll(
            collection_name=collection, limit=1000, offset=offset,
            with_payload=["source"], with_vectors=False,
        )
        points, offset = results
        if not points:
            break
        for p in points:
            chunk_counts[p.payload.get("source", "")] += 1
        if offset is None:
            break

    print(f"Total sources in Qdrant: {len(chunk_counts)}")

    # Classify each source
    sources: dict[str, dict] = {}
    for src, cnt in chunk_counts.items():
        sources[src] = {
            "type": classify_source(src, cnt),
            "chunks": cnt,
            "document_id": src,  # default: self
        }

    # Show type distribution
    type_counts = Counter(s["type"] for s in sources.values())
    print("\n── Source type distribution ──")
    for t, n in type_counts.most_common():
        chunks = sum(s["chunks"] for s in sources.values() if s["type"] == t)
        print(f"  {t:20s}  {n:4d} sources  {chunks:6d} chunks")

    # Build filesystem mappings
    stem_student = build_stem_to_student()
    print(f"\nFilesystem: {len(stem_student)} stems mapped to students")

    # Group chapters and detect duplicates
    group_chapters(sources, stem_student)
    detect_duplicates(sources)
    from qdrant_client.models import FieldCondition, Filter, MatchValue
    _cache: dict[str, list[str]] = {}

    def texts(src: str) -> list[str]:
        if src not in _cache:
            out, off = [], None
            while True:
                pts, off = client.scroll(collection, scroll_filter=Filter(must=[FieldCondition(
                    key="source", match=MatchValue(value=src))]), limit=2000, offset=off, with_payload=["text"])
                out += [pt.payload.get("text") or "" for pt in pts]
                if off is None:
                    break
            _cache[src] = out
        return _cache[src]
    confirm_duplicates(sources, texts)
    mark_chapter_copies(sources, client, collection)

    # Assign document IDs to ungrouped sources
    for src, info in sources.items():
        if "document_id" not in info or info["document_id"] == src:
            info["document_id"] = src

    # Count logical documents (unique document_ids, excluding images/admin)
    exclude_types = {"image_artifact", "admin", "misc"}
    doc_ids = set()
    for src, info in sources.items():
        if info["type"] not in exclude_types:
            doc_ids.add(info.get("document_id", src))

    grouped = sum(1 for s in sources.values()
                  if s.get("parent") or s.get("duplicate_of"))

    print(f"\n── Deduplication results ──")
    print(f"  Original sources:     {len(sources)}")
    print(f"  Grouped/duplicated:   {grouped}")
    print(f"  Excluded (img/admin): {sum(1 for s in sources.values() if s['type'] in exclude_types)}")
    print(f"  Logical documents:    {len(doc_ids)}")

    # Write registry
    registry_path = out_path
    with open(registry_path, "w") as f:
        json.dump(sources, f, indent=2, ensure_ascii=False)
    print(f"\nRegistry written to {registry_path}")

    # Show some examples of grouping
    print("\n── Example groupings ──")
    students_shown = set()
    for src, info in sorted(sources.items()):
        student = info.get("student")
        if student and student not in students_shown and info.get("parent"):
            students_shown.add(student)
            siblings = [(s, i) for s, i in sources.items()
                        if i.get("student") == student]
            print(f"\n  Student: {student}")
            for s, i in siblings:
                role = ""
                if i.get("duplicate_of"):
                    role = f" [DUPLICATE of {i['duplicate_of'][:40]}]"
                elif i.get("parent"):
                    role = f" [CHAPTER of {i['parent'][:40]}]"
                elif i["type"] == "defense":
                    role = f" [DEFENSE]"
                elif i["type"] == "admin":
                    role = f" [ADMIN]"
                elif i["type"] == "dissertation":
                    role = f" [MAIN DISS]"
                print(f"    {i['chunks']:4d} ch  {i['type']:15s}  {s[:60]}{role}")
            if len(students_shown) >= 5:
                break

    # Show duplicates
    print("\n── Detected duplicates ──")
    for src, info in sorted(sources.items()):
        if info.get("duplicate_of"):
            print(f"  {src[:50]:50s} → {info['duplicate_of'][:50]}")


if __name__ == "__main__":
    import argparse
    from memento.config.settings import COLLECTION
    from memento.graph.documents import registry_path as _default_registry_path
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--collection", default=COLLECTION)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    build_registry(a.collection, a.out or _default_registry_path(a.domain))
