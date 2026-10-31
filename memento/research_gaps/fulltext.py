"""Full texts of papers held in an offline database (ALEXANDRIA_DB), for the literature check.

Limitations and untested conditions rarely appear in abstracts. For the papers the check already judges, this reads the
paper's own text when the database holds it and gives the judge the passages that best match the question, beside the
abstract. The database is only read: nothing is written, moved or added there. A file being present is not full text:
a publisher's XML can hold metadata or citations only, and a PDF can be a scan. Such a paper is judged from its abstract.

Off unless LITERATURE_FULLTEXT=1, ALEXANDRIA_DB is set and the lookup exists (scripts/build_fulltext_paths.py).
"""
from __future__ import annotations

import contextlib
import os
import re
import sqlite3
import xml.etree.ElementTree as ET
from pathlib import Path

from memento.common.pdf import PYMUPDF
from memento.config.paths import PROJECT_ROOT

INDEX = PROJECT_ROOT / "data" / "alexandria" / "fulltext_paths.sqlite"
MIN_BODY = 3000             # shorter than this is not a paper's text
PASSAGE_CHARS = 1200
PREFILTER = 15              # passages per paper that reach the reranker
TOP_PASSAGES = 3
NEAR_COPY = 0.8             # share of a passage's words in the abstract: it says nothing new
MARK = "From the full text (excerpts):"
REFERENCES = re.compile(r"(?im)^[#*\s\d.]*(references|bibliography|literature cited|notes and references)[*:\s]*$")
HEADER = re.compile(r"©|\bwww\.", re.I)     # two of these opening a paragraph: a page header or title block
BODY_TAGS = {"body"}
PARA_TAGS = {"p", "para", "simple-para"}


def enabled(index=None) -> bool:
    return (os.getenv("LITERATURE_FULLTEXT", "0") == "1" and bool(os.getenv("ALEXANDRIA_DB"))
            and Path(index or INDEX).is_file())


def paper_folder(doi: str, *, index=None, root=None) -> Path | None:
    """The paper's folder in the database, or None when it is not held. DOIs in the lookup are lower case."""
    from memento.research_gaps.literature import _doi
    key = re.sub(r"^doi:\s*", "", _doi(doi))
    if not key:
        return None
    with contextlib.closing(sqlite3.connect(f"file:{index or INDEX}?mode=ro", uri=True)) as db:
        row = db.execute("SELECT path FROM papers WHERE doi = ?", (key,)).fetchone()
    if not row:
        return None
    base = Path(root or os.environ["ALEXANDRIA_DB"]).resolve()
    folder = (base / row[0]).resolve()
    return folder if folder.is_relative_to(base) and folder.is_dir() else None


def _markdown(folder: Path) -> str:
    return (folder / "paper.md").read_text(encoding="utf-8", errors="replace")


def _pdf(folder: Path) -> str:
    import fitz
    with PYMUPDF, fitz.open(folder / "Paper.pdf") as doc:
        blocks = [b[4] for page in doc for b in page.get_text("blocks") if b[6] == 0]
    return "\n\n".join(re.sub(r"-\n(?=[a-z])", "", b).replace("\n", " ").strip() for b in blocks if b.strip())


def _xml(folder: Path) -> str:
    """The body paragraphs of the publisher's XML (JATS and similar), whatever the file is called."""
    def local(tag) -> str:
        return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""
    best = ""
    for path in sorted(folder.glob("*.xml")):
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError:
            continue
        for body in (e for e in root.iter() if local(e.tag) in BODY_TAGS):
            paras = (" ".join("".join(p.itertext()).split()) for p in body.iter() if local(p.tag) in PARA_TAGS)
            text = "\n\n".join(p for p in paras if p)
            best = max(best, text, key=len)
    return best


def _body(text: str) -> str:
    cuts = [m.start() for m in REFERENCES.finditer(text) if m.start() > 0.4 * len(text)]
    return (text[:cuts[-1]] if cuts else text).strip()


def read_body(folder: Path) -> tuple[str, str] | None:
    """("md" | "pdf" | "xml", the paper's text without references), or None when no file holds a real body."""
    for kind, read in (("md", _markdown), ("pdf", _pdf), ("xml", _xml)):
        try:
            body = _body(read(Path(folder)))
        except Exception:               # a missing or damaged file: try the next kind
            continue
        if len(body) >= MIN_BODY:
            return kind, body
    return None


def passages(body: str, *, chars: int = PASSAGE_CHARS) -> list[tuple[int, int]]:
    """Offsets of passages of at most `chars`, on paragraph boundaries; a longer paragraph ends at a sentence end."""
    spans = []
    for m in re.finditer(r"\S(?:.*?\S)??(?=\s*\n\s*\n|\s*\Z)", body, flags=re.S):
        start, end = m.span()
        while end - start > chars:
            cut = body.rfind(". ", start, start + chars) + 1
            if cut <= start:
                cut = start + chars
            spans.append((start, cut))
            start = cut + (body[cut:cut + 1] == " ")
        if start < end:
            spans.append((start, end))
    merged: list[list[int]] = []
    for start, end in spans:                # short paragraphs (headings, captions) join the next
        if merged and end - merged[-1][0] <= chars and body[merged[-1][1]:start].strip() == "":
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return [(a, b) for a, b in merged]


def _terms(text: str) -> set[str]:
    return {w for w in re.findall(r"[^\W\d_]{3,}", text.lower())}


def best_passages(question: str, phrase: str, abstract: str, body: str, *, rank, k: int = TOP_PASSAGES) -> list[str]:
    """The `k` passages of `body` that best match the question: those sharing most words with it, minus near copies of
    the abstract, ordered by `rank` (texts → texts, best first)."""
    wanted, known = _terms(f"{question} {phrase}"), _terms(abstract)
    body = "\n\n".join(p for p in re.split(r"\n\s*\n", body) if len(HEADER.findall(p[:200])) < 2)
    scored = []
    for a, b in passages(body):
        text = body[a:b].strip()
        words = _terms(text)
        shared = len(words & wanted)
        if shared and not (words and len(words & known) / len(words) >= NEAR_COPY):
            scored.append((shared, text))
    scored.sort(key=lambda x: -x[0])
    candidates = [t for _, t in scored[:PREFILTER]]
    return list(rank(candidates))[:k] if candidates else []


def attach(works: list, question: str, phrase: str, *, rank, index=None, root=None) -> int:
    """Set `excerpts` on each paper the database holds in full; the others stay as they are. Returns the papers read.
    A failure on one paper leaves that paper to its abstract."""
    n = 0
    for w in works:
        w.excerpts = []
        try:
            folder = paper_folder(w.doi, index=index, root=root)
            read = read_body(folder) if folder else None
            if read:
                w.excerpts = best_passages(question, phrase, w.abstract or "", read[1], rank=rank)
                n += bool(w.excerpts)
        except Exception:
            w.excerpts = []
    return n


def ranker(question: str, reranker_url: str, instruction: str | None = None):
    """`rank` for best_passages with the production reranker."""
    from memento.retrieval.search import rerank

    def rank(texts: list[str]) -> list[str]:
        hits = [{"text": t, "type": "text"} for t in texts]
        return [h["text"] for h in rerank(question, hits, reranker_url, TOP_PASSAGES, 0.0, 0.0,
                                          instruction=instruction)]
    return rank


def judged_text(work) -> str:
    """What the judge read of a paper, as stored with the verdict."""
    excerpts = getattr(work, "excerpts", None)
    return f"{work.abstract}\n\n{MARK}\n" + "\n\n".join(excerpts) if excerpts else work.abstract


def read_in_full(stored_text: str | None) -> bool:
    return MARK in (stored_text or "")


def public_text(stored_text: str | None) -> str:
    """The stored text without its excerpts: they stay on this server, so citations and paper cards show this."""
    return (stored_text or "").split(MARK)[0].strip()
