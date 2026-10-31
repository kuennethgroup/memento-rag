"""Shared pieces for the RQ2 question builders."""
from __future__ import annotations

import dataclasses
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PILOT_DIR = ROOT / "data" / "rq2"
EXCLUDED_SECTIONS = frozenset({"References", "Front Matter", "Declarations", "Appendix",
                               "Abbildungsverzeichnis", "Tabellenverzeichnis", "Inhaltsverzeichnis"})
# List sections under any heading (an abbreviation list labelled "Nomenclature" must not answer a question). "notations"
# is plural on purpose: "Space Group Notation" is content.
_LIST_SECTION_RE = re.compile(r"abkürzung|abbreviation|nomenclature|nomenklatur|symbol|glossar|acronym|notations\b"
                              r"|formelzeichen|verzeichnis|\bindices\b", re.I)


def is_excluded_section(section) -> bool:
    return bool(section) and (section in EXCLUDED_SECTIONS or _LIST_SECTION_RE.search(section) is not None)


@dataclass
class Question:
    qid: str
    domain: str
    qtype: str
    question: str
    reference_answer: str
    lang: str
    universe: str
    gold_pages: list = field(default_factory=list)
    gold_docs: list = field(default_factory=list)
    gold_works: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)


def write_jsonl(path: Path, qs: list[Question]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(dataclasses.asdict(q), ensure_ascii=False) + "\n" for q in qs))


def read_jsonl(path: Path) -> list[Question]:
    return [Question(**json.loads(l)) for l in Path(path).read_text().splitlines() if l.strip()]


def universe_ids(domain: str, universe: str) -> set[str] | None:
    if universe == "full":
        return None
    lines = (PILOT_DIR / f"pilot_{domain}.txt").read_text().splitlines()
    return {l for l in lines if l.strip()}


_CHUNK_CACHE: dict[str, list[dict]] = {}


def load_chunks(collection: str, *, client=None) -> list[dict]:
    if collection in _CHUNK_CACHE:
        return _CHUNK_CACHE[collection]
    if client is None:
        from qdrant_client import QdrantClient
        from memento.config.settings import QDRANT_URL
        client = QdrantClient(url=QDRANT_URL, timeout=120)
    rows, off = [], None
    while True:
        pts, off = client.scroll(collection, limit=5000, offset=off,
                                 with_payload=["source", "page", "type", "section", "text"])
        rows += [{"point_id": str(p.id), "source": p.payload.get("source"), "page": p.payload.get("page", 0),
                  "type": p.payload.get("type"), "section": p.payload.get("section"),
                  "text": p.payload.get("text") or ""} for p in pts]
        if off is None:
            break
    _CHUNK_CACHE[collection] = rows
    return rows


def content_chunks(chunks, registry: dict, allowed: set[str] | None) -> list[dict]:
    from memento.graph.documents import extracts_entities
    return [c for c in chunks
            if (c["text"] or "").strip()
            and not is_excluded_section(c.get("section"))
            and extracts_entities(registry.get(c["source"]))
            and (allowed is None or c["source"] in allowed)]


def llm_json(client, model: str, system: str, user: str, *, temperature: float = 0.0, max_tokens: int = 3000):
    r = client.chat.completions.create(
        model=model, temperature=temperature, max_tokens=max_tokens,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    raw = re.sub(r"<think>.*?</think>", "", r.choices[0].message.content or "", flags=re.S)
    m = re.search(r"```(?:json)?\s*(.*?)```", raw, re.S)
    body = m.group(1) if m else raw
    pairs = sorted((("{", "}"), ("[", "]")),
                   key=lambda oc: body.find(oc[0]) if body.find(oc[0]) >= 0 else len(body))
    for opener, closer in pairs:                 # the span that starts first wins
        a, b = body.find(opener), body.rfind(closer)
        if a >= 0 and b > a:
            try:
                return json.loads(body[a:b + 1])
            except json.JSONDecodeError:
                continue
    return None


_DE = {"der", "die", "das", "und", "mit", "von", "für", "ist", "wird", "werden", "eine", "einer", "den", "dem"}


def lang_of(texts: list[str]) -> str:
    words = re.findall(r"[A-Za-zÄÖÜäöüß]+", " ".join(texts).lower())
    return "de" if words and sum(w in _DE for w in words) / len(words) >= 0.02 else "en"


LANG_NAME = {"en": "English", "de": "German"}
# Questions must not reveal how they were built ("in Passage A genannte …").
_META_RE = re.compile(r"\b(?:passages?|passagen?)\b", re.I)


def language_instruction(lang: str) -> str:
    return f"Write the question and the answer in {LANG_NAME[lang]}."


def question_problem(question: str, lang: str) -> str | None:
    """Deterministic post-check of a composed question: the model must write in the
    target language and must not refer to the passages it was built from."""
    if _META_RE.search(question):
        return "meta_reference"
    if lang_of([question]) != lang:
        return "language"
    return None
