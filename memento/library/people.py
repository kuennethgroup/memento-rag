# memento/library/people.py: the people a document names, read once by the library's answer model
"""The front-matter rules (memento/library/front_matter.py) read a paper's author line, but thesis and talk title pages name
people in other ways ("Dissertation von …", "Betreut von: …"). The library's answer model reads each document's stored
front matter and its library title once (no thinking) and lists its authors (a thesis' candidate, a talk's speaker) and,
apart, its supervisors and reviewers. A name is kept only when the text has its surname, so the model cannot add a
person. Kept per document version in `memento_doc_people`; documents without a row keep the front-matter rules. Filled
by each library's worker while the model is idle (query_pipe._fill_people), a few documents per round."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, MetaData, String, Table, Text, delete, insert, select

from memento.library import front_matter as da

METADATA = MetaData()
PEOPLE = Table(
    "memento_doc_people", METADATA,
    Column("collection", String(200), primary_key=True),
    Column("source", String(500), primary_key=True),
    Column("content_hash", String(128)),
    Column("authors", Text),          # JSON list of names; NULL: the model's reply could not be read (not asked again)
    Column("supervisors", Text),
    Column("model", String(200)),
    Column("updated_at", DateTime(timezone=True)),
)

SYSTEM_PROMPT = ("You read the first page of a document from a university chair's library (a thesis, the slides of a "
                 "talk, a paper, a book or a chapter) and list the people it names as its authors and as its "
                 "supervisors or reviewers. Reply with JSON only.")
USER_PROMPT = """The document's title in the library: {title}
Its file name: {source}

Its first page:
<<<
{front}
>>>

Reply with JSON only: {{"authors": [...], "supervisors": [...]}}
- "authors": the people who wrote the document: for a thesis the candidate, for a talk the speaker, for a paper or a book
  every author. When the first page names nobody, a person's name in the file name or in brackets in the library title
  is the author (for example "Vortrag_Lindemann_100607", "Dissertation (Weber)", "Chapter 1 (Lindemann)").
- "supervisors": the people named as supervisor, advisor, reviewer or examiner (Betreuer, betreut von, Gutachter,
  Erstgutachter, Zweitgutachter, Berichterstatter, Prüfer, Vorsitzender).
- Each name with the first name(s) first, then the surname, without academic titles, degrees or affiliations
  ("Prof. Dr.-Ing.", "Dipl.-Ing.", "M.Sc.").
- Leave out people who are only thanked, cited or mentioned in the text. Empty lists when nobody is named."""

# academic titles and degrees, folded and without punctuation ("Dr.-Ing." → "dring")
_TITLES = {"prof", "dr", "dring", "ing", "dipl", "dipling", "diplchem", "diplphys", "diplinf", "chem", "msc", "bsc",
           "meng", "beng", "mba", "phd", "rer", "nat", "drrernat", "phil", "drphil", "habil", "med", "jun"}


def ensure_table(engine) -> None:
    METADATA.create_all(engine, tables=[PEOPLE])


def _clean_name(name) -> str:
    """A name without titles, degrees, a birth name in brackets or stray punctuation ("Dr.-Ing. Anna Muster (geb.
    Beispiel)" → "Anna Muster")."""
    if not isinstance(name, str):
        return ""
    name = re.sub(r"\([^)]*\)", " ", name)
    words = []
    for w in re.findall(r"[^\s,;]+", name):
        if re.sub(r"[^a-z]", "", da._fold(w)) in _TITLES:
            continue
        w = w.strip(",;:")
        if w.strip("."):
            words.append(w)
    while len(words) > 1 and re.fullmatch(r"[^\W\d_]\.?", words[-1]):   # a trailing initial ("Anna M. Muster B.")
        words.pop()
    return " ".join(w if re.fullmatch(r"[^\W\d_]\.", w) else w.strip(".") for w in words)


def _in_text(name: str, folded_text: str) -> bool:
    surname = re.sub(r"[^a-z\-]", "", da._fold(name.split()[-1])) if name.split() else ""
    return len(surname) >= 2 and re.search(rf"(?<![a-z]){re.escape(surname)}(?![a-z])", folded_text) is not None


def parse_people(reply: str, front: str, title: str = "", source: str = "") -> tuple[list[str], list[str]] | None:
    """(authors, supervisors) from the model's reply, or None when it cannot be read. Names the text (front matter,
    library title, file name) does not have are dropped; a person listed as an author is not also a supervisor."""
    text = re.sub(r"(?s)<think>.*?</think>", "", reply or "")
    m = re.search(r"(?s)\{.*\}", text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except ValueError:
        return None
    if not isinstance(obj, dict) or not isinstance(obj.get("authors"), list):
        return None
    sups = obj.get("supervisors") if isinstance(obj.get("supervisors"), list) else []
    folded = da._fold(f"{title}\n{source}\n{front}").replace("_", " ")
    out: list[list[str]] = [[], []]
    seen: set = set()
    for k, names in enumerate((obj["authors"], sups)):
        for raw in names:
            name = _clean_name(raw)
            key = da._fold(name)
            if not name or key in seen or not _in_text(name, folded):
                continue
            seen.add(key)
            out[k].append(name)
    return out[0], out[1]


def put(engine, *, collection: str, source: str, content_hash: str | None, authors, supervisors, model: str) -> bool:
    """Store one document's people, only for the version in the library (its row's fingerprint known and equal, as
    doc_authors.put); authors None: the reply could not be read (kept, so it is not asked again)."""
    from memento.library import documents as ldb
    if not content_hash:
        return False
    with engine.begin() as c:
        have = c.execute(select(ldb.DOCUMENTS.c.content_hash).where(
            ldb.DOCUMENTS.c.collection == collection, ldb.DOCUMENTS.c.source == source)).scalar()
        if have != content_hash:
            return False
        c.execute(delete(PEOPLE).where(PEOPLE.c.collection == collection, PEOPLE.c.source == source))
        c.execute(insert(PEOPLE).values(
            collection=collection, source=source, content_hash=content_hash,
            authors=None if authors is None else json.dumps(authors, ensure_ascii=False),
            supervisors=None if supervisors is None else json.dumps(supervisors, ensure_ascii=False),
            model=model, updated_at=datetime.now(timezone.utc)))
    return True


def _rows(engine, collection: str) -> dict:
    with engine.connect() as c:
        rows = c.execute(select(PEOPLE.c.source, PEOPLE.c.content_hash, PEOPLE.c.authors, PEOPLE.c.supervisors)
                         .where(PEOPLE.c.collection == collection)).all()
    return {r.source: r for r in rows}


def current(engine, collection: str, library: dict) -> dict:
    """{source: (authors, supervisors)} of the library's documents read for their current version (`library`:
    {source: content_hash}); unread, older or unreadable ones are left out (the front-matter rules decide for them)."""
    out = {}
    for s, r in _rows(engine, collection).items():
        h = library.get(s)
        if h and r.content_hash == h and r.authors is not None:
            out[s] = (json.loads(r.authors), json.loads(r.supervisors or "[]"))
    return out


def todo(engine, collection: str, library: dict) -> list[str]:
    """The library's documents whose current version has not been read yet (an unreadable reply counts as read)."""
    rows = _rows(engine, collection)
    return sorted(s for s, h in library.items() if h and (s not in rows or rows[s].content_hash != h))


def classify(people: dict, person: str) -> dict:
    """{"author": [...], "supervisor": [...], "possible_identity": [...]}: the documents by how they name `person`
    (the name rules of doc_authors: a full match, or only an initial or a surname on record)."""
    surname, first = da._person_parts(person)
    out: dict = {"author": [], "supervisor": [], "possible_identity": []}
    if len(surname) < 2:
        return out
    def kinds(names):
        # a name written surname first ("Muster Anna") is tried both ways
        out_k = []
        for n in names:
            words = re.findall(r"[^\s,]+", n)
            k = da._entry_kind(words, surname, first, set())
            if k != "author" and len(words) == 2:
                k = "author" if da._entry_kind(words[::-1], surname, first, set()) == "author" else k
            out_k.append(k)
        return out_k
    for src, (authors, supervisors) in people.items():
        kinds_a, sup = kinds(authors), kinds(supervisors)
        if "author" in kinds_a:
            out["author"].append(src)
        elif "author" in sup:
            out["supervisor"].append(src)
        elif "possible_identity" in kinds_a or "possible_identity" in sup:
            # only an initial on record ("V Sandhäger" in a talk's footer): possibly the same person, said, not dropped
            out["possible_identity"].append(src)
    return {k: sorted(v) for k, v in out.items()}


def messages(front: str, title: str, source: str = "") -> list[dict]:
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(title=title or "(none)", source=source or "(none)",
                                                           front=front)}]


def fill(engine, *, collection: str, library: dict, fronts: dict, titles: dict, client, model: str,
         limit: int) -> tuple[int, int]:
    """Read up to `limit` documents not read yet (those with front matter) with `client` (OpenAI-compatible, sync):
    (stored with names, failed: a failed call, tried again later, or an unreadable reply, kept as such)."""
    stored = failed = asked = 0
    for source in todo(engine, collection, library):
        front = fronts.get(source)
        if not front:
            continue
        if asked >= limit:
            break
        asked += 1
        try:
            r = client.chat.completions.create(model=model, temperature=0, max_tokens=600,
                                               extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                                               messages=messages(front, titles.get(source) or "", source))
            parsed = parse_people(r.choices[0].message.content or "", front, titles.get(source) or "", source)
        except Exception:
            failed += 1
            continue
        if parsed is None:
            failed += 1
            put(engine, collection=collection, source=source, content_hash=library.get(source), authors=None,
                supervisors=None, model=model)
            continue
        if put(engine, collection=collection, source=source, content_hash=library.get(source), authors=parsed[0],
               supervisors=parsed[1], model=model):
            stored += 1
    return stored, failed
