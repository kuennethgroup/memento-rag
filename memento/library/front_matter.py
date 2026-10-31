# memento/library/front_matter.py: which documents name a person as an author
"""A person's documents are found deterministically: the name is matched against each document's front matter (the head
of its parsed text, before any references heading), kept per document in `memento_front_matter` (Open WebUI's Postgres;
SQLite in tests), written when a document is stored. No model call, so long author lists are read in full, which a
meaning-based search does not do."""
from __future__ import annotations

import glob
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import Column, DateTime, MetaData, String, Table, Text, delete, inspect, insert, select, text

FRONT_CHARS = 4000             # the head that holds title, authors and affiliations
METADATA = MetaData()
FRONT = Table(
    "memento_front_matter", METADATA,
    Column("collection", String(200), primary_key=True),
    Column("source", String(500), primary_key=True),
    Column("content_hash", String(128)),
    Column("front", Text, nullable=False),
    Column("updated_at", DateTime(timezone=True)),
)
_REFS_RE = re.compile(r"(?im)^#{1,6}\s*(references|bibliography|literatur|literaturverzeichnis|works cited)\b")
# authors come before the abstract (a model named after a person in an abstract must not match as an author)
_ABSTRACT_RE = re.compile(r"(?im)^(?:#{1,6}\s*)?(?:\d+\.?\s*)?(abstract|a b s t r a c t|summary|introduction|"
                          r"zusammenfassung|einleitung|keywords|index terms)\b")
_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LATEX_RE = re.compile(r"\$[^$]*\$")


def ensure_table(engine) -> None:
    METADATA.create_all(engine, tables=[FRONT])


def front_matter(md_text: str) -> str:
    """The head of a document's parsed text: images removed, cut before the abstract or introduction (else before the
    references), FRONT_CHARS at most."""
    text = md_text or ""
    m = _ABSTRACT_RE.search(text, 30) or _REFS_RE.search(text)
    if m:
        text = text[:m.start()]
    return _IMAGE_RE.sub("", text)[:FRONT_CHARS]


def parsed_text_path(out_dir, source: str) -> Path | None:
    """The parsed Markdown of `source` under OUTPUT_ROOT/<domain dir> (the vlm backend's first). The name is matched as
    text, not as a pattern."""
    if not out_dir:
        return None
    found = sorted((Path(out_dir) / source).glob(f"*/{glob.escape(source)}.md"), key=lambda p: p.parent.name != "vlm")
    return found[0] if found else None


def read_front(out_dir, source: str) -> str | None:
    p = parsed_text_path(out_dir, source)
    return front_matter(p.read_text(errors="replace")) if p else None


def put(engine, *, collection: str, source: str, content_hash: str | None, front: str) -> bool:
    """Insert or replace one document's front matter, only for the version in the library: its row must exist and its
    fingerprint be known and equal to this job's. Anything else (a newer version, a removed document, an unknown
    fingerprint on either side, no library table) leaves the stored list as it is (False). The check and the write share
    one transaction; with the document locks around it no other write can come between them."""
    from memento.library import locks as db_locks
    if not content_hash or not inspect(engine).has_table("memento_documents"):
        return False
    with engine.begin() as c:
        db_locks.document(c, collection, source)                     # no reset or replacement in between
        have = c.execute(text("SELECT content_hash FROM memento_documents WHERE collection = :c AND source = :s"),
                         {"c": collection, "s": source}).first()
        if have is None or have[0] != content_hash:
            return False
        c.execute(delete(FRONT).where(FRONT.c.collection == collection, FRONT.c.source == source))
        c.execute(insert(FRONT).values(collection=collection, source=source, content_hash=content_hash, front=front,
                                       updated_at=datetime.now(timezone.utc)))
    return True


def remove(engine, *, collection: str, source: str) -> None:
    with engine.begin() as c:
        c.execute(delete(FRONT).where(FRONT.c.collection == collection, FRONT.c.source == source))


def load(engine, collection: str, with_hash: bool = False) -> dict:
    """{source: front matter} of a collection; with_hash: {source: (front matter, its document's fingerprint)}."""
    with engine.connect() as c:
        rows = c.execute(select(FRONT.c.source, FRONT.c.front, FRONT.c.content_hash)
                         .where(FRONT.c.collection == collection)).all()
    return {r.source: ((r.front, r.content_hash) if with_hash else r.front) for r in rows}


def fresh(rows: dict, library: dict) -> tuple[dict, set]:
    """({source: front matter} of the lists that belong to the version in the library, {the library's other documents}).
    A list counts only when both fingerprints are known and equal: an unknown one is not proof of being current. `rows`:
    load(with_hash=True); `library`: {source: content_hash}."""
    current = {s: rows[s][0] for s, h in library.items()
               if s in rows and h and rows[s][1] and rows[s][1] == h}
    return current, set(library) - set(current)


def _fold(text: str) -> str:
    """Lower case, accents off, German umlauts and their "ae / oe / ue" spellings made the same ("Müller", "Mueller" →
    "muller"); LaTeX superscripts ($^{e}$) and escape backslashes removed."""
    text = _LATEX_RE.sub(" ", text or "").replace("\\", " ").lower().replace("ß", "ss")
    text = "".join(ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch))
    return text.replace("ae", "a").replace("oe", "o").replace("ue", "u")


# ── documents by a person, from positive evidence ──
# The author lines are found first: lines of names only (superscripts, markers and separators removed), never a line
# with a role, an organisation, an e-mail or an acronym. A document is then "author", "possible_identity" (only an
# initial matches), "possible_role" (the full name appears elsewhere in the front matter) or "no" (a title, a role, an
# institute named after a person, nothing). A heuristic.
_ROLE_RE = re.compile(r"\b(?:supervis\w*|betreu\w*|gutacht\w*|erstgutacht\w*|zweitgutacht\w*|advis\w*|referee\w*|"
                      r"prüfer\w*|pruefer\w*|editor\w*|edited|herausgeb\w*|reviewer\w*|examin\w*|mentor\w*|contact|"
                      r"correspond\w*)\b", re.I)
_ORG_RE = re.compile(r"\b(?:universit\w*|institut\w*|department|dept|faculty|fakultät|school|college|laborator\w*|"
                     r"lab|gmbh|ggmbh|inc|ltd|llc|corporation|corp|company|centre|center|zentrum|lehrstuhl|chair|"
                     r"hochschule|academy|akademie|foundation|stiftung|meta|facebook|google|microsoft)\b", re.I)
_EMAIL_RE = re.compile(r"\S+@\S+")
_SCRIPT_RE = re.compile(r"\$[^$\n]*\$")                    # $^{a,1}$, $_{a,f}$: affiliation marks become separators
_LEAD_RE = re.compile(r"^(?:by|von|vorgelegt\s+von|eingereicht\s+von|submitted\s+by|presented\s+by|written\s+by|"
                      r"authors?:?|autor(?:in|en)?:?)\s+", re.I)
_PARTICLES = {"van", "von", "de", "der", "den", "di", "da", "du", "del", "la", "le", "dos", "das", "ten", "ter", "bin"}
_HONORIFICS = {"prof", "dr", "rer", "nat", "ing", "habil", "phd", "msc", "bsc"}
_NAME_TOKEN_RE = re.compile(r"^(?:[A-ZÀ-ÖØ-Þ][\w'’.\-]*|[A-ZÀ-ÖØ-Þ]\.(?:-?[A-ZÀ-ÖØ-Þ]\.)*)$")


def _name_entries(line: str) -> list[list[str]] | None:
    """The names on a line of names only (each a list of its words), else None."""
    t = _SCRIPT_RE.sub(",", _EMAIL_RE.sub(" ", line))
    t = re.sub(r"\bet\s+al\.?", " ", t)
    t = re.sub(r"[*†‡✉§¶#\\]|\(\s*\)", " ", t)
    t = re.sub(r"(?<=\w)\d+\b", " ", t)                    # "Kessler1", "Carter1"
    t = _LEAD_RE.sub("", t.strip())
    entries = []
    for part in re.split(r"[,;|&]|\s+(?:and|und)\s+", t):
        words = [w for w in part.split() if not re.fullmatch(r"[a-z]", w)]     # a lone affiliation letter ("…a,c")
        if not words:
            continue
        if any(not (_NAME_TOKEN_RE.match(w) or w.lower() in _PARTICLES) or (w.isupper() and len(w.strip(".")) >= 3)
               for w in words):
            return None                                     # a word that is no part of a name: not a name line
        if len(words) >= 2:
            entries.append(words)
    return entries or None


def _person_parts(person: str) -> tuple[str, str | None]:
    """(folded surname, folded first name or initial, or None) of a requested person."""
    words = [w for w in re.findall(r"[^\s,]+", person or "") if w.strip(".").lower() not in _HONORIFICS]
    parts = [_fold(w).strip(".") for w in words]
    parts = [p for p in parts if p]
    if not parts:
        return "", None
    return re.sub(r"[^a-z\-]", "", parts[-1]), (parts[0] if len(parts) > 1 else None)


def _affiliation_letters(front: str) -> set:
    """Affiliation letters glued to the next word ("aNaMLab", "eHAW-München"): their letter may be glued to a name too."""
    return set(re.findall(r"(?m)(?:^|[,;]\s*)([a-z])(?=[A-ZÄÖÜ])", front or ""))


def _entry_kind(words: list[str], surname: str, first: str | None, glued: set) -> str | None:
    last = re.sub(r"[^a-z\-]", "", _fold(words[-1]))
    if last != surname and not (last[:-1] == surname and last[-1:] in glued):
        return None
    firsts = [w for w in words[:-1] if w.lower() not in _PARTICLES and _fold(w).strip(".") not in _HONORIFICS]
    if any(_fold(w).strip(".") in _HONORIFICS for w in words[:-1]):
        return "role"                                      # "Prof. Dr. …": an examiner's or supervisor's title page line
    if first is None:
        return "author"
    if not firsts:
        return "possible_identity"
    have = _fold(firsts[0]).strip(".")
    if len(have) == 1 or len(first) == 1:
        return "possible_identity" if have[:1] == first[:1] else None
    if have == first:
        return "author"
    # a short form ("Chris" / "Christopher") is not proof of the same person, nor is a shared prefix ("Anna" /
    # "Annabelle"): possibly the same person
    if min(len(have), len(first)) >= 4 and (have.startswith(first) or first.startswith(have)):
        return "possible_identity"
    return None


def match_person(front: str, person: str) -> str:
    """"author", "possible_identity", "possible_role" or "no": how the front matter names `person` (see above)."""
    surname, first = _person_parts(person)
    if len(surname) < 2:
        return "no"
    glued, best, role_label = _affiliation_letters(front), "no", False
    rank = {"no": 0, "possible_role": 1, "possible_identity": 2, "author": 3}
    for line in (front or "").splitlines():
        s = line.strip()
        if not s:
            role_label = False
            continue
        org = _ORG_RE.search(s)
        if org and not _ROLE_RE.search(s):
            # an author line that runs into the first affiliation ("… T. Novakowski $^{a,c}$ $^{a}$ NaMLab gGmbH"):
            # the names before that affiliation's mark count, the affiliation does not
            marks = [m.start() for m in _SCRIPT_RE.finditer(s, 0, org.start())]
            head = s[:marks[-1]] if marks else ""
            entries = _name_entries(head.lstrip("#").strip()) if head.strip() else None
        else:
            entries = None if _ROLE_RE.search(s) else _name_entries(s.lstrip("#").strip())
        if entries is None:
            # a role word with no name after it ("Betreuer:") makes the next lines of names roles too
            role_label = bool(_ROLE_RE.search(s)) and _name_entries(_ROLE_RE.sub(" ", s).strip(" :")) is None
            if (first and not _ROLE_RE.search(s) and not _ORG_RE.search(s) and not s.startswith("#")
                    and re.search(rf"(?<![a-z]){re.escape(first)}\w*\s+(?:\w\.\s*)*{re.escape(surname)}(?![a-z])",
                                  _fold(_EMAIL_RE.sub(" ", s)))):
                best = max(best, "possible_role", key=rank.get)
            continue
        for words in entries:
            kind = _entry_kind(words, surname, first, glued)
            if kind == "role" or (kind and role_label):
                continue
            if kind:
                best = max(best, kind, key=rank.get)
    return best


def names_person(front: str, person: str) -> bool:
    """Does the front matter list `person` among its authors (match_person "author")?"""
    return match_person(front, person) == "author"


def classify(fronts: dict, person: str) -> dict:
    """{"author": [...], "possible_identity": [...], "possible_role": [...]}: the sources by how they name `person`,
    sorted; documents that do not name them are left out."""
    out: dict = {"author": [], "possible_identity": [], "possible_role": []}
    for src, front in fronts.items():
        kind = match_person(front, person)
        if kind in out:
            out[kind].append(src)
    return {k: sorted(v) for k, v in out.items()}


def documents_by(fronts: dict, person: str) -> list[str]:
    """The sources whose front matter lists `person` among its authors, sorted."""
    return classify(fronts, person)["author"]


def fill_missing(engine, *, collection: str, out_dir) -> tuple[int, list[str]]:
    """Front matter for each of the library's documents without a current list (none stored, or one of an earlier
    version), from its parsed text: (lists stored, documents without parsed text). No model call; best effort, never
    raises."""
    if engine is None or not out_dir:
        return 0, []
    try:
        from memento.library import documents as ldb
        library = {r["source"]: r.get("content_hash") for r in ldb.list_documents(engine, collection)}
        ensure_table(engine)
        _, todo = fresh(load(engine, collection, with_hash=True), library)
    except Exception:
        return 0, []
    stored, missing = 0, []
    for source in sorted(todo):
        if parsed_text_path(out_dir, source) is None:
            missing.append(source)
        elif store_front(engine, collection=collection, source=source, out_dir=out_dir,
                         content_hash=library.get(source)):
            stored += 1
    return stored, missing


def store_front(engine, *, collection: str, source: str, out_dir, content_hash: str | None) -> bool:
    """After an upload: the document's front matter read from its parsed text and stored. Best effort: False (and
    nothing stored) when there is no parsed text, the write is not for the library's version (put), or the database
    fails; an upload never fails on it."""
    try:
        front = read_front(out_dir, source)
        if not front:
            return False
        ensure_table(engine)
        return put(engine, collection=collection, source=source, content_hash=content_hash, front=front)
    except Exception:
        return False
