"""Gap ledger step 3, the literature check: searches Semantic Scholar (the deployment's key) and OpenAlex (free, with a
small daily allowance without a key; its headers say how many are left), merged by DOI or title, the library's own
documents left out, cached for 30 days per service and phrase. Only the short search phrase leaves the server, never
document text.

An offline source joins them: abstracts in a local keyword index (SQLite FTS5, BM25) built by
scripts/build_alexandria_index.py into data/alexandria/abstracts.sqlite, with results shaped like the online
services' (Work) and merged with them by DOI; nothing leaves the server, and it is skipped when the index is missing.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from sqlalchemy import Column, DateTime, MetaData, String, Table, Text, delete, insert, select, update

import sqlite3
from pathlib import Path
from memento.config.paths import PROJECT_ROOT


OPENALEX_URL = "https://api.openalex.org/works"
S2_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
CACHE_DAYS = 30

METADATA = MetaData()
CACHE = Table(
    "memento_literature_cache", METADATA,
    Column("service", String(30), primary_key=True),
    Column("query", String(500), primary_key=True),
    Column("results", Text, nullable=False),               # the service's records (JSON), as returned
    Column("fetched_at", DateTime(timezone=True), nullable=False),
)


class SearchFailed(Exception):
    pass


def _safe(service: str, e: Exception) -> str:
    """An error without the request URL, which carries the API key: no secret is ever shown."""
    status = getattr(getattr(e, "response", None), "status_code", None)
    return f"{service}: {type(e).__name__}" + (f" (HTTP {status})" if status else "")


@dataclass
class Work:
    title: str
    year: int | None
    doi: str
    abstract: str
    url: str
    source: str                                             # openalex | semantic_scholar (the first one found)
    work_id: str
    sources: set = field(default_factory=set)
    excerpts: list = field(default_factory=list)            # passages of its full text, when read


def ensure_tables(engine) -> None:
    METADATA.create_all(engine, tables=[CACHE])


def _doi(raw) -> str:
    d = (raw or "").strip().lower()
    return re.sub(r"^https?://(dx\.)?doi\.org/", "", d)


def _norm_title(t: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", (t or "").lower()))


def _abstract(inv: dict | None) -> str:
    if not inv:
        return ""
    pos = sorted((p, w) for w, ps in inv.items() for p in ps)
    return " ".join(w for _p, w in pos)


def from_openalex(w: dict) -> Work:
    doi = _doi(w.get("doi"))
    return Work(title=" ".join((w.get("title") or "").split()), year=w.get("publication_year"), doi=doi,
                abstract=_abstract(w.get("abstract_inverted_index")),
                url=f"https://doi.org/{doi}" if doi else (w.get("id") or ""), source="openalex",
                work_id=w.get("id") or "", sources={"openalex"})


def from_s2(p: dict) -> Work:
    doi = _doi((p.get("externalIds") or {}).get("DOI"))
    return Work(title=" ".join((p.get("title") or "").split()), year=p.get("year"), doi=doi,
                abstract=(p.get("abstract") or "").strip(),
                url=f"https://doi.org/{doi}" if doi else (p.get("url") or ""), source="semantic_scholar",
                work_id=p.get("paperId") or "", sources={"semantic_scholar"})


def merge(works: list[Work]) -> list[Work]:
    """One entry per paper (same DOI, else the same title), in first-seen order; the longer abstract is kept."""
    out: list[Work] = []
    for w in works:
        key_t = _norm_title(w.title)
        same = next((o for o in out if (w.doi and o.doi == w.doi) or (key_t and _norm_title(o.title) == key_t)), None)
        if same is None:
            out.append(Work(**{**w.__dict__, "sources": set(w.sources)}))
            continue
        same.sources |= w.sources
        if len(w.abstract) > len(same.abstract):
            same.abstract = w.abstract
        same.doi = same.doi or w.doi
        if not same.url.startswith("https://doi.org/") and w.url.startswith("https://doi.org/"):
            same.url = w.url
        same.year = same.year or w.year
    return out


def drop_library(works: list[Work], library_titles: list[str]) -> list[Work]:
    """Leave out works that are documents of the library: the same title, or one title the start of the other (at
    least 4 words; library titles may be cut)."""
    lib = [t for t in (_norm_title(x) for x in library_titles) if t]

    def is_lib(w: Work) -> bool:
        t = _norm_title(w.title)
        for x in lib:
            if t == x:
                return True
            short, long_ = sorted((t, x), key=len)
            if len(short.split()) >= 4 and long_.startswith(short + " "):
                return True
        return False
    return [w for w in works if not is_lib(w)]


def openalex_searches_left(headers) -> int | None:
    """Searches left today from OpenAlex's own headers (credits as dollars); None when it does not say."""
    try:
        h = {k.lower(): v for k, v in dict(headers or {}).items()}
        left, cost = float(h["x-ratelimit-remaining-usd"]), float(h["x-ratelimit-cost-usd"])
        return int(left / cost + 1e-9) if cost > 0 else None
    except (KeyError, ValueError, TypeError):
        return None


# ── cache ──

def _cached(engine, service: str, query: str) -> list | None:
    with engine.connect() as c:
        row = c.execute(select(CACHE.c.results, CACHE.c.fetched_at).where(
            CACHE.c.service == service, CACHE.c.query == query)).first()
    if row is None:
        return None
    at = row[1] if row[1].tzinfo else row[1].replace(tzinfo=timezone.utc)
    if datetime.now(timezone.utc) - at > timedelta(days=CACHE_DAYS):
        return None
    return json.loads(row[0])


def _store(engine, service: str, query: str, records: list) -> None:
    with engine.begin() as c:
        c.execute(delete(CACHE).where(CACHE.c.service == service, CACHE.c.query == query))
        c.execute(insert(CACHE).values(service=service, query=query, results=json.dumps(records),
                                       fetched_at=datetime.now(timezone.utc)))


def cached_abstracts(engine) -> dict:
    """{work id: abstract} of every cached search result (OpenAlex ids, Semantic Scholar paper ids)."""
    out = {}
    with engine.connect() as c:
        for service, results in c.execute(select(CACHE.c.service, CACHE.c.results)).all():
            for rec in json.loads(results):
                w = from_openalex(rec) if service == "openalex" else from_s2(rec)
                if w.work_id and w.abstract and len(w.abstract) > len(out.get(w.work_id, "")):
                    out[w.work_id] = w.abstract
    return out


def age_cache(engine, by: timedelta) -> None:
    """Tests: make every cache entry older."""
    with engine.begin() as c:
        for q, at in c.execute(select(CACHE.c.query, CACHE.c.fetched_at)).all():
            c.execute(update(CACHE).where(CACHE.c.query == q).values(fetched_at=at - by))


# ── searches ──

def _http():
    import httpx
    return httpx


def openalex_phrase(query: str) -> str:
    """The phrase without OpenAlex search syntax: wildcards are refused, and quotes, brackets, colons and boolean marks
    would change the search. Hyphens inside words stay."""
    return " ".join(re.sub(r'[*?"“”()\[\]{}:|~!^\\]', " ", query or "").split())


def search_openalex(query: str, *, engine, http=None, n: int = 25, mailto: str = "", api_key: str = "",
                    timeout: float = 20) -> tuple[list[Work], int | None]:
    """(works, searches left today or None when not known / from the cache)."""
    hit = _cached(engine, "openalex", query)
    if hit is not None:
        return [from_openalex(w) for w in hit], None
    params = {"search": openalex_phrase(query), "per_page": n,
              "select": "id,title,doi,publication_year,abstract_inverted_index"}
    if mailto:
        params["mailto"] = mailto
    if api_key:
        params["api_key"] = api_key
    try:
        r = (http or _http()).get(OPENALEX_URL, params=params, timeout=timeout)
        r.raise_for_status()
        records = r.json().get("results") or []
    except Exception as e:
        raise SearchFailed(_safe("openalex", e)) from None
    _store(engine, "openalex", query, records)
    return [from_openalex(w) for w in records], openalex_searches_left(r.headers)


def search_s2(query: str, *, engine, api_key: str, http=None, n: int = 25, timeout: float = 20,
              sleep=None, retries: int = 3) -> list[Work]:
    """Semantic Scholar enforces its rate strictly (a 429 can come even within the key's 1 request a second): a 429 is
    retried after 2, 4, … seconds, `retries` times."""
    import time as _time
    hit = _cached(engine, "semantic_scholar", query)
    if hit is not None:
        return [from_s2(p) for p in hit]
    sleep = sleep or _time.sleep
    for attempt in range(retries + 1):
        try:
            r = (http or _http()).get(S2_URL, params={"query": query, "limit": n,
                                                      "fields": "title,year,abstract,externalIds,url"},
                                      headers={"x-api-key": api_key}, timeout=timeout)
            if r.status_code == 429 and attempt < retries:
                sleep(2 ** (attempt + 1))
                continue
            r.raise_for_status()
            records = r.json().get("data") or []
            break
        except Exception as e:
            raise SearchFailed(_safe("semantic_scholar", e)) from None
    _store(engine, "semantic_scholar", query, records)
    return [from_s2(p) for p in records]


# ── Offline abstracts (local keyword index) ───────────────────────────────────
DEFAULT = PROJECT_ROOT / "data" / "alexandria" / "abstracts.sqlite"


def available(path=DEFAULT) -> bool:
    return Path(path).is_file()


def fts_query(phrase: str) -> str:
    """Each word as a quoted term, joined with OR (ranked by BM25: more matching terms rank higher); FTS5 syntax in
    the phrase (quotes, *, NOT, brackets, colons) cannot reach the query."""
    words = re.findall(r"[\w]+", phrase or "", flags=re.UNICODE)
    words = [w for w in words if w.upper() not in ("AND", "OR", "NOT", "NEAR")]
    return " OR ".join(f'"{w}"' for w in dict.fromkeys(words))


def search(phrase: str, *, path=DEFAULT, n: int = 25) -> list[Work]:
    if not available(path):
        return []
    q = fts_query(phrase)
    if not q:
        return []
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
        rows = db.execute("""SELECT w.doi, w.title, w.abstract FROM works_fts f JOIN works w ON w.id = f.rowid
                             WHERE works_fts MATCH ? ORDER BY bm25(works_fts, 3.0, 1.0) LIMIT ?""", (q, n)).fetchall()
    return [Work(title=t or "", year=None, doi=d, abstract=a or "", url=f"https://doi.org/{d}", source="alexandria",
                 work_id=d, sources={"alexandria"}) for d, t, a in rows]


def journal(doi: str, *, path=DEFAULT) -> str:
    """The journal of a paper by its DOI, "" when the index does not have it (paper cards). DOIs are case-insensitive;
    the index keeps them as written, so the lower and upper case spellings are looked up through its DOI index (a
    mixed-case one may be missed: then no journal is shown)."""
    from memento.research_gaps.report import doi_key
    d = doi_key(doi)
    if not d or not available(path):
        return ""
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
        row = db.execute("SELECT journal FROM works WHERE doi IN (?, ?) AND journal IS NOT NULL AND journal != '' "
                         "LIMIT 1", (d, d.upper())).fetchone()
    return (row[0] or "").strip() if row else ""
