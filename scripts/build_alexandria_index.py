#!/usr/bin/env python3
# scripts/build_alexandria_index.py: a keyword index of an offline abstract database (ALEXANDRIA_DB), the gap ledger's
# offline literature source.
#
# The source database is read only: files are opened for reading, nothing there is written, moved or added, and its
# config.json (API keys) is never opened.
# Every journal list (<publisher>/<ISSN>/all_papers.json) is read once; each record with an abstract → title, abstract
# (JATS markup removed), DOI, journal, publisher into one SQLite file under data/alexandria/ with an FTS5 full-text
# index (BM25 ranking). Built into a new file and swapped in at the end, so a failed build never breaks a working one.
#   nohup setsid .venv/bin/python scripts/build_alexandria_index.py > logs/alexandria_index.log 2>&1 &
import ast
import glob
import html
import json
import os
import re
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(os.getenv("ALEXANDRIA_DB", ""))   # the source database folder (<publisher>/<ISSN>/all_papers.json)
OUT = ROOT / "data" / "alexandria" / "abstracts.sqlite"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def clean(text: str) -> str:
    """JATS / HTML markup out, entities decoded, a leading "Abstract" label dropped, whitespace normalised."""
    t = html.unescape(re.sub(r"<[^>]+>", " ", text or ""))
    t = " ".join(t.split())
    return re.sub(r"^(abstract|summary)\s*[:.]?\s+", "", t, flags=re.I)


def title_of(raw) -> str:
    """Titles are stored as a list, or as the text of a Python list ("['Title']")."""
    if isinstance(raw, list):
        raw = raw[0] if raw else ""
    s = str(raw or "").strip()
    if s.startswith("["):
        try:
            v = ast.literal_eval(s)
            s = v[0] if isinstance(v, list) and v else s
        except (ValueError, SyntaxError):
            s = s.strip("[]'\"")
    return clean(s)


def main() -> int:
    if not os.getenv("ALEXANDRIA_DB"):
        sys.exit("set ALEXANDRIA_DB to the source database folder")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".building")
    if tmp.exists():
        tmp.unlink()
    db = sqlite3.connect(tmp)
    db.executescript("""
        PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;
        CREATE TABLE works (id INTEGER PRIMARY KEY, doi TEXT, title TEXT, abstract TEXT, journal TEXT,
                            issn TEXT, publisher TEXT);
        CREATE VIRTUAL TABLE works_fts USING fts5(title, abstract, content='works', content_rowid='id',
                                                  tokenize='porter unicode61');
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
    """)
    files = sorted(glob.glob(str(SRC / "*" / "*" / "all_papers.json")))
    log(f"{len(files)} journal lists in {SRC}")
    t0, n, seen = time.time(), 0, set()
    for k, f in enumerate(files, start=1):
        try:
            with open(f, "rb") as fh:                         # read only
                d = json.loads(fh.read())
        except Exception as e:
            log(f"skipped {f}: {type(e).__name__}")
            continue
        pub, issn = Path(f).parent.parent.name, Path(f).parent.name
        journal = str(d.get("title") or "")
        rows = []
        for r in d.get("dois") or []:
            a = clean(r.get("abstract") or "")
            doi = (r.get("DOI") or "").strip().lower()
            if len(a) < 80 or not doi or doi in seen:
                continue
            seen.add(doi)
            rows.append((doi, title_of(r.get("title")), a, journal, issn, pub))
        if rows:
            db.executemany("INSERT INTO works (doi, title, abstract, journal, issn, publisher) VALUES (?,?,?,?,?,?)",
                           rows)
            n += len(rows)
        if k % 50 == 0:
            db.commit()
            log(f"{k}/{len(files)} lists, {n:,} abstracts, {time.time() - t0:.0f} s")
    db.commit()
    log(f"indexing {n:,} abstracts ...")
    db.execute("INSERT INTO works_fts (rowid, title, abstract) SELECT id, title, abstract FROM works")
    db.execute("CREATE INDEX works_doi ON works (doi)")
    db.executemany("INSERT INTO meta VALUES (?, ?)", [("source", str(SRC)), ("built", time.strftime("%Y-%m-%d %H:%M")),
                                                      ("abstracts", str(n)), ("lists", str(len(files)))])
    db.commit()
    db.execute("INSERT INTO works_fts (works_fts) VALUES ('optimize')")
    db.commit()
    db.close()
    os.replace(tmp, OUT)
    log(f"finished: {n:,} abstracts in {time.time() - t0:.0f} s, {OUT.stat().st_size / 1e9:.1f} GB → {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
