#!/usr/bin/env python3
# scripts/build_fulltext_paths.py: which papers of an offline database (ALEXANDRIA_DB) can be read in full, by DOI, for
# the literature check (memento/research_gaps/fulltext.py). The source database is read only: files are opened for
# reading, nothing there is written, moved or added, and its config.json (API keys) is never opened.
# Reads the tab-separated topic lists beside the database (columns doi, path, downloaded, markdown) into one SQLite file
# under data/alexandria/. Built into a new file and swapped in at the end, so a failed build never breaks a worker.
#   ALEXANDRIA_DB=... python scripts/build_fulltext_paths.py [--lists FOLDER] [--out FILE]
import argparse
import contextlib
import csv
import os
import sqlite3
import sys
import time
from pathlib import Path

from memento.research_gaps.fulltext import INDEX


def rows(lists: list[Path]):
    """(doi, path) of every paper with a downloaded file, lower-case DOIs, each paper once per list."""
    for path in lists:
        with open(path, encoding="utf-8", errors="replace", newline="") as f:
            for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
                if (r.get("doi") or "").strip() and "1" in (r.get("downloaded"), r.get("markdown")):
                    yield r["doi"].strip().lower(), r["path"]


def build(lists: list[Path], out: Path) -> int:
    tmp = out.with_suffix(".building")
    tmp.unlink(missing_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.closing(sqlite3.connect(tmp)) as db, db:
        db.execute("CREATE TABLE papers (doi TEXT PRIMARY KEY, path TEXT NOT NULL)")
        db.executemany("INSERT OR IGNORE INTO papers VALUES (?, ?)", rows(lists))
        n = db.execute("SELECT count(*) FROM papers").fetchone()[0]
    tmp.replace(out)
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lists", type=Path, help="folder of the topic lists (default: beside the database)")
    ap.add_argument("--out", type=Path, default=INDEX)
    args = ap.parse_args()
    db = os.getenv("ALEXANDRIA_DB", "")
    folder = args.lists or (Path(db).parent / "ressources" / "topic_index" if db else None)
    lists = sorted(folder.glob("*.tsv")) if folder else []
    if not lists:
        print("no topic lists found: set ALEXANDRIA_DB or --lists", file=sys.stderr)
        return 1
    t0 = time.time()
    n = build(lists, args.out)
    print(f"finished: {n} papers from {len(lists)} lists in {time.time() - t0:.0f} s -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
