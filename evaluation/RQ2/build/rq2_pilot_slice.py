#!/usr/bin/env python3
"""Pick the RQ2 pilot slice (~50 documents per corpus), link-dense on purpose:
a random slice of 405 papers would hold almost no in-slice citations or
same-work pairs, leaving G1 and cross-document bridges untestable.

  python evaluation/RQ2/build/rq2_pilot_slice.py --domain ceramics --n 50 --seed 7
Writes data/rq2/pilot_<domain>.json and .txt (ids for build_graph.py --papers-file)."""
import argparse
import json
import random
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from memento.config.cli_domain import apply_domain_arg  # noqa: E402
apply_domain_arg()


def pick_ceramics_slice(citation_pairs, sizes, n, seed):
    rng = random.Random(seed)
    adj = defaultdict(set)
    for a, b in citation_pairs:
        if a != b:
            adj[a].add(b)
            adj[b].add(a)
    target_linked = int(round(0.8 * n))
    chosen, seen = [], set()
    for start in sorted(adj, key=lambda x: (-len(adj[x]), x)):
        if len(chosen) >= target_linked:
            break
        if start in seen:
            continue
        q = deque([start])
        seen.add(start)
        while q and len(chosen) < target_linked:
            x = q.popleft()
            chosen.append(x)
            for y in sorted(adj[x], key=lambda y: (-len(adj[y]), y)):
                if y not in seen:
                    seen.add(y)
                    q.append(y)
    rest = sorted((s for s in sizes if s not in set(chosen)), key=lambda s: (sizes[s], s))
    k = len(rest)
    terciles = [rest[:k // 3], rest[k // 3:2 * k // 3], rest[2 * k // 3:]]   # small / medium / large
    for t in terciles:
        rng.shuffle(t)
    i = 0
    while len(chosen) < n and any(terciles):
        if terciles[i % 3]:
            chosen.append(terciles[i % 3].pop())
        i += 1
    ids = set(chosen)
    inside = sum(1 for a, b in citation_pairs if a in ids and b in ids and a != b)
    return {"ids": chosen, "n_linked": sum(1 for x in chosen if x in adj), "n_edges_inside": inside}


def pick_polymer_slice(registry, n, seed, max_chapters=4):
    from memento.graph.documents import NON_RESEARCH_TYPES, is_person_folder
    rng = random.Random(seed)
    folders = defaultdict(list)
    for src, info in registry.items():
        if is_person_folder(info.get("student")):
            folders[info["student"]].append(src)
    eligible = [st for st, docs in folders.items()
                if any(registry[d]["type"] == "dissertation" for d in docs)
                and any(registry[d]["type"] in ("defense", "chapter") for d in docs)]
    rng.shuffle(eligible)
    ids, students = [], []
    for st in eligible:
        if len(ids) >= n - 6:
            break
        docs = [d for d in sorted(folders[st]) if registry[d]["type"] not in NON_RESEARCH_TYPES]
        # chapters are mostly drafts/copies of the dissertation: a few per student keep the
        # same-work pairs without letting one large folder fill the whole slice
        chapters = [d for d in docs if registry[d]["type"] == "chapter"]
        keep = set(rng.sample(chapters, min(max_chapters, len(chapters))))
        ids += [d for d in docs if registry[d]["type"] != "chapter" or d in keep]
        students.append(st)
    non_research = sorted(s for s, i in registry.items() if i["type"] in NON_RESEARCH_TYPES and s not in ids)
    theses = sorted(s for s, i in registry.items() if i["type"] == "student_thesis" and s not in ids)
    ids += rng.sample(non_research, min(4, len(non_research))) + rng.sample(theses, min(2, len(theses)))
    idset = set(ids)
    same_work = sum(1 for s in ids for f in ("duplicate_of", "parent", "parent_diss")
                    if registry[s].get(f) in idset)
    return {"ids": ids, "students": students, "n_same_work_pairs": same_work,
            "types": dict(Counter(registry[s]["type"] for s in ids))}


def _ceramics_inputs(output_dir):
    from memento.graph.build.build_graph import _find_content_list, _ledger_title, _ledger_year, list_collection_sources
    from memento.graph.build.citations import resolve_citations
    from memento.ingestion.extract_metadata import extract_paper_metadata, parse_bibliography
    from memento.ingestion.run_pipeline import find_paper_dir_in
    from memento.config.settings import COLLECTION, MINERU_OUT_SUBDIR
    papers, refs, sizes = [], [], {}
    for pid in list_collection_sources(COLLECTION):
        d = find_paper_dir_in(output_dir, pid, MINERU_OUT_SUBDIR)
        cl = d and next(iter(sorted(d.glob("*_content_list_enriched.json"))), None)
        if not cl:
            continue
        items = json.loads(cl.read_text())
        meta = extract_paper_metadata(items)
        papers.append({"id": pid, "title": _ledger_title(pid) or meta["title"],
                       "year": meta["year"] or _ledger_year(pid)})
        sizes[pid] = len(items)
        for r in parse_bibliography(items):
            refs.append({"id": f"{pid}::ref::{r['marker']}", "paper_id": pid, "raw": r["raw"], "year": r["year"]})
    res = resolve_citations(papers, refs)
    return [(m["citing"], m["cited"]) for m in res["matches"]], sizes, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--output-dir", type=Path)
    a = ap.parse_args()
    out = Path("data/rq2")
    out.mkdir(parents=True, exist_ok=True)
    if a.domain == "ceramics":
        pairs, sizes, res = _ceramics_inputs(a.output_dir or Path("data/output/Ceramics"))
        r = pick_ceramics_slice(pairs, sizes, a.n, a.seed)
        r["corpus_citations_resolved"] = len(res["matches"])
    else:
        from memento.graph.documents import load_document_registry
        r = pick_polymer_slice(load_document_registry(a.domain), a.n, a.seed)
    r |= {"domain": a.domain, "seed": a.seed, "n": len(r["ids"])}
    (out / f"pilot_{a.domain}.json").write_text(json.dumps(r, indent=1, ensure_ascii=False))
    (out / f"pilot_{a.domain}.txt").write_text("\n".join(r["ids"]) + "\n")
    print(json.dumps({k: v for k, v in r.items() if k != "ids"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
