"""List the figure claims awaiting review, grouped by figure (both sets), and record verdicts.
  list:   .venv/bin/python evaluation/RQ3/benchmark/review_figures.py list <corpus> [start] [n]
  record: .venv/bin/python evaluation/RQ3/benchmark/review_figures.py record <corpus> '<json {row_id@SET: ok|wrong|unsure}>'
Review rule: "ok" = the stated label is right (supported claims are visible in the figure/caption; unsupported
claims are NOT shown by it); "wrong" = the label is wrong; "unsure" = cannot be decided from the figure."""
import json, sys
from collections import defaultdict
from pathlib import Path
D = Path(__file__).resolve().parents[3] / "data/rq3/benchmark"
cmd, corpus = sys.argv[1], sys.argv[2]
if cmd == "list":
    start, n = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (0, 5)
    figs, caps = defaultdict(list), {}
    for s in ("V", "J"):
        f = D / f"claims_{s}_{corpus}.jsonl"
        if not f.exists(): continue
        for line in open(f):
            rec = json.loads(line)
            for i, r in enumerate(rec["rows"]):
                if r["needs_review"] and not r["dropped"]:
                    figs[r["img_path"]].append((f"{r['passage_key']}#{i}@{s}", r["kind"], r["label"], r["text"]))
                    caps[r["img_path"]] = r["source_text"]
    for k, (img, rows) in enumerate(sorted(figs.items())[start:start + n], start):
        print(f"\n### figure {k}: {img}\ncaption: {caps[img][:500]}")
        for rid, kind, label, text in rows:
            print(f"- {rid} [{kind} → {label}] {text}")
    print(f"\n(total figures: {len(figs)})")
else:
    verdicts = json.loads(sys.argv[3])
    for s in ("V", "J"):
        path = D / f"figure_review_{s}_{corpus}.json"
        cur = json.loads(path.read_text()) if path.exists() else {}
        cur.update({k.rsplit("@", 1)[0]: v for k, v in verdicts.items() if k.endswith("@" + s)})
        path.write_text(json.dumps(cur, indent=1))
    print("recorded", len(verdicts))
