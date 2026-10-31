"""
RQ3 step 3.3/3.4: measure a checker on a labelled claim set.

  --set V --model qwen   : the system verifier (qwen3.8) on the Gemma-written set  → verifier accuracy, false alarms
  --set J --model gemma  : the evaluation judge (Gemma 4 12B) on the qwen-written set → gate ≥ 0.80

Each claim is checked against its own source (text/table: text verifier; figure: vision verifier with the image) —
exactly the production components (memento/agentic_validation/verifiers.py), plus, for the system view, the deterministic
checks (memento/agentic_validation/scientific_checks.py) with figure rule A (skip), B (caption conflict) or C (caption +
the figure's printed text, extracted once by qwen). Figure claims count only if their label was confirmed by review
(figure_review_<set>_<corpus>.json: {row_id: "ok" | "wrong" | "unsure"}).
Verdicts are cached per row (resumable). Binary scoring: supported vs not; unjudged counts as wrong.
"""
import argparse, json, os, sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT))
for l in (ROOT / "config/.env").read_text().splitlines():
    l = l.strip()
    if l and not l.startswith("#") and "=" in l:
        k, v = l.split("=", 1); os.environ.setdefault(k, v.strip().strip('"').strip("'"))
from openai import OpenAI
from memento.agentic_validation.claims import Claim
from memento.agentic_validation.scientific_checks import run_checks
from memento.agentic_validation.verifiers import LLMJudgeVerifier, VerdictCache, VLMFigureVerifier, verify_claims

D = ROOT / "data/rq3/benchmark"
MODELS = {"qwen": ("http://localhost:8090/v1", "qwen3.8-27b", os.environ.get("REASONING_API_KEY", "none")),
          "gemma": ("http://localhost:8095/v1", "gemma-4-12b", "none"),
          # Qwen3.8-Flash-Next-FP8 on the local model server, measured as the system verifier like "qwen"
          "flash": (os.getenv("FLASH_NEXT_BASE_URL", "http://localhost:8001/v1"), "qwen-flash-next",
                    os.environ.get("QWEN_FLASH_NEXT_API_KEY", "none"))}


def load_rows(set_name, corpus):
    rows = []
    for line in open(D / f"claims_{set_name}_{corpus}.jsonl"):
        rec = json.loads(line)
        for i, r in enumerate(rec["rows"]):
            if not r["dropped"]:
                rows.append({**r, "id": f"{r['passage_key']}#{i}"})
    review = D / f"figure_review_{set_name}_{corpus}.json"
    rv = json.loads(review.read_text()) if review.exists() else {}
    for r in rows:
        r["usable"] = (not r["needs_review"]) or rv.get(r["id"]) == "ok"
    return rows


def verdict_for(row, text_v, fig_v, figure_thinking, labels=False):
    hit = {"type": "image" if row["passage_type"] == "figure" and row["img_path"] else row["passage_type"],
           "text": row["source_text"], "img_path": row["img_path"]}
    if labels:                      # the "<document>, page n" line the answer model and the live verifier see
        src, page = row["passage_key"].split("|")[:2]
        hit.update(source=src, page=int(page))
    c = Claim("c1", row["text"], (1,), (0, 1))
    v = verify_claims([c], [hit], text_verifier=text_v, figure_verifier=fig_v, cache=VerdictCache())["c1"]
    return {"label": v.label, "via": v.via, "rationale": v.rationale[:200]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["V", "J"], required=True)
    ap.add_argument("--model", choices=["qwen", "gemma", "flash"], required=True)
    ap.add_argument("--figure-thinking", action="store_true", help="vision check with thinking (qwen only)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--labels", action="store_true", help="show the source label line (verifier input since 2026-09-27)")
    a = ap.parse_args()
    url, model, key = MODELS[a.model]
    llm = OpenAI(base_url=url, api_key=key)
    text_v = LLMJudgeVerifier(llm, model, workers=1)
    fig_v = VLMFigureVerifier(llm, model, enable_thinking=a.figure_thinking)
    tag = f"{a.set}_{a.model}" + ("_figthink" if a.figure_thinking else "") + ("_labels" if a.labels else "")
    cache_path = D / f"verdicts_{tag}.jsonl"
    cache = {json.loads(l)["id"]: json.loads(l) for l in open(cache_path)} if cache_path.exists() else {}
    report = []
    for corpus in ("ceramics", "polymer-engineering"):
        rows = load_rows(a.set, corpus)
        if a.figure_thinking:
            rows = [r for r in rows if r["passage_type"] == "figure"]
        todo = [r for r in rows if r["id"] not in cache]
        with ThreadPoolExecutor(a.workers) as pool, open(cache_path, "a") as f:
            for r, v in zip(todo, pool.map(lambda r: verdict_for(r, text_v, fig_v, a.figure_thinking, a.labels), todo)):
                rec = {"id": r["id"], **v}
                cache[r["id"]] = rec
                f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
        report.append((corpus, rows))
    lines = [f"# Benchmark {tag}\n"]
    for corpus, rows in report + [("both", [r for _, rs in report for r in rs])]:
        lines += score(corpus, rows, cache, system=(a.model in ("qwen", "flash") and a.set == "V"))
    text = "\n".join(lines)
    print(text)
    (D / f"report_{tag}.md").write_text(text + "\n")


def _bal(pairs):
    """pairs of (gold_supported, predicted_supported) → balanced accuracy, recall on each class, n"""
    pos = [p for g, p in pairs if g]; neg = [p for g, p in pairs if not g]
    tpr = sum(pos) / len(pos) if pos else float("nan")
    tnr = sum(1 - p for p in neg) / len(neg) if neg else float("nan")
    return (tpr + tnr) / 2 if pos and neg else float("nan"), tpr, tnr, len(pairs)


def score(corpus, rows, cache, system=False):
    usable = [r for r in rows if r["usable"] and r["id"] in cache]
    pending = sum(1 for r in rows if not r["usable"])
    out = [f"\n## {corpus} (n = {len(usable)}; figure rows excluded — label wrong, unsure or not reviewed: {pending})\n"]

    def pred(r, mode=None):
        ok = cache[r["id"]]["label"] == "supported"
        if mode and ok:
            hit = {"type": "image" if r["passage_type"] == "figure" else r["passage_type"], "text": r["source_text"],
                   "img_path": r["img_path"]}
            if mode == "C" and r["passage_type"] == "figure":
                hit["text"] += "\n" + figure_text(r)
            ok = not run_checks("c1", r["text"], [hit], figure_mode={"A": "skip", "B": "conflict", "C": "conflict"}[mode])
        return ok

    views = [("verifier", None)] + ([("system A", "A"), ("system B", "B"), ("system C", "C")] if system else [])
    out.append("| view | balanced acc. | true claims kept (TPR) | errors caught (TNR) | unjudged |")
    out.append("|---|---|---|---|---|")
    unj = sum(cache[r["id"]]["label"] == "unjudged" for r in usable)
    for name, mode in views:
        b, tpr, tnr, n = _bal([(r["label"] == "supported", pred(r, mode)) for r in usable])
        out.append(f"| {name} | {b:.3f} | {tpr:.3f} | {tnr:.3f} | {unj} |")
    out.append("\n| subset | n | balanced acc. | false alarms on true claims | errors caught |")
    out.append("|---|---|---|---|---|")
    groups = defaultdict(list)
    for r in usable:
        groups[f"type:{r['passage_type']}"].append(r)
        groups[f"claim lang:{r['claim_lang']}"].append(r)
        groups["cross-language" if r["claim_lang"] != r["passage_lang"] else "same language"].append(r)
    for g, rs in sorted(groups.items()):
        b, tpr, tnr, n = _bal([(r["label"] == "supported", pred(r)) for r in rs])
        out.append(f"| {g} | {n} | {b:.3f} | {1 - tpr:.3f} | {tnr:.3f} |")
    out.append("\n| error kind | n | caught |")
    out.append("|---|---|---|")
    for k in ("number", "unit", "material", "negation", "wrong_source", "extrapolation"):
        rs = [r for r in usable if r["kind"] == k]
        if rs:
            out.append(f"| {k} | {len(rs)} | {sum(not pred(r) for r in rs) / len(rs):.3f} |")
    return out


_FIG_TEXT = {}


def figure_text(r):
    """Option C: the figure's printed text (axis labels, ticks, legend), extracted once by qwen and cached on disk."""
    path = D / "figure_text.json"
    if not _FIG_TEXT and path.exists():
        _FIG_TEXT.update(json.loads(path.read_text()))
    return _FIG_TEXT.get(r["img_path"] or "", "")


if __name__ == "__main__":
    main()
