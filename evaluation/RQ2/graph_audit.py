#!/usr/bin/env python3
"""RQ2 graph-quality gate, for every rung of the graph.

Per corpus graph, a sample audit of:
  entity    — is the MENTIONed entity really named in the chunk, as that type?   (G2, >= 0.80)
  relation  — does the chunk assert the RELATION triple? July's SUPPORTED axis   (G3, >= 0.70)
  merge     — do a merged entity and its representative denote one concept?      (G2, >= 0.90)
  same_work — are two SAME_WORK-linked documents parts/versions of one work?     (G1, >= 0.90)
  census    — chunk coverage, hub concentration (reported, no threshold)

Thresholds were fixed before any audit ran. Verdict:
PASS if the point estimate >= threshold, FAIL if the 95% Wilson upper bound
< threshold, otherwise INCONCLUSIVE -> a second disjoint sample of the same
size is judged once and the union re-decided. Deliberately false negative
controls are mixed into every judged kind; a kind whose control false-positive
rate exceeds 0.10 is marked judge-invalid. The main judge is the local
qwen3.8-27b (the extractor); ~100 items are re-judged by a different model
family (mistral-medium-3.5 on a cloud API, capped at 20 calls) and the
agreement is reported.

  python evaluation/RQ2/graph_audit.py --domain ceramics \
      --merge-map data/rq2/merge_ceramics_0.85.json data/rq2/merge_ceramics_0.90.json data/rq2/merge_ceramics_0.95.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from memento.config.cli_domain import apply_domain_arg  # noqa: E402
apply_domain_arg()

THRESHOLDS = {"relation_support": 0.70, "entity_precision": 0.80,
              "merge_precision": 0.90, "same_work_precision": 0.90}
MAX_CONTROL_FP = 0.10
BATCH = 10
OUT_DIR = Path(__file__).resolve().parent / "pilot_2026-09"


# ── statistics (ported: kg_audit_stats.py) ───────────────────────────────────
def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """(point estimate, lo, hi), rounded to 4 dp; (0, 0, 0) for n = 0."""
    if n <= 0:
        return (0.0, 0.0, 0.0)
    p = successes / n
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + (z * z) / (4 * n * n))
    return (round(p, 4), round(max(0.0, center - half), 4), round(min(1.0, center + half), 4))


def sqrt_proportional_allocation(counts: dict[str, int], total: int) -> dict[str, int]:
    """Slots per stratum proportional to sqrt(count): rare strata stay estimable."""
    weights = {k: math.sqrt(v) for k, v in counts.items() if v > 0}
    if not weights:
        return {k: 0 for k in counts}
    wsum = sum(weights.values())
    exact = {k: total * w / wsum for k, w in weights.items()}
    alloc = {k: min(int(v), counts[k]) for k, v in exact.items()}
    short = total - sum(alloc.values())
    order = sorted(exact, key=lambda k: exact[k] - int(exact[k]), reverse=True)
    i = 0
    while short > 0 and any(alloc[k] < counts[k] for k in alloc):
        k = order[i % len(order)]
        if alloc[k] < counts[k]:
            alloc[k] += 1
            short -= 1
        i += 1
    for k in counts:
        alloc.setdefault(k, 0)
    return alloc


_SUB_SUP = str.maketrans("₀₁₂₃₄₅₆₇₈₉⁰¹²³⁴⁵⁶⁷⁸⁹", "01234567890123456789")
_GREEK = {"α": " alpha ", "β": " beta ", "γ": " gamma ", "δ": " delta ", "θ": " theta ",
          "κ": " kappa ", "λ": " lambda ", "μ": " mu ", "σ": " sigma ", "τ": " tau "}


def normalize_entity_name(name: str | None) -> str:
    if not name:
        return ""
    s = unicodedata.normalize("NFKC", str(name)).strip().lower().translate(_SUB_SUP)
    for g, a in _GREEK.items():
        s = s.replace(g, a)
    return " ".join(re.sub(r"[^a-z0-9]+", " ", s).split())


def verdict(k: int, n: int, threshold: float) -> str:
    p, lo, hi = wilson_ci(k, n)
    if n and p >= threshold:
        return "PASS"
    if n and hi < threshold:
        return "FAIL"
    return "INCONCLUSIVE"


def cohen_kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    if not n:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in set(a) | set(b))
    return 1.0 if pe == 1.0 else (po - pe) / (1 - pe)


# ── judge I/O ───────────────────────────────────────────────────────────────
def _extract_json(raw: str):
    raw = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.S)
    m = re.search(r"```(?:json)?\s*(.*?)```", raw, re.S)
    body = m.group(1) if m else raw
    a, b = body.find("["), body.rfind("]")
    return json.loads(body[a:b + 1] if a >= 0 and b > a else body)


def parse_labels(raw: str, refs: list[str], allowed: set[str], key: str = "label") -> dict[str, str]:
    """{ref: label}; unknown refs, invalid labels and duplicated refs are dropped
    (an unparsed item leaves the denominator instead of guessing)."""
    try:
        data = _extract_json(raw)
    except Exception:
        return {}
    if not isinstance(data, list):
        return {}
    valid, out, dup = set(refs), {}, set()
    for d in data:
        if not isinstance(d, dict) or d.get("ref") not in valid:
            continue
        lab = str(d.get(key, "")).strip().lower()
        if lab not in allowed:
            continue
        if d["ref"] in out:
            dup.add(d["ref"])
        out[d["ref"]] = lab
    return {k: v for k, v in out.items() if k not in dup}


# Fields that describe the swapped value and must travel with it (else the judge sees the donor's name next to the
# original entity's aliases).
_COMPANIONS = {"entity_name": ("aliases", "entity_type", "canonical")}


def negative_controls(items: list[dict], n: int, seed: int, field: str) -> list[dict]:
    """Deliberately false items: `field` (and its companion fields) swapped for a
    donor from another paper none of whose surface forms (name or aliases) occur
    in this item's passage. Any control judged true is a judge false positive
    (no hand labels needed)."""
    rng = random.Random(seed)
    if len({i["paper_id"] for i in items}) < 2:
        return []
    companions = _COMPANIONS.get(field, ())

    def forms(d):
        return {normalize_entity_name(x) for x in [d[field], *(d.get("aliases") or [] if companions else [])] if x}
    out = []
    for it in rng.sample(items, min(n, len(items))):
        own = normalize_entity_name(it[field])
        passage = normalize_entity_name(it.get("passage", ""))
        donors = [d for d in items if d["paper_id"] != it["paper_id"]
                  and not any(f in passage for f in forms(d))
                  and normalize_entity_name(d[field]) != own]
        if donors:
            d = rng.choice(donors)
            out.append(dict(it, ref=f"ctrl_{it['ref']}", is_control=True,
                            **{k: d[k] for k in (field, *companions) if k in d}))
    return out


PROMPTS = {
    # Registry entities are shown with their aliases (the passage often uses the abbreviation), and the schema's entity
    # types are spelled out (else the judge rejects registry types such as "temperature" or "X-ray diffraction" as
    # "generic").
    "entity": (
        "You audit entity mentions extracted from scientific documents. For each numbered item you see "
        "an entity name (with its known aliases/abbreviations), its type, and the source passage. The "
        "entity types are: Material (a specific material, compound, phase or element), Process (a "
        "processing or synthesis method), Property (a material property or measured quantity), "
        "Condition (a processing/measurement condition such as temperature, pressure, atmosphere, time "
        "or heating rate), Characterization (a measurement or analysis technique). Label \"yes\" if the "
        "passage mentions this entity, by its name or any alias or abbreviation, as that kind of thing. "
        "Label \"no\" if it does not occur in the passage, if the type is wrong, or if it is not a "
        "scientific entity at all: a person or organisation name, a funding/grant number, software, or "
        "a vague phrase such as 'experimental plan' or 'results'. Judge only against the passage shown.\n"
        "Output ONLY a JSON array: [{\"ref\": \"<ref>\", \"label\": \"yes|no\"}, ...]"),
    # July's two-axis prompt (kg_audit_judge.JUDGE_SYSTEM_PROMPT), unchanged in substance.
    "relation": (
        "You are auditing a knowledge graph extracted from scientific papers. For each numbered item you "
        "see one extracted triple (head, relation, tail) — the relation is a short free-text description "
        "— and the exact source text passage it was extracted from.\nGive each item TWO INDEPENDENT labels:\nSUPPORTED — does the source passage "
        "actually assert this triple?\n  \"yes\" the passage states this relation between these two "
        "things\n  \"partial\" the passage supports part of it (e.g. both things appear and are related, "
        "but not by this relation, or one side is imprecise)\n  \"no\" the passage does not support it "
        "at all\nINFORMATIVE — would this triple help answer a real research question about this "
        "material or process?\n  \"yes\" it carries specific, usable scientific content\n  \"no\" it is "
        "vacuous, circular, or too generic to be useful\nJudge SUPPORTED strictly against the passage "
        "shown, and nothing you know beyond it. If the passage ends with [passage truncated], judge only "
        "what is present. Judge INFORMATIVE on the triple's own scientific content.\nOutput ONLY a JSON "
        "array: [{\"ref\": \"<ref>\", \"supported\": \"yes|partial|no\", \"informative\": \"yes|no\"}, ...]"),
    "merge": (
        "You check synonym merges in a {domain} research knowledge graph. For each numbered pair A / B "
        "(with their entity type), answer \"yes\" if A and B denote the same concept, so that a question "
        "about one is fully about the other (spelling variants, translations, abbreviations); answer "
        "\"no\" if they are different concepts, including a subtype, a derivative or a related but "
        "distinct thing.\nOutput ONLY a JSON array: [{\"ref\": \"<ref>\", \"label\": \"yes|no\"}, ...]"),
    "same_work": (
        "You check links between documents of a {domain} research archive. For each numbered pair you "
        "see two documents (file name, title and opening text). Answer \"yes\" if they are parts or "
        "versions of the same scientific work (e.g. a dissertation and its defence talk, a chapter file "
        "or draft of it, two copies of one paper); answer \"no\" if they are different works, even by "
        "the same author.\nOutput ONLY a JSON array: [{\"ref\": \"<ref>\", \"label\": \"yes|no\"}, ...]"),
}


def render(kind: str, it: dict) -> str:
    if kind == "entity":
        al = [x for x in (it.get("aliases") or []) if x and x != it["entity_name"]][:8]
        also = f" — also written as: {', '.join(al)}" if al else ""
        return (f"Entity: {it['entity_name']} (type: {it['entity_type']}){also}\n"
                f"Passage:\n{_cut(it['passage'])}")
    if kind == "relation":
        return f"Triple: ({it['head']}) -[{it['rel_type']}]-> ({it['tail']})\nSource passage:\n{_cut(it['passage'])}"
    if kind == "merge":
        return f"A: {it['name_a']}\nB: {it['name_b']}\nType: {it['entity_type']}"
    return (f"Document 1: {it['doc_a']}\n{_cut(it['text_a'], 1500)}\n\n"
            f"Document 2: {it['doc_b']}\n{_cut(it['text_b'], 1500)}")


def _cut(t: str, n: int = 3000) -> str:
    t = t or ""
    return t if len(t) <= n else t[:n] + "\n[passage truncated]"


class Judge:
    """OpenAI-compatible judge, batches of <= 10 items, JSON labels."""

    def __init__(self, base_url: str, model: str, api_key: str = "none", *,
                 min_interval_s: float = 0.0, max_calls: int | None = None, local: bool = True,
                 workers: int = 1):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=300, max_retries=1)
        self.model, self.min_interval, self.max_calls, self.local = model, min_interval_s, max_calls, local
        # parallel batches only for a local server; a rate-limited cloud judge stays sequential
        self.workers = workers if min_interval_s <= 0 and max_calls is None else 1
        self.calls, self._last = 0, 0.0

    def _chat(self, system: str, user: str) -> str:
        import openai
        for attempt in range(4):
            if self.max_calls is not None and self.calls >= self.max_calls:
                raise RuntimeError("cloud call budget exhausted")
            wait = self.min_interval - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            self.calls += 1
            try:
                kw = {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}} if self.local else {}
                r = self.client.chat.completions.create(
                    model=self.model, temperature=0, max_tokens=2000,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}], **kw)
                return r.choices[0].message.content or ""
            except openai.RateLimitError as e:
                ra = getattr(getattr(e, "response", None), "headers", {}).get("retry-after")
                time.sleep(max(30.0, float(ra) if ra else 0.0))
        raise RuntimeError("rate-limited 4 times")

    def judge(self, kind: str, items: list[dict], domain: str = "") -> dict[str, dict]:
        """{ref: {"label": ..., ["informative": ...]}} for parsed items."""
        system = PROMPTS[kind].replace("{domain}", domain)

        def one(batch):
            user = "".join(f"\n--- item {it['ref']} ---\n{render(kind, it)}\n" for it in batch)
            raw = self._chat(system, user)
            refs = [it["ref"] for it in batch]
            if kind == "relation":
                sup = parse_labels(raw, refs, {"yes", "partial", "no"}, key="supported")
                inf = parse_labels(raw, refs, {"yes", "no"}, key="informative")
                return {r: {"label": sup[r], "informative": inf.get(r)} for r in sup}
            return {r: {"label": l} for r, l in parse_labels(raw, refs, {"yes", "no"}).items()}

        batches = [items[i:i + BATCH] for i in range(0, len(items), BATCH)]
        out: dict[str, dict] = {}
        if getattr(self, "workers", 1) > 1 and len(batches) > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                for part in pool.map(one, batches):
                    out.update(part)
        else:
            for b in batches:
                out.update(one(b))
        return out


def cloud_api_key() -> str:
    """The Academic Cloud key: CLOUD_API_KEY (project convention, as in
    evaluation/RQ1/auto_annotate.py), falling back to OPENAI_API_KEY."""
    return os.getenv("CLOUD_API_KEY") or os.getenv("OPENAI_API_KEY", "none")


# ── samplers (Neo4j) ─────────────────────────────────────────────────────────
_MENTIONS = ("MATCH (c:Chunk)-[:MENTIONS]->(e:Entity) WHERE c.text IS NOT NULL AND size(c.text) > 0 "
             "RETURN c.id AS chunk_id, c.source AS paper_id, c.text AS passage, e.id AS entity_id, "
             "e.name AS entity_name, e.type AS entity_type, coalesce(e.canonical, false) AS canonical, "
             "coalesce(e.aliases, []) AS aliases "
             "ORDER BY c.id, e.id")
_RELATIONS = ("MATCH (a:Entity)-[r:RELATION]->(b:Entity) MATCH (c:Chunk {id: r.chunk_id}) "
              "WHERE c.text IS NOT NULL AND size(c.text) > 0 "
              "RETURN a.name AS head, b.name AS tail, r.rel_type AS rel_type, r.paper_id AS paper_id, "
              "r.chunk_id AS chunk_id, c.text AS passage ORDER BY elementId(r)")


def _stratified(rows: list[dict], key, total: int, seed: int, exclude: set) -> list[dict]:
    groups = defaultdict(list)
    for r in rows:
        if r["_uid"] not in exclude:
            groups[key(r)].append(r)
    alloc = sqrt_proportional_allocation({k: len(v) for k, v in groups.items()}, total)
    rng = random.Random(seed)
    out = []
    for k in sorted(alloc):
        out += rng.sample(groups[k], min(alloc[k], len(groups[k])))
    return out


def sample_entities(rows, total, seed, exclude=frozenset()):
    return _stratified(rows, lambda r: f"{'registry' if r['canonical'] else 'llm'}:{r['entity_type']}",
                       total, seed, exclude)


def sample_relations(rows, total, seed, exclude=frozenset()):
    # Open relations (free-text descriptions) make every relation its own
    # "type", so per-type strata would degenerate to database order: sample
    # one random pool instead.
    return _stratified(rows, lambda r: "all", total, seed, exclude)


def _with_refs(items, prefix):
    for i, it in enumerate(items):
        it["ref"] = f"{prefix}{i}"
        it.setdefault("is_control", False)
    return items


def load_rows(store):
    ment = [dict(r) | {"_uid": f"{r['chunk_id']}|{r['entity_id']}"} for r in store._run(_MENTIONS).records]
    rel = [dict(r) | {"_uid": f"{r['chunk_id']}|{r['head']}|{r['rel_type']}|{r['tail']}"}
           for r in store._run(_RELATIONS).records]
    return ment, rel


def merge_items(store, map_paths, per_map, seed):
    """Applied merges = (member, representative) pairs of each map; one sample per map."""
    names = {r["id"]: (r["name"], r["type"]) for r in store._run(
        "MATCH (e:Entity) RETURN e.id AS id, e.name AS name, e.type AS type").records}
    rng = random.Random(seed)
    by_map, seen = {}, {}
    for p in map_paths:
        m = json.loads(Path(p).read_text())
        pairs = sorted((a, b) for a, b in m["mapping"].items() if a in names and b in names)
        pick = rng.sample(pairs, min(per_map, len(pairs)))
        by_map[str(m["threshold"])] = pick
        for a, b in pick:
            seen[(a, b)] = {"name_a": names[a][0], "name_b": names[b][0], "entity_type": names[a][1],
                            "paper_id": a}
    items = [dict(v, pair=list(k)) for k, v in sorted(seen.items())]
    # controls: random same-type pairs that no map merges
    merged = {tuple(sorted(k)) for k in seen}
    by_type = defaultdict(list)
    for eid, (nm, t) in names.items():
        by_type[t].append(eid)
    ctrl = []
    types = [t for t, v in by_type.items() if len(v) > 1]
    while types and len(ctrl) < 20:
        t = rng.choice(types)
        a, b = rng.sample(by_type[t], 2)
        if tuple(sorted((a, b))) in merged or set(normalize_entity_name(names[a][0]).split()) & \
                set(normalize_entity_name(names[b][0]).split()):
            continue
        ctrl.append({"name_a": names[a][0], "name_b": names[b][0], "entity_type": t,
                     "paper_id": a, "pair": [a, b], "is_control": True})
    return items, ctrl, by_map


def same_work_items(store, seed):
    def doc(pid):
        rows = store._run("MATCH (c:Chunk {source: $p}) RETURN c.text AS t ORDER BY c.page, c.id LIMIT 12",
                          p=pid).records
        title = store._run("MATCH (p:Paper {id: $p}) RETURN p.title AS t", p=pid).records
        return (title[0]["t"] if title and title[0]["t"] else ""), "\n".join(r["t"] or "" for r in rows)
    edges = [(r["a"], r["b"], r["kind"]) for r in store._run(
        "MATCH (a:Paper)-[r:SAME_WORK]->(b:Paper) RETURN a.id AS a, b.id AS b, r.kind AS kind "
        "ORDER BY a.id, b.id").records]
    rng = random.Random(seed)
    edges = rng.sample(edges, min(100, len(edges)))
    papers = [(r["id"], r["student"]) for r in store._run(
        "MATCH (p:Paper) RETURN p.id AS id, p.student AS student ORDER BY p.id").records]
    linked = {frozenset((a, b)) for a, b, _ in edges}
    items, ctrl = [], []
    for a, b, kind in edges:
        (ta, xa), (tb, xb) = doc(a), doc(b)
        items.append({"doc_a": f"{a} — {ta}", "doc_b": f"{b} — {tb}", "text_a": xa, "text_b": xb,
                      "kind": kind, "paper_id": a, "pair": [a, b]})
    tries = 0
    while len(ctrl) < min(20, len(items)) and len(papers) > 1 and tries < 500:
        tries += 1
        (a, sa), (b, sb) = rng.sample(papers, 2)
        if frozenset((a, b)) in linked or (sa and sa == sb):
            continue
        (ta, xa), (tb, xb) = doc(a), doc(b)
        ctrl.append({"doc_a": f"{a} — {ta}", "doc_b": f"{b} — {tb}", "text_a": xa, "text_b": xb,
                     "kind": "control", "paper_id": a, "pair": [a, b], "is_control": True})
    return items, ctrl


def census(store) -> dict:
    q = lambda c, **k: store._run(c, **k).records
    n_chunks = q("MATCH (c:Chunk) RETURN count(c) AS n")[0]["n"]
    covered = q("MATCH (c:Chunk) WHERE (c)-[:MENTIONS]->() RETURN count(c) AS n")[0]["n"]
    per_entity = [r["n"] for r in q("MATCH (e:Entity)<-[m:MENTIONS]-() "
                                    "WITH e, count(m) AS n RETURN n ORDER BY n DESC")]
    total_m = sum(per_entity)
    by_type = {r["t"]: r["n"] for r in q("MATCH (e:Entity) RETURN e.type AS t, count(e) AS n")}
    canon = q("MATCH (e:Entity) RETURN sum(CASE WHEN e.canonical THEN 1 ELSE 0 END) AS c, count(e) AS n")[0]
    rels = {r["t"]: r["n"] for r in q("MATCH ()-[r:RELATION]->() RETURN r.rel_type AS t, count(r) AS n")}
    edges = {r["t"]: r["n"] for r in q("MATCH ()-[r]->() RETURN type(r) AS t, count(r) AS n")}
    return {"n_chunks": n_chunks, "chunks_with_mentions": covered,
            "coverage": round(covered / n_chunks, 4) if n_chunks else 0.0,
            "n_entities": len(per_entity), "n_mentions": total_m,
            "top50_mention_share": round(sum(per_entity[:50]) / total_m, 4) if total_m else 0.0,
            "mentions_per_covered_chunk": round(total_m / covered, 2) if covered else 0.0,
            "entities_by_type": by_type, "canonical_entities": canon["c"], "relations_by_type": rels,
            "edges_by_type": edges}


# ── scoring ─────────────────────────────────────────────────────────────────
POSITIVE = {"entity": {"yes"}, "relation": {"yes", "partial"}, "merge": {"yes"}, "same_work": {"yes"}}
METRIC = {"entity": "entity_precision", "relation": "relation_support",
          "merge": "merge_precision", "same_work": "same_work_precision"}


def score(kind: str, items: list[dict], judged: dict[str, dict]) -> dict:
    real = [i for i in items if not i.get("is_control")]
    ctrl = [i for i in items if i.get("is_control")]
    rj = [judged[i["ref"]]["label"] for i in real if i["ref"] in judged]
    cj = [judged[i["ref"]]["label"] for i in ctrl if i["ref"] in judged]
    k = sum(1 for l in rj if l in POSITIVE[kind])
    p, lo, hi = wilson_ci(k, len(rj))
    fp = sum(1 for l in cj if l in POSITIVE[kind])
    thr = THRESHOLDS[METRIC[kind]]
    out = {"metric": METRIC[kind], "threshold": thr, "n": len(rj), "k": k, "estimate": p, "ci": [lo, hi],
           "verdict": verdict(k, len(rj), thr), "n_unparsed": len(real) - len(rj),
           "n_controls": len(cj), "control_fp_rate": round(fp / len(cj), 4) if cj else None}
    out["judge_valid"] = out["control_fp_rate"] is None or out["control_fp_rate"] <= MAX_CONTROL_FP
    if kind == "relation":
        inf = [judged[i["ref"]].get("informative") for i in real if i["ref"] in judged]
        out["informative_rate"] = round(sum(1 for x in inf if x == "yes") / len(inf), 4) if inf else None
    return out


def _binary(kind, label):
    return "pos" if label in POSITIVE[kind] else "neg"


# ── main ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--merge-map", nargs="*", default=[])
    ap.add_argument("--n-entity", type=int, default=200)
    ap.add_argument("--n-relation", type=int, default=200)
    ap.add_argument("--n-merge", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--local-url", default="http://localhost:8090/v1")
    ap.add_argument("--local-model", default="qwen3.8-27b")
    ap.add_argument("--judge-workers", type=int, default=8, help="parallel judge batches (local server only)")
    ap.add_argument("--cross-family", type=int, default=100, help="items re-judged by the second family (0 = off)")
    ap.add_argument("--cross-family-url", default=None)
    ap.add_argument("--cross-family-model", default="mistral-medium-3.5-128b")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    a = ap.parse_args()
    from memento.graph.store import get_store
    local = Judge(a.local_url, a.local_model, workers=a.judge_workers)
    report = {"domain": a.domain, "thresholds": THRESHOLDS, "seed": a.seed, "local_judge": a.local_model,
              "metrics": {}, "samples": {}}
    all_items: dict[str, list[dict]] = {}
    judged_all: dict[str, dict[str, dict]] = {}
    with get_store() as store:
        report["census"] = census(store)
        ment, rel = load_rows(store)

        def run(kind, items, field=None, controls=None):
            items = _with_refs(items, kind[0])
            ctrl = controls if controls is not None else negative_controls(items, 30, a.seed, field)
            ctrl = _with_refs(ctrl, f"{kind[0]}c") if controls is not None else ctrl
            batch = items + ctrl
            random.Random(a.seed).shuffle(batch)
            judged = local.judge(kind, batch, a.domain)
            return items + ctrl, judged

        # entity / relation, with one resample on INCONCLUSIVE
        for kind, rows, n, sampler, field in (("entity", ment, a.n_entity, sample_entities, "entity_name"),
                                              ("relation", rel, a.n_relation, sample_relations, "tail")):
            if not rows:
                report["metrics"][kind] = {"metric": METRIC[kind], "verdict": "NO DATA", "n": 0}
                continue
            first = sampler(rows, n, a.seed)
            items, judged = run(kind, first, field)
            res = score(kind, items, judged)
            if res["verdict"] == "INCONCLUSIVE":
                second = sampler(rows, n, a.seed + 1, exclude={r["_uid"] for r in first})
                more = _with_refs(second, f"{kind[0]}2_")
                judged.update(local.judge(kind, more, a.domain))
                items = items + more
                res = score(kind, items, judged) | {"resampled": True}
            report["metrics"][kind], all_items[kind], judged_all[kind] = res, items, judged

        if a.merge_map:
            m_items, m_ctrl, by_map = merge_items(store, a.merge_map, a.n_merge, a.seed)
            items, judged = run("merge", m_items, controls=m_ctrl)
            per_t = {}
            for t, pairs in by_map.items():
                keys = {tuple(p) for p in pairs}
                sub = [i for i in items if not i.get("is_control") and tuple(i["pair"]) in keys]
                per_t[t] = score("merge", sub + [i for i in items if i.get("is_control")], judged)
            report["metrics"]["merge"] = {"by_threshold": per_t}
            all_items["merge"], judged_all["merge"] = items, judged

        sw_items, sw_ctrl = same_work_items(store, a.seed)
        if sw_items:
            items, judged = run("same_work", sw_items, controls=sw_ctrl)
            res = score("same_work", items, judged)
            kinds = Counter(i["kind"] for i in items if not i.get("is_control"))
            res["by_kind"] = {k: score("same_work", [i for i in items if i.get("kind") == k or i.get("is_control")],
                                       judged)["estimate"] for k in kinds}
            report["metrics"]["same_work"], all_items["same_work"], judged_all["same_work"] = res, items, judged
        else:
            report["metrics"]["same_work"] = {"metric": METRIC["same_work"], "verdict": "NO DATA", "n": 0}

    # self-consistency: 50 real items re-judged in a different batch order
    rng = random.Random(a.seed + 7)
    pool = [(k, i) for k, its in all_items.items() for i in its
            if not i.get("is_control") and i["ref"] in judged_all[k]]
    sc = rng.sample(pool, min(50, len(pool)))
    same = 0
    for kind in {k for k, _ in sc}:
        its = [dict(i) for k, i in sc if k == kind]
        rng.shuffle(its)
        again = local.judge(kind, its, a.domain)
        same += sum(1 for i in its if i["ref"] in again
                    and again[i["ref"]]["label"] == judged_all[kind][i["ref"]]["label"])
    report["self_consistency"] = {"n": len(sc), "agreement": round(same / len(sc), 4) if sc else None}

    # cross-family comparison on the same items
    if a.cross_family > 0 and pool:
        url = a.cross_family_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        cloud = a.cross_family_url is None
        other = Judge(url, a.cross_family_model, cloud_api_key() if cloud else "none",
                      min_interval_s=6.0 if cloud else 0.0, max_calls=20 if cloud else None, local=not cloud)
        per_kind = defaultdict(list)
        for k, i in pool:
            per_kind[k].append(i)
        share = max(1, a.cross_family // max(1, len(per_kind)))
        cf = {}
        try:
            for kind, its in sorted(per_kind.items()):
                pick = random.Random(a.seed + 3).sample(its, min(share, len(its)))
                theirs = other.judge(kind, pick, a.domain)
                both = [i["ref"] for i in pick if i["ref"] in theirs]
                la = [_binary(kind, judged_all[kind][r]["label"]) for r in both]
                lb = [_binary(kind, theirs[r]["label"]) for r in both]
                cf[kind] = {"n": len(both), "agreement": round(sum(x == y for x, y in zip(la, lb)) / len(both), 4)
                            if both else None, "kappa": round(cohen_kappa(la, lb), 4) if both else None}
            report["cross_family"] = {"model": a.cross_family_model, "url": url, "calls": other.calls, "by_kind": cf}
        except Exception as e:
            report["cross_family"] = {"model": a.cross_family_model, "url": url, "calls": other.calls,
                                      "by_kind": cf, "error": repr(e)}

    a.out_dir.mkdir(parents=True, exist_ok=True)
    out = a.out_dir / f"graph_audit_{a.domain}.json"
    report["items"] = {k: [{kk: vv for kk, vv in i.items() if kk not in ("passage", "text_a", "text_b")}
                           | {"judged": judged_all[k].get(i["ref"])} for i in its]
                       for k, its in all_items.items()}
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    out.with_suffix(".md").write_text(render_markdown(report))
    print(f"wrote {out} (+ .md)")


def render_markdown(r: dict) -> str:
    L = [f"# Graph-quality audit: {r['domain']}", "",
         "| metric | n | estimate | 95% CI | threshold | verdict | control FP | judge valid | unparsed |",
         "|---|---:|---:|---|---:|---|---:|---|---:|"]

    def row(name, m):
        if m.get("verdict") == "NO DATA":
            L.append(f"| {name} | 0 | – | – | – | NO DATA | – | – | – |")
            return
        L.append(f"| {name} | {m['n']} | {m['estimate']:.3f} | {m['ci'][0]:.3f}–{m['ci'][1]:.3f} | "
                 f"{m['threshold']:.2f} | {m['verdict']}{' (resampled)' if m.get('resampled') else ''} | "
                 f"{m['control_fp_rate'] if m['control_fp_rate'] is not None else '–'} | "
                 f"{'yes' if m['judge_valid'] else 'NO'} | {m['n_unparsed']} |")
    for kind in ("same_work", "entity", "relation"):
        if kind in r["metrics"]:
            row(r["metrics"][kind].get("metric", kind), r["metrics"][kind])
    for t, m in sorted(r["metrics"].get("merge", {}).get("by_threshold", {}).items()):
        row(f"merge_precision @ {t}", m)
    c = r["census"]
    L += ["", "## Census", "",
          f"Chunks {c['n_chunks']}, with ≥1 mention {c['chunks_with_mentions']} (coverage {c['coverage']:.1%}); "
          f"entities {c['n_entities']} (registry {c['canonical_entities']}); mentions {c['n_mentions']}; "
          f"top-50 entities hold {c['top50_mention_share']:.1%} of mentions; "
          f"{c['mentions_per_covered_chunk']} mentions per covered chunk.", "",
          "Edges: " + ", ".join(f"{k} {v}" for k, v in sorted(c["edges_by_type"].items())), "",
          "## Judge validity", "",
          f"Self-consistency: {r['self_consistency']['agreement']} over {r['self_consistency']['n']} items."]
    cf = r.get("cross_family")
    if cf:
        L += ["", f"Cross-family ({cf['model']}, {cf['calls']} calls{', ERROR: ' + cf['error'] if cf.get('error') else ''}):", "",
              "| kind | n | agreement | kappa |", "|---|---:|---:|---:|"]
        for k, v in sorted(cf["by_kind"].items()):
            L.append(f"| {k} | {v['n']} | {v['agreement']} | {v['kappa']} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
