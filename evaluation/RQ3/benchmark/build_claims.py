"""
RQ3 step 3.2 — labelled claim set for the verifier benchmark and the judge gate.

Passages: real retrieved sources (RQ2 G0 contexts), per corpus 40 text (polymer: 20 German + 20 English), 15 table,
20 figure. For each passage the WRITER model writes 2 supported claims (one in the passage's language, one in the
other of EN/DE) and 1 plausible claim the passage does not state. Written crosswise:
  set "V" (measures the qwen verifier) is written by Gemma; set "J" (gates the Gemma judge) is written by qwen.
Deterministic corruptions of each supported claim (no model): number, unit, material (entity registry), negation;
plus "wrong source": a supported claim paired with a different passage of the same question.
Labels: supported | contradicted (number/unit/material/negation) | unsupported (extrapolation, wrong source).
Label quality: a supported text/table claim must pass the number check against its own passage, else it is dropped;
figure claims are flagged needs_review (checked by looking at the figure). Resumable: one JSONL line per passage.

    .venv/bin/python evaluation/RQ3/benchmark/build_claims.py --set V --corpus ceramics
"""
import argparse, importlib, json, os, random, re, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT))
for l in (ROOT / "config/.env").read_text().splitlines():
    l = l.strip()
    if l and not l.startswith("#") and "=" in l:
        k, v = l.split("=", 1); os.environ.setdefault(k, v.strip().strip('"').strip("'"))
from openai import OpenAI
from memento.agentic_validation.scientific_checks import _QTY, _U, extract_quantities, number_fidelity, plain
from memento.retrieval.answer_pipeline import image_data_url

OUT = ROOT / "data/rq3/benchmark"
WRITERS = {"V": ("http://localhost:8095/v1", "gemma-4-12b", "none"),                    # Gemma writes the qwen test set
           "J": ("http://localhost:8090/v1", "qwen3.8-27b", os.environ.get("REASONING_API_KEY", "none"))}
REG = {"ceramics": "ceramics", "polymer-engineering": "polymer"}
DE = re.compile(r"\b(der|die|das|und|wird|werden|wurde|wurden|ist|sind|mit|von|für|bei|nicht|eine?)\b", re.I)


def lang(t: str) -> str:
    return "de" if len(DE.findall(t or "")) >= 4 else "en"


# ── passages ──────────────────────────────────────────────────────────────────
def sample_passages(corpus: str, seed: int = 27) -> list[dict]:
    seen, pool = set(), {"text_de": [], "text_en": [], "table": [], "figure": []}
    contexts = {}
    for l in open(ROOT / f"data/rq2/runs/final/{corpus}/arms.jsonl"):
        r = json.loads(l)
        if r.get("arm") == "g0" and "context" in r:
            contexts[r["qid"]] = r["context"]
    for qid, ctx in sorted(contexts.items()):
        for h in ctx:
            t = (h.get("text") or "").strip()
            key = (h.get("source"), h.get("page"), t[:120])
            if key in seen:
                continue
            seen.add(key)
            p = {"qid": qid, "source": h.get("source"), "page": h.get("page"), "type": h.get("type"), "text": t,
                 "img_path": h.get("img_path"), "key": f"{h.get('source')}|{h.get('page')}|{abs(hash(t[:200])) % 10**8}"}
            if h.get("type") == "text" and 250 <= len(t) <= 2500:
                pool["text_" + lang(t)].append(p)
            elif h.get("type") == "table" and 150 <= len(t) <= 4000:
                pool["table"].append(p)
            elif h.get("type") in ("image", "chart") and h.get("img_path") and image_data_url(h["img_path"]):
                pool["figure"].append(p)
    rng = random.Random(seed)
    want = ({"text_de": 20, "text_en": 20} if corpus == "polymer-engineering" else {"text_de": 0, "text_en": 40})
    want.update(table=15, figure=20)
    out = []
    for k, n in want.items():
        rng.shuffle(pool[k]); out += pool[k][:n]
    for p in out:                                            # a different passage of the same question (wrong source)
        others = [h for h in contexts[p["qid"]] if h.get("type") in ("text", "table")
                  and (h.get("source"), h.get("page")) != (p["source"], p["page"]) and len(h.get("text") or "") > 200]
        p["other_text"] = rng.choice(others)["text"] if others else None
    return out


# ── writing ───────────────────────────────────────────────────────────────────
WRITE_PROMPT = """\
You write test claims about a {what} from a scientific document.
1. Two SUPPORTED claims: each one atomic statement the {what} states explicitly. Prefer claims with a number and unit, \
a material, or a condition. Copy values exactly. Claim 1 in {lang1}, claim 2 in {lang2}.{fig}
2. One UNSUPPORTED claim: plausible and on the same topic, but NOT stated by the {what} (and not contradicting it \
either) — e.g. a value, cause or comparison the {what} does not give. In {lang1}.
Each claim must be understandable without the {what} (name the material/sample; no "this figure", no "the table").
Output ONLY JSON: {{"supported": ["...", "..."], "unsupported": "..."}}"""
LANG_NAME = {"de": "German", "en": "English"}


def write(llm, model: str, p: dict) -> dict:
    l1 = lang(p["text"]) if p["type"] == "text" else "en"
    l2 = "en" if l1 == "de" else "de"
    what = {"text": "text passage", "table": "table", "image": "figure", "chart": "figure"}[p["type"]]
    fig = (" For the figure: at least one claim must be about something only visible in the image (a plotted value, "
           "trend or legend entry), not just the caption.") if what == "figure" else ""
    sys_p = WRITE_PROMPT.format(what=what, lang1=LANG_NAME[l1], lang2=LANG_NAME[l2], fig=fig)
    if what == "figure":
        content = [{"type": "image_url", "image_url": {"url": image_data_url(p["img_path"])}},
                   {"type": "text", "text": f"Caption / text of the figure:\n{p['text'][:2000]}"}]
    else:
        content = f"{what.capitalize()}:\n{p['text']}"
    extra = {"chat_template_kwargs": {"enable_thinking": False}} if "qwen" in model else None
    for attempt in (0, 1):
        r = llm.chat.completions.create(model=model, temperature=0.0, max_tokens=800, extra_body=extra,
                                        messages=[{"role": "system", "content": sys_p}, {"role": "user", "content": content}])
        raw = r.choices[0].message.content or ""
        m = re.search(r"\{.*\}", raw, re.S)
        try:
            d = json.loads(m.group(0))
            sup = [s for s in d.get("supported", []) if isinstance(s, str) and s.strip()][:2]
            if sup and isinstance(d.get("unsupported"), str):
                return {"supported": list(zip(sup, [l1, l2])), "unsupported": (d["unsupported"], l1)}
        except Exception:
            pass
    return {}


# ── deterministic corruptions ─────────────────────────────────────────────────
def _fmt(v: float, like: str) -> str:
    dec = len(re.split(r"[.,]", like)[-1]) if re.search(r"[.,]\d", like) else 0
    s = f"{v:.{dec}f}"
    return s.replace(".", ",") if "," in like else s


def corrupt_number(claim: str, passage: str):
    for m in _QTY.finditer(plain(claim)):
        raw = m.group("a")
        try:
            v = float(raw.replace(",", "."))
        except ValueError:
            continue
        if v == 0:
            continue
        for factor in (1.5, 0.6, 2.0):
            new = _fmt(v * factor, raw)
            if new != raw and new not in passage:
                txt = plain(claim)
                return txt[:m.start("a")] + new + txt[m.end("a"):]
    return None


UNIT_SWAP = {"MPa": "GPa", "GPa": "MPa", "kPa": "MPa", "µm": "nm", "nm": "µm", "mm": "µm", "min": "h", "h": "min",
             "s": "min", "wt%": "vol%", "vol%": "wt%", "mN": "N", "µN": "mN", "N": "kN", "°C": "K", "K": "°C",
             "g/cm³": "kg/m³", "J/g": "kJ/m²", "Hz": "kHz", "kHz": "Hz"}


def corrupt_unit(claim: str):
    txt = plain(claim)
    for m in _QTY.finditer(txt):
        canon = _U[m.group("unit")][0]
        if canon in UNIT_SWAP:
            return txt[:m.start("unit")] + UNIT_SWAP[canon] + txt[m.end("unit"):]
    return None


def corrupt_material(claim: str, passage: str, registry):
    mats = [e for e in registry.REGISTRY if e.type == "Material"]
    present = {e.canonical_id for e in registry_obj(registry).find_entities_in_text(passage)}
    for e in registry_obj(registry).find_entities_in_text(claim):
        if e.type != "Material":
            continue
        for alias in sorted([e.name] + list(getattr(e, "aliases", ())), key=len, reverse=True):
            m = re.search(rf"(?<![\w-]){re.escape(alias)}(?![\w-])", claim, re.I)
            if not m:
                continue
            rng = random.Random(claim)
            cands = [x for x in mats if x.canonical_id not in present and x.canonical_id != e.canonical_id]
            if cands:
                repl = rng.choice(cands).name
                return claim[:m.start()] + repl + claim[m.end():]
    return None


_REG_CACHE = {}


def registry_obj(module):
    if module.__name__ not in _REG_CACHE:
        from memento.graph.registry_base import build_registry
        _REG_CACHE[module.__name__] = build_registry(module.REGISTRY, module.ENTITY_TYPES)
    return _REG_CACHE[module.__name__]


NEG_EN = re.compile(r"\b(is|are|was|were|has|have|had|can|could|will|does|did)\b")
NEG_DE = re.compile(r"\b(ist|sind|war|waren|wird|werden|wurde|wurden|hat|haben|hatte|kann|können|zeigt|zeigen)\b")


def corrupt_negation(claim: str, lang_: str):
    rx, word = (NEG_DE, "nicht") if lang_ == "de" else (NEG_EN, "not")
    m = rx.search(claim)
    if not m or re.search(rf"\b{word}\b", claim, re.I):
        return None
    return claim[:m.end()] + f" {word}" + claim[m.end():]


# ── one passage ───────────────────────────────────────────────────────────────
def build_one(llm, model, p, registry, set_name, corpus):
    w = write(llm, model, p)
    rows, fig = [], p["type"] in ("image", "chart")
    if not w:
        return {"key": p["key"], "failed": True, "rows": []}
    base = dict(set=set_name, corpus=corpus, passage_key=p["key"], passage_type="figure" if fig else p["type"],
                passage_lang=lang(p["text"]) if p["type"] == "text" else "en", source_text=p["text"],
                img_path=p["img_path"] if fig else None, writer=model)
    kinds = ["number", "unit", "material", "negation"]
    for k, (claim, cl) in enumerate(w["supported"]):
        dropped = (not fig) and bool(number_fidelity("x", claim, [p["text"]]))
        rows.append({**base, "kind": "true", "label": "supported", "text": claim, "claim_lang": cl,
                     "needs_review": fig, "dropped": dropped})
        if dropped:
            continue
        order = kinds[k % 4:] + kinds[:k % 4]
        for kind in order:
            t = {"number": lambda: corrupt_number(claim, p["text"]), "unit": lambda: corrupt_unit(claim),
                 "material": lambda: corrupt_material(claim, p["text"], registry),
                 "negation": lambda: corrupt_negation(claim, cl)}[kind]()
            if t and t != claim:
                rows.append({**base, "kind": kind, "label": "contradicted",
                             "text": t, "claim_lang": cl, "needs_review": False, "dropped": False})
                break
        # wrong source: only when the claim has values and NONE of them is in the other passage (then that passage
        # cannot support it); claims without values could be stated there too, so they are not used
        if k == 0 and p.get("other_text") and extract_quantities(claim) and \
                len(number_fidelity("x", claim, [p["other_text"]])) == len(extract_quantities(claim)):
            if True:
                rows.append({**base, "kind": "wrong_source", "label": "unsupported", "text": claim, "claim_lang": cl,
                             "source_text": p["other_text"], "img_path": None, "passage_type": "text",
                             "needs_review": False, "dropped": False})
    ex, el = w["unsupported"]
    rows.append({**base, "kind": "extrapolation", "label": "unsupported", "text": ex, "claim_lang": el,
                 "needs_review": fig, "dropped": False})
    return {"key": p["key"], "failed": False, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["V", "J"], required=True)
    ap.add_argument("--corpus", choices=["ceramics", "polymer-engineering"], required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    url, model, key = WRITERS[a.set]
    llm = OpenAI(base_url=url, api_key=key)
    registry = importlib.import_module(f"memento.graph.registries.entity_registry_{REG[a.corpus]}")
    passages = sample_passages(a.corpus)
    (OUT / f"passages_{a.corpus}.json").write_text(json.dumps(passages, ensure_ascii=False, indent=1))
    out_path = OUT / f"claims_{a.set}_{a.corpus}.jsonl"
    done = {json.loads(l)["key"] for l in open(out_path)} if out_path.exists() else set()
    todo = [p for p in passages if p["key"] not in done]
    print(f"set {a.set} {a.corpus}: {len(passages)} passages, {len(todo)} to write (writer {model})", flush=True)
    with ThreadPoolExecutor(a.workers) as pool, open(out_path, "a") as f:
        for res in pool.map(lambda p: build_one(llm, model, p, registry, a.set, a.corpus), todo):
            f.write(json.dumps(res, ensure_ascii=False) + "\n"); f.flush()
    rows = [r for l in open(out_path) for r in json.loads(l)["rows"] if not r["dropped"]]
    from collections import Counter
    print("DONE", a.set, a.corpus, dict(Counter((r["kind"], r["passage_type"]) for r in rows)), flush=True)


if __name__ == "__main__":
    main()
