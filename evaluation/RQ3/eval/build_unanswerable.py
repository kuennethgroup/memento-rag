"""RQ3 step 4.2 — verified-unanswerable questions (near misses) per corpus.

Gemma writes candidate questions from a real passage: about the same material/topic, asking for a specific fact the
passage does not give. A candidate is kept only if the WHOLE corpus does not answer it:
  1. hybrid search, top 100 passages (no rerank cut-off) → Gemma judges, in batches, whether any passage answers it;
  2. the production top 20 (with rerank) → qwen judges the same independently (cross-family).
Kept only if both find no answering passage. 12 per corpus (polymer: 6 German + 6 English).
Output: data/rq3/eval/unanswerable_<corpus>.jsonl (+ a log of rejected candidates).
"""
import json, os, random, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT))
for l in (ROOT / "config/.env").read_text().splitlines():
    l = l.strip()
    if l and not l.startswith("#") and "=" in l:
        k, v = l.split("=", 1); os.environ.setdefault(k, v.strip().strip('"').strip("'"))
import yaml
from openai import OpenAI
from qdrant_client import QdrantClient
from memento.retrieval.search import search
from memento.agentic_validation.scientific_checks import plain

# a question pointing at an unnamed context cannot be judged "not in the corpus" (and no user would ask it that way)
DEICTIC = re.compile(r"\b(this|the present|the current|in the) (study|work|paper|article|thesis|investigation)\b|"
                     r"\b(diese[rnms]?|vorliegende[rnms]?) (Arbeit|Studie|Untersuchung|Dissertation)\b|"
                     r"\bthe (western blot|figure|table|sample|samples|experiment)\b(?! of| for| from| with)", re.I)


def clean_question(q: str) -> str | None:
    q = re.sub(r"\s+", " ", plain(q).replace("$", "")).strip().strip('"')
    if not q.endswith("?") or len(q) < 30 or DEICTIC.search(q):
        return None
    return q

GEMMA = (OpenAI(base_url="http://localhost:8095/v1", api_key="none"), "gemma-4-12b")
QWEN = (OpenAI(base_url="http://localhost:8090/v1", api_key=os.environ.get("REASONING_API_KEY", "none")), "qwen3.8-27b")
COLL = {"ceramics": "memento-ceramics-rag", "polymer-engineering": "memento-polymer-rag"}
WANT = {"ceramics": {"en": 12}, "polymer-engineering": {"de": 6, "en": 6}}

GEN = """\
Below is a passage from a scientific document. Write ONE question in {lang} that a researcher could plausibly ask \
about the SAME material, sample or topic, but whose answer is NOT in the passage: ask for a specific value, condition, \
comparison or result the passage does not give. The question must be self-contained (name the material/topic), \
specific, and answerable in principle by an experiment. Output only the question."""

JUDGE = """\
Question: {q}

Below are numbered passages. Does ANY passage state the answer to the question (fully or the key part of it)? \
Answer with JSON: {{"answers": true|false, "passage": <number or null>, "why": "short"}}"""


def ask(client_model, messages, max_tokens=300):
    client, model = client_model
    extra = {"chat_template_kwargs": {"enable_thinking": False}} if "qwen" in model else None
    r = client.chat.completions.create(model=model, messages=messages, temperature=0.0, max_tokens=max_tokens,
                                       extra_body=extra)
    return (r.choices[0].message.content or "").strip()


def answered(client_model, q, hits, batch=10):
    for k in range(0, len(hits), batch):
        chunk = hits[k:k + batch]
        text = "\n\n".join(f"[{k + i + 1}] {h['text'][:2500]}" for i, h in enumerate(chunk))
        out = ask(client_model, [{"role": "user", "content": JUDGE.format(q=q) + "\n\n" + text}])
        m = re.search(r"\{.*\}", out, re.S)
        try:
            d = json.loads(m.group(0))
        except Exception:
            return True, "unparseable judge output — treated as answered (conservative)"
        if d.get("answers") is True:
            return True, f"passage {d.get('passage')}: {d.get('why', '')[:150]}"
    return False, ""


FIG = """\
Question: {q}

Caption / text of the figure:
{cap}

Does this figure (plotted values, trends, legend, labels) or its caption give the answer to the question, fully or \
the key part of it? Output ONLY JSON: {{"answers": true|false, "why": "short"}}"""


def figure_answers(q, hits):
    """A text-only check misses answers shown in figures, so every figure among the production top-20 is shown to
    Gemma."""
    from memento.retrieval.answer_pipeline import image_data_url
    for h in hits:
        if h.get("type") not in ("image", "chart") or not h.get("img_path"):
            continue
        url = image_data_url(h["img_path"])
        if not url:
            continue
        out = GEMMA[0].chat.completions.create(model=GEMMA[1], temperature=0.0, max_tokens=300, messages=[
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                         {"type": "text", "text": FIG.format(q=q, cap=h["text"][:1500])}]}]
        ).choices[0].message.content or ""
        m = re.search(r"\{.*\}", out, re.S)
        try:
            if json.loads(m.group(0)).get("answers") is True:
                return True, f"figure {h.get('source')} p.{h.get('page')}"
        except Exception:
            return True, "unparseable figure check — treated as answered"
    return False, ""


def main(corpus):
    dom = yaml.safe_load((ROOT / f"config/domains/{corpus}.yaml").read_text())
    client = QdrantClient(url="http://localhost:6333", timeout=60)
    passages = json.loads((ROOT / f"data/rq3/benchmark/passages_{corpus}.json").read_text())
    pool = [p for p in passages if p["type"] in ("text", "table")]
    rng = random.Random(11); rng.shuffle(pool)
    kept, rejected, need = [], [], dict(WANT[corpus])
    out = ROOT / f"data/rq3/eval/unanswerable_{corpus}.jsonl"
    used, retired = set(), set()                        # retired qids are never reused
    if out.exists():                                    # top-up: re-check existing questions with the filter
        for r in (json.loads(l) for l in open(out)):
            q = clean_question(r["question"])
            used.add(r.get("seed_passage"))
            if q and r.get("figure_checked") is not True:
                hits = search(client, COLL[corpus], q, embedder_url="http://localhost:8021",
                              query_instruction=dom["instruction"], retrieve_k=80, hybrid=True, hybrid_prefetch_k=80,
                              rerank_enabled=True, reranker_url="http://localhost:8022", rerank_top=20,
                              rerank_min_score=0.03, image_rerank_ratio=0.2)[0]
                fa, why = figure_answers(q, hits)
                if fa:
                    retired.add(r["qid"])
                    rejected.append({"question": r["question"], "why": "figure answers it: " + why})
                    print("rejected (figure):", q[:90], "|", why, flush=True)
                    continue
                r["figure_checked"] = True
            if q and need.get(r["lang"], 0) > 0:
                need[r["lang"]] -= 1
                kept.append({**r, "question": q})            # qid kept: frozen records are matched by qid
            else:
                retired.add(r["qid"])
                rejected.append({"question": r["question"], "why": "filter: not self-contained / format"})
    pool = [p for p in pool if p["key"] not in used]
    for p in pool:
        lang = next((l for l, n in need.items() if n > 0), None)
        if lang is None:
            break
        q = clean_question(ask(GEMMA, [{"role": "system", "content": GEN.format(lang={"de": "German", "en": "English"}[lang])},
                                       {"role": "user", "content": p["text"][:3000]}]))
        if not q:
            continue

        def s(k, rerank):
            hits, _ = search(client, COLL[corpus], q, embedder_url="http://localhost:8021",
                             query_instruction=dom["instruction"], retrieve_k=k, hybrid=True, hybrid_prefetch_k=k,
                             rerank_enabled=rerank, reranker_url="http://localhost:8022", rerank_top=20 if rerank else k,
                             rerank_min_score=0.03, image_rerank_ratio=0.2)
            return hits
        a1, why1 = answered(GEMMA, q, s(100, False))
        top20 = [] if a1 else s(80, True)
        a2, why2 = (False, "") if a1 else answered(QWEN, q, top20)
        a3, why3 = (False, "") if (a1 or a2) else figure_answers(q, top20)
        if a1 or a2 or a3:
            why = why1 or why2 or why3
            rejected.append({"question": q, "why": why}); print("rejected:", q[:100], "|", why[:80], flush=True)
            continue
        need[lang] -= 1
        nxt = 1 + max([int(k["qid"].rsplit(":", 1)[1]) for k in kept] + [0] +
                      [int(x.rsplit(":", 1)[1]) for x in retired])
        kept.append(dict(qid=f"{corpus}:unanswerable:{nxt}", group="unanswerable", question=q, lang=lang,
                         reference=None, gold_pages=None, gold_docs=None, seed_passage=p["key"], figure_checked=True))
        print("KEPT:", q, flush=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept))
    (ROOT / f"data/rq3/eval/unanswerable_rejected_{corpus}.json").write_text(json.dumps(rejected, ensure_ascii=False, indent=1))
    print(f"DONE {corpus}: kept {len(kept)}, rejected {len(rejected)}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
