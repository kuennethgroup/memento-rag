"""RQ3 step 4.3 — reference nuggets for the control questions (RQ1 GT v4 gold passages).

Gemma (the judge family) writes 1–4 nuggets per question from the gold passages (+ images for figure/chart targets):
key facts a complete answer must contain, each "vital" or "okay". qwen cross-checks every nugget (stated in the gold
passages AND relevant to the question); rejected nuggets are dropped, agreement is reported.
Output: data/rq3/eval/nuggets_<corpus>.jsonl  {qid, nuggets: [{text, importance, qwen_ok}], gold_passages}
"""
import ast, json, os, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT))
for l in (ROOT / "config/.env").read_text().splitlines():
    l = l.strip()
    if l and not l.startswith("#") and "=" in l:
        k, v = l.split("=", 1); os.environ.setdefault(k, v.strip().strip('"').strip("'"))
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue
from memento.retrieval.answer_pipeline import image_data_url
from memento.agentic_validation.jsonfix import loads_lenient

GEMMA = (OpenAI(base_url="http://localhost:8095/v1", api_key="none"), "gemma-4-12b")
QWEN = (OpenAI(base_url="http://localhost:8090/v1", api_key=os.environ.get("REASONING_API_KEY", "none")), "qwen3.8-27b")
COLL = {"ceramics": "memento-ceramics-rag", "polymer-engineering": "memento-polymer-rag"}
GT = {"ceramics": "evaluation/RQ1/ground_truth_auto_v4_qwenjudge.json",
      "polymer-engineering": "evaluation/RQ1/polymer_ground_truth_auto_v4_qwenjudge.json"}

WRITE = """\
Question: {q}

Below are the gold source passages that answer this question (with a note on what each provides). Write the key facts \
a complete answer must contain, as 1–4 short, self-contained statements ("nuggets"), copying values and units exactly. \
Mark each "vital" (without it the answer is wrong or incomplete) or "okay" (useful supporting detail). Use only what \
the passages state. Output ONLY JSON: [{{"text": "...", "importance": "vital"}}]"""

CHECK = """\
Question: {q}

Gold passages:
{p}

Nuggets:
{n}

For each nugget: is it stated by the gold passages (or clearly shown in the attached figure) AND relevant to answering \
the question? Output ONLY JSON: [{{"id": 1, "ok": true, "why": "short"}}]"""


def chat(cm, content, max_tokens=800):
    c, m = cm
    extra = {"chat_template_kwargs": {"enable_thinking": False}} if "qwen" in m else None
    r = c.chat.completions.create(model=m, temperature=0.0, max_tokens=max_tokens, extra_body=extra,
                                  messages=[{"role": "user", "content": content}])
    return r.choices[0].message.content or ""


def parse_list(raw):
    m = re.search(r"\[.*\]", raw, re.S)
    return loads_lenient(m.group(0)) if m else []


def gold_passages(client, corpus, rc):
    out = []
    for g in rc:
        pts, _ = client.scroll(COLL[corpus], limit=50, with_payload=True, scroll_filter=Filter(must=[
            FieldCondition(key="source", match=MatchValue(value=g["paper_id"])),
            FieldCondition(key="page", match=MatchValue(value=g["page"]))]))
        for p in pts:
            pl = p.payload or {}
            if g.get("type") in (None, pl.get("type")) and pl.get("text"):
                out.append({"source": g["paper_id"], "page": g["page"], "type": pl.get("type"), "text": pl["text"],
                            "img_path": pl.get("img_path"), "rationale": g.get("rationale", "")})
    return out[:6]


def content_for(prompt, passages):
    text = prompt + "\n\n" + "\n\n".join(
        f"[{i + 1}] ({p['source']}, p. {p['page'] + 1}, {p['type']}; note: {p['rationale']})\n{p['text'][:2500]}"
        for i, p in enumerate(passages))
    imgs = [image_data_url(p["img_path"]) for p in passages if p["type"] in ("image", "chart") and p.get("img_path")]
    imgs = [u for u in imgs if u][:1]                      # Gemma is served with 1 image per prompt
    return [{"type": "image_url", "image_url": {"url": imgs[0]}}, {"type": "text", "text": text}] if imgs else text


def main(corpus):
    client = QdrantClient(url="http://localhost:6333", timeout=60)
    rq1 = {q["query"]: q for q in json.load(open(ROOT / GT[corpus]))["queries"]}
    qs = [json.loads(l) for l in open(ROOT / f"data/rq3/eval/questions_{corpus}.jsonl") if '"control"' in l]
    out_path = ROOT / f"data/rq3/eval/nuggets_{corpus}.jsonl"
    done = {json.loads(l)["qid"] for l in open(out_path)} if out_path.exists() else set()
    agree = total = 0
    with open(out_path, "a") as f:
        for q in qs:
            if q["qid"] in done:
                continue
            rc = rq1[q["question"]]["relevant_chunks"]
            rc = ast.literal_eval(rc) if isinstance(rc, str) else rc
            passages = gold_passages(client, corpus, rc)
            nuggets = [n for n in parse_list(chat(GEMMA, content_for(WRITE.format(q=q["question"]), passages)))
                       if isinstance(n, dict) and n.get("text")][:4]
            listing = "\n".join(f"{i + 1}. {n['text']}" for i, n in enumerate(nuggets))
            ptxt = "\n\n".join(f"[{i + 1}] {p['text'][:2500]}" for i, p in enumerate(passages))
            chk = {}
            try:
                imgs = [image_data_url(p["img_path"]) for p in passages if p["type"] in ("image", "chart") and p.get("img_path")]
                imgs = [u for u in imgs if u][:2]
                c = CHECK.format(q=q["question"], p=ptxt, n=listing)
                chk = {int(d["id"]): bool(d.get("ok")) for d in parse_list(
                    chat(QWEN, ([{"type": "image_url", "image_url": {"url": u}} for u in imgs] + [{"type": "text", "text": c}]) if imgs else c))}
            except Exception as e:
                print("check failed", q["qid"], repr(e)[:80], flush=True)
            for i, n in enumerate(nuggets):
                n["qwen_ok"] = chk.get(i + 1)
                total += 1; agree += bool(n["qwen_ok"])
            f.write(json.dumps({"qid": q["qid"], "nuggets": nuggets, "gold_passages": passages}, ensure_ascii=False) + "\n")
            f.flush()
            print(q["qid"], [(n["importance"], n["qwen_ok"], n["text"][:60]) for n in nuggets], flush=True)
    print(f"DONE {corpus}: qwen agreed with {agree}/{total} nuggets", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
