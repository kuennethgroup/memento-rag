"""RQ3 step 3 — option C for figure claims: the printed text of each benchmark figure (axis titles, tick labels,
legend entries, panel labels, annotations), transcribed once by qwen (the system model) and cached in
data/rq3/benchmark/figure_text.json. Only the benchmark's figures; a production version would do this at ingestion."""
import json, os, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT))
for l in (ROOT / "config/.env").read_text().splitlines():
    l = l.strip()
    if l and not l.startswith("#") and "=" in l:
        k, v = l.split("=", 1); os.environ.setdefault(k, v.strip().strip('"').strip("'"))
from openai import OpenAI
from memento.retrieval.answer_pipeline import image_data_url

PROMPT = ("Transcribe ALL text printed inside this figure, exactly as written: axis titles with units, every tick "
          "label, legend entries, panel labels, and annotations. Do not describe the figure or read values off the "
          "curves — only printed text. One item per line.")
D = ROOT / "data/rq3/benchmark"
llm = OpenAI(base_url="http://localhost:8090/v1", api_key=os.environ.get("REASONING_API_KEY", "none"))
out_path = D / "figure_text.json"
done = json.loads(out_path.read_text()) if out_path.exists() else {}
imgs = sorted({p["img_path"] for c in ("ceramics", "polymer-engineering")
               for p in json.loads((D / f"passages_{c}.json").read_text()) if p.get("img_path")} - set(done))


def one(path):
    r = llm.chat.completions.create(model="qwen3.8-27b", temperature=0.0, max_tokens=1500,
                                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                                    messages=[{"role": "user", "content": [
                                        {"type": "image_url", "image_url": {"url": image_data_url(path)}},
                                        {"type": "text", "text": PROMPT}]}])
    return path, (r.choices[0].message.content or "").strip()


with ThreadPoolExecutor(4) as pool:
    for path, text in pool.map(one, imgs):
        done[path] = text
        out_path.write_text(json.dumps(done, ensure_ascii=False, indent=1))
print("FIGURE TEXT DONE", len(done))
