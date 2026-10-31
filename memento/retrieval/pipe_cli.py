"""The command-line answer path: the Library Chat pipe itself.

run_query.sh and `python -m memento.retrieval.generate "question"` run openwebui/functions/query_pipe.py, the code Open
WebUI runs, with the pipe's saved settings (its Valves in Open WebUI's database, DATABASE_URL), so a question on the
command line takes exactly the path it takes in the chat.

Always the served library: no collection or domain override. Settings are changed in the pipe's Valves, not here.

bash run_query.sh "<question>"
python -m memento.retrieval.generate "question" --thorough --json out.json"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from memento.config.paths import PROJECT_ROOT

ROOT = PROJECT_ROOT
PIPE_FILE = ROOT / "openwebui" / "functions" / "query_pipe.py"
OLD_FLAGS = {
    "--think": "thinking is decided per question by the pipe (THINKING_AUTO)",
    "--paper": "name the document in the question, as in the chat",
    "--no-rerank": "search settings are the pipe's Valves",
    "--no-expand": "search settings are the pipe's Valves",
    "--retrieve-k": "search settings are the pipe's Valves",
    "--rerank-top": "search settings are the pipe's Valves",
    "--max-tokens": "the answer length is the pipe's Valve MAX_TOKENS",
    "--reasoning-url": "the answer model is chosen with --model (config/query_models.yaml)",
    "--strategy": "the check is the pipe's Thorough check (--thorough)",
    "--validate": "use --thorough (the chat's Thorough check)",
    "--no-validate": "the check is off unless --thorough is given",
}


def load_pipe_module():
    """query_pipe.py as a module that is not OpenWebUI's (`function_…`), so its background warm-up worker stays off."""
    spec = importlib.util.spec_from_file_location("memento_query_cli", PIPE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def saved_valves(engine, function_id: str = "memento_query") -> dict | None:
    """A copy's saved Valves; None when it is not installed. Encrypted Valves: refused (never defaults, N11)."""
    from sqlalchemy import text
    with engine.connect() as c:
        v = c.execute(text("SELECT valves FROM function WHERE id = :id"), {"id": function_id}).scalar()
    if v is None:
        return None
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except ValueError:
            raise SystemExit(f"{function_id}: its saved settings cannot be read (encrypted Valves?)")
    return v if isinstance(v, dict) else None


def make_pipe(mod, engine, function_id: str = "memento_query"):
    """A Pipe of the copy `function_id` with its saved Valves, reading and writing OpenWebUI's database through
    `engine` (inside OpenWebUI, _db_engine is OpenWebUI's own engine on the same database)."""
    saved = saved_valves(engine, function_id)
    if saved is None:
        raise SystemExit(f"Memento Query has no saved settings in this database (function '{function_id}'): "
                         "set it up first with scripts/setup_openwebui.py")
    p = mod.Pipe()
    p.function_id = function_id             # the module is not Open WebUI's `function_<id>`: name the copy
    p.engine = lambda: engine
    v = p.Valves()
    for k, val in saved.items():
        if val is not None and hasattr(v, k):
            setattr(v, k, val)
    p.valves = v
    return p


def build(function_id: str = "memento_query"):
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise SystemExit("DATABASE_URL is not set: source config/.env first (run_query.sh does)")
    from sqlalchemy import create_engine
    return make_pipe(load_pipe_module(), create_engine(url), function_id)


async def ask(pipe, question: str, *, model: str | None = None, thorough: bool = False, history=None,
              on_status=None) -> dict:
    """One question as OpenWebUI sends it: {answer, sources ([N] → name, in order), events}."""
    events = []

    async def emit(e):
        events.append(e)
        if on_status and isinstance(e, dict) and e.get("type") == "status":
            on_status((e.get("data") or {}).get("description") or "")
    body = {"model": f"{getattr(pipe, 'function_id', 'memento_query')}.{model or 'local'}", "stream": True,
            "messages": list(history or []) + [{"role": "user", "content": question}]}
    meta = {"filter_ids": ["memento_thorough"]} if thorough else {}
    chunks = [c async for c in pipe.pipe(body, {"id": "cli", "role": "admin", "name": "command line"}, emit,
                                         None, meta)]
    sources = [((e.get("data") or {}).get("source") or {}).get("name") or "?"
               for e in events if isinstance(e, dict) and e.get("type") == "citation"]
    return {"answer": "".join(c for c in chunks if isinstance(c, str)), "sources": sources, "events": events}


def clean_output(text: str, *, show_thinking: bool = False) -> str:
    """The chat's markdown for a terminal: figures as their caption (no base64), collapsible blocks as plain text."""
    def think(m):
        return f"Thinking:\n{m.group(1).strip()}\n\n" if show_thinking else ""
    t = re.sub(r"(?s)<think>(.*?)</think>", think, text)
    t = re.sub(r"!\[([^\]]*)\]\(data:image/[^)]*\)", r"[figure: \1]", t)
    t = re.sub(r"<details[^>]*>\s*<summary>(.*?)</summary>", r"\1:", t, flags=re.S)
    t = t.replace("</details>", "")
    return re.sub(r"\n{3,}", "\n\n", t).strip()


def model_ids() -> tuple[str, ...]:
    """The enabled answer models of config/query_models.yaml, in its order; empty when the file cannot be read (then
    the pipe decides)."""
    try:
        from memento.query.models import load_variants
        return tuple(v.id for v in load_variants())
    except Exception:
        return ()


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    old = [a.split("=", 1)[0] for a in argv if a.split("=", 1)[0] in OLD_FLAGS]
    if old:
        for a in old:
            print(f"{a} no longer applies: the command line runs Memento Query with its saved settings; "
                  f"{OLD_FLAGS[a]}.", file=sys.stderr)
        raise SystemExit(2)
    ap = argparse.ArgumentParser(description="Ask Memento Query from the command line (the same pipe as the chat).")
    ap.add_argument("query", help="the question, as typed in the chat")
    ap.add_argument("--model", choices=model_ids() or None, default=None, help="answer model (default: local)")
    ap.add_argument("--thorough", action="store_true", help="Thorough check, as the chat's switch")
    ap.add_argument("--show-thinking", action="store_true", help="print the model's reasoning too")
    ap.add_argument("--no-sources", action="store_true", help="do not print the sources list")
    ap.add_argument("--json", metavar="PATH", help="save question, answer and sources as JSON")
    ap.add_argument("--function-id", default="memento_query",
                    help="the Memento Query copy to ask (a second library: memento_query_<SECOND_LIBRARY_ID>)")
    ap.add_argument("--verbose", "-v", action="store_true", help="print the progress lines")
    args = ap.parse_args(argv)

    pipe = build(args.function_id)
    status = (lambda s: print(f"  … {s}", file=sys.stderr)) if args.verbose else None
    out = asyncio.run(ask(pipe, args.query, model=args.model, thorough=args.thorough, on_status=status))
    print(clean_output(out["answer"], show_thinking=args.show_thinking))
    if out["sources"] and not args.no_sources:
        print("\nSources Memento read:")
        for i, name in enumerate(out["sources"], start=1):
            print(f"  [{i}] {name}")
    if args.json:
        Path(args.json).write_text(json.dumps({"question": args.query, "model": args.model or "local",
                                               "thorough": args.thorough, "answer": out["answer"],
                                               "sources": out["sources"]}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
