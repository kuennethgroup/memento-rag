"""Config manifest for every graph build."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from memento.config.paths import PROJECT_ROOT

ROOT = PROJECT_ROOT


def git_state() -> dict:
    def run(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}


def file_sha256(path: Path) -> str:
    p = Path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.exists() else ""


def make_manifest(*, domain, args, summaries, extra=None):
    from memento.config.settings import CONTEXT_BATCH_SIZE, CONTEXT_WORKERS, DOMAIN, REASONING_URL
    ok = [s for s in summaries if not s.get("failed")]
    totals: dict[str, float] = {}
    for s in ok:
        for k, v in s.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                totals[k] = totals.get(k, 0) + v
    reg = DOMAIN.get("entity_registry", "generic")
    body = {
        "domain": domain, "args": args, "git": git_state(),
        "extractor": {"model": os.getenv("REASONING_MODEL_NAME", ""), "url": REASONING_URL,
                      "temperature": 0.1, "batch_size": CONTEXT_BATCH_SIZE, "workers": CONTEXT_WORKERS},
        "hashes": {"registry": file_sha256(ROOT / "memento" / "graph" / "registries" / f"entity_registry_{reg}.py"),
                   "registry_base": file_sha256(ROOT / "memento" / "graph" / "registry_base.py"),
                   "llm_enrich": file_sha256(ROOT / "memento" / "graph" / "build" / "llm_enrich.py"),
                   "document_registry": file_sha256(ROOT / "data" / f"document_registry_{domain}.json")},
        "totals": totals, "n_papers": len(ok),
        "failed": sorted(s["paper_id"] for s in summaries if s.get("failed")),
        "extra": extra or {},
    }
    body["manifest_id"] = hashlib.sha1(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()[:12]
    return body
