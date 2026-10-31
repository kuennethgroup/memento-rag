# Pipes as Open WebUI actually stores them. Open WebUI's loader rewrites "from config" → "from open_webui.config"
# (utils/plugin.replace_imports) when it saves a function, so a "from memento.config.settings import ..." would always fail inside
# Open WebUI and the pipe would run on its fallback instruction. These tests load the rewritten source.
import ast
import types
from pathlib import Path

import pytest
import yaml
from memento.config.paths import PROJECT_ROOT

ROOT = PROJECT_ROOT
replace_imports = pytest.importorskip("open_webui.utils.plugin").replace_imports


def _stored(name: str) -> str:
    return replace_imports((ROOT / "openwebui/functions" / name).read_text())


@pytest.mark.parametrize("name", ["query_pipe.py", "ingestion_pipe.py", "qdrant_pipe.py"])
def test_no_import_is_mangled_by_openwebui(name):
    src = (ROOT / "openwebui/functions" / name).read_text()
    before = {n.module for n in ast.walk(ast.parse(src)) if isinstance(n, ast.ImportFrom)}
    after = {n.module for n in ast.walk(ast.parse(_stored(name))) if isinstance(n, ast.ImportFrom)}
    assert before == after, f"OpenWebUI would rewrite: {sorted(after - before)}"


def test_stored_query_pipe_loads_the_valve_domain(monkeypatch):
    monkeypatch.setenv("MEMENTO_DIR", str(ROOT))
    mod = types.ModuleType("stored_query_pipe")
    mod.__dict__["__file__"] = str(ROOT / "openwebui/functions/query_pipe.py")
    exec(_stored("query_pipe.py"), mod.__dict__)
    want = yaml.safe_load((ROOT / "config/domains/generic.yaml").read_text())["instruction"]
    assert mod.request.domain("generic")["instruction"] == want != mod.request.FALLBACK_INSTRUCTION
    assert mod.request.domain("ceramics")["instruction"] != want


def _load_stored(name: str, monkeypatch, **env):
    """The stored function as a freshly started Open WebUI loads it (the package's domain cache empty, as at start)."""
    from memento.query import request
    monkeypatch.setattr(request, "_DOMAINS", {})
    monkeypatch.setenv("MEMENTO_DIR", str(ROOT))
    for k, v in env.items():
        monkeypatch.setenv(k, v) if v is not None else monkeypatch.delenv(k, raising=False)
    mod = types.ModuleType(f"stored_{name[:-3]}")
    mod.__dict__["__file__"] = str(ROOT / "openwebui/functions" / name)
    exec(_stored(name), mod.__dict__)
    return mod


@pytest.mark.parametrize("name", ["query_pipe.py", "ingestion_pipe.py"])
def test_default_collection_follows_the_domain_profile(name, monkeypatch):
    mod = _load_stored(name, monkeypatch, DOMAIN_NAME="generic", COLLECTION_PREFIX=None)
    assert mod.Pipe.Valves().COLLECTION == "memento-generic-rag"
    mod = _load_stored(name, monkeypatch, DOMAIN_NAME="ceramics", COLLECTION_PREFIX=None)
    assert mod.Pipe.Valves().COLLECTION == "memento-ceramics-rag"
    mod = _load_stored(name, monkeypatch, DOMAIN_NAME="ceramics", COLLECTION_PREFIX="custom")
    assert mod.Pipe.Valves().COLLECTION == "custom-rag"


def test_answer_uses_the_domain_system_prompt():
    # each domain gets its own prompt, never a hard-coded one
    src = _stored("query_pipe.py")
    # via prompts.answer_prompt (the domain's, else generic.yaml's; behaviour tested in test_pipe_logic)
    assert "turn.system_prompt = prompts.answer_prompt(turn.dom)" in (ROOT / "memento/query/flows/answer.py").read_text()
    assert 'if (dom or {}).get("system_prompt"):' in (ROOT / "memento/query/prompts.py").read_text()
