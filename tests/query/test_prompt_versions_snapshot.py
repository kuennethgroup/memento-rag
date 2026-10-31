# What a stored answer is keyed by and what the answer model is told must not change when the pipes' code moves:
# the prompt versions of every picker answer, the answer prompt per domain, and every text and setting the two pipes
# define (their hashes). Run through scripts/test_offline.sh (the snapshot was recorded in its environment).
import ast
import os
from pathlib import Path

import pytest

from _snapshots import check_snapshot, plain, recorded_env, sha
from memento.query import cache, prompts, request

ROOT = Path(__file__).resolve().parents[2]
DOMAINS = sorted(p.stem for p in (ROOT / "config" / "domains").glob("*.yaml"))
SUMMARY_MODES = (None, "whole", "partial")       # summary.summary_hits: the whole document, a selection, or nothing
TITLES = ("A study of sample hardness", "Talk: Results of the group")   # gap_system_prompt: a paper's, a talk's
CONSTANT = r"^_?[A-Z][A-Z0-9_]*$"


def _load(path: str, function_id: str):
    from _owui_loader import load_function
    with recorded_env():
        return load_function(ROOT / path, function_id)


@pytest.fixture(scope="module")
def qp():
    return _load("openwebui/functions/query_pipe.py", "memento_query")


def _defined_constants(path: str) -> list[str]:
    """Module-level names in UPPER_CASE the file itself assigns (not the ones it imports)."""
    import re
    names = []
    for node in ast.parse((ROOT / path).read_text()).body:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        names += [t.id for t in targets if isinstance(t, ast.Name) and re.match(CONSTANT, t.id)]
    return sorted(set(names))


# Module-level caches (empty at import, filled while the code runs; design §6.2): not constants, their content depends
# on the tests that ran before.
CACHES = {"_DOMAINS", "_ALIASES", "_TITLES", "_LIB_TITLES", "_TABLES_READY", "_STATE_CACHE"}
# The package modules a pipe's definitions moved to (the query pipe's: every module of memento.query, and Document
# Chat's query side; the upload pipe's: memento.ingestion.upload); a moved constant is looked up there.
def _owners(path: str) -> list:
    import importlib
    import importlib.util
    import pkgutil
    if path.endswith("ingestion_pipe.py"):
        import memento.ingestion.upload as u
        return [importlib.import_module(m.name) for m in pkgutil.walk_packages(u.__path__, "memento.ingestion.upload.")]
    if not path.endswith("query_pipe.py"):
        return []
    import memento.query as q
    names = sorted(m.name for m in pkgutil.walk_packages(q.__path__, "memento.query."))
    for extra in ("memento.document_chat.chat", "memento.document_chat.cleanup"):
        if importlib.util.find_spec(extra):
            names.append(extra)
    return [importlib.import_module(n) for n in names]


def _recorded_constants(path: str) -> list[str]:
    """The constants the file had when its snapshot was recorded (the file's own today, before a first recording)."""
    snap = ROOT / "tests" / "fixtures" / f"constants_{Path(path).stem}_snapshot.json"
    import json
    names = set(json.loads(snap.read_text())) if snap.exists() else set(_defined_constants(path))
    return sorted(names - CACHES)


# Constants a pipe repeated from the package: only the package's own definition is left (design anti-goal: two copies),
# so the value is read there.
PACKAGE_OWN = {"_JUNK_PDF_TITLE": "memento.library.documents", "STATE_CACHE_S": "memento.query.library_state"}


def _lookup(modules: list, name: str):
    """A constant by its name in the modules that hold it now (a moved name may have lost its underscore)."""
    import importlib
    if name in PACKAGE_OWN:
        modules = [importlib.import_module(PACKAGE_OWN[name])]
    for m in modules:
        for n in (name, name.lstrip("_")):
            if hasattr(m, n):
                return getattr(m, n)
    raise AttributeError(name)


def _constant_hashes(modules: list, path: str) -> dict:
    import json
    out = {}
    for name in _recorded_constants(path):
        value = _lookup(modules, name)
        if callable(value) and not isinstance(value, type):
            continue
        try:
            text = value if isinstance(value, str) else json.dumps(plain(value), sort_keys=True, ensure_ascii=False)
        except TypeError:
            continue                    # not data (a class, a client): not a text or a setting
        tmp = os.environ.get("TMPDIR")
        text = text.replace(tmp, "<TMP>") if tmp else text            # test_offline.sh's per-run folder
        out[name] = sha(text.replace(str(ROOT), "<ROOT>"))              # the checkout's own path
    return out


def test_prompt_versions_unchanged(qp):
    with recorded_env():
        pipe = qp.Pipe()
    got = {}
    for d in DOMAINS:
        dom = request.domain(d)
        for mode in SUMMARY_MODES:
            got[f"summary|{d}|{mode}"] = cache.prompt_version("summary", mode, dom, pipe.valves, think=False)
        for think in (True, False):
            for title in TITLES:
                got[f"gaps|{d}|think={think}|{title}"] = cache.prompt_version("gaps", "capped", dom, pipe.valves,
                                                                                      think=think, title=title)
    check_snapshot("prompt_versions_snapshot", {"prompt_versions": got})


def test_answer_prompts_unchanged(qp):
    got = {d: sha(prompts.answer_prompt(request.domain(d))) for d in DOMAINS}
    got["<no domain>"] = sha(prompts.answer_prompt(None))
    check_snapshot("answer_prompts_snapshot", {"answer_prompt_sha256": got})


@pytest.mark.parametrize("path", ["openwebui/functions/query_pipe.py", "openwebui/functions/ingestion_pipe.py"])
def test_pipe_constants_unchanged(qp, path):
    modules = ([qp] if path.endswith("query_pipe.py") else [_load(path, "memento_ingestion")]) + _owners(path)
    check_snapshot(f"constants_{Path(path).stem}_snapshot", _constant_hashes(modules, path))
