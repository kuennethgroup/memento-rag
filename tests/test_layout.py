# The package layout: one memento/ package with one project root, no module of the old flat layout left, every
# documented command pointing at something that exists, and a package whose import starts nothing.
import ast
import importlib
import importlib.util
import pkgutil
import re
import subprocess
import threading
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ("retrieval", "ingestion", "graph", "agentic_validation", "config", "query", "library", "research_gaps",
          "document_chat", "common")
SHELL = ("run_query.sh", "run_ingestion.sh", "scripts/services.sh")


def _py(*dirs):
    for d in dirs:
        for p in sorted((ROOT / d).rglob("*.py")):
            if "fixtures" not in p.relative_to(ROOT).parts and "__pycache__" not in p.parts:
                yield p


def test_project_root_is_repo_root():
    from memento.config.paths import PROJECT_ROOT
    assert (PROJECT_ROOT / "pyproject.toml").is_file()
    assert (PROJECT_ROOT / "memento" / "__init__.py").is_file()
    assert (PROJECT_ROOT / "config" / "domains").is_dir()


def test_no_module_computes_its_own_root():
    listed = [f"{p.relative_to(ROOT)}:{n.lineno}" for p in _py("memento") if p != ROOT / "memento/config/paths.py"
              for n in ast.walk(ast.parse(p.read_text())) if isinstance(n, ast.Name) and n.id == "__file__"]
    assert listed == []


def _call_name(func) -> str:
    return func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else ""


def test_no_legacy_top_level_imports():
    found = []
    for p in _py("memento", "openwebui/functions", "scripts", "evaluation", "tests"):
        for n in ast.walk(ast.parse(p.read_text())):
            names = []
            if isinstance(n, ast.Import):
                names = [a.name for a in n.names]
            elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
                names = [n.module]
            elif isinstance(n, ast.Call) and _call_name(n.func) in ("import_module", "setattr", "patch") and n.args \
                    and isinstance(n.args[0], ast.Constant) and isinstance(n.args[0].value, str):
                names = [n.args[0].value] if "." in n.args[0].value else []
            found += [f"{p.relative_to(ROOT)}:{n.lineno} {x}" for x in names if x.split(".")[0] in LEGACY]
    assert found == []


def _commands():
    """(where, kind, target): `python -m <module>` and `python <file>.py` in the shell scripts and in the fenced
    shell blocks of every README.md (a *_SCRIPT= assignment counts as a file)."""
    texts = [(f, (ROOT / f).read_text()) for f in SHELL]
    for readme in sorted(ROOT.rglob("README.md")):
        if ".venv" in readme.parts or "build" in readme.relative_to(ROOT).parts:
            continue
        blocks = re.findall(r"```(?:bash|sh|shell|console)?\n(.*?)```", readme.read_text(), re.S)
        texts.append((str(readme.relative_to(ROOT)), "\n".join(blocks)))
    for where, text in texts:
        for ln in text.splitlines():
            ln = ln.split("#", 1)[0] if not ln.lstrip().startswith("#") else ""
            for m in re.finditer(r"python[0-9.]*['\"]?\s+(?:-[A-Za-z]\s+)*-m\s+([\w.]+)", ln):
                yield where, "module", m.group(1)
            for m in re.finditer(r"(?:python[0-9.]*['\"]?\s+|_SCRIPT=)['\"]?([\w$/{}.-]+\.py)\b", ln):
                yield where, "file", m.group(1)


def test_commands_point_at_existing_targets():
    bad = []
    for where, kind, target in _commands():
        if kind == "module":
            if importlib.util.find_spec(target) is None:
                bad.append(f"{where}: -m {target}")
        else:
            rel = re.sub(r"^(\$\{?MEMENTO_DIR\}?|\$\{?ROOT\}?|\.)/", "", target)
            if "$" in rel:
                continue                                  # a variable path: checked where it is assigned
            if not (ROOT / rel).is_file() or rel.startswith("memento/"):
                bad.append(f"{where}: {target}")
    assert bad == []


def test_pyproject_packages():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert data["tool"]["setuptools"]["packages"]["find"]["include"] == ["memento*"]


def test_importing_the_package_starts_no_thread(monkeypatch):
    # command-line modules read --domain when imported (memento.config.cli_domain.apply_domain_arg): give them the
    # current one
    import os
    import sys
    import memento
    domain = os.environ.get("DOMAIN_NAME", "polymer-engineering")
    monkeypatch.setattr(sys, "argv", [sys.argv[0], "--domain", domain])
    monkeypatch.setenv("DOMAIN_NAME", domain)
    before = threading.active_count()
    for mod in pkgutil.walk_packages(memento.__path__, "memento."):
        importlib.import_module(mod.name)
    assert threading.active_count() == before


def test_runtime_links_are_ignored():
    # `-q` takes one path at a time; a path that is not there yet (the release links) must match too
    for name in ("data", "logs", "backups", ".venv"):
        assert subprocess.run(["git", "check-ignore", "-q", name], cwd=ROOT).returncode == 0, name


MERGED_SOURCES = ("memento/retrieval/fusion.py", "memento/agentic_validation/check_notes.py",
                  "memento/research_gaps/literature_search.py", "memento/research_gaps/alexandria.py",
                  "memento/library/copy_state.py", "memento/library/model_links.py", "memento/library/library_db.py",
                  "memento/library/library_list.py", "memento/graph/query/predicate_classify.py",
                  "memento/graph/query/predicate_menu.py", "memento/graph/query/operands.py",
                  "memento/ingestion/point_ids.py", "memento/ingestion/sparse.py", "memento/ingestion/linked_text.py")


def test_merged_sources_gone():
    assert [f for f in MERGED_SOURCES if (ROOT / f).exists()] == []
    defined = {n.name for n in ast.parse((ROOT / "memento/ingestion/run_pipeline.py").read_text()).body
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    assert defined & {"build_embed_input", "enumerate_embeddable", "_fold_context"} == set()


def _top_definitions(path: Path) -> dict:
    """{name: its definition's syntax tree} of a module's top-level functions, classes and assignments."""
    out = {}
    for n in ast.parse(path.read_text()).body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out[n.name] = ast.dump(ast.Module(body=n.body, type_ignores=[]))
        elif isinstance(n, (ast.Assign, ast.AnnAssign)):
            for t in (n.targets if isinstance(n, ast.Assign) else [n.target]):
                if isinstance(t, ast.Name):
                    out[t.id] = ast.dump(n.value)
    return out


def test_the_pipes_moved_code_repeats_no_package_definition():
    # what moved out of each pipe uses the package's own definition, never a second copy of it (design anti-goal): the
    # query pipe's code against the rest of the package, the upload pipe's against the rest including the query side
    groups = [sorted((ROOT / "memento/query").rglob("*.py")) + [ROOT / "memento/document_chat/chat.py"],
              sorted((ROOT / "memento/ingestion/upload").rglob("*.py"))]
    twice = []
    for moved in groups:
        theirs = {}
        for p in sorted((ROOT / "memento").rglob("*.py")):
            if p not in moved:
                for name, body in _top_definitions(p).items():
                    theirs.setdefault((name, body), p)
        twice += [f"{p.relative_to(ROOT)}: {name} (also in {theirs[(name, body)].relative_to(ROOT)})"
                  for p in moved for name, body in _top_definitions(p).items() if (name, body) in theirs]
    assert twice == []


def test_the_query_modules_import_each_other_without_a_cycle():
    # a step may hand its reply to another (gap answer → report), never both ways
    graph = {}
    for p in sorted((ROOT / "memento/query").rglob("*.py")):
        name = ".".join(p.relative_to(ROOT).with_suffix("").parts)
        deps = set()
        for n in ast.parse(p.read_text()).body:
            if isinstance(n, ast.ImportFrom) and (n.module or "").startswith("memento.query"):
                deps |= {f"{n.module}.{a.name}" for a in n.names} | {n.module}
            elif isinstance(n, ast.Import):
                deps |= {a.name for a in n.names if a.name.startswith("memento.query")}
        graph[name] = {d for d in deps if d != name}
    def path_back(start):
        stack, seen = [(start, [start])], set()
        while stack:
            node, path = stack.pop()
            for nxt in graph.get(node, ()):
                if nxt == start:
                    return path + [start]
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append((nxt, path + [nxt]))
        return None
    cycles = [c for c in (path_back(m) for m in graph) if c]
    assert cycles == [], cycles[0]
