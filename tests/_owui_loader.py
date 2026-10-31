"""An Open WebUI function file loaded the way Open WebUI 0.11.4 loads it (open_webui/utils/plugin.py,
load_function_module_by_id): the source through the installed loader's replace_imports, a temporary file as
`__file__`, executed as module `function_<id>` registered in sys.modules, with MEMENTO_DIR on sys.path.

The module name starts with `function_`, so the module's own guard (a top-level `if` on `__name__`) would start the
warm-up worker and publish the copy's model map into OUTPUT_ROOT; with start_workers=False that guarded block is left
out and no thread named memento-warmup* starts."""
import atexit
import ast
import contextlib
import functools
import importlib.util
import os
import sys
import tempfile
import threading
import types
from pathlib import Path

from memento.config.paths import PROJECT_ROOT


def _plugin_file() -> Path:
    spec = importlib.util.find_spec("open_webui")              # the installed package, not imported
    return Path(spec.submodule_search_locations[0]) / "utils" / "plugin.py"


@functools.cache
def _replace_imports():
    path = _plugin_file()
    fn = next(n for n in ast.parse(path.read_text(encoding="utf-8")).body
              if isinstance(n, ast.FunctionDef) and n.name == "replace_imports")
    ns: dict = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), ns)
    return ns["replace_imports"]


def replace_imports(src: str) -> str:
    return _replace_imports()(src)


@contextlib.contextmanager
def _no_warmup_threads():
    start = threading.Thread.start

    def guarded(self):
        if not (self.name or "").startswith("memento-warmup"):
            start(self)
    threading.Thread.start = guarded
    try:
        yield
    finally:
        threading.Thread.start = start


def _without_name_guards(tree: ast.Module) -> ast.Module:
    """The module without its top-level `if` statements that test `__name__` (the "inside Open WebUI" block)."""
    def guard(n):
        return isinstance(n, ast.If) and any(isinstance(x, ast.Name) and x.id == "__name__" for x in ast.walk(n.test))
    tree.body = [n for n in tree.body if not guard(n)]
    return tree


def load_function(path: Path, function_id: str, *, start_workers: bool = False) -> types.ModuleType:
    src = replace_imports(Path(path).read_text(encoding="utf-8"))
    name = f"function_{function_id}"
    fd, tmp = tempfile.mkstemp(prefix=f"{name}_", suffix=".py")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(src)
    atexit.register(lambda: os.path.exists(tmp) and os.remove(tmp))
    root = os.getenv("MEMENTO_DIR") or str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    mod = types.ModuleType(name)
    mod.__dict__["__file__"] = tmp
    sys.modules[name] = mod
    code = src if start_workers else _without_name_guards(ast.parse(src, tmp))
    with contextlib.nullcontext() if start_workers else _no_warmup_threads():
        exec(compile(code, tmp, "exec"), mod.__dict__)
    return mod
