"""The adapters load the way Open WebUI 0.11.4 loads a function (tests/_owui_loader.py): the loader's import rewrite
leaves them unchanged, they run from a temporary file, a copy takes its id from the module name, and the helper
starts no warm-up worker and publishes nothing."""
import threading

import pytest

from _owui_loader import load_function, replace_imports
from memento.config.paths import PROJECT_ROOT

FUNCTIONS = PROJECT_ROOT / "openwebui" / "functions"


@pytest.mark.parametrize("file", ["query_pipe.py", "ingestion_pipe.py", "qdrant_pipe.py", "thorough_filter.py"])
def test_adapter_source_is_fixed_point_of_replace_imports(file):
    src = (FUNCTIONS / file).read_text(encoding="utf-8")
    assert replace_imports(src) == src


def test_function_loads_from_a_temporary_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mod = load_function(FUNCTIONS / "query_pipe.py", "memento_query")
    assert hasattr(mod.Pipe, "Valves")
    assert mod.__name__ == "function_memento_query" and mod.__file__ != str(FUNCTIONS / "query_pipe.py")


def test_copy_id_from_module_name():
    a = load_function(FUNCTIONS / "query_pipe.py", "memento_query")
    b = load_function(FUNCTIONS / "query_pipe.py", "memento_query_polymer")
    assert (a._OWN_ID, b._OWN_ID) == ("memento_query", "memento_query_polymer")


def test_no_worker_started_by_the_helper():
    load_function(FUNCTIONS / "query_pipe.py", "memento_query")
    assert not [t for t in threading.enumerate() if t.name.startswith("memento-warmup") and t.is_alive()]


def test_no_model_map_published_by_the_helper(tmp_path, monkeypatch):
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "output"))
    load_function(FUNCTIONS / "query_pipe.py", "memento_query")
    assert not (tmp_path / "output" / "_runtime" / "query_models.json").exists()


def test_pipe_is_an_async_generator():
    import inspect
    mod = load_function(FUNCTIONS / "query_pipe.py", "memento_query")
    assert inspect.isasyncgenfunction(mod.Pipe.pipe)
    assert "contextlib.aclosing(service.handle(" in inspect.getsource(mod.Pipe.pipe)    # the stream closes the service's


def test_upload_pipe_is_an_async_generator():
    import inspect
    from memento.ingestion.upload import workflow
    mod = load_function(FUNCTIONS / "ingestion_pipe.py", "memento_ingestion")
    assert inspect.isasyncgenfunction(mod.Pipe.pipe) and inspect.isasyncgenfunction(workflow.run)
    assert "contextlib.aclosing(workflow.run(" in inspect.getsource(mod.Pipe.pipe)      # the stream closes the workflow's


def _thread_starts(monkeypatch):
    """Names of the threads started from now on (none actually runs)."""
    started = []
    monkeypatch.setattr(threading.Thread, "start", lambda self: started.append(self.name))
    return started


def test_adapter_starts_one_worker_per_load(monkeypatch):
    from memento.query import workers
    monkeypatch.setattr(workers, "warmup_loop", lambda *a, **k: None)
    started = _thread_starts(monkeypatch)
    load_function(FUNCTIONS / "query_pipe.py", "memento_query", start_workers=True)
    load_function(FUNCTIONS / "query_pipe.py", "memento_query", start_workers=True)
    workers_started = [n for n in started if n.startswith("memento-warmup")]
    assert len(workers_started) == 2
    gens = [int(n.rsplit("-", 1)[1]) for n in workers_started]
    assert workers_started == [f"memento-warmup-memento_query-{g}" for g in gens] and gens[1] == gens[0] + 1


def test_cli_module_starts_no_worker(monkeypatch):
    from memento.retrieval import pipe_cli
    started = _thread_starts(monkeypatch)
    pipe_cli.load_pipe_module()
    assert not [n for n in started if n.startswith("memento-warmup")]
