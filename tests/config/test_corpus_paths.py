# One rule for corpus folders: documents in INPUT_ROOT/<corpus_dir>, parsed output in OUTPUT_ROOT/<corpus_dir>.
# INPUT_ROOT/OUTPUT_ROOT come from config/.env (or Open WebUI Valves), corpus_dir from the domain YAML.
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from memento.config.paths import PROJECT_ROOT
from memento.ingestion.upload import document

ROOT = PROJECT_ROOT
DOMAINS = sorted((ROOT / "config" / "domains").glob("*.yaml"))


def test_every_domain_names_its_corpus_folder():
    for p in DOMAINS:
        assert (yaml.safe_load(p.read_text()) or {}).get("corpus_dir"), f"{p.name} has no corpus_dir"


def test_corpus_dirs_join_roots_and_corpus_dir(tmp_path, monkeypatch):
    from memento.config.paths import corpus_dirs
    monkeypatch.setenv("INPUT_ROOT", str(tmp_path / "in"))
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "out"))
    assert corpus_dirs("ceramics") == (tmp_path / "in" / "Ceramics", tmp_path / "out" / "Ceramics")
    assert corpus_dirs({"corpus_dir": "X"}, input_root="/a", output_root="/b") == (Path("/a/X"), Path("/b/X"))


def test_roots_default_to_the_repository_data_folder(monkeypatch):
    from memento.config.paths import roots
    monkeypatch.delenv("INPUT_ROOT", raising=False)
    monkeypatch.delenv("OUTPUT_ROOT", raising=False)
    assert roots() == (ROOT / "data" / "input", ROOT / "data" / "output")


def test_missing_corpus_dir_names_the_yaml():
    from memento.config.paths import corpus_dirs
    with pytest.raises(ValueError, match="corpus_dir"):
        corpus_dirs({"name": "nameless"})


def _config_paths(env):
    code = ("from memento.config.settings import INPUT_DIR, OUTPUT_DIR, INPUT_ROOT, OUTPUT_ROOT; import json; "
            "print(json.dumps([str(INPUT_DIR), str(OUTPUT_DIR), str(INPUT_ROOT), str(OUTPUT_ROOT)]))")
    clean = {k: v for k, v in os.environ.items() if k not in ("INPUT_DIR", "OUTPUT_DIR", "INPUT_ROOT", "OUTPUT_ROOT")}
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=clean | env, capture_output=True, text=True,
                         check=True).stdout.strip().splitlines()[-1]
    return json.loads(out)


def test_config_follows_domain_and_roots():
    got = _config_paths({"DOMAIN_NAME": "ceramics", "INPUT_ROOT": "/r/in", "OUTPUT_ROOT": "/r/out"})
    assert got == ["/r/in/Ceramics", "/r/out/Ceramics", "/r/in", "/r/out"]


def test_explicit_input_dir_still_overrides_for_scripts():
    got = _config_paths({"DOMAIN_NAME": "ceramics", "INPUT_ROOT": "/r/in", "OUTPUT_DIR": "/exact/out"})
    assert got[0] == "/r/in/Ceramics" and got[1] == "/exact/out"


def test_ingestion_pipe_writes_into_the_valve_domain_folder():
    sys.path.insert(0, str(ROOT / "openwebui" / "functions"))
    import ingestion_pipe
    p = document.Ingester(ingestion_pipe.Pipe().valves)
    p.valves.DOMAIN_NAME, p.valves.INPUT_ROOT, p.valves.OUTPUT_ROOT = "ceramics", "/v/in", "/v/out"
    assert p.input_dir() == Path("/v/in/Ceramics") and p.output_dir() == Path("/v/out/Ceramics")


def test_build_graph_output_dir_defaults_to_the_domain_folder():
    code = ("import sys; sys.argv=['x','--domain','ceramics']; "
            "from memento.config.cli_domain import apply_domain_arg; apply_domain_arg(); "
            "import memento.graph.build.build_graph as bg; from memento.config.settings import OUTPUT_DIR; "
            "print(bg._parser().parse_args(['--domain','ceramics','--all']).output_dir == OUTPUT_DIR)")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
    assert out.stdout.strip().splitlines()[-1] == "True"


def test_compose_mounts_use_the_roots():
    text = (ROOT / "docker-compose.yml").read_text()
    assert "${MEMENTO_DIR}/data/input" not in text and "${MEMENTO_DIR}/data/output" not in text
    assert "${INPUT_ROOT" in text and "${OUTPUT_ROOT" in text


def test_startup_banner_shows_the_resolved_folders():
    code = "from memento.config.cli_domain import domain_banner; domain_banner()"
    clean = {k: v for k, v in os.environ.items() if k not in ("INPUT_DIR", "OUTPUT_DIR")}
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True,
                         env=clean | {"DOMAIN_NAME": "ceramics", "INPUT_ROOT": "/r/in", "OUTPUT_ROOT": "/r/out"}).stdout
    assert "documents /r/in/Ceramics" in out and "parsed output /r/out/Ceramics" in out
