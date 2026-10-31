# Where the configuration is read from must not change when the modules move into memento/: every domain profile,
# the missing-profile fallback, the corpus folders, both model-variant files and the .env file, each read from another
# working directory. Recorded on the code before the move (tests/fixtures/config_snapshot.json).
import json
import subprocess
import sys
from pathlib import Path

import pytest

from _snapshots import check_snapshot, sha

ROOT = Path(__file__).resolve().parents[2]
DOMAINS = sorted(p.stem for p in (ROOT / "config" / "domains").glob("*.yaml"))


@pytest.fixture
def elsewhere(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _profile(dom: dict) -> dict:
    return {"name": dom.get("name"), "corpus_dir": dom.get("corpus_dir"),
            "collection_prefix": dom.get("collection_prefix"), "instruction": sha(dom.get("instruction") or ""),
            "system_prompt": sha(dom.get("system_prompt") or ""), "keys": sorted(dom)}


def test_every_domain_profile_loads_from_another_cwd(elsewhere):
    from memento.config.domain_loader import load_domain
    from memento.config import paths
    got = {}
    for d in DOMAINS:
        dom = load_domain(d)
        assert dom["name"] and dom["corpus_dir"]
        inp, out = paths.corpus_dirs(d, input_root="/in", output_root="/out")
        got[d] = dict(_profile(dom), corpus_dirs=[str(inp), str(out)])
    check_snapshot("config_profiles_snapshot", got)


def test_missing_profile_behaves_as_recorded(elsewhere):
    from memento.config.domain_loader import load_domain
    try:
        outcome = {"profile": _profile(load_domain("no-such-domain"))}
    except Exception as e:                          # recorded either way: an error type or the fallback profile
        outcome = {"error": type(e).__name__}
    check_snapshot("config_missing_profile_snapshot", outcome)


def test_variant_files_load(elsewhere):
    from memento.query.models import load_variants
    got = {"default": [v.id for v in load_variants()]}
    for f in ("query_models.yaml", "query_models_study.yaml"):
        got[f] = [v.id for v in load_variants(ROOT / "config" / f)]
    assert got["default"] == got["query_models.yaml"]
    check_snapshot("config_variants_snapshot", got)


def test_env_file_path_and_precedence(tmp_path):
    """Importing the settings module loads <repo>/config/.env without override: a variable already set wins."""
    probe = (
        "import json, sys, dotenv\n"
        "calls = []\n"
        "dotenv.load_dotenv = lambda *a, **k: calls.append([str(a[0] if a else k.get('dotenv_path')),"
        " bool(k.get('override', False))])\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "import memento.config.settings\n"
        "print(json.dumps(calls))\n")
    out = subprocess.run([sys.executable, "-c", probe], cwd=tmp_path, capture_output=True, text=True, check=True)
    calls = json.loads(out.stdout.strip().splitlines()[-1])
    assert calls == [[str(ROOT / "config" / ".env"), False]]
