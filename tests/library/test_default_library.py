# The default library (whose list is also the single-library file and whose attachments use the manifest's old key) is
# the configured domain's collection, or MEMENTO_DEFAULT_COLLECTION; the file server reads the same setting.
import json
import os
import subprocess
import sys

from memento.config.paths import PROJECT_ROOT
from memento.library import documents


def _in_fresh_process(code: str, **env) -> str:
    base = {k: v for k, v in os.environ.items() if k not in ("MEMENTO_DEFAULT_COLLECTION", "COLLECTION_PREFIX")}
    out = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, env={**base, **env, "PYTHONPATH": str(PROJECT_ROOT)},
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    return out.stdout.strip()


def _default(**env) -> str:
    return _in_fresh_process("from memento.library import documents; print(documents.LEGACY_COLLECTION)", **env)


def test_the_default_library_follows_the_configured_domain():
    assert _default(DOMAIN_NAME="computational-materials") == "memento-comp-mat-rag"
    assert _default(DOMAIN_NAME="ceramics") == "memento-ceramics-rag"
    assert _default(DOMAIN_NAME="ceramics", COLLECTION_PREFIX="lab") == "lab-rag"
    assert _default(DOMAIN_NAME="ceramics", MEMENTO_DEFAULT_COLLECTION="other-rag") == "other-rag"


def test_an_unknown_domain_still_names_a_default_library():
    assert _default(DOMAIN_NAME="no-such-domain") == "memento-generic-rag"        # the generic profile's


def test_only_the_default_library_writes_the_single_library_file(tmp_path):
    entry = {"source": "a", "title": "A"}
    documents.write_library(tmp_path, [entry], collection="other-rag")
    assert not documents.library_path(tmp_path).exists()
    documents.write_library(tmp_path, [entry], collection=documents.LEGACY_COLLECTION)
    one = json.loads(documents.library_path(tmp_path).read_text())
    own = json.loads(documents.library_path(tmp_path, documents.LEGACY_COLLECTION).read_text())
    assert one["documents"] == own["documents"] == [entry]


def test_the_file_server_reads_the_same_setting(tmp_path):
    code = ("import importlib.util as u; s = u.spec_from_file_location('fs', 'services/file_server.py'); "
            "m = u.module_from_spec(s); s.loader.exec_module(m); print(repr(m.LEGACY_COLLECTION))")
    dirs = {"INPUT_DIR": str(tmp_path), "OUTPUT_DIR": str(tmp_path)}
    assert _in_fresh_process(code, MEMENTO_DEFAULT_COLLECTION="memento-comp-mat-rag", **dirs).splitlines()[-1] \
        == "'memento-comp-mat-rag'"
    assert _in_fresh_process(code, **dirs).splitlines()[-1] == "''"   # not set: no old single-library file to fall back to
