# The browser script with two libraries in one Open WebUI: its pure model and library rules run in Node
# (tests/openwebui/js/loader_models.test.mjs), and its list of the default model ids matches the default model list
# (config/query_models.yaml).
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from memento.config.paths import PROJECT_ROOT

ROOT = PROJECT_ROOT
sys.path.insert(0, str(ROOT))


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_the_loaders_model_and_library_rules():
    r = subprocess.run(["node", str(ROOT / "tests/openwebui/js/loader_models.test.mjs")], capture_output=True, text=True,
                       timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_the_pickers_page_wiring_never_shows_another_librarys_list():
    # the picker fragment of the page code with a fake fetch
    r = subprocess.run(["node", str(ROOT / "tests/openwebui/js/loader_picker_wiring.test.mjs")], capture_output=True, text=True,
                       timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr


def test_the_loaders_today_ids_are_todays_models():
    from memento.query import models as qm
    src = (ROOT / "openwebui/loader.js").read_text()
    block = src[src.index("/* memento:models-begin"):src.index("/* memento:models-end */")]
    ids = set(re.findall(r"'([a-z0-9-]+)'", block[block.index("TODAY_IDS"):block.index("]", block.index("TODAY_IDS"))]))
    every = {qm.workspace_id(v) for v in qm.load_variants()}
    assert every <= ids and {"memento-document-chat", "memento-ingestion-prod"} <= ids


def test_the_loader_names_no_model_id_outside_its_rules():
    # every literal link in the page code goes through MementoModels.relink (today's three ids only)
    src = (ROOT / "openwebui/loader.js").read_text()
    for m in re.finditer(r"/\?model=([a-z0-9-]+)", src):
        assert m.group(1) in ("memento-query", "memento-ingestion-prod", "memento-document-chat"), m.group(0)


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_the_library_loads_right_after_signing_in():
    r = subprocess.run(["node", str(ROOT / "tests/openwebui/js/loader_load.test.mjs")], capture_output=True, text=True,
                       timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
