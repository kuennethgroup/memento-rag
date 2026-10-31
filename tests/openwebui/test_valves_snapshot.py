# The four Open WebUI functions' Valves (names, defaults, order) must not change with the reorganization: saved
# Valves are matched by name, and a new default would change every copy that never saved that Valve. Run through
# scripts/test_offline.sh (defaults read the environment it sets).
import os
import re
from pathlib import Path

import pytest

from _snapshots import check_snapshot, recorded_env

ROOT = Path(__file__).resolve().parents[2]
FUNCTIONS = {"query_pipe.py": "Pipe", "ingestion_pipe.py": "Pipe", "qdrant_pipe.py": "Pipe", "thorough_filter.py": "Filter"}
IDS = {"query_pipe.py": "memento_query", "ingestion_pipe.py": "memento_ingestion", "qdrant_pipe.py": "qdrant_manager",
       "thorough_filter.py": "memento_thorough"}


def _valves(file: str, cls: str) -> list:
    from _owui_loader import load_function
    with recorded_env():
        mod = load_function(ROOT / "openwebui" / "functions" / file, IDS[file])
    tmp = os.environ.get("TMPDIR")
    def norm(v):                                     # test_offline.sh's per-run folder; pytest's folder names the user
        if not (tmp and isinstance(v, str)):
            return v
        return re.sub(r"pytest-of-[^/]+/pytest-\d+/", "<PYTEST>/", v.replace(tmp, "<TMP>"))
    return [[n, norm(f.default)] for n, f in getattr(mod, cls).Valves.model_fields.items()]


@pytest.mark.parametrize("file", sorted(FUNCTIONS))
def test_valves_unchanged(file):
    check_snapshot(f"valves_{Path(file).stem}_snapshot", {"valves": _valves(file, FUNCTIONS[file])})
