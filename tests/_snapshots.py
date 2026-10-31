# Snapshots of what the reorganization must not change, recorded on the code before any move: compared with
# tests/fixtures/<name>.json; MEMENTO_UPDATE_SNAPSHOTS=1 writes them (only when the change is intended).
import hashlib
import json
import os
import re
from pathlib import Path

from _contract_fakes import _diff

FIXTURES = Path(__file__).parent / "fixtures"
UPDATE = os.getenv("MEMENTO_UPDATE_SNAPSHOTS") == "1"


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def plain(value):
    """A constant as JSON data: sets sorted, patterns as (pattern, flags), paths as text."""
    if isinstance(value, re.Pattern):
        return {"pattern": value.pattern, "flags": value.flags}
    if isinstance(value, (set, frozenset)):
        return sorted((plain(v) for v in value), key=lambda v: json.dumps(v, sort_keys=True, default=str))
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def check_snapshot(name: str, data) -> None:
    path = FIXTURES / f"{name}.json"
    got = json.loads(json.dumps(plain(data), ensure_ascii=False, default=str))
    if UPDATE or not path.exists():
        path.write_text(json.dumps(got, indent=1, ensure_ascii=False, sort_keys=True) + "\n")
        if not UPDATE:
            raise AssertionError(f"{path} written; inspect it and run again")
        return
    want = json.loads(path.read_text())
    if got != want:
        diffs = "\n".join(_diff(want, got)[:12])
        raise AssertionError(f"{name}: differs from {path}\n{diffs}")


def check_snapshot_entry(name: str, key: str, data) -> None:
    """One entry of tests/fixtures/<name>.json (parametrized tests share one file)."""
    path = FIXTURES / f"{name}.json"
    got = json.loads(json.dumps(plain(data), ensure_ascii=False, default=str))
    stored = json.loads(path.read_text()) if path.exists() else {}
    if UPDATE or key not in stored:
        stored[key] = got
        path.write_text(json.dumps(stored, indent=1, ensure_ascii=False, sort_keys=True) + "\n")
        if not UPDATE:
            raise AssertionError(f"{path} [{key}] written; inspect it and run again")
        return
    if got != stored[key]:
        diffs = "\n".join(_diff(stored[key], got)[:12])
        raise AssertionError(f"{name}[{key}]: differs from {path}\n{diffs}")


# The domain settings the snapshots were recorded with (tests/conftest.py's and scripts/test_offline.sh's): other
# test modules change DOMAIN_NAME at import (memento.config.cli_domain.apply_domain_arg), and the pipes read it at load time.
RECORDED_ENV = {"DOMAIN_NAME": "polymer-engineering", "COLLECTION_PREFIX": None, "ENTITY_REGISTRY": None}


def recorded_env():
    """A pytest.MonkeyPatch context with RECORDED_ENV set (None: unset)."""
    import contextlib
    import pytest

    @contextlib.contextmanager
    def ctx():
        with pytest.MonkeyPatch.context() as mp:
            for k, v in RECORDED_ENV.items():
                mp.delenv(k, raising=False) if v is None else mp.setenv(k, v)
            yield mp
    return ctx()
