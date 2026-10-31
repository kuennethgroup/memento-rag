# The cloud answer models are optional examples: any OpenAI-compatible chat API can serve them. One is offered only while
# its key variable is set, so a new installation sends no question to a cloud service it did not configure, and the
# shipped entries name no provider, link or data policy for an endpoint someone else chooses.
import shutil

import pytest
import yaml

from memento.config.paths import PROJECT_ROOT
from memento.query import models

LISTS = ("query_models.yaml", "query_models_study.yaml")


def test_a_cloud_model_is_offered_only_while_its_key_is_set(monkeypatch):
    for name in LISTS:
        monkeypatch.delenv("CLOUD_API_KEY", raising=False)
        assert [v.id for v in models.load_variants(PROJECT_ROOT / "config" / name)] == ["local"], name
        monkeypatch.setenv("CLOUD_API_KEY", "some-key")
        assert [v.id for v in models.load_variants(PROJECT_ROOT / "config" / name) if v.is_cloud], name


def test_the_shipped_cloud_entries_name_no_provider_link_or_data_policy():
    for name in LISTS:
        for e in yaml.safe_load((PROJECT_ROOT / "config" / name).read_text()):
            if e.get("provider") != "cloud":
                continue
            text = " ".join(str(e.get(k, "")) for k in ("name", "label", "description", "host"))
            assert "http" not in text and "Academic Cloud" not in text and "store" not in text, (name, e["id"])


def test_the_docs_say_any_openai_compatible_api_can_serve_them():
    readme = (PROJECT_ROOT / "README.md").read_text()
    section = readme[readme.index("### Answer models in the model menu"):readme.index("### Important settings")]
    assert "OpenAI-compatible" in section and "only while" in section
    assert "OpenAI-compatible" in (PROJECT_ROOT / "config" / "query_models.yaml").read_text()


def test_the_cloud_entries_not_offered_are_named(monkeypatch):
    path = PROJECT_ROOT / "config" / "query_models.yaml"
    monkeypatch.delenv("CLOUD_API_KEY", raising=False)
    off = [v.id for v in models.unoffered_cloud_variants(path)]
    assert off and "local" not in off and set(off) == {e["id"] for e in yaml.safe_load(path.read_text())
                                                        if e.get("provider") == "cloud"}
    monkeypatch.setenv("CLOUD_API_KEY", "some-key")
    assert [v.id for v in models.unoffered_cloud_variants(path)] == [
        e["id"] for e in yaml.safe_load(path.read_text()) if e.get("provider") == "cloud" and e.get("enabled") is False]


def _setup_module(monkeypatch):
    import importlib.util
    monkeypatch.delenv("CLOUD_API_KEY", raising=False)
    spec = importlib.util.spec_from_file_location("so_cloud_off", PROJECT_ROOT / "scripts/setup_openwebui.py")
    so = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(so)
    return so


def _fake_api(monkeypatch, so, existing, bases=None):
    posted, bases = [], bases or {}

    class R:
        def __init__(self, body, status_code=200): self.body, self.status_code = body, status_code
        def json(self): return self.body
        def raise_for_status(self): pass

    def get(url, **kw):
        path = url.replace("http://x", "")
        if path == "/api/v1/groups/":
            return R([{"id": "gid-1", "name": "contributors"}, {"id": "gid-2", "name": "members"}])
        if path == "/api/v1/configs/models":
            return R({"DEFAULT_MODELS": "memento-query", "MODEL_ORDER_LIST": []})
        if path.startswith("/api/v1/models/model?id=") and path.split("=", 1)[1] in existing:
            mid = path.split("=", 1)[1]
            return R({"id": mid, "base_model_id": bases.get(mid, "memento_query." + mid.rsplit("-", 1)[-1])})
        return R(None, 404)

    def post(url, json=None, **kw):
        path = url.replace("http://x", "")
        posted.append((path, json))
        if path == "/api/v1/models/model/delete":
            existing.discard(json["id"])
            return R(True)
        return R({"ok": True})
    monkeypatch.setattr(so.requests, "get", get)
    monkeypatch.setattr(so.requests, "post", post)
    return posted


def test_a_restart_removes_the_models_no_longer_offered(monkeypatch):
    # a cloud key removed after its model was set up: the next start's sync deletes that workspace model, leaves the
    # local one and every model the config does not name, and creates no cloud model
    so = _setup_module(monkeypatch)
    existing = {"memento-query", "memento-query-qwen35", "someone-elses-model"}
    posted = _fake_api(monkeypatch, so, existing)
    so.cmd_sync_models("http://x", {}, dry=False)
    deleted = [b["id"] for u, b in posted if u == "/api/v1/models/model/delete"]
    assert deleted == ["memento-query-qwen35"]
    created = {b["id"] for u, b in posted if u == "/api/v1/models/create"}
    assert "memento-query" in created and not any(i.startswith("memento-query-") for i in created)
    assert existing == {"memento-query", "someone-elses-model"}


def test_a_dry_run_only_reports_the_models_it_would_remove(monkeypatch, capsys):
    so = _setup_module(monkeypatch)
    posted = _fake_api(monkeypatch, so, {"memento-query-qwen35"})
    so.cmd_sync_models("http://x", {}, dry=True)
    assert posted == [] and "memento-query-qwen35" in capsys.readouterr().out


def test_a_model_with_that_id_but_another_function_is_kept(monkeypatch, capsys):
    # the id alone is not enough: a record whose base is not this library's Library Chat function stays
    so = _setup_module(monkeypatch)
    existing = {"memento-query-qwen35"}
    posted = _fake_api(monkeypatch, so, existing, bases={"memento-query-qwen35": "memento_query_polymer.local"})
    so.cmd_sync_models("http://x", {}, dry=False)
    assert not [b for u, b in posted if u == "/api/v1/models/model/delete"]
    assert existing == {"memento-query-qwen35"} and "memento-query-qwen35" in capsys.readouterr().out


def test_the_two_libraries_never_share_a_model_id(tmp_path):
    import shutil
    so_ids = __import__("importlib").util
    spec = so_ids.spec_from_file_location("so_ids", PROJECT_ROOT / "scripts/setup_openwebui.py")
    so = so_ids.module_from_spec(spec); spec.loader.exec_module(so)
    study = PROJECT_ROOT / "config" / "query_models_study.yaml"
    assert so.shared_model_ids(PROJECT_ROOT / "config" / "query_models.yaml", study) == set()
    clash = tmp_path / "query_models.yaml"
    shutil.copy(PROJECT_ROOT / "config" / "query_models.yaml", clash)
    entries = yaml.safe_load(clash.read_text())
    entries.append({**next(e for e in entries if e.get("provider") == "cloud"), "id": "polymer"})
    clash.write_text(yaml.safe_dump(entries))
    assert so.shared_model_ids(clash, study) == {"memento-query-polymer"}


def _clashing_config(tmp_path):
    clash = tmp_path / "query_models.yaml"
    shutil.copy(PROJECT_ROOT / "config" / "query_models.yaml", clash)
    entries = yaml.safe_load(clash.read_text())
    entries.append({**next(e for e in entries if e.get("provider") == "cloud"), "id": "polymer"})
    clash.write_text(yaml.safe_dump(entries))
    return clash


SETUP_COMMANDS = {"sync-models": lambda so, dry: so.cmd_sync_models("http://x", {}, dry),
                  "trial": lambda so, dry: so.cmd_trial("http://x", {}, "generic", None, dry),
                  "study": lambda so, dry: so.cmd_study("http://x", {}, dry, None, None)}


@pytest.mark.parametrize("dry", [False, True], ids=["real", "dry-run"])
@pytest.mark.parametrize("command", SETUP_COMMANDS)
def test_a_shared_model_id_stops_every_setup_command_before_any_change(monkeypatch, tmp_path, capsys, command, dry):
    so = _setup_module(monkeypatch)
    posted = _fake_api(monkeypatch, so, set())
    monkeypatch.setattr(so, "VARIANTS_CONFIG", _clashing_config(tmp_path))
    with pytest.raises(SystemExit) as stop:
        SETUP_COMMANDS[command](so, dry)
    assert stop.value.code not in (0, None) and posted == []
    assert "memento-query-polymer" in capsys.readouterr().out


class _Reached(Exception):
    pass


@pytest.mark.parametrize("command,first_step", [("sync-models", "ensure_group"), ("trial", "get_valves"),
                                                ("study", "ensure_group")])
def test_the_shipped_config_passes_the_check_and_reaches_the_setup(monkeypatch, command, first_step):
    so = _setup_module(monkeypatch)
    _fake_api(monkeypatch, so, set())

    def reached(*a, **kw):
        raise _Reached
    monkeypatch.setattr(so, first_step, reached)
    with pytest.raises(_Reached):
        SETUP_COMMANDS[command](so, False)


def test_a_shared_model_id_stops_the_full_first_setup_too(monkeypatch, tmp_path, capsys):
    so = _setup_module(monkeypatch)
    posted = _fake_api(monkeypatch, so, set())
    api_get = so.requests.get

    class Admin:
        status_code = 200
        def json(self): return {"role": "admin"}

    monkeypatch.setattr(so.requests, "get", lambda url, **kw: Admin() if url.endswith("/api/v1/auths/") else api_get(url, **kw))
    monkeypatch.setattr(so, "load_env", lambda: {"OPENWEBUI_API_KEY": "some-key"})
    monkeypatch.setattr(so, "VARIANTS_CONFIG", _clashing_config(tmp_path))
    monkeypatch.setattr(so.sys, "argv", ["setup_openwebui.py", "--url", "http://x"])
    with pytest.raises(SystemExit) as stop:
        so.main()
    assert stop.value.code not in (0, None) and posted == []
    assert "memento-query-polymer" in capsys.readouterr().out
