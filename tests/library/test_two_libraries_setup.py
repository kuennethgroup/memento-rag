# Setting up, keeping and removing the second copy (the `study` library), on a fake Open WebUI API
# (tests/_fake_openwebui.py). `study` mode installs the copy inactive, saves every setting, restores archived Document
# Chat rows, publishes, creates the models, then activates; `sync-models` (every Open WebUI start) handles the copy only
# while it is enabled and never brings back a removed one; `--remove-study` switches it off, drains, archives (transfer
# rule), deletes, and can put the default functions back from a backup.
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.pool import StaticPool
from memento.config.paths import PROJECT_ROOT
from memento.query import workers

ROOT = PROJECT_ROOT
sys.path.insert(0, str(ROOT))
from _fake_openwebui import FakeOpenWebUI  # noqa: E402
from memento.library import copies as cs  # noqa: E402
from memento.document_chat import store as td  # noqa: E402

spec = importlib.util.spec_from_file_location("so_study", ROOT / "scripts/setup_openwebui.py")
so = importlib.util.module_from_spec(spec)
spec.loader.exec_module(so)

POLY_TEMP = "memento-polymer-temp"
CHAT = "3f2b8c1e-6a4d-4e0b-9c7a-1d2e3f4a5b6c"


@pytest.fixture
def api(monkeypatch):
    fake = FakeOpenWebUI()
    fake.install(monkeypatch, so)
    # the default copy, installed and active
    for fid, file in (("memento_query", "query_pipe.py"), ("memento_ingestion", "ingestion_pipe.py")):
        fake.functions[fid] = {"id": fid, "name": fid, "content": (ROOT / "openwebui/functions" / file).read_text(),
                               "meta": {}, "valves": {"COLLECTION": "memento-comp-mat-rag",
                                                      "DOMAIN_NAME": "computational-materials"},
                               "is_active": True}
    fake.groups["g-1"] = {"name": "contributors", "users": []}
    fake.groups["g-2"] = {"name": "members", "users": []}
    monkeypatch.setattr(so, "stop_study_updater", lambda dry: None)
    return fake


@pytest.fixture
def world(tmp_path, monkeypatch):
    from memento.library import documents as ldb
    e = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    td.ensure_tables(e)
    ldb.ensure_tables(e)
    out = tmp_path / "output"
    monkeypatch.setenv("OUTPUT_ROOT", str(out))
    return type("W", (), {"engine": e, "out": out, "backups": tmp_path / "backups"})


def study(api, world, dry=False):
    return so.cmd_study("http://x", {}, dry=dry, engine=world.engine, output_root=world.out)


def remove(api, world, dry=False, restore_from=None):
    return so.cmd_remove_study("http://x", {}, dry=dry, engine=world.engine, backup_root=world.backups,
                               restore_from=restore_from)


def sync(api):
    so.cmd_sync_models("http://x", {}, dry=False)


# ── the second copy's settings ──

def test_the_study_valves_name_every_required_setting_and_the_study_library(api):
    q, i = so.study_valves(api.functions["memento_query"]["valves"], api.functions["memento_ingestion"]["valves"])
    assert cs.missing(q, "query") == [] and cs.missing(i, "ingestion") == []
    assert (q["COLLECTION"], q["DOMAIN_NAME"]) == ("memento-polymer-rag", "polymer-engineering")
    assert (q["LLM_BASE_URL"], q["LLM_MODEL"], q["LLM_API_KEY_ENV"]) == (
        "http://localhost:8023/v1", "qwen-flash-next", "REASONING_API_KEY")
    assert q["VARIANTS_FILE"] == "query_models_study.yaml" and q["INGESTION_FUNCTION_ID"] == "memento_ingestion_polymer"
    assert q["QUERY_MODEL_ID"] == "memento-query-polymer" and q["GRAPH_RETRIEVAL"] is False
    assert i["CONTRIBUTOR_GROUP"] == "User study" and i["GRAPH_BUILD"] is False
    assert i["LLM_API_KEY_ENV"] == "REASONING_API_KEY"


def test_the_study_document_chat_has_no_page_limit_whatever_the_chair_library_saved(api):
    api.functions["memento_query"]["valves"]["DOC_CHAT_MAX_PAGES"] = 150
    q, _i = so.study_valves(api.functions["memento_query"]["valves"], api.functions["memento_ingestion"]["valves"])
    assert q["DOC_CHAT_MAX_PAGES"] == 0


def test_the_saved_study_valves_are_the_effective_values_in_a_comp_mat_process(api, monkeypatch):
    # Open WebUI's process has config/.env's domain: every required setting must come from the saved Valves
    monkeypatch.setenv("DOMAIN_NAME", "computational-materials")
    sys.path.insert(0, str(ROOT / "openwebui/functions"))
    import query_pipe as qp
    q, _i = so.study_valves(api.functions["memento_query"]["valves"], api.functions["memento_ingestion"]["valves"])
    v = workers.valves_from(qp.Pipe.Valves, q)
    for k in cs.REQUIRED_QUERY_VALVES:
        assert getattr(v, k) == q[k], k
    every = set(qp.Pipe.Valves.model_fields)
    assert set(cs.REQUIRED_QUERY_VALVES) <= every                              # every required name is a real Valve


def test_every_required_ingestion_setting_is_a_real_valve():
    sys.path.insert(0, str(ROOT / "openwebui/functions"))
    import ingestion_pipe as ip
    assert set(cs.REQUIRED_INGESTION_VALVES) <= set(ip.Pipe.Valves.model_fields)


def test_todays_modes_never_touch_the_study_copy():
    ids = {f["id"] for f in so.FUNCTIONS}
    assert not ids & {"memento_query_polymer", "memento_ingestion_polymer"}
    src = (ROOT / "scripts/setup_openwebui.py").read_text()
    assert "empties the library's stored answers" in src                   # the single-library mode's warning


# ── study mode ──

def test_a_dry_run_changes_nothing_and_says_what_it_would_do(api, world, capsys):
    before = api.snapshot()
    study(api, world, dry=True)
    assert api.snapshot() == before and not [c for c in api.calls if c[0] != "GET"]
    out = capsys.readouterr().out
    assert "memento_query_polymer" in out and "User study" in out and "qwen-flash-next" in out


def test_study_setup_installs_saves_publishes_and_then_activates(api, world):
    study(api, world)
    q = api.functions["memento_query_polymer"]
    i = api.functions["memento_ingestion_polymer"]
    assert q["is_active"] and i["is_active"]
    assert cs.missing(q["valves"], "query") == [] and cs.missing(i["valves"], "ingestion") == []
    assert "User study" in [g["name"] for g in api.groups.values()]
    study_gid = next(gid for gid, g in api.groups.items() if g["name"] == "User study")
    for mid in ("memento-query-polymer", "memento-query-polymer-qwen35", "memento-document-chat-polymer",
                "memento-ingestion-polymer", "memento_query_polymer.local", "memento_ingestion_polymer"):
        grants = [g["principal_id"] for g in api.models[mid]["access_grants"]]
        assert grants == [study_gid], mid
    assert api.models["memento-query-polymer"]["meta"]["suggestion_prompts"] == so.STUDY_QUERY_SUGGESTIONS
    assert api.config["DEFAULT_MODELS"] == "memento-query,memento-query-polymer"
    assert set(so.STUDY_MODEL_ORDER) <= set(api.config["MODEL_ORDER_LIST"])
    data = json.loads((world.out / "_runtime" / "query_models.json").read_text())
    assert "memento_query_polymer" in data["copies"]


def test_the_study_functions_are_switched_on_only_after_everything_else(api, world):
    study(api, world)
    posts = [p for m, p, _ in api.calls if m == "POST"]
    first_toggle = next(i for i, p in enumerate(posts) if p.endswith("/toggle"))
    last_other = max(i for i, p in enumerate(posts) if not p.endswith("/toggle") and p != "/api/v1/configs/models")
    assert first_toggle > last_other


def test_a_second_run_changes_nothing(api, world):
    study(api, world)
    before = api.snapshot()
    study(api, world)
    assert api.snapshot() == before


def test_a_rerun_during_the_study_never_switches_it_off(api, world):
    study(api, world)
    n = len(api.calls)
    study(api, world)
    assert api.functions["memento_query_polymer"]["is_active"]
    assert api.functions["memento_ingestion_polymer"]["is_active"]
    assert not [p for _m, p, _ in api.calls[n:] if p.endswith("/toggle")]       # no toggle at all on a re-run


def test_an_unknown_group_stops_before_any_change(api, world, monkeypatch):
    monkeypatch.setattr(so, "STUDY_MODELS", so.STUDY_MODELS + [{"id": "x", "name": "x", "base_model_id": None,
                                                                "meta": {}, "params": {}, "groups": ["Nope"]}])
    before = api.snapshot()
    with pytest.raises(SystemExit, match="Nope"):
        study(api, world)
    assert api.snapshot() == before


def test_an_interrupted_first_setup_leaves_the_copy_off(api, world, monkeypatch):
    monkeypatch.setattr(so, "create_model", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("network")))
    with pytest.raises(RuntimeError):
        study(api, world)
    assert not api.functions["memento_query_polymer"]["is_active"]
    assert api.functions["memento_query_polymer"]["valves"]                 # settings saved, but not enabled
    assert so.study_state("http://x", {}) is False


@pytest.mark.parametrize("fault", ["toggle", "toggle-noop"])
def test_a_failed_or_ignored_activation_stops_the_setup(api, world, fault):
    api.fail.add(fault)
    with pytest.raises(SystemExit, match="memento_"):
        study(api, world)


# ── restarts: sync-models never brings back a removed copy ──

def _state(api):
    study_ids = set(so.STUDY_MODEL_ORDER)
    return {"models": study_ids & set(api.models), "group": "User study" in [g["name"] for g in api.groups.values()],
            "default": api.config["DEFAULT_MODELS"],
            "order_has_study": bool(study_ids & set(api.config["MODEL_ORDER_LIST"]))}


def test_restart_matrix(api, world, monkeypatch):
    sync(api)                                                       # before the copy is installed
    assert _state(api) == {"models": set(), "group": False, "default": "memento-query", "order_has_study": False}
    study(api, world)
    api.models = {k: v for k, v in api.models.items() if k not in so.STUDY_MODEL_ORDER}    # wrappers edited away
    sync(api)                                                       # after a successful setup: restored
    s = _state(api)
    assert s["models"] == set(so.STUDY_MODEL_ORDER) and s["default"] == "memento-query,memento-query-polymer"
    remove(api, world)
    sync(api)                                                       # after removal: nothing comes back
    assert _state(api) == {"models": set(), "group": False, "default": "memento-query", "order_has_study": False}
    study(api, world)                                               # reinstalled
    sync(api)
    assert _state(api)["default"] == "memento-query,memento-query-polymer"
    so.activate("http://x", {}, "memento_ingestion_polymer", False)   # an interrupted state: one function off
    sync(api)
    assert _state(api)["default"] == "memento-query" and not _state(api)["order_has_study"]


def test_an_unreadable_state_leaves_the_default_model_alone(api, world, monkeypatch):
    study(api, world)
    before = dict(api.config)
    monkeypatch.setattr(so.requests, "get", lambda url, **kw: (_ for _ in ()).throw(ConnectionError("down"))
                        if "/functions/" in url else api.request("GET", url, **kw))
    with pytest.raises(SystemExit):                     # reported (services.sh logs a warning)
        sync(api)
    assert api.config["DEFAULT_MODELS"] == before["DEFAULT_MODELS"]


# ── removal and reinstallation (R1, N12) ──────────────────────────────────────

def _rows(engine, table):
    with engine.connect() as c:
        return sorted((dict(r) for r in c.execute(select(table)).mappings()), key=lambda r: r["file_id"])


def _claim(world, fid, coll=POLY_TEMP, chat=CHAT):
    src = td.claim(world.engine, chat_id=chat, user_id="u1", file_id=fid, name=f"{fid}.pdf", collection=coll)
    td.set_status(world.engine, chat, fid, "ready", pages=1)
    return src


def test_removal_switches_off_archives_deletes_and_keeps_todays_copy(api, world):
    study(api, world)
    _claim(world, "f1")
    _claim(world, "c1", coll=td.DEFAULT_TEMP_COLLECTION, chat="9a8b7c6d-5e4f-4a3b-8c2d-1e0f9a8b7c6d")
    live = _rows(world.engine, td.TEMP_DOCS)
    remove(api, world)
    assert "memento_query_polymer" not in api.functions and "memento_ingestion_polymer" not in api.functions
    assert not set(so.STUDY_MODEL_ORDER) & set(api.models)
    assert "User study" not in [g["name"] for g in api.groups.values()]
    assert api.config["DEFAULT_MODELS"] == "memento-query"
    assert api.functions["memento_query"]["is_active"]                  # today's copy untouched
    assert [r["file_id"] for r in _rows(world.engine, td.TEMP_DOCS)] == ["c1"]
    assert _rows(world.engine, td.ARCHIVE) == [r for r in live if r["file_id"] == "f1"]
    # the authoritative backup holds the rows; the global cleaner finds no row of the copy
    backups = sorted(world.backups.glob("*remove_study*/study_rows_after_drain.json"))
    assert backups and [r["file_id"] for r in json.loads(backups[-1].read_text())] == ["f1"]
    data = json.loads((world.out / "_runtime" / "query_models.json").read_text())
    assert "memento_query_polymer" not in data.get("copies", {})


def test_removal_twice_and_reinstallation_twice(api, world):
    study(api, world)
    _claim(world, "f1")
    before = _rows(world.engine, td.TEMP_DOCS)
    remove(api, world)
    remove(api, world)                                                  # nothing left to remove: no error
    study(api, world)
    assert _rows(world.engine, td.TEMP_DOCS) == before and _rows(world.engine, td.ARCHIVE) == []
    study(api, world)
    assert _rows(world.engine, td.TEMP_DOCS) == before


def test_a_removal_dry_run_changes_nothing(api, world):
    study(api, world)
    _claim(world, "f1")
    before, rows = api.snapshot(), _rows(world.engine, td.TEMP_DOCS)
    remove(api, world, dry=True)
    assert api.snapshot() == before and _rows(world.engine, td.TEMP_DOCS) == rows


def test_a_conflicting_archive_stops_removal_before_anything_is_deleted(api, world):
    study(api, world)
    _claim(world, "f1")
    live = _rows(world.engine, td.TEMP_DOCS)
    with world.engine.begin() as c:
        c.execute(td.ARCHIVE.insert().values(**{**live[0], "status": "failed"}))
    with pytest.raises(SystemExit, match="differ"):
        remove(api, world)
    assert "memento_query_polymer" in api.functions and not api.functions["memento_query_polymer"]["is_active"]
    assert _rows(world.engine, td.TEMP_DOCS) == live


def test_a_restore_conflict_keeps_the_copy_off(api, world):
    study(api, world)
    _claim(world, "f1")
    remove(api, world)
    _claim(world, "f1")                                                 # the same key live again, another source
    with pytest.raises(SystemExit, match="differ"):
        study(api, world)
    assert not api.functions["memento_query_polymer"]["is_active"]


def test_todays_functions_are_put_back_from_the_deployment_backup(api, world, tmp_path):
    backup = tmp_path / "deploy_backup"
    backup.mkdir()
    saved = {fid: {"content": "OLD CODE " + fid, "valves": {"COLLECTION": "memento-comp-mat-rag", "K": fid}}
             for fid in ("memento_query", "memento_ingestion")}
    (backup / "functions.json").write_text(json.dumps(saved))
    study(api, world)
    remove(api, world, restore_from=backup)
    for fid, f in saved.items():
        assert api.functions[fid]["content"] == f["content"] and api.functions[fid]["valves"] == f["valves"]
        assert api.functions[fid]["is_active"]


# ── API failures are reported, never success ──
NOT_FOUND = "We could not find what you're looking for :/"     # OpenWebUI 0.11.4 answers a missing item with 401


def _failing(api, monkeypatch, when, status=500, body=None):
    real = api.request

    def request(method, url, json=None, **kw):
        if when(method, url, json):
            api.calls.append((method, url, json))
            return api._r(status, {"detail": "temporarily unavailable"} if body is None else body)
        return real(method, url, json=json, **kw)
    monkeypatch.setattr(api, "request", request)


def test_setup_stops_before_switching_on_when_a_model_cannot_be_registered(api, world, monkeypatch):
    _failing(api, monkeypatch, lambda m, u, j: m == "POST" and ("/models/create" in u or "/models/model/update" in u))
    with pytest.raises(SystemExit):
        study(api, world)
    assert not api.functions[so.STUDY_QUERY_FN]["is_active"] and not api.functions[so.STUDY_INGEST_FN]["is_active"]
    assert api.config["DEFAULT_MODELS"] == "memento-query"                 # no default for models that are not there


def test_setup_stops_when_the_model_map_is_not_written(api, world, monkeypatch):
    import memento.query.models as qm
    monkeypatch.setattr(qm, "merge_model_map", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(SystemExit):
        study(api, world)
    assert not api.functions[so.STUDY_QUERY_FN]["is_active"]


def test_setup_stops_when_the_default_model_cannot_be_saved(api, world, monkeypatch):
    _failing(api, monkeypatch, lambda m, u, j: m == "POST" and u.endswith("/api/v1/configs/models"))
    with pytest.raises(SystemExit):
        study(api, world)


def test_removal_stops_before_deleting_functions_when_a_model_delete_fails(api, world, monkeypatch):
    study(api, world)
    _failing(api, monkeypatch, lambda m, u, j: m == "POST" and "/models/model/delete" in u)
    with pytest.raises(SystemExit):
        remove(api, world)
    assert so.STUDY_QUERY_FN in api.functions and so.STUDY_INGEST_FN in api.functions      # still there, off
    assert not api.functions[so.STUDY_QUERY_FN]["is_active"]
    assert any(g["name"] == so.STUDY_GROUP for g in api.groups.values())


def test_removal_counts_a_model_that_is_already_gone_as_deleted(api, world, monkeypatch):
    study(api, world)
    for m in so.STUDY_MODELS[:2]:
        api.models.pop(m["id"], None)                                       # deleted by hand before
    real = api.request

    def request(method, url, json=None, **kw):                                # the real API: 401 "not found"
        if method == "POST" and "/models/model/delete" in url and json["id"] not in api.models:
            return api._r(401, {"detail": NOT_FOUND})
        if method == "GET" and "/models/model" in url and url.split("id=")[-1] not in api.models:
            return api._r(401, {"detail": NOT_FOUND})
        return real(method, url, json=json, **kw)
    monkeypatch.setattr(api, "request", request)
    remove(api, world)
    assert set(api.functions) == {"memento_query", "memento_ingestion"}          # the copy is gone, the default one
                                                                                 # kept
    assert not [m for m in api.models if m.startswith("memento-")]


def test_removal_stops_when_the_group_cannot_be_deleted(api, world, monkeypatch):
    study(api, world)
    _failing(api, monkeypatch, lambda m, u, j: m == "DELETE" and "/groups/id/" in u)
    with pytest.raises(SystemExit):
        remove(api, world)


def test_a_failed_valve_read_is_unknown_and_keeps_the_study_default(api, world, monkeypatch):
    study(api, world)
    before = (api.config["DEFAULT_MODELS"], list(api.config["MODEL_ORDER_LIST"]))
    _failing(api, monkeypatch, lambda m, u, j: m == "GET" and f"/functions/id/{so.STUDY_QUERY_FN}/valves" in u)
    assert so.study_state("http://x", {}) is None
    with pytest.raises(SystemExit):                     # services.sh logs a warning; OpenWebUI starts anyway
        sync(api)
    assert (api.config["DEFAULT_MODELS"], api.config["MODEL_ORDER_LIST"]) == before


def test_an_unauthorised_read_is_unknown_not_absent(api, world, monkeypatch):
    study(api, world)
    _failing(api, monkeypatch, lambda m, u, j: m == "GET" and f"/functions/id/{so.STUDY_INGEST_FN}" in u,
             status=401, body={"detail": "401 Unauthorized"})
    assert so.study_state("http://x", {}) is None


def test_a_missing_function_is_absent_as_the_real_api_says_it(api, world, monkeypatch):
    _failing(api, monkeypatch, lambda m, u, j: m == "GET" and f"/functions/id/{so.STUDY_QUERY_FN}" in u,
             status=401, body={"detail": NOT_FOUND})
    assert so.study_state("http://x", {}) is False


def test_removal_stops_before_switching_off_when_its_settings_backup_cannot_be_read(api, world, monkeypatch):
    study(api, world)
    _failing(api, monkeypatch, lambda m, u, j: m == "GET" and f"/functions/id/{so.STUDY_QUERY_FN}/valves" in u)
    with pytest.raises((SystemExit, RuntimeError)):
        remove(api, world)
    assert api.functions[so.STUDY_QUERY_FN]["is_active"]                    # nothing done without its backup


# ── the deployment backup is run from a worktree before the merge ──
