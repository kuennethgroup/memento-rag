# One research-gap updater per library. Each copy's updater (scripts/update_gap_ledger.py --function-id X) runs only
# while its copy is enabled (memento/library/copies.py), holds a lock per collection, stops when the copy is switched off
# or its Valves name another collection, and asks its own copy's model server whether there is room. services.sh starts
# the second copy's updater only when that copy is enabled (scripts/copy_state_cli.py).
import json
import sys
import types
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text
from memento.config.paths import PROJECT_ROOT

ROOT = PROJECT_ROOT
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "openwebui/functions"))
from memento.library import copies as cs  # noqa: E402

POLY = "memento-polymer-rag"


@pytest.fixture
def db(tmp_path, monkeypatch):
    url = f"sqlite:///{tmp_path / 'owui.sqlite'}"
    e = create_engine(url)
    with e.begin() as c:
        c.execute(text("CREATE TABLE function (id TEXT PRIMARY KEY, valves TEXT, is_active BOOLEAN)"))
    monkeypatch.setenv("DATABASE_URL", url)
    return e


def put(engine, fid, valves, active=True):
    with engine.begin() as c:
        c.execute(text("DELETE FROM function WHERE id = :i"), {"i": fid})
        c.execute(text("INSERT INTO function VALUES (:i, :v, :a)"), {"i": fid, "v": json.dumps(valves), "a": active})


def study(engine, active=True, **over):
    import query_pipe as qp
    have = qp.Pipe.Valves().model_dump()
    q = {k: have.get(k, f"{k.lower()}-v") for k in cs.REQUIRED_QUERY_VALVES}
    q.update(COLLECTION=POLY, DOMAIN_NAME="polymer-engineering", INGESTION_FUNCTION_ID="memento_ingestion_polymer",
             METRICS_URL="http://llm.test:8001/metrics", WARMUP_MAX_RUNNING=2)
    q.update(over)
    put(engine, "memento_ingestion_polymer", {k: "x" for k in cs.REQUIRED_INGESTION_VALVES}, active)
    put(engine, "memento_query_polymer", q, active)


def test_saved_valves_say_when_a_copy_is_missing(db):
    import build_gap_ledger as bgl
    assert bgl.saved_valves(db, "memento_query_polymer") is None
    study(db)
    assert bgl.saved_valves(db, "memento_query_polymer")["COLLECTION"] == POLY
    assert bgl.saved_valves(db) is None                          # today's copy not saved in this database either


def test_each_collection_has_its_own_lock_and_one_updater(tmp_path):
    import update_gap_ledger as ugl
    a = ugl.take_lock(tmp_path, "memento-comp-mat-rag")
    b = ugl.take_lock(tmp_path, POLY)
    assert a is not None and b is not None
    assert ugl.take_lock(tmp_path, POLY) is None                 # the same library twice: refused
    assert (tmp_path / f".gap_updater.{POLY}.lock").exists()
    b.close()
    assert ugl.take_lock(tmp_path, POLY) is not None             # free again once released


@pytest.mark.parametrize("setup", ["missing", "inactive", "incomplete"])
def test_an_updater_of_a_copy_that_is_not_enabled_exits_and_touches_nothing(db, tmp_path, monkeypatch, setup):
    import update_gap_ledger as ugl
    if setup == "inactive":
        study(db, active=False)
    elif setup == "incomplete":
        study(db, COLLECTION="")
    monkeypatch.setattr(ugl, "one_round", lambda *a, **k: pytest.fail("a round ran"))
    assert ugl.main(["--function-id", "memento_query_polymer", "--once", "--lock-dir", str(tmp_path)]) == 0
    assert not list(tmp_path.glob(".gap_updater.*.lock"))


def test_an_enabled_copy_runs_its_round_on_its_own_collection(db, tmp_path, monkeypatch):
    import update_gap_ledger as ugl
    study(db)
    seen = []
    monkeypatch.setattr(ugl, "model_has_room", lambda V: True)
    monkeypatch.setattr(ugl, "upload_in_progress", lambda e: False)
    monkeypatch.setattr(ugl, "other_gap_job_running", lambda: False)
    monkeypatch.setattr(ugl, "one_round", lambda engine, V: seen.append(V.COLLECTION) or "nothing new")
    assert ugl.main(["--function-id", "memento_query_polymer", "--once", "--lock-dir", str(tmp_path)]) == 0
    assert seen == [POLY]


@pytest.mark.parametrize("change", ["switched_off", "other_collection"])
def test_the_updater_stops_when_its_copy_is_switched_off_or_moves_library(db, tmp_path, monkeypatch, change):
    import update_gap_ledger as ugl
    study(db)
    rounds = []
    monkeypatch.setattr(ugl, "model_has_room", lambda V: True)
    monkeypatch.setattr(ugl, "upload_in_progress", lambda e: False)
    monkeypatch.setattr(ugl, "other_gap_job_running", lambda: False)

    def one_round(engine, V):
        rounds.append(V.COLLECTION)
        if change == "switched_off":
            study(db, active=False)
        else:
            study(db, COLLECTION="memento-other-rag")
        return "nothing new"
    monkeypatch.setattr(ugl, "one_round", one_round)
    assert ugl.main(["--function-id", "memento_query_polymer", "--every", "0", "--lock-dir", str(tmp_path)]) == 0
    assert rounds == [POLY]                                      # one round, then it stopped by itself
    assert ugl.take_lock(tmp_path, POLY) is not None              # and released its lock


def test_a_setting_saved_while_the_updater_runs_applies_at_its_next_round(db, tmp_path, monkeypatch):
    # each round reads the saved Valves: an admin's change in Open WebUI needs no restart
    import update_gap_ledger as ugl
    study(db, LLM_MODEL="model-a")
    seen = []
    monkeypatch.setattr(ugl, "model_has_room", lambda V: True)
    monkeypatch.setattr(ugl, "upload_in_progress", lambda e: False)
    monkeypatch.setattr(ugl, "other_gap_job_running", lambda: False)

    def one_round(engine, V):
        seen.append(V.LLM_MODEL)
        if len(seen) == 1:
            study(db, LLM_MODEL="model-b")                        # saved while the updater runs
        else:
            study(db, active=False)                               # then switched off: it stops
        return "nothing new"
    monkeypatch.setattr(ugl, "one_round", one_round)
    assert ugl.main(["--function-id", "memento_query_polymer", "--every", "0", "--lock-dir", str(tmp_path)]) == 0
    assert seen == ["model-a", "model-b"]


def test_room_on_the_model_is_asked_of_the_copys_own_server(monkeypatch):
    import update_gap_ledger as ugl
    import query_pipe as qp
    monkeypatch.setenv("MEMENTO_TEST_STUDY_KEY", "k1")
    asked = []
    metrics = 'vllm:num_requests_running{m="x"} 2.0\nvllm:num_requests_waiting{m="x"} 0.0\n'

    def get(url, timeout=5, headers=None):
        asked.append((url, headers))
        return types.SimpleNamespace(text=metrics)
    monkeypatch.setattr("requests.get", get)
    V = qp.Pipe.Valves(METRICS_URL="http://llm.test:8001/metrics", METRICS_API_KEY_ENV="MEMENTO_TEST_STUDY_KEY",
                       WARMUP_MAX_RUNNING=2)
    assert ugl.model_has_room(V) is True
    V.WARMUP_MAX_RUNNING = 1
    assert ugl.model_has_room(V) is False
    assert asked[0] == ("http://llm.test:8001/metrics", {"Authorization": "Bearer k1"})
    ugl.model_has_room(qp.Pipe.Valves())                          # today's copy: no key header
    assert asked[-1][1] is None


def test_the_state_cli_says_whether_a_copy_is_enabled(db, capsys):
    import copy_state_cli as cli
    assert cli.main(["--function-id", "memento_query_polymer"]) == 1
    study(db)
    assert cli.main(["--function-id", "memento_query_polymer"]) == 0
    assert cli.main(["--function-id", "memento_query"]) == 0
    assert "enabled" in capsys.readouterr().out


def test_services_starts_the_study_updater_only_when_its_copy_is_enabled():
    src = (ROOT / "scripts/services.sh").read_text()
    assert 'SECOND_ID="${SECOND_LIBRARY_ID:-polymer}"' in src and 'GAP_UPDATER_SECOND_SESSION="gap-updater-$SECOND_ID"' in src
    block = src[src.index("start_gap_updater() {"):src.index("stop_gap_updater() {")]
    assert 'copy_state_cli.py --function-id "memento_query_$SECOND_ID"' in block
    assert "update_gap_ledger.py --function-id 'memento_query_$SECOND_ID'" in block
    stop = src[src.index("stop_gap_updater() {"):]
    stop = stop[:stop.index("\n}\n")]
    assert "$GAP_UPDATER_SECOND_SESSION" in stop
