# Is a copy of the Memento pipes ready to work? A second copy works only when its function row exists, its saved Valves
# name every library-relevant setting (Open WebUI saves only explicitly set fields; anything unsaved would silently take
# the default value from config/.env), and its function is active. The default ids are always ready.
import json
import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool
from memento.config.paths import PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT))
from memento.library import copies as cs  # noqa: E402


@pytest.fixture
def engine():
    e = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    with e.begin() as c:
        c.execute(text("CREATE TABLE function (id TEXT PRIMARY KEY, valves TEXT, is_active BOOLEAN)"))
    return e


def complete(role: str, **over) -> dict:
    v = {k: "x" for k in (cs.REQUIRED_QUERY_VALVES if role == "query" else cs.REQUIRED_INGESTION_VALVES)}
    if role == "query":
        v["INGESTION_FUNCTION_ID"] = "memento_ingestion_polymer"
    v.update(over)
    return v


def put(engine, fid, valves, active=True):
    with engine.begin() as c:
        c.execute(text("DELETE FROM function WHERE id = :i"), {"i": fid})
        c.execute(text("INSERT INTO function (id, valves, is_active) VALUES (:i, :v, :a)"),
                  {"i": fid, "v": None if valves is None else json.dumps(valves), "a": active})


@pytest.fixture
def study(engine):
    put(engine, "memento_ingestion_polymer", complete("ingestion"))
    put(engine, "memento_query_polymer", complete("query"))
    return engine


def test_the_lists_name_the_settings_that_decide_the_library_and_the_cache_key():
    for k in ("DOMAIN_NAME", "COLLECTION", "LLM_BASE_URL", "LLM_MODEL", "LLM_API_KEY_ENV", "METRICS_URL",
              "INGESTION_FUNCTION_ID", "VARIANTS_FILE", "TEMPERATURE", "GAP_THINKING_BUDGET", "GRAPH_RETRIEVAL"):
        assert k in cs.REQUIRED_QUERY_VALVES
    for k in ("DOMAIN_NAME", "COLLECTION", "CONTRIBUTOR_GROUP", "BACKEND", "RESEARCH_GAP_TAG", "GRAPH_BUILD",
              "QUERY_MODEL_ID", "UPLOAD_MODEL_ID", "DOC_CHAT_MODEL_ID"):
        assert k in cs.REQUIRED_INGESTION_VALVES
    assert not any("*" in k for k in cs.REQUIRED_QUERY_VALVES + cs.REQUIRED_INGESTION_VALVES)


def test_todays_copies_are_always_ready_without_reading_the_database():
    # the comp-mat copy behaves as today: no database read, no requirement
    for fid, role in (("memento_query", "query"), ("memento_ingestion", "ingestion")):
        st = cs.copy_state(None, fid, role)
        assert st.configured and st.enabled and tuple(st)[:2] == (True, True)


def test_a_complete_active_study_copy_is_enabled(study):
    st = cs.copy_state(study, "memento_query_polymer", "query")
    assert (st.configured, st.enabled) == (True, True)
    assert cs.copy_state(study, "memento_ingestion_polymer", "ingestion").enabled


def test_a_missing_row_is_not_configured(engine):
    st = cs.copy_state(engine, "memento_query_polymer", "query")
    assert (st.configured, st.enabled) == (False, False) and "not installed" in st.reason


@pytest.mark.parametrize("value", [None, "", "   "])
def test_a_missing_none_or_blank_setting_is_not_configured(study, value):
    v = complete("query")
    if value is None:
        v.pop("COLLECTION")
    else:
        v["COLLECTION"] = value
    put(study, "memento_query_polymer", v)
    st = cs.copy_state(study, "memento_query_polymer", "query")
    assert not st.configured and not st.enabled and "COLLECTION" in st.reason


def test_a_setting_saved_as_null_is_missing(study):
    put(study, "memento_query_polymer", complete("query", LLM_MODEL=None))
    st = cs.copy_state(study, "memento_query_polymer", "query")
    assert not st.configured and "LLM_MODEL" in st.reason


def test_false_and_zero_are_real_values(study):
    put(study, "memento_query_polymer", complete("query", GRAPH_RETRIEVAL=False, TEMPERATURE=0, HYBRID_ENABLED=False))
    put(study, "memento_ingestion_polymer", complete("ingestion", GRAPH_BUILD=False))
    assert cs.copy_state(study, "memento_query_polymer", "query").enabled


def test_an_inactive_complete_copy_is_configured_but_not_enabled(study):
    put(study, "memento_query_polymer", complete("query"), active=False)
    st = cs.copy_state(study, "memento_query_polymer", "query")
    assert (st.configured, st.enabled) == (True, False) and "inactive" in st.reason


def test_a_query_copy_needs_its_linked_ingestion_copy(study):
    put(study, "memento_ingestion_polymer", complete("ingestion", CONTRIBUTOR_GROUP=" "))
    st = cs.copy_state(study, "memento_query_polymer", "query")
    assert (st.configured, st.enabled) == (False, False) and "memento_ingestion_polymer" in st.reason
    put(study, "memento_ingestion_polymer", complete("ingestion"), active=False)
    st = cs.copy_state(study, "memento_query_polymer", "query")
    assert (st.configured, st.enabled) == (True, False) and "memento_ingestion_polymer" in st.reason
    put(study, "memento_ingestion_polymer", None)
    assert not cs.copy_state(study, "memento_query_polymer", "query").configured


def test_a_query_copy_linked_to_todays_ingestion_copy_needs_nothing_more(study):
    put(study, "memento_query_polymer", complete("query", INGESTION_FUNCTION_ID="memento_ingestion"))
    assert cs.copy_state(study, "memento_query_polymer", "query").enabled


def test_encrypted_or_unreadable_valves_are_not_configured(study):
    with study.begin() as c:
        c.execute(text("UPDATE function SET valves = :v WHERE id = 'memento_query_polymer'"),
                  {"v": "gAAAAABk-encrypted-looking-text"})
    st = cs.copy_state(study, "memento_query_polymer", "query")
    assert not st.configured and "valves" in st.reason


def test_valves_stored_as_a_json_string_inside_json_are_read(study):
    # OpenWebUI's JSONField can hand back a dict or a JSON text depending on the database
    with study.begin() as c:
        c.execute(text("UPDATE function SET valves = :v WHERE id = 'memento_query_polymer'"),
                  {"v": json.dumps(json.dumps(complete("query")))})
    assert cs.copy_state(study, "memento_query_polymer", "query").enabled


def test_an_unreadable_database_fails_closed():
    broken = create_engine("sqlite://")          # no function table
    st = cs.copy_state(broken, "memento_query_polymer", "query")
    assert (st.configured, st.enabled) == (False, False) and st.reason


def test_an_unknown_role_is_refused(study):
    with pytest.raises(ValueError):
        cs.copy_state(study, "memento_query_polymer", "filter")


def test_missing_lists_the_absent_settings_in_order():
    v = complete("ingestion")
    v.pop("BACKEND")
    v["COLLECTION"] = None
    assert cs.missing(v, "ingestion") == [k for k in cs.REQUIRED_INGESTION_VALVES if k in ("BACKEND", "COLLECTION")]
