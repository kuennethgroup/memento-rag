# The second library's settings (scripts/setup_openwebui.py `study`) come from config/.env, with the values of the
# two-library deployment as defaults; each library's start-screen suggestions come from its domain profile.
import json
import os
import subprocess
import sys

import yaml

from memento.config.paths import PROJECT_ROOT

KEYS = ("SECOND_LIBRARY_ID", "SECOND_LIBRARY_DOMAIN", "SECOND_LIBRARY_GROUP", "REASONING_MODEL_NAME",
        "COLLECTION_PREFIX")


def _setup_values(**env) -> dict:
    code = ("import importlib.util as u, json; s = u.spec_from_file_location('so', 'scripts/setup_openwebui.py'); "
            "m = u.module_from_spec(s); s.loader.exec_module(m); "
            "print(json.dumps({k: getattr(m, k) for k in ('STUDY_GROUP', 'STUDY_QUERY_FN', 'STUDY_INGEST_FN', "
            "'STUDY_IDS', 'STUDY_DOMAIN', 'STUDY_COLLECTION', 'FLASH_NEXT', 'QUERY_SUGGESTIONS', "
            "'STUDY_QUERY_SUGGESTIONS')}))")
    base = {k: v for k, v in os.environ.items() if k not in KEYS}
    out = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=120,
                         env={**base, "PYTHONPATH": str(PROJECT_ROOT), **env})
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def _profile(name: str) -> dict:
    return yaml.safe_load((PROJECT_ROOT / "config" / "domains" / f"{name}.yaml").read_text())


def test_the_second_library_defaults_to_the_two_library_deployment():
    v = _setup_values(DOMAIN_NAME="computational-materials")
    assert v["STUDY_GROUP"] == "User study"
    assert (v["STUDY_QUERY_FN"], v["STUDY_INGEST_FN"]) == ("memento_query_polymer", "memento_ingestion_polymer")
    assert v["STUDY_IDS"] == {"query": "memento-query-polymer", "upload": "memento-ingestion-polymer",
                              "doc": "memento-document-chat-polymer"}
    assert (v["STUDY_DOMAIN"], v["STUDY_COLLECTION"]) == ("polymer-engineering", "memento-polymer-rag")
    assert (v["FLASH_NEXT"]["LLM_MODEL"], v["FLASH_NEXT"]["LLM_API_KEY_ENV"]) == ("qwen-flash-next",
                                                                                  "REASONING_API_KEY")


def test_the_second_library_is_configured_in_the_env():
    v = _setup_values(DOMAIN_NAME="computational-materials", SECOND_LIBRARY_ID="lab2", SECOND_LIBRARY_DOMAIN="ceramics",
                      SECOND_LIBRARY_GROUP="Lab two", REASONING_MODEL_NAME="local-model")
    assert v["STUDY_GROUP"] == "Lab two"
    assert (v["STUDY_QUERY_FN"], v["STUDY_INGEST_FN"]) == ("memento_query_lab2", "memento_ingestion_lab2")
    assert v["STUDY_IDS"]["doc"] == "memento-document-chat-lab2"
    assert (v["STUDY_DOMAIN"], v["STUDY_COLLECTION"]) == ("ceramics", "memento-ceramics-rag")
    assert (v["FLASH_NEXT"]["LLM_MODEL"], v["FLASH_NEXT"]["LLM_API_KEY_ENV"]) == ("local-model", "REASONING_API_KEY")


def test_start_suggestions_come_from_each_librarys_domain_profile():
    v = _setup_values(DOMAIN_NAME="computational-materials")
    assert v["QUERY_SUGGESTIONS"] == _profile("computational-materials")["start_suggestions"]
    assert v["STUDY_QUERY_SUGGESTIONS"] == _profile("polymer-engineering")["start_suggestions"]
    w = _setup_values(DOMAIN_NAME="ceramics", SECOND_LIBRARY_DOMAIN="generic")       # no suggestions of its own
    assert w["QUERY_SUGGESTIONS"] == w["STUDY_QUERY_SUGGESTIONS"] == _profile("generic")["start_suggestions"]


def test_services_sh_names_the_second_library_by_its_setting():
    sh = (PROJECT_ROOT / "scripts" / "services.sh").read_text()
    assert "memento_query_polymer" not in sh and "gap-updater-polymer" not in sh
    assert 'SECOND_ID="${SECOND_LIBRARY_ID:-polymer}"' in sh and sh.count("memento_query_$SECOND_ID") == 2
