# Each copy's model list, model map and publication. The second copy answers on its own local entry
# (config/query_models_study.yaml) plus the two cloud models; its workspace models are named from its QUERY_MODEL_ID
# ("memento-query-polymer", "memento-query-polymer-qwen35"). The browser's model map (_runtime/query_models.json) holds
# every copy's entries side by side; the top-level "models" list is kept for the default copy, its entries unchanged.
import ast
import json
import sys
import threading
from pathlib import Path

import pytest
from memento.config.paths import PROJECT_ROOT
from memento.query import cache, generation, service, workers

ROOT = PROJECT_ROOT
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from memento.query import models as qm  # noqa: E402
from test_two_libraries_worker import QUERY, load_copy  # noqa: E402

STUDY = dict(VARIANTS_FILE="query_models_study.yaml", QUERY_MODEL_ID="memento-query-polymer",
             UPLOAD_MODEL_ID="memento-ingestion-polymer", DOC_CHAT_MODEL_ID="memento-document-chat-polymer",
             COLLECTION="memento-polymer-rag", LLM_BASE_URL="http://llm.test:8001/v1", LLM_MODEL="qwen-flash-next")


@pytest.fixture
def qp(monkeypatch):
    return load_copy(QUERY, "memento_query", monkeypatch)


@pytest.fixture
def study(monkeypatch):
    return load_copy(QUERY, "memento_query_polymer", monkeypatch)


# ── the second copy's model list ──

def test_the_study_list_has_flash_next_and_the_two_cloud_models():
    vs = qm.load_variants(ROOT / "config" / "query_models_study.yaml")
    local = [v for v in vs if not v.is_cloud]
    assert len(local) == 1 and "Flash-Next" in local[0].name and local[0].model == "qwen-flash-next"
    assert [v.id for v in vs if v.is_cloud] == ["qwen35", "qwen36"]
    today = {v.id: v for v in qm.load_variants()}
    for v in vs:                                       # the cloud models exactly as in today's list
        if v.is_cloud:
            assert v == today[v.id]


def test_each_copy_reads_its_own_model_list(qp):
    assert qp.Pipe.Valves().VARIANTS_FILE == "query_models.yaml"
    today = generation.copy_variants(qp.Pipe.Valves())
    assert [v.name for v in today] == [v.name for v in qm.load_variants()]
    study = generation.copy_variants(qp.Pipe.Valves(**STUDY))
    assert "Flash-Next" in next(v for v in study if not v.is_cloud).name


@pytest.mark.parametrize("bad", ["../query_models.yaml", "/etc/hosts", "missing.yaml", "  "])
def test_a_bad_list_name_leaves_only_the_local_model(qp, bad):
    vs = generation.copy_variants(qp.Pipe.Valves(VARIANTS_FILE=bad))
    assert len(vs) == 1 and not vs[0].is_cloud


def test_every_list_read_in_the_pipe_names_the_copys_valves():
    trees = [ast.parse(QUERY.read_text())] + [ast.parse(f.read_text()) for f in sorted((ROOT / "memento/query").rglob("*.py"))]
    calls = [n for t in trees for n in ast.walk(t) if isinstance(n, ast.Call)
             and (getattr(n.func, "id", None) or getattr(n.func, "attr", None)) == "copy_variants"]
    assert len(calls) >= 8
    for c in calls:
        assert c.args and ast.unparse(c.args[0]).endswith("valves"), ast.unparse(c)


# ── workspace ids and resolving the chosen model ──────────────────────────────

def test_workspace_ids_follow_the_copys_query_model_id():
    vs = qm.load_variants(ROOT / "config" / "query_models_study.yaml")
    local, q35 = next(v for v in vs if not v.is_cloud), next(v for v in vs if v.id == "qwen35")
    assert qm.workspace_id(local) == "memento-query" and qm.workspace_id(q35) == "memento-query-qwen35"   # today
    assert qm.workspace_id(local, "memento-query-polymer") == "memento-query-polymer"
    assert qm.workspace_id(q35, "memento-query-polymer") == "memento-query-polymer-qwen35"


def test_resolve_finds_the_study_models_and_falls_back_to_the_study_local_model():
    vs = qm.load_variants(ROOT / "config" / "query_models_study.yaml")
    base = "memento-query-polymer"
    assert qm.resolve("memento_query_polymer.qwen35", vs, base).id == "qwen35"
    assert qm.resolve("memento-query-polymer-qwen36", vs, base).id == "qwen36"
    assert qm.resolve("memento_query_polymer.local", vs, base).model == "qwen-flash-next"
    assert qm.resolve("memento-query-polymer-gone", vs, base).model == "qwen-flash-next"     # unknown → local
    today = qm.load_variants()
    assert qm.resolve("memento-query-qwen35", today).id == "qwen35"                          # today unchanged


# ── the model map: every copy's entries ───────────────────────────────────────

def _today_entries(variants):
    """The single-library write_model_colours output, the reference for the default copy's entries."""
    import re
    out = []
    for v in variants:
        light = v.accent.lower() if v.accent else None
        hint = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", v.full_description.split("\n", 1)[1]) if v.is_cloud else None
        out.append({"id": qm.workspace_id(v), "name": v.name, "cloud": v.is_cloud, "accent": light,
                    "accent_dark": v.accent_dark.lower() if v.accent_dark else light, "hint": hint})
    return out


def _publish(mod, tmp_path, monkeypatch, **valves):
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path))
    workers.publish_model_colours(mod.Pipe.Valves(**valves), mod._OWN_ID)
    return json.loads((tmp_path / "_runtime" / "query_models.json").read_text())


def test_two_copies_share_the_map_without_loss_and_todays_entries_are_unchanged(qp, study, tmp_path, monkeypatch):
    _publish(qp, tmp_path, monkeypatch)
    data = _publish(study, tmp_path, monkeypatch, **STUDY)
    today = _today_entries(qm.load_variants())
    assert data["models"][:len(today)] == today                                        # byte-identical, first
    ids = [m["id"] for m in data["models"]]
    assert "memento-query-polymer" in ids and "memento-query-polymer-qwen35" in ids
    assert set(data["copies"]) == {"memento_query", "memento_query_polymer"}
    entry = next(e for e in data["copies"]["memento_query_polymer"] if e["id"] == "memento-query-polymer")
    assert entry["kind"] == "query" and entry["collection"] == "memento-polymer-rag"
    assert entry["siblings"] == {"query": "memento-query-polymer", "upload": "memento-ingestion-polymer",
                                 "doc": "memento-document-chat-polymer"}
    kinds = {e["id"]: e["kind"] for e in data["copies"]["memento_query_polymer"]}
    assert kinds["memento-document-chat-polymer"] == "doc" and kinds["memento-ingestion-polymer"] == "upload"
    today_copy = {e["id"]: e for e in data["copies"]["memento_query"]}
    assert today_copy["memento-query"]["siblings"]["doc"] == "memento-document-chat"


def test_publishing_again_replaces_and_never_duplicates(qp, study, tmp_path, monkeypatch):
    _publish(study, tmp_path, monkeypatch, **STUDY)
    _publish(qp, tmp_path, monkeypatch)
    data = _publish(study, tmp_path, monkeypatch, **STUDY)
    ids = [m["id"] for m in data["models"]]
    assert len(ids) == len(set(ids))
    assert ids[:len(_today_entries(qm.load_variants()))] == [e["id"] for e in _today_entries(qm.load_variants())]


def test_concurrent_publications_lose_nothing(tmp_path):
    path = tmp_path / "_runtime" / "query_models.json"
    vs = qm.load_variants()
    errors = []

    def go(fid, base):
        try:
            for _ in range(40):
                qm.merge_model_map(path, fid, qm.map_entries(vs, base=base, collection="c",
                                                             siblings={"query": base, "upload": "u", "doc": "d"}))
        except Exception as e:  # noqa: BLE001
            errors.append(repr(e))
    ts = [threading.Thread(target=go, args=("memento_query", "memento-query")),
          threading.Thread(target=go, args=("memento_query_polymer", "memento-query-polymer"))]
    [t.start() for t in ts]
    [t.join(60) for t in ts]
    data = json.loads(path.read_text())
    assert errors == [] and set(data["copies"]) == {"memento_query", "memento_query_polymer"}


# ── the shared cloud quota lists both copies' cloud models ─────────────────────

def test_the_quota_state_lists_both_copies_cloud_models(qp, study, tmp_path, monkeypatch):
    _publish(qp, tmp_path, monkeypatch)
    _publish(study, tmp_path, monkeypatch, **STUDY)
    p = study.Pipe()
    p.valves = p.Valves(**STUDY)
    headers = {"x-ratelimit-remaining-minute": "29", "x-ratelimit-remaining-hour": "199",
               "x-ratelimit-remaining-day": "999", "x-ratelimit-limit-day": "1000"}
    generation.record_quota(p.valves, headers)
    state = json.loads((tmp_path / "_runtime" / "cloud_quota.json").read_text())
    assert {"memento-query-qwen35", "memento-query-qwen36", "memento-query-polymer-qwen35",
            "memento-query-polymer-qwen36"} <= set(state["cloud_models"])


# ── the second copy's warm-up asks the model a picker click asks, under the same cache key ──

def test_the_study_warmup_and_a_picker_click_use_one_model_and_one_key(study, monkeypatch):
    import asyncio
    v = study.Pipe.Valves(**STUDY)
    seen = []

    async def handle(body, *a, **k):                  # the copy's own answer path
        seen.append(body["model"])
        yield "x"
    monkeypatch.setattr(service, "handle", handle)
    asyncio.run(workers.run_warmup_job(v, {"action": "summary"}, "T", function_id=study._OWN_ID, db_engine=study._db_engine))
    variants = generation.copy_variants(v)
    warm = qm.resolve(seen[0], variants, v.QUERY_MODEL_ID)
    click = qm.resolve("memento_query_polymer.local", variants, v.QUERY_MODEL_ID)   # memento-query-polymer's base
    assert warm == click and not warm.is_cloud
    assert cache.model_key(warm, v) == cache.model_key(click, v) == "local:qwen-flash-next"


def test_the_study_answers_go_to_flash_next(monkeypatch):
    from test_nothing_found_skip import _library_run
    import test_rq3_answer_pipeline as rq3
    made = []
    real_init = rq3._RecordingAsyncOpenAI.__init__

    def init(self, *a, **k):
        made.append(k.get("base_url"))
        real_init(self, *a, **k)
    monkeypatch.setattr(rq3._RecordingAsyncOpenAI, "__init__", init)
    _library_run(monkeypatch, [0.6, 0.5], QDRANT_URL="http://127.0.0.1:9",
                 **{k: v for k, v in STUDY.items() if k != "COLLECTION"})
    assert made and set(made) == {"http://llm.test:8001/v1"}
