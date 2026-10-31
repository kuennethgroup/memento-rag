# Each copy's answer model gets its own key. The Valve LLM_API_KEY_ENV names the environment variable (the second copy's
# local model and the default copy: REASONING_API_KEY, the one answer-model key); every client of the copy's own model reads
# it: the pipe's answers, checks and gap work, the upload's captions and gap tags, Document Chat (it copies the
# ingestion settings), the gap updater and the gap-ledger build script. The warm-up's metrics request sends a key only
# when its own Valve names one.
import ast
import asyncio
import os
import sys
import types
from pathlib import Path

import pytest
from memento.config.paths import PROJECT_ROOT
from memento.query import library_state, request, workers
from memento.ingestion.upload import document

ROOT = PROJECT_ROOT
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "openwebui/functions"))
sys.path.insert(0, str(ROOT / "scripts"))

STUDY_KEY, DEFAULT_KEY = "study-key-value", "default-key-value"


@pytest.fixture(autouse=True)
def keys(monkeypatch):
    monkeypatch.setenv("MEMENTO_TEST_STUDY_KEY", STUDY_KEY)
    monkeypatch.setenv("REASONING_API_KEY", DEFAULT_KEY)
    monkeypatch.setenv("OPENAI_API_KEY", DEFAULT_KEY)          # a client given no key at all falls back to it


def _qp():
    import query_pipe
    return query_pipe


# ── the query pipe ───────────────────────────────────────────────────────────

def test_the_key_comes_from_the_variable_the_copy_names():
    qp = _qp()
    assert qp.Pipe.Valves().LLM_API_KEY_ENV == "REASONING_API_KEY"              # the answer model's key
    assert request.llm_key(qp.Pipe.Valves()) == DEFAULT_KEY
    assert request.llm_key(qp.Pipe.Valves(LLM_API_KEY_ENV="MEMENTO_TEST_STUDY_KEY")) == STUDY_KEY
    assert request.llm_key(qp.Pipe.Valves(LLM_API_KEY_ENV="MEMENTO_TEST_UNSET")) == ""   # never another copy's key
    assert request.llm_key(qp.Pipe.Valves(LLM_API_KEY_ENV="  ")) == ""


def test_no_other_read_of_the_default_key_is_left_in_the_query_pipe():
    src = (ROOT / "openwebui/functions/query_pipe.py").read_text()
    tree = ast.parse(src)
    reads = [n for n in ast.walk(tree) if isinstance(n, ast.Constant) and n.value in ("OPENAI_API_KEY", "REASONING_API_KEY")]
    # the Valve's default (the answer model's key) and nothing else: every client reads request.llm_key(self.valves)
    assert [n.value for n in reads] == ["REASONING_API_KEY"], [ast.get_source_segment(src, n) for n in reads]


def test_an_answer_of_the_study_copy_uses_the_study_key_and_endpoint(monkeypatch):
    from test_nothing_found_skip import _library_run
    import test_rq3_answer_pipeline as rq3
    made = []
    real_init = rq3._RecordingAsyncOpenAI.__init__

    def init(self, *a, **k):                    # the answer pipeline's fake model client, recording its key
        made.append((k.get("base_url"), k.get("api_key")))
        real_init(self, *a, **k)
    monkeypatch.setattr(rq3._RecordingAsyncOpenAI, "__init__", init)
    _library_run(monkeypatch, [0.6, 0.5], LLM_API_KEY_ENV="MEMENTO_TEST_STUDY_KEY",
                 LLM_BASE_URL="http://llm.test:8001/v1", QDRANT_URL="http://127.0.0.1:9")
    local = [k for url, k in made if url == "http://llm.test:8001/v1"]
    assert local and set(local) == {STUDY_KEY}
    assert DEFAULT_KEY not in {k for _u, k in made}


def test_the_metrics_request_carries_a_key_only_when_the_copy_names_one(monkeypatch):
    qp = _qp()
    sent = []

    class R:
        ok, text = True, "vllm:num_requests_running 0"
    import requests
    monkeypatch.setattr(requests, "get", lambda url, timeout=2, headers=None: sent.append((url, headers)) or R())
    assert qp.Pipe.Valves().METRICS_API_KEY_ENV == ""                         # today's copy: no key sent
    workers.vllm_metrics("http://127.0.0.1:8090/metrics", None)
    workers.vllm_metrics("http://llm.test:8001/metrics", "MEMENTO_TEST_STUDY_KEY")
    workers.vllm_metrics("http://llm.test:8001/metrics", "MEMENTO_TEST_UNSET")
    assert sent[0][1] in (None, {})
    assert sent[1][1] == {"Authorization": f"Bearer {STUDY_KEY}"}
    assert sent[2][1] in (None, {})


def test_the_warmup_passes_the_copys_metrics_key(monkeypatch):
    qp = _qp()
    seen = []
    monkeypatch.setattr(workers, "vllm_metrics", lambda url=None, key_env=None: seen.append((url, key_env)) or None)
    monkeypatch.setattr(library_state, "in_progress_sources", lambda *a: {"busy"})
    pipe = types.SimpleNamespace(valves=types.SimpleNamespace(COLLECTION="c", METRICS_URL="http://m",
                                                              METRICS_API_KEY_ENV="MEMENTO_TEST_STUDY_KEY",
                                                              WARMUP_MAX_RUNNING=2))
    assert asyncio.run(workers.warmup_step(pipe.valves, None, function_id="memento_query_polymer", db_engine=lambda: None)) == "busy"
    assert seen == [("http://m", "MEMENTO_TEST_STUDY_KEY")]


# ── the ingestion side ───────────────────────────────────────────────────────

class _RecOpenAI:
    made: list = []

    def __init__(self, base_url=None, api_key=None, **k):
        _RecOpenAI.made.append((base_url, api_key))
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    def _create(self, **k):
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="[]"))])


def test_captions_use_the_given_key(monkeypatch, tmp_path):
    import memento.ingestion.caption_image as ci
    from PIL import Image
    img = tmp_path / "f.png"
    Image.new("RGB", (8, 8)).save(img)
    _RecOpenAI.made = []
    monkeypatch.setattr(ci, "OpenAI", _RecOpenAI)
    ci.caption_image(img, base_url="http://llm.test/v1", model="m", api_key=STUDY_KEY)
    ci.caption_image(img, base_url="http://llm.test/v1", model="m")                 # no key given: as today
    assert _RecOpenAI.made == [("http://llm.test/v1", STUDY_KEY), ("http://llm.test/v1", DEFAULT_KEY)]


def test_cached_captions_forward_the_key(monkeypatch, tmp_path):
    import memento.ingestion.caption_image as ci
    seen = []
    monkeypatch.setattr(ci, "caption_image", lambda path, base_url=None, model=None, api_key=None:
                        seen.append(api_key) or "a caption")
    (tmp_path / "images").mkdir()
    assert ci.cached_caption(tmp_path, "images/a.jpg", base_url="u", model="m", api_key=STUDY_KEY) == "a caption"
    assert seen == [STUDY_KEY]


def test_gap_tags_use_the_given_key_and_never_fall_back_from_an_empty_one(monkeypatch):
    import memento.research_gaps.tagging as rg
    _RecOpenAI.made = []
    monkeypatch.setattr(rg, "OpenAI", _RecOpenAI)
    chunks = [{"text": "Further work is needed.", "section": "Conclusions"}]
    rg.tag_paper_chunks(chunks, base_url="http://llm.test/v1", model="m", api_key=STUDY_KEY, workers=1)
    rg.tag_paper_chunks(chunks, base_url="http://llm.test/v1", model="m", api_key="", workers=1)
    rg.tag_paper_chunks(chunks, base_url="http://llm.test/v1", model="m", workers=1)     # none given: as today
    assert [k for _u, k in _RecOpenAI.made] == [STUDY_KEY, "", DEFAULT_KEY]


def test_the_ingestion_pipe_sends_its_copys_key_to_captions_and_gap_tags():
    src = (ROOT / "memento/ingestion/upload/document.py").read_text()
    tree = ast.parse(src)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and getattr(n.func, "id", None) in ("cached_caption", "tag_paper_chunks")]
    assert {c.func.id for c in calls} == {"cached_caption", "tag_paper_chunks"}
    for c in calls:
        kw = {k.arg: ast.unparse(k.value) for k in c.keywords}
        assert kw.get("api_key") == "llm_key(self.valves)", ast.unparse(c)[:120]


def test_the_ingestion_key_valve_defaults_to_the_answer_model_key(monkeypatch):
    from test_two_libraries_worker import INGEST, load_copy
    ip = load_copy(INGEST, "memento_ingestion", monkeypatch)
    assert ip.Pipe.Valves().LLM_API_KEY_ENV == "REASONING_API_KEY"
    assert document.llm_key(ip.Pipe.Valves(LLM_API_KEY_ENV="MEMENTO_TEST_STUDY_KEY")) == STUDY_KEY


def test_document_chat_copies_the_key_setting(monkeypatch, tmp_path):
    import memento.document_chat.intake as ti
    assert "LLM_API_KEY_ENV" in ti.COPY_VALVES
    p = ti.make_ingester("3f2b8c1e-6a4d-4e0b-9c7a-1d2e3f4a5b6c", "u1", input_root=tmp_path / "in",
                         output_root=tmp_path / "out", collection="memento-polymer-rag",
                         valves={"LLM_API_KEY_ENV": "MEMENTO_TEST_STUDY_KEY", "LLM_BASE_URL": "http://llm.test/v1"})
    assert p.valves.LLM_API_KEY_ENV == "MEMENTO_TEST_STUDY_KEY"


# ── the scripts ──────────────────────────────────────────────────────────────

def test_the_gap_updater_uses_the_copys_key(monkeypatch):
    import qdrant_client
    import update_gap_ledger as ugl
    qp = _qp()
    monkeypatch.setattr(qdrant_client, "QdrantClient", lambda **k: None)
    V = qp.Pipe.Valves(DOMAIN_NAME="polymer-engineering", COLLECTION="memento-polymer-rag",
                       LLM_API_KEY_ENV="MEMENTO_TEST_STUDY_KEY", LLM_BASE_URL="http://llm.test/v1")
    ctx = ugl.make_ctx(None, V, 1)
    assert ctx.llm_client.api_key == STUDY_KEY and str(ctx.llm_client.base_url).startswith("http://llm.test")
    ctx = ugl.make_ctx(None, qp.Pipe.Valves(), 1)
    assert ctx.llm_client.api_key == DEFAULT_KEY


def test_the_gap_ledger_build_script_uses_the_copys_key():
    tree = ast.parse((ROOT / "scripts/build_gap_ledger.py").read_text())
    clients = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "OpenAI"]
    assert clients
    for c in clients:
        kw = {k.arg: ast.unparse(k.value) for k in c.keywords}
        assert kw.get("api_key") == "request.llm_key(V)", ast.unparse(c)[:120]
