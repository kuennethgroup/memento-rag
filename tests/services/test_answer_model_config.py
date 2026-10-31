# One answer model, named once in config/.env: REASONING_URL (the server, without /v1), REASONING_MODEL_NAME and
# REASONING_API_KEY. Library Chat and Document Chat (Memento Query), Library Upload (Memento Ingestion), the setup
# script and command-line uploads all take it from there.
import json
import subprocess
import sys

from memento.config.paths import PROJECT_ROOT

ENV = {"REASONING_URL": "http://llm.test:9000", "REASONING_MODEL_NAME": "answer-model", "DOMAIN_NAME": "generic"}
OLD = ("FLASH_NEXT_BASE_URL", "FLASH_NEXT_MODEL", "FLASH_NEXT_KEY_ENV", "QWEN_FLASH_NEXT_API_KEY", "MY_LLM_KEY")


def _run(code: str, **env) -> dict:
    import os
    base = {k: v for k, v in os.environ.items() if k not in ENV and k not in OLD}
    out = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=180,
                         env={**base, "PYTHONPATH": str(PROJECT_ROOT), **ENV, **env})
    assert out.returncode == 0, out.stderr[-2500:]
    return json.loads(out.stdout.strip().splitlines()[-1])


PIPES = ("import json, sys; sys.path.insert(0, 'openwebui/functions'); import query_pipe, ingestion_pipe; "
         "f = lambda v: [v.LLM_BASE_URL, v.LLM_MODEL, v.LLM_API_KEY_ENV]; "
         "print(json.dumps([f(query_pipe.Pipe.Valves()), f(ingestion_pipe.Pipe.Valves())]))")


def test_both_pipes_default_to_the_answer_model():
    q, i = _run(PIPES)
    assert q == i == ["http://llm.test:9000/v1", "answer-model", "REASONING_API_KEY"]


def test_a_url_given_with_v1_is_not_doubled():
    q, i = _run(PIPES, REASONING_URL="http://llm.test:9000/v1/")
    assert q[0] == i[0] == "http://llm.test:9000/v1"


def test_the_setup_writes_the_answer_model_into_both_pipes():
    code = ("import importlib.util as u, json; s = u.spec_from_file_location('so', 'scripts/setup_openwebui.py'); "
            "m = u.module_from_spec(s); s.loader.exec_module(m); "
            "q = m.trial_valves('memento_query', 'generic', {}); i = m.trial_valves('memento_ingestion', 'generic', {}); "
            "k = ('LLM_BASE_URL', 'LLM_MODEL', 'LLM_API_KEY_ENV'); "
            "print(json.dumps([[q.get(x) for x in k], [i.get(x) for x in k], q.get('METRICS_URL')]))")
    q, i, metrics = _run(code)
    assert q == i == ["http://llm.test:9000/v1", "answer-model", "REASONING_API_KEY"]
    assert metrics == "http://llm.test:9000/metrics"


def test_command_line_gap_tagging_uses_the_answer_model(monkeypatch):
    from memento.research_gaps import tagging
    seen = {}

    class FakeOpenAI:
        def __init__(self, base_url, api_key, **kw):
            seen.update(base_url=base_url, api_key=api_key)

    monkeypatch.setattr(tagging, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(tagging, "classify_batch", lambda client, model, batch, **kw: seen.update(model=model) or {})
    monkeypatch.setenv("REASONING_MODEL_NAME", "answer-model")
    monkeypatch.setenv("REASONING_API_KEY", "answer-key")
    monkeypatch.setenv("CLOUD_REASONING_MODEL", "some-cloud-model")
    tagging.tag_paper_chunks([{"text": "The limitation is that no test above 900 C was run."}], "http://llm.test:9000")
    assert seen == {"base_url": "http://llm.test:9000/v1", "api_key": "answer-key", "model": "answer-model"}


def test_the_docs_name_one_answer_model():
    example = (PROJECT_ROOT / "config/.env.example").read_text()
    readme = (PROJECT_ROOT / "README.md").read_text()
    ow = (PROJECT_ROOT / "openwebui/README.md").read_text()
    for var in ("REASONING_URL", "REASONING_MODEL_NAME", "REASONING_API_KEY"):
        assert f"\n{var}=" in example, var
        assert f"`{var}`" in readme, var
    assert [v for v in OLD if v in example + readme + ow] == []
