# The Memento Query variants.
import sys
from pathlib import Path

import pytest
from memento.config.paths import PROJECT_ROOT

ROOT = PROJECT_ROOT
sys.path.insert(0, str(ROOT))
from memento.query import models as qm  # noqa: E402


def test_shipped_config_loads_in_order():
    vs = qm.load_variants()
    assert [v.id for v in vs] == ["local", "qwen35", "qwen36"]
    assert vs[0].provider == "local" and all(v.is_cloud for v in vs[1:])
    assert all("—" not in v.full_description + v.name + v.label for v in vs)          # no em dashes
    assert vs[1].full_description.endswith("Runs on a cloud service.")   # no provider claimed
    # short descriptions: where it runs, data, speed; no test results
    assert all("in our tests" not in v.full_description for v in vs)
    # every start screen says what Memento does first, in one line; a cloud model adds one line: speed hint + host
    first = "Ask about the shared library in English or German; every answer cites its sources."
    assert vs[0].full_description == first and "GWDG" not in vs[0].full_description
    for v in vs[1:]:
        line1, line2 = v.full_description.split("\n")
        assert line1 == first and line2 == v.description + " Runs on a cloud service."
        assert len(line2.replace("[", "").split("](")[0]) + len("; does not store your data.") <= 82


def test_a_cloud_model_without_a_host_line_names_no_provider():
    v = qm.Variant(id="c", name="C", label="C", provider="cloud", model="m", description="Fast.",
                   base_url_env="X_URL", key_env="X_KEY")
    assert v.full_description.split("\n")[1] == "Fast. Runs on a cloud service."


def test_resolve_manifold_plain_and_workspace_ids():
    vs = qm.load_variants()
    assert qm.resolve("memento_query.qwen35", vs).id == "qwen35"
    assert qm.resolve("qwen36", vs).id == "qwen36"
    assert qm.resolve("memento-query-qwen36", vs).id == "qwen36"
    assert qm.resolve("memento-query-gemma", vs).id == "local"         # a disabled variant falls back to the local one
    assert qm.resolve("memento-query", vs).id == "local"


def test_resolve_unknown_or_disabled_is_local(tmp_path):
    vs = qm.load_variants()
    assert qm.resolve("memento_query.mistral", vs).id == "local"      # a removed variant falls back to the local one
    assert qm.resolve(None, vs).id == "local"
    cfg = tmp_path / "m.yaml"
    text = (ROOT / "config" / "query_models.yaml").read_text()
    cfg.write_text(text.replace('enabled: true\n  description: "Fast and well-rounded.', 'enabled: false\n  description: "Fast and well-rounded.'))
    vs2 = qm.load_variants(cfg)
    assert "qwen36" not in [v.id for v in vs2]
    assert qm.resolve("memento_query.qwen36", vs2).id == "local"


def test_workspace_ids():
    vs = {v.id: v for v in qm.load_variants()}
    assert qm.workspace_id(vs["local"]) == "memento-query"
    assert qm.workspace_id(vs["qwen35"]) == "memento-query-qwen35"


@pytest.mark.parametrize("bad, msg", [
    ("- {id: a, name: A, label: A, provider: cloud, model: m, description: d}", "base_url_env"),
    ("- {id: a, name: A, label: A, provider: moon, model: m, description: d}", "provider"),
    ("- {id: a, name: A, label: A, provider: cloud, model: m, description: d, base_url_env: X, key_env: Y}", "local"),
    ("- {id: a, name: A, label: A, provider: local, model: m, description: d}\n"
     "- {id: a, name: B, label: B, provider: local, model: m, description: d}", "duplicate"),
    ("- {id: a, name: A, label: A, provider: local, model: m, description: d, accent: blue}", "accent"),
    ("- {id: a, name: A, label: A, provider: local, model: m, description: d, accent_dark: '#12345'}", "accent_dark"),
])
def test_invalid_configs_are_rejected(tmp_path, bad, msg):
    cfg = tmp_path / "bad.yaml"
    cfg.write_text(bad)
    with pytest.raises(ValueError, match=msg):
        qm.load_variants(cfg)


# ── Model colours: Memento's green, cloud models in their provider's colour, set per model here ──
def test_shipped_cloud_models_use_gwdg_blue():
    # the provider's link blue for light mode (its logo blue is too light for text on white), the logo blue for dark
    vs = qm.load_variants()
    assert all(v.accent is None and v.accent_dark is None for v in vs if not v.is_cloud)
    assert all((v.accent, v.accent_dark) == ("#0369a1", "#00adee") for v in vs if v.is_cloud)


def test_model_colours_file(tmp_path, monkeypatch):
    monkeypatch.setenv("Y", "key")                      # a cloud model is offered only while its key is set
    cfg = tmp_path / "m.yaml"
    cfg.write_text("- {id: local, name: L, label: L, provider: local, model: m, description: d}\n"
                   "- {id: c, name: 'Memento Query (C, cloud)', label: C, provider: cloud, model: m, description: d,"
                   " base_url_env: X, key_env: Y, accent: '#AABBCC'}\n")
    out = tmp_path / "_runtime" / "query_models.json"
    qm.write_model_colours(out, qm.load_variants(cfg))
    import json
    data = json.loads(out.read_text())
    assert set(data) == {"models", "copies"}            # each copy's full entries
    assert {"models": data["models"]} == {"models": [
        {"id": "memento-query", "name": "L", "cloud": False, "accent": None, "accent_dark": None, "hint": None},
        {"id": "memento-query-c", "name": "Memento Query (C, cloud)", "cloud": True, "accent": "#aabbcc",
         "accent_dark": "#aabbcc",            # no dark colour given: the light one is used in both modes
         # the dropdown's info tooltip of a cloud model: its second description line, as plain text
         "hint": "d Runs on a cloud service."}]}


def test_model_names_leave_local_and_cloud_to_the_dropdown_label():
    # the dropdown tags each model "Local" / "Cloud" (loader.js), so the name does not repeat it; the answer footer's
    # label keeps it, since nothing else there says where the model runs
    vs = qm.load_variants()
    # the local model in both libraries
    assert [v.name for v in vs] == ["Library Chat (Qwen 3.8 Flash-Next)", "Library Chat (Qwen 3.5 397B)",
                                    "Library Chat (Qwen 3.6 35B)"]
    assert vs[0].label == "Qwen 3.8 Flash-Next (Local)" and all(v.label.endswith("(Cloud)") for v in vs[1:])
