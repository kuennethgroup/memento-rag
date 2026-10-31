# tests/graph/test_rq2_registry_loading.py
import importlib
from pathlib import Path

import pytest
import yaml
from memento.config.paths import PROJECT_ROOT

DOMAINS = sorted((PROJECT_ROOT / "config" / "domains").glob("*.yaml"))


@pytest.mark.parametrize("yml", DOMAINS, ids=lambda p: p.stem)
def test_every_domain_registry_imports(yml):
    name = yaml.safe_load(yml.read_text()).get("entity_registry", "ceramics")
    mod = importlib.import_module(f"memento.graph.registries.entity_registry_{name}")
    for attr in ("REGISTRY", "ENTITY_TYPES", "RELATION_TYPES", "get_by_id",
                 "canonicalize", "find_entities_in_text", "normalize"):
        assert hasattr(mod, attr), f"{mod.__name__} lacks {attr}"


def test_ceramics_domain_uses_ceramics_registry():
    y = yaml.safe_load((PROJECT_ROOT / "config/domains/ceramics.yaml").read_text())
    assert y["entity_registry"] == "ceramics"
