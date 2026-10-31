# Open (free-text) relations, as in LightRAG / HippoRAG 2.
import sys

sys.argv = [sys.argv[0], "--domain", "polymer-engineering"]
from memento.config.cli_domain import apply_domain_arg  # noqa: E402
apply_domain_arg()
from memento.graph.build.extract_entities import extract_entities  # noqa: E402
from memento.graph.build.llm_enrich import build_enrich_prompt  # noqa: E402


def test_prompt_asks_for_open_relations_without_a_type_list():
    p = build_enrich_prompt("overview", [(0, "text")], want_context=False, want_entities=True)
    assert '"description"' in p and "Allowed relation types" not in p and "rel_type" not in p


def test_fusion_accepts_open_relation_descriptions():
    r = extract_entities("The epoxy resin was toughened with core-shell rubber particles.", use_llm=True,
                         precomputed={"entities": [{"name": "epoxy resin", "type": "Material"},
                                                   {"name": "core-shell rubber particles", "type": "Material"}],
                                      "relations": [{"source": "epoxy resin", "target": "core-shell rubber particles",
                                                     "description": "  toughened   with "}]})
    assert [(x.source, x.target, x.rel_type) for x in r.relations] == [("epoxy", "core_shell_rubber_particles", "toughened with")]


def test_fusion_drops_relations_without_description_or_with_value_endpoints():
    r = extract_entities("Tg of the epoxy was 180 C.", use_llm=True,
                         precomputed={"entities": [{"name": "epoxy", "type": "Material"}, {"name": "Tg", "type": "Property"}],
                                      "relations": [{"source": "epoxy", "target": "Tg", "description": ""},
                                                    {"source": "epoxy", "target": "180 °C", "description": "has Tg of"}]})
    assert r.relations == []
