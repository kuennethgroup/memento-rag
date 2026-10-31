#!/usr/bin/env python3
"""
memento/graph/registries/entity_registry_generic.py — empty registry for domains without a
curated vocabulary. Dictionary matching finds nothing; LLM extraction still
runs against the default entity types and this small relation vocabulary.
"""
from memento.graph.registry_base import DEFAULT_ENTITY_TYPES, EntityDef, build_registry, normalize  # noqa: F401

REGISTRY: tuple[EntityDef, ...] = ()
ENTITY_TYPES = DEFAULT_ENTITY_TYPES
RELATION_TYPES = {
    "produces":         "a process produces a material",
    "measured_by":      "a property is measured using a characterization technique",
    "characterized_by": "a material/process is characterized by a property",
    "compared_to":      "an entity's value/behaviour is compared against another",
    "consists_of":      "a material consists of another material or phase",
}

_registry = build_registry(REGISTRY, ENTITY_TYPES)
get_by_id = _registry.get_by_id
canonicalize = _registry.canonicalize
find_entities_in_text = _registry.find_entities_in_text
