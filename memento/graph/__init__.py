"""
graph/ — Knowledge-graph layer for Memento (thesis RQ2: dual-graph
retrieval augmentation).

Domain loader: re-exports the entity-registry interface from whichever
domain module `config.DOMAIN["entity_registry"]` names (default "ceramics"
-> memento.graph.registries.entity_registry_ceramics, see memento/graph/registry_base.py). This is the
*only* place a domain is named — memento/graph/build/extract_entities.py,
memento/graph/build/build_graph.py, and memento/graph/query/graph_retrieval.py import from here,
never from a specific `entity_registry_<domain>` module, so swapping domains
is a one-line config change (same pattern as config.DOMAIN's prompts/
instructions).

Public surface:
    from graph import (
        REGISTRY, ENTITY_TYPES, RELATION_TYPES,
        EntityDef, canonicalize, find_entities_in_text, get_by_id, normalize,
    )
    from memento.graph.store import GraphStore, Neo4jStore, GraphNode, GraphEdge, get_store
"""
import importlib

from memento.config.settings import DOMAIN  # noqa: E402

from memento.graph.registry_base import EntityDef  # noqa: E402  (domain-agnostic dataclass)

_registry_name = DOMAIN.get("entity_registry", "ceramics")
_registry_module = importlib.import_module(f"memento.graph.registries.entity_registry_{_registry_name}")

REGISTRY = _registry_module.REGISTRY
ENTITY_TYPES = _registry_module.ENTITY_TYPES
RELATION_TYPES = _registry_module.RELATION_TYPES
canonicalize = _registry_module.canonicalize
find_entities_in_text = _registry_module.find_entities_in_text
get_by_id = _registry_module.get_by_id
normalize = _registry_module.normalize
