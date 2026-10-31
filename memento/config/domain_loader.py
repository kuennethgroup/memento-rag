"""
Load a domain profile from config/domains/<name>.yaml and return a
DOMAIN-compatible dict that config.py can use as a drop-in replacement
for its inline DOMAIN block.

Usage (in config.py):
    from memento.config.domain_loader import load_domain
    DOMAIN = load_domain(os.getenv("DOMAIN_NAME", "polymer-engineering"))
"""

import os
from pathlib import Path

import yaml
from memento.config.paths import PROJECT_ROOT


_DOMAINS_DIR = PROJECT_ROOT / "config" / "domains"


def load_domain(name: str) -> dict:
    """Load a domain profile YAML and return a DOMAIN dict.

    Handles:
    - frozenset conversion for parsing.title_banner_tokens
    - Injects set_query_vocabulary (language-bound, not domain-bound)
    - Falls back to generic.yaml if the named profile doesn't exist
    """
    path = _DOMAINS_DIR / f"{name}.yaml"
    if not path.exists():
        fallback = _DOMAINS_DIR / "generic.yaml"
        if not fallback.exists():
            raise FileNotFoundError(
                f"Domain profile '{name}' not found at {path} "
                f"and no generic fallback at {fallback}"
            )
        path = fallback

    with open(path) as f:
        domain = yaml.safe_load(f)

    # Convert title_banner_tokens list → frozenset (config.py consumers expect it)
    parsing = domain.get("parsing", {})
    if "title_banner_tokens" in parsing:
        parsing["title_banner_tokens"] = frozenset(parsing["title_banner_tokens"])

    # collection_prefix can be overridden by env var
    domain["collection_prefix"] = os.getenv(
        "COLLECTION_PREFIX", domain.get("collection_prefix", "memento")
    )

    # entity_registry can be overridden by env var
    domain["entity_registry"] = os.getenv(
        "ENTITY_REGISTRY", domain.get("entity_registry", "generic")
    )

    # Embed/rerank instruction — derived from the domain's instruction field
    domain["rerank_instruction"] = domain.get("instruction", "")

    # set_query_vocabulary is language-bound, not domain-bound.
    # Every domain gets the same vocabulary; a non-English corpus would
    # override this at the profile level if needed.
    if "set_query_vocabulary" not in domain:
        domain["set_query_vocabulary"] = _SET_QUERY_VOCABULARY

    return domain


def default_collection(name: str | None = None) -> str:
    """A domain profile's primary collection, `<collection_prefix>-rag` (DOMAIN_NAME by default); "memento-rag" when no
    profile loads."""
    try:
        prefix = load_domain(name or os.getenv("DOMAIN_NAME", "polymer-engineering")).get("collection_prefix")
    except Exception:
        prefix = None
    return (prefix or "memento") + "-rag"


def default_library_collection() -> str:
    """The default library's collection: MEMENTO_DEFAULT_COLLECTION, else the configured domain's (config/.env is
    read first, without overriding the environment)."""
    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / "config" / ".env")
    except Exception:
        pass
    return os.getenv("MEMENTO_DEFAULT_COLLECTION", "").strip() or default_collection()


# ── Language-bound set-query vocabulary (shared across domains) ──────────────

_SET_QUERY_VOCABULARY = {
    "corpus_nouns": [
        "papers", "publications", "studies", "articles", "works",
        "documents", "manuscripts", "references", "datasets", "literature",
        "corpus",
        "papiere", "publikationen", "studien", "artikel", "arbeiten",
        "dokumente", "veroeffentlichungen", "veröffentlichungen", "literatur",
    ],
    "corpus_nouns_singular": [
        "paper", "publication", "study", "article", "work", "document",
        "manuscript", "dataset",
        "papier", "publikation", "studie", "artikel", "arbeit", "dokument",
    ],
    "list_cues": [
        "list", "enumerate", "catalogue", "catalog", "name", "show", "give",
        "find", "retrieve", "which", "what", "every", "all",
        "liste", "zeige", "zeig", "nenne", "welche", "was", "alle", "jede",
    ],
    "count_cues": [
        "how many", "number of", "count of", "total number", "tally",
        "wie viele", "anzahl der", "anzahl von",
    ],
    "universal_quantifiers": ["all", "every", "each", "alle", "jede", "jeder"],
    "distributive_preps": [
        "in", "of", "for", "from", "across", "within", "during", "per",
        "between", "among", "aus", "von", "für", "in", "je", "pro",
    ],
    "passage_cues": [
        "say", "says", "state", "report", "conclude", "explain", "describe",
        "why", "how does", "how did",
        "sagt", "berichtet", "erklärt", "beschreibt", "warum", "wieso",
    ],
    "negation_cues": [
        "but not", "but without", "without", "excluding", "except for",
        "except", "other than", "rather than", "instead of",
        "aber nicht", "ohne", "ausser", "außer", "ausgenommen",
    ],
    "conjunction_cues": [
        "both", "as well as", "together with", "in combination with",
        "and also", "plus", "intersection",
        "sowohl", "als auch", "zusammen mit", "in kombination mit",
    ],
}
