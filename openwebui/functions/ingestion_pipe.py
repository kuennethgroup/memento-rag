"""
title: Memento Ingestion
author: rangan
version: 1.8.7
type: pipe
requirements: qdrant-client,pypdf
"""

import contextlib
import os
import sys
from pathlib import Path
from typing import AsyncGenerator

from pydantic import BaseModel

_MEMENTO      = Path(os.getenv("MEMENTO_DIR", ""))
if str(_MEMENTO) and str(_MEMENTO) not in sys.path:      # the packages (memento/) come from MEMENTO_DIR
    sys.path.insert(0, str(_MEMENTO))

from memento.config.paths import roots  # noqa: E402
from memento.ingestion.upload import messages, workflow  # noqa: E402
from memento.query import request  # noqa: E402


def _own_id(module_name: str) -> str:
    """This copy's Open WebUI function id: Open WebUI loads a function as module `function_<id>`, so two copies of this
    file can run side by side; `memento_ingestion` elsewhere (tests, scripts)."""
    return module_name.removeprefix("function_") if module_name.startswith("function_") else "memento_ingestion"


_OWN_ID = _own_id(__name__)


def _db_engine():
    from open_webui.internal.db import engine
    return engine


def _relink(text, valves):
    """Links to the default models in `text` pointing at this copy's (memento/library/copies.py); after the German twin.
    Links to the default models in `text` pointing at this copy's (memento/library/copies.py); after the German twin."""
    if not isinstance(text, str) or "/?model=" not in text:
        return text
    from memento.library.copies import link_ids, relink
    return relink(text, link_ids(valves))


# Corpus folders follow one rule (memento/config/paths.py): INPUT_ROOT/<corpus_dir> and OUTPUT_ROOT/<corpus_dir>,
# with DOMAIN_NAME / INPUT_ROOT / OUTPUT_ROOT as Valves (defaults: config/.env). See document.Ingester.input_dir().
INPUT_ROOT, OUTPUT_ROOT = (str(p) for p in roots())


def _default_mineru_backend() -> str:
    """memento.config.settings.MINERU_BACKEND (env-driven, default "vlm-engine"), loaded through importlib: Open WebUI rewrites a
    "from" + " config" import (see document.Ingester.out_subdir())."""
    try:
        import importlib
        return importlib.import_module("memento.config.settings").MINERU_BACKEND
    except Exception:
        return "vlm-engine"  # matches memento.config.settings's own fallback 


_DEFAULT_BACKEND = _default_mineru_backend()
_DEFAULT_COLLECTION = request.default_collection()


class Pipe:
    class Valves(BaseModel):
        DOMAIN_NAME:        str  = os.getenv("DOMAIN_NAME",     "polymer-engineering")
        INPUT_ROOT:         str  = INPUT_ROOT     # documents: INPUT_ROOT/<corpus_dir of DOMAIN_NAME>
        OUTPUT_ROOT:        str  = OUTPUT_ROOT    # parsed output: OUTPUT_ROOT/<corpus_dir of DOMAIN_NAME>
        MINERU_URL:         str  = os.getenv("MINERU_API_URL",  "http://localhost:8010")
        EMBEDDER_URL:       str  = os.getenv("EMBEDDER_URL",    "http://localhost:8021")
        QDRANT_URL:         str  = os.getenv("QDRANT_URL",      "http://localhost:6333")
        COLLECTION:         str  = _DEFAULT_COLLECTION   # <collection_prefix of DOMAIN_NAME>-rag
        BACKEND:            str  = _DEFAULT_BACKEND  # sourced from memento.config.settings.MINERU_BACKEND
        PARSE_WORKERS:      int  = int(os.getenv("PARSE_WORKERS", "8"))
        EMBED_WORKERS:      int  = int(os.getenv("EMBED_WORKERS", "8"))
        EMBED_BATCH_SIZE:   int  = int(os.getenv("EMBED_BATCH_SIZE", "2"))
        SKIP_PARSE:          bool = False
        SKIP_EMBED:          bool = False
        FORCE_REEMBED:       bool = False
        FORCE_REPARSE:       bool = False
        # the answer model of config/.env (REASONING_URL without /v1, REASONING_MODEL_NAME), for figure descriptions
        # and gap tags; the environment variable holding its key
        LLM_BASE_URL:        str  = (os.getenv("REASONING_URL", "http://localhost:8023").rstrip("/")
                                     .removesuffix("/v1") + "/v1")
        LLM_API_KEY_ENV:     str  = "REASONING_API_KEY"
        # this copy's models in reply links (defaults: the single-copy ids)
        QUERY_MODEL_ID:      str  = "memento-query"
        UPLOAD_MODEL_ID:     str  = "memento-ingestion-prod"
        DOC_CHAT_MODEL_ID:   str  = "memento-document-chat"
        LLM_MODEL:           str  = os.getenv("REASONING_MODEL_NAME", "qwen-flash-next")
        NO_ENRICH:           bool = False  # mirrors run_pipeline.py's --no-enrich
        GRAPH_BUILD:         bool = os.getenv("GRAPH_BUILD", "false").strip().lower() in ("1", "true", "yes", "on")
        RESEARCH_GAP_TAG:    bool = os.getenv("RESEARCH_GAP_TAG", "false").strip().lower() in ("1", "true", "yes", "on")
        # who may ingest besides admins: members of these Open WebUI groups, comma-separated; empty = admins only
        CONTRIBUTOR_GROUP:   str  = "contributors"
        CLEANUP_CHAT_UPLOAD: bool = False  # off by default: uploads stay visible in the chat. When on, the upload is
                                           # removed from this Open WebUI chat only (see workflow.cleanup_webui_files);
                                           # Memento's own copies (data/input/, data/output/, Qdrant vectors,
                                           # manifest.json) are untouched either way

    def __init__(self):
        self.type   = "pipe"
        self.name   = "Memento Ingestion"
        self.valves = self.Valves()

    # ── main handler: memento/ingestion/upload/workflow.py (async, the work on a background thread) ──
    async def pipe(self, body: dict, __user__: dict, __files__: list = [],
                   __event_emitter__=None, __task__=None, __chat_id__=None, __metadata__=None) -> AsyncGenerator:
        """The reply, in German when the typed message is German; tasks (titles) unchanged."""
        de = not __task__ and messages.upload_lang(body) == "de"
        async with contextlib.aclosing(workflow.run(body, __user__, __files__, __event_emitter__, __task__, __chat_id__,
                                                    __metadata__, function_id=_OWN_ID, valves=self.valves,
                                                    engine=_db_engine)) as reply:
            async for item in reply:
                # German first (found by the English text), then the links to this copy's models
                yield _relink(messages.in_german(item) if de and isinstance(item, str) else item, self.valves)
