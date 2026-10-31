"""
title: Memento Query
author: rangan
version: 1.19.0
type: pipe
requirements: qdrant-client
"""

import contextlib
import os
import sys
from pathlib import Path

from pydantic import BaseModel

_MEMENTO  = Path(os.getenv("MEMENTO_DIR", ""))
if str(_MEMENTO) and str(_MEMENTO) not in sys.path:      # the packages (memento/) come from MEMENTO_DIR
    sys.path.insert(0, str(_MEMENTO))
from memento.query import generation, request, service, workers  # noqa: E402
from memento.document_chat import chat  # noqa: E402


def _own_id(module_name: str) -> str:
    """This copy's Open WebUI function id: Open WebUI loads a function as module `function_<id>`, so two copies of this
    file can run side by side; `memento_query` elsewhere (tests, scripts)."""
    return module_name.removeprefix("function_") if module_name.startswith("function_") else "memento_query"


_OWN_ID = _own_id(__name__)


def _db_engine():
    from open_webui.internal.db import engine
    return engine


class Pipe:
    class Valves(BaseModel):
        DOMAIN_NAME:    str   = os.getenv("DOMAIN_NAME",     "polymer-engineering")
        EMBEDDER_URL:   str   = os.getenv("EMBEDDER_URL",    "http://localhost:8021")
        EMBEDDER_MODEL: str   = os.getenv("EMBEDDER_MODEL",  "Qwen3-VL-Embedding-8B")
        QDRANT_URL:     str   = os.getenv("QDRANT_URL",      "http://localhost:6333")
        RERANKER_URL:   str   = os.getenv("RERANKER_URL",    "http://localhost:8022")
        FILES_URL:      str   = os.getenv("FILES_URL",       "http://localhost:8024")
        COLLECTION:     str   = request.default_collection()   # <collection_prefix of DOMAIN_NAME>-rag
        # the answer model of config/.env (REASONING_URL without /v1, REASONING_MODEL_NAME) and the environment
        # variable holding its key
        LLM_BASE_URL:   str   = (os.getenv("REASONING_URL", "http://localhost:8023").rstrip("/")
                                 .removesuffix("/v1") + "/v1")
        LLM_MODEL:      str   = os.getenv("REASONING_MODEL_NAME", "qwen-flash-next")
        LLM_API_KEY_ENV: str  = "REASONING_API_KEY"
        # this copy's models users can pick: a file in config/
        VARIANTS_FILE:  str   = "query_models.yaml"
        # this copy's models in reply links, and the ingestion copy whose settings Document Chat reads (defaults: the
        # single-copy ids)
        QUERY_MODEL_ID: str   = "memento-query"
        UPLOAD_MODEL_ID: str  = "memento-ingestion-prod"
        DOC_CHAT_MODEL_ID: str = "memento-document-chat"
        INGESTION_FUNCTION_ID: str = "memento_ingestion"
        # the upload warm-up's model server and slot rule, per copy (defaults from .env)
        METRICS_URL:    str   = os.getenv("WARMUP_METRICS_URL", "http://127.0.0.1:8090/metrics")
        METRICS_API_KEY_ENV: str = ""       # the variable of a key the metrics endpoint needs; empty: none sent
        WARMUP_MAX_RUNNING: int = int(os.getenv("WARMUP_MAX_RUNNING", "2"))
        RETRIEVE_K:     int   = 80
        RERANK_TOP:     int   = 20
        RERANK_MIN_SCORE: float = 0.03
        IMAGE_RERANK_RATIO: float = 0.2
        # soft per-document cap on the RERANK_TOP passages: a document above PER_PAPER_CAP gives up its weakest passage
        # to another document's best while that one scores at least PER_PAPER_SWAP_RATIO × the replaced one; 0 = off
        PER_PAPER_CAP:  int   = int(os.getenv("PER_PAPER_CAP", "6"))
        PER_PAPER_SWAP_RATIO: float = float(os.getenv("PER_PAPER_SWAP_RATIO", "0.5"))
        # documents not used / closest passages under an answer: a per-document search while the answer is written; the
        # reranker reads the best RELATED_RERANK_K of up to RELATED_GROUPS documents; a document is listed when its
        # passage scores ≥ RELATED_RATIO × the answer's best passage and ≥ RELATED_MIN_SCORE
        RELATED_DOCS:   bool  = os.getenv("RELATED_DOCS", "true").strip().lower() in ("1", "true", "yes", "on")
        RELATED_GROUPS: int   = int(os.getenv("RELATED_GROUPS", "200"))
        RELATED_RERANK_K: int = int(os.getenv("RELATED_RERANK_K", "30"))
        RELATED_RATIO:  float = float(os.getenv("RELATED_RATIO", "0.12"))
        RELATED_MIN_SCORE: float = float(os.getenv("RELATED_MIN_SCORE", "0.05"))
        RELATED_LOW_SCORE: float = float(os.getenv("RELATED_LOW_SCORE", "0.1"))   # best passage below: nothing found
        TEMPERATURE:    float = 0.15
        MAX_TOKENS:     int   = 16384
        ENABLE_THINKING: bool = True
        # With ENABLE_THINKING on: let the query router switch thinking off for simple questions (one stated fact, or a
        # follow-up that only reworks the previous answer). Off = always think.
        THINKING_AUTO:  bool  = os.getenv("THINKING_AUTO", "false").strip().lower() in ("1", "true", "yes", "on")
        # live answers think without Thorough check too? Off (generation.live_thinks); the warm-up keeps ENABLE_THINKING
        LIVE_THINKING:  bool  = os.getenv("LIVE_THINKING", "false").strip().lower() in ("1", "true", "yes", "on")
        # Cap on thinking tokens per answer (vLLM thinking_token_budget), so a model that keeps deliberating still
        # leaves room for the answer within MAX_TOKENS; 0 = no cap (an endpoint that rejects the field).
        THINKING_BUDGET: int = int(os.getenv("THINKING_BUDGET", "12288"))
        # research gaps on the local model: thinking capped lower (nearly the same gaps, much faster); "Thorough check"
        # on = the normal cap; 0 = the normal cap always. Cloud models keep THINKING_BUDGET.
        GAP_THINKING_BUDGET: int = int(os.getenv("GAP_THINKING_BUDGET", "2000"))
        # an answer with Thorough check on, and the check's revisions, on the local model: a lower cap kept the results
        # at half the time; 0 = THINKING_BUDGET
        THOROUGH_THINKING_BUDGET: int = int(os.getenv("THOROUGH_THINKING_BUDGET", "2000"))
        # research gaps think at all? On: without thinking the model did not tell books and reviews from studies; leaked
        # reasoning is removed from the display (research_gaps.passages.strip_leaked) and never stored
        # (cache.gap_storable)
        GAP_THINKING: bool = os.getenv("GAP_THINKING", "true").strip().lower() in ("1", "true", "yes")
        # list / count questions on the local model think? Off: the same documents without thinking, several times
        # faster (generation.list_thinks)
        LIST_THINKING: bool = os.getenv("LIST_THINKING", "false").strip().lower() in ("1", "true", "yes")
        # nothing relevant in the library: a short model reply naming genuinely related documents, else the fixed
        # NOTHING_IN_LIBRARY_MSG
        NOTHING_FOUND_REPLY: bool = True
        RERANK_ENABLED: bool  = True
        MAX_IMAGES_TO_LLM: int = int(os.getenv("MAX_IMAGES_TO_LLM", "6"))
        # figures shown under an answer, those it discusses first; 0 = all cited ones
        MAX_FIGURES_SHOWN: int = int(os.getenv("MAX_FIGURES_SHOWN", "6"))
        # list / count questions over the library: LIST_SCAN_PER_DOC passages of each of the LIST_SCAN_DOCS documents
        # closest to the topic (in the documents of the kind asked for) are reranked; a passage scoring at least
        # LIST_SCAN_RATIO × the best joins the context, for at most LIST_SCAN_MAX documents (the answer says when more
        # matched). 0.08 keeps the relevant documents while admitting few others; 0.25 lost many.
        LIST_SCAN: bool = os.getenv("LIST_SCAN", "true").strip().lower() in ("1", "true", "yes")
        LIST_SCAN_RATIO: float = float(os.getenv("LIST_SCAN_RATIO", "0.08"))
        LIST_SCAN_MAX: int = int(os.getenv("LIST_SCAN_MAX", "50"))
        LIST_SCAN_DOCS: int = int(os.getenv("LIST_SCAN_DOCS", "60"))
        LIST_SCAN_PER_DOC: int = int(os.getenv("LIST_SCAN_PER_DOC", "2"))
        # "summarize <document>": the whole document goes to the model when its text fits in about this many tokens
        # (window − MAX_TOKENS − prompt and history); longer ones get their budgeted main parts
        SUMMARY_MAX_TOKENS: int = int(os.getenv("SUMMARY_MAX_TOKENS", "100000"))
        # research gaps: at most this many tokens of the document's gap passages, most important types first
        GAP_BUDGET_TOKENS: int = int(os.getenv("GAP_BUDGET_TOKENS", "20000"))
        HYBRID_ENABLED: bool  = os.getenv("HYBRID_ENABLED", "false").strip().lower() in ("1", "true", "yes", "on")
        HYBRID_PREFETCH_K: int = int(os.getenv("HYBRID_PREFETCH_K", "80"))
        GRAPH_RETRIEVAL: bool = os.getenv("GRAPH_RETRIEVAL", "false").strip().lower() in ("1", "true", "yes", "on")
        VALIDATION_ENABLED: bool = os.getenv("VALIDATION", "false").strip().lower() in ("1", "true", "yes", "on")
        # RQ3 verification loop: "A" = verify → revise failing statements, "C" = verify → remove them. Only used when
        # VALIDATION_ENABLED.
        VALIDATION_STRATEGY: str = os.getenv("VALIDATION_STRATEGY", "A")
        VALIDATION_MAX_ROUNDS: int = int(os.getenv("VALIDATION_MAX_ROUNDS", "2"))
        # verifier endpoint/model; empty = the answer model's own endpoint and model (never a cloud default)
        VERIFIER_URL:   str   = os.getenv("VERIFIER_URL", "")
        VERIFIER_MODEL: str   = os.getenv("VERIFIER_MODEL", "")
        # time limits: writing one answer (thinking + text, any model), and Thorough check on top; past them the model
        # is stopped and the reader is told to try again (answer) or gets the unchecked answer (check)
        ANSWER_TIMEOUT: float = float(os.getenv("ANSWER_TIMEOUT", "480"))
        # a finished answer (stored, or held for a check) typed out like a generated one: characters per second (0 = at
        # once; 300 is about the local model's speed) and an optional cap in seconds (0 = none)
        REPLAY_CHARS_PER_S: float = float(os.getenv("REPLAY_CHARS_PER_S", "300"))
        REPLAY_MAX_S: float = float(os.getenv("REPLAY_MAX_S", "0"))
        # only the first this many characters of a finished answer typed out, the rest at once; 0 = all
        REPLAY_TYPED_CHARS: int = int(os.getenv("REPLAY_TYPED_CHARS", "4000"))
        VERIFY_TIMEOUT: float = float(os.getenv("VERIFY_TIMEOUT", "480"))
        # answers within a time budget: seconds from the request to the answer; thinking stops this many seconds before
        # the end (then the answer without thinking); optional steps need SUGGEST_MIN_S left
        ANSWER_BUDGET_S: float = float(os.getenv("ANSWER_BUDGET_S", "120"))
        THOROUGH_BUDGET_S: float = float(os.getenv("THOROUGH_BUDGET_S", "270"))
        ANSWER_RESERVE_S: float = float(os.getenv("ANSWER_RESERVE_S", "45"))
        SUGGEST_MIN_S: float = float(os.getenv("SUGGEST_MIN_S", "20"))
        ENABLE_QUERY_REWRITE: bool = True
        HISTORY_TURNS:  int   = 3  # user/assistant pairs kept for rewrite + generation
        # open research questions report: questions shown in full
        GAP_REPORT_FULL: int = int(os.getenv("GAP_REPORT_FULL", "10"))
        # Memento Document Chat: files per chat (loader.js MAX_FILES, keep the two in step), pages per file (0: no
        # limit; a read still ends at MinerU's 30 minutes)
        DOC_CHAT_MAX_FILES: int = int(os.getenv("DOC_CHAT_MAX_FILES", "5"))
        DOC_CHAT_MAX_PAGES: int = int(os.getenv("DOC_CHAT_MAX_PAGES", "0"))

    def pipes(self) -> list[dict]:
        """One model per variant (OpenWebUI manifold: models memento_query.<id>), and Memento Document Chat;
        workspace models wrap them."""
        return [{"id": v.id, "name": v.name} for v in generation.copy_variants(self.valves)] + [
            {"id": chat.DOC_CHAT_ID, "name": "Document Chat"}]

    def __init__(self):
        self.type   = "pipe"
        self.name   = "Memento Query"
        self.valves = self.Valves()
        self.function_id = _OWN_ID              # this copy's id (Open WebUI's module name for the function)
        self.engine = _db_engine                # Open WebUI's database (a getter)

    async def pipe(self, body: dict, __user__: dict, __event_emitter__=None, __files__: list = None,
                   __metadata__: dict = None, __task__=None):
        """The message answered by memento/query/service.py, on this copy's id, a copy of its Valves and its
        database; closing the stream closes the service's at once."""
        async with contextlib.aclosing(service.handle(
                body, __user__, __event_emitter__, __files__, __metadata__, __task__,
                function_id=self.function_id, valves=self.valves.model_copy(), engine=self.engine)) as stream:
            async for chunk in stream:
                yield chunk


def _should_start_worker(module_name: str) -> bool:
    """Only inside OpenWebUI, which loads this file as `function_<id>`; never in tests or scripts."""
    return module_name.startswith("function_")


if _should_start_worker(__name__):
    workers.start(_OWN_ID, Pipe.Valves, _db_engine)
