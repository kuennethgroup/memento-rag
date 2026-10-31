"""
memento/config/settings.py — Central configuration for Memento

All Python files import from here. Loads config/.env automatically.
Override any value by setting environment variables before running.

Usage (with the repo root on sys.path):
    from memento.config.settings import DOMAIN, HYBRID_COLLECTION, OCR_COLLECTION

To serve a different scientific domain, set DOMAIN_NAME or add a profile under config/domains/.
"""
import os
from pathlib import Path

from memento.config.paths import PROJECT_ROOT

# ── Load .env ─────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / "config" / ".env")
except ImportError:
    pass   # python-dotenv not installed — fall back to env vars and defaults

# ── Data paths ────────────────────────────────────────────────────────────────
DATA_DIR   = Path(os.getenv("DATA_DIR",   str(PROJECT_ROOT / "data")))
# INPUT_DIR / OUTPUT_DIR (the active corpus's folders) are set below, after DOMAIN is loaded.

# ── Model paths ───────────────────────────────────────────────────────────────
EMBEDDER_MODEL_PATH  = os.getenv(
    "MODEL_PATH",
    str(Path.home() / "models" / "Qwen3-VL-Embedding-8B")
)
RERANKER_MODEL_PATH  = os.getenv(
    "RERANKER_MODEL_PATH",
    str(Path.home() / "models" / "Qwen3-VL-Reranker-8B")
)
REASONING_MODEL_PATH = os.getenv(
    "REASONING_MODEL_PATH",
    str(Path.home() / "models" / "reasoning-model")  # vision-capable model required
)

# ── Service URLs ──────────────────────────────────────────────────────────────
EMBEDDER_URL  = os.getenv("EMBEDDER_URL",  "http://localhost:8021")
RERANKER_URL   = os.getenv("RERANKER_URL",   "http://localhost:8022")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "Qwen3-VL-Reranker-8B")
EMBEDDER_MODEL = os.getenv("EMBEDDER_MODEL", "Qwen3-VL-Embedding-8B")
REASONING_URL = os.getenv("REASONING_URL", "http://localhost:8023")
QDRANT_URL    = os.getenv("QDRANT_URL",    "http://localhost:6333")
FILES_URL     = os.getenv("FILES_URL",     "http://localhost:8024")

# ── Neo4j (knowledge-graph store — dual-graph retrieval) ──────────────────
NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

# ── Graph retrieval) ────────────
# Adds graph-traversal candidates (via memento/graph/query/graph_retrieval.py) alongside
# the dense vector search results before reranking. Off by default so existing
# behavior (vector-only) is unchanged unless explicitly enabled.
GRAPH_RETRIEVAL_ENABLED = os.getenv("GRAPH_RETRIEVAL", "false").strip().lower() in ("1", "true", "yes", "on")
GRAPH_MAX_HOPS          = int(os.getenv("GRAPH_MAX_HOPS", "2"))
GRAPH_LIMIT             = int(os.getenv("GRAPH_LIMIT", "20"))
GRAPH_QUERY_USE_LLM     = os.getenv("GRAPH_QUERY_USE_LLM", "false").strip().lower() in ("1", "true", "yes", "on")

# HippoRAG2-inspired traversal/seeding upgrades (all default to the original
# behavior — "bfs" traversal, "dict" registry-match seeding:
#   GRAPH_TRAVERSAL: "bfs" = subgraph_chunks unweighted union (RAG-Anything
#     style); "ppr" = Personalized PageRank over Entity edges, chunks ranked
#     by the entity mass they MENTION (memento/graph/ppr.py).
#   GRAPH_SEED: "dict" = registry dictionary match; "dense" = embedding
#     similarity against the memento-entities collection (built by
#     memento/graph/build/embed_entities.py); "both" = union, dense scores as weights.
GRAPH_TRAVERSAL         = os.getenv("GRAPH_TRAVERSAL", "bfs").strip().lower()
GRAPH_SEED_MODE         = os.getenv("GRAPH_SEED", "dict").strip().lower()
GRAPH_PPR_DAMPING       = float(os.getenv("GRAPH_PPR_DAMPING", "0.5"))   # HippoRAG's default
GRAPH_PPR_TOP_ENTITIES  = int(os.getenv("GRAPH_PPR_TOP_ENTITIES", "100"))
GRAPH_SEED_TOP_K        = int(os.getenv("GRAPH_SEED_TOP_K", "5"))
GRAPH_SEED_MIN_SCORE    = float(os.getenv("GRAPH_SEED_MIN_SCORE", "0.5"))
# Knowledge-graph ablation stream selector, the graph factor of RQ2's
# structural x entity x rerank factorial:
#   "structural" = Layer A: REFERENCES + CITES_PAPER only, no entity layer
#   "entity"     = Layer B: entity seeding + traversal (+ entity-filtered
#                  cross-paper hop) only, no structural streams
#   "all"        = A + B, the full dual graph
# Layer B must be isolatable or neither main effect nor the A x B interaction
# is identifiable.
GRAPH_STREAMS           = os.getenv("GRAPH_STREAMS", "all").strip().lower()
# Raise instead of degrading to an empty candidate list. Off in production
# (a graph outage should not break retrieval); required for evaluation, where
# a silent empty result is indistinguishable from a genuine null.
GRAPH_STRICT            = os.getenv("GRAPH_STRICT", "false").strip().lower() in ("1", "true", "yes", "on")
# Hub cap for entity traversal: entities MENTIONed by more than this many
# chunks are skipped (unset = no cap; tuned in the RQ2 pilot).
GRAPH_MAX_DF: int | None = int(os.getenv("GRAPH_MAX_DF")) if os.getenv("GRAPH_MAX_DF") else None
# Multi-hop bridge stream: seed traversal from the entities of the top-K dense
# chunks (the bridge term is absent from the question by construction).
GRAPH_FIRST_HOP   = os.getenv("GRAPH_FIRST_HOP", "false").strip().lower() in ("1", "true", "yes", "on")
GRAPH_FIRST_HOP_K = int(os.getenv("GRAPH_FIRST_HOP_K", "10"))
# Cached entity-graph CSR matrices for PPR (memento/graph/ppr_cache.py), keyed by the graph's build_id
GRAPH_PPR_CACHE_DIR = os.getenv("GRAPH_PPR_CACHE_DIR",
                                str(PROJECT_ROOT / "data" / "rq2" / "ppr_cache"))
# memento/graph/query/predicates.py: LLM fallback for structural set-query predicates (cites_paper, authored_by, …) that
# neither the mentions/temporal path nor the deterministic triggers (memento/graph/query/predicates.py) reach. Off by default:
# it only fires after a query failed both the trigger check and entity seeding; off, that case falls back to ranked
# retrieval. Switch on only for a measured run of the citation and authorship strata.
PREDICATE_LLM_ENABLED   = os.getenv("PREDICATE_LLM", "false").strip().lower() in ("1", "true", "yes", "on")
# One instruction string for BOTH entity-name embedding (ingestion) and the
# query-side seed lookup — single-sourced here on purpose (see the
# embedding-instruction-unification lesson: asymmetric instructions cost
# real retrieval quality).


ENTITY_EMBED_INSTRUCTION = os.getenv(
    "ENTITY_EMBED_INSTRUCTION",
    "Represent this scientific term for semantic matching with related terms and queries.")

# ── Hybrid dense+BM25 search (query-time fusion via Qdrant RRF) ───────────────
# Off by default — dense-only is unchanged unless explicitly enabled. Hybrid
# tracks dense closely on paraphrased queries and rescues exact-term queries
# (numbers, formulas, named techniques) that dense alone blurs.
HYBRID_ENABLED      = os.getenv("HYBRID_ENABLED", "false").strip().lower() in ("1", "true", "yes", "on")
HYBRID_PREFETCH_K   = int(os.getenv("HYBRID_PREFETCH_K", "80"))

# ── Multimodal generation (image/chart pixels sent to the reasoning model) ────
# image/chart hits are sent as images, at most MAX_IMAGES_TO_LLM per turn (cost, latency)
MAX_IMAGES_TO_LLM  = int(os.getenv("MAX_IMAGES_TO_LLM", "6"))

# ── Contextual Retrieval (situating-context enrichment for embedding) ─────────
# Anthropic's Contextual Retrieval: before embedding a chunk, prepend a short LLM-generated sentence that situates it
# in its source document. This changes the embedding input only; the displayed and cited `payload['text']` stays the
# original chunk. Off by default.
CONTEXTUALIZE_ENABLED = os.getenv("CONTEXTUALIZE", "false").strip().lower() in ("1", "true", "yes", "on")
CONTEXT_MODEL         = os.getenv("CONTEXT_MODEL", "qwen3.6-35b-a3b")
CONTEXT_GRANULARITY   = os.getenv("CONTEXT_GRANULARITY", "per_doc").strip().lower()  # per_doc | per_chunk
CONTEXT_BATCH_SIZE    = int(os.getenv("CONTEXT_BATCH_SIZE", "20"))   # chunks per LLM call in per_doc mode
CONTEXT_WORKERS       = int(os.getenv("CONTEXT_WORKERS", "4"))       # concurrent LLM calls
CONTEXT_BM25          = os.getenv("CONTEXT_BM25", "false").strip().lower() in ("1", "true", "yes", "on")
CONTEXT_MAX_OVERVIEW_CHARS = int(os.getenv("CONTEXT_MAX_OVERVIEW_CHARS", "4000"))

# ── Combined ingestion-time LLM enrichment (context + graph extraction) ──────
# When both Contextual Retrieval and the graph build are enabled, one batched LLM call per batch of chunks produces the
# situating context and the entity/relation extraction together (memento/graph/build/llm_enrich.py), instead of a
# context pass plus one extraction call per prose chunk. GRAPH_ENRICH_COMBINED=false restores the separate passes.
GRAPH_ENRICH_COMBINED = os.getenv("GRAPH_ENRICH_COMBINED", "true").strip().lower() in ("1", "true", "yes", "on")
ENRICH_MODEL          = os.getenv("CLOUD_REASONING_MODEL", "qwen3.8-27b")
EXTRACT_WORKERS       = int(os.getenv("EXTRACT_WORKERS", "1"))

# ── Vision-conditioned extraction for image/chart chunks ─────────────────────
# Off by default. 
GRAPH_VISION_EXTRACTION = os.getenv("GRAPH_VISION_EXTRACTION", "false").strip().lower() in ("1", "true", "yes", "on")

# ── Graph build (materialize the KG during ingestion) ─────────────────────────
# Off by default — ingestion throughput/cost is unchanged unless explicitly
# enabled. When on, memento/graph/build/build_graph.py's build_paper() runs (with full
# hybrid entity/relation extraction, including the per-chunk LLM pass) after
# each paper is embedded, keeping Neo4j in sync incrementally. The graph can
# also always be (re)built in bulk via `memento/graph/build/build_graph.py --all`.
GRAPH_BUILD_ENABLED = os.getenv("GRAPH_BUILD", "false").strip().lower() in ("1", "true", "yes", "on")

# Gap tagging: when enabled, memento/research_gaps/tagging.py classifies each text chunk
# as gap-relevant or not (limitation, future_work, untested_condition,
# uncertainty) and writes the result into the Qdrant payload. Query-time gap
# retrieval reads these tags via a metadata filter.
GAP_TAG_ENABLED = os.getenv("RESEARCH_GAP_TAG", "false").strip().lower() in ("1", "true", "yes", "on")
GAP_TAG_WORKERS = int(os.getenv("GAP_TAG_WORKERS", "4"))             # concurrent LLM calls for gap tagging (1 = sequential)

# ── Post-generation answer validation (RQ3) ──
# When enabled, every answer passes through agentic_validation's checks (citation contract, source anchoring,
# plausibility) after generation, and the strategy below attempts to fix failing statements. Independent of the graph
# feature.
VALIDATION_ENABLED = os.getenv("VALIDATION", "false").strip().lower() in ("1", "true", "yes", "on")
# RQ3 strategy (memento/agentic_validation/strategies.py): "A" verify → revise, "C" verify → remove. Verifier empty = the answer
# model itself.
VALIDATION_STRATEGY   = os.getenv("VALIDATION_STRATEGY", "A").strip().upper()
VALIDATION_MAX_ROUNDS = int(os.getenv("VALIDATION_MAX_ROUNDS", "2"))
VERIFIER_URL          = os.getenv("VERIFIER_URL", "")
VERIFIER_MODEL        = os.getenv("VERIFIER_MODEL", "")

# ── Domain configuration ──────────────────────────────────────────────────────
# DOMAIN (loaded below) holds every domain-specific instruction and prompt of the embedder, reranker and answer model.
# The model servers receive the instruction with every request, so a profile is the one place to edit.

# Shared instruction for the embedder at index time (every text, image and table chunk) and query time, and for the
# reranker (rerank_instruction is a separate key only because it is delivered differently). It is baked into every
# document embedding, so changing it means re-indexing. One short instruction for the query and every modality, as the
# Qwen3-VL-Embedding/Reranker report (arXiv:2601.04720) does, chosen by an instruction ablation on the pooled ground
# truth. English instructions work for German documents.
_EMBED_RETRIEVE_RERANK_INSTRUCTION = os.getenv("EMBED_INSTRUCTION", "")

# ── Domain profile ───────────────────────────────────────────────────────────
# Loaded from config/domains/<DOMAIN_NAME>.yaml (memento/config/domain_loader.py). For a different scientific domain,
# set DOMAIN_NAME in config/.env or add a profile under config/domains/.
from memento.config.domain_loader import load_domain   # noqa: E402

DOMAIN = load_domain(os.getenv("DOMAIN_NAME", "polymer-engineering"))

# Each corpus has its own Neo4j instance (Community edition = one database):
# the domain's neo4j_uri wins over the .env default so graphs never mix.
if DOMAIN.get("neo4j_uri"):
    NEO4J_URI = DOMAIN["neo4j_uri"]
# RQ2 G2-lite rungs live in their own Neo4j instances (docker-compose *-no-llm*); a run selects one with
# NEO4J_URI_OVERRIDE. Unset (default): no change.
if os.getenv("NEO4J_URI_OVERRIDE"):
    NEO4J_URI = os.environ["NEO4J_URI_OVERRIDE"]

# Override instruction only if explicitly set via env var (EMBED_INSTRUCTION);
# otherwise the domain YAML's instruction is authoritative.
if _EMBED_RETRIEVE_RERANK_INSTRUCTION:
    DOMAIN["instruction"] = _EMBED_RETRIEVE_RERANK_INSTRUCTION
    DOMAIN["rerank_instruction"] = _EMBED_RETRIEVE_RERANK_INSTRUCTION

# ── Corpus folders ──
# One rule (memento/config/paths.py): documents in INPUT_ROOT/<corpus_dir>, parsed output in OUTPUT_ROOT/<corpus_dir>.
# INPUT_ROOT / OUTPUT_ROOT come from config/.env (default <repo>/data/input and <repo>/data/output); corpus_dir comes
# from the domain YAML, so switching DOMAIN_NAME (or --domain) switches corpus, collection, graph and folders together.
# INPUT_DIR / OUTPUT_DIR in the environment override the exact folder for one-off scripts; they are not user settings.
from memento.config.paths import corpus_dirs as _corpus_dirs, roots as _roots   # noqa: E402

INPUT_ROOT, OUTPUT_ROOT = _roots()
_CORPUS_IN, _CORPUS_OUT = _corpus_dirs(DOMAIN, INPUT_ROOT, OUTPUT_ROOT)
INPUT_DIR  = Path(os.environ["INPUT_DIR"])  if os.getenv("INPUT_DIR")  else _CORPUS_IN
OUTPUT_DIR = Path(os.environ["OUTPUT_DIR"]) if os.getenv("OUTPUT_DIR") else _CORPUS_OUT
CORPUS_DIR = DOMAIN["corpus_dir"]

# ── Derived Qdrant collection names ───────────────────────────────────────────
_pfx = DOMAIN["collection_prefix"]
COLLECTION        = f"{_pfx}-rag"          # primary collection (dense + BM25 named vectors)
OCR_COLLECTION    = f"{_pfx}-rag-ocr"      # MinerU pipeline / OCR baseline
TEXT_COLLECTION   = f"{_pfx}-rag-text"     # local text-only embedding ablation (Qwen3-Embedding-8B)
COLQWEN_COLLECTION = f"{_pfx}-rag-colqwen" # UniDoc-Bench sub-study — visual-document retrieval (page images)
CONTEXT_COLLECTION = f"{_pfx}-rag-context" # Contextual Retrieval ablation — same VLM parse, CONTEXTUALIZE=true embed
NOENRICH_COLLECTION = f"{_pfx}-rag-noenrich"  # enrichment ablation — same VLM parse, --no-enrich (raw MinerU
                                              # content_list.json embedded as-is: no figure/table/equation context
                                              # grounding, no section assignment, no junk-fragment filtering)
NOENRICH_TEXT_COLLECTION = f"{_pfx}-rag-noenrich-text"  # NoEnrich re-embedded with text-only Qwen3-Embedding-8B
                                                        # (clean VL-vs-text embedder ablation without enrichment)
ENTITY_COLLECTION  = f"{_pfx}-entities"    # Entity-name embeddings for dense graph seeding (memento/graph/build/embed_entities.py)
# Not to be confused with HYBRID_COLLECTION below ("hybrid" = dense+BM25
# hybrid SEARCH, an unrelated meaning) — this is content parsed with
# MinerU's hybrid-engine BACKEND (PaddleOCR text + VLM figures/charts).
HYBRID_ENGINE_COLLECTION = f"{_pfx}-rag-hybrid"
HYBRID_COLLECTION = COLLECTION       # backwards-compat alias

# ── Visual-document retrieval (UniDoc-Bench sub-study) ──
# TomoroAI/tomoro-colqwen3-embed-4b, a late-interaction (MaxSim) multi-vector model that embeds whole page images (no
# text extraction). Runs in its own .venv-colvision/: its remote-code model classes need an older transformers than the
# main .venv.
COLQWEN_MODEL = os.getenv("COLQWEN_MODEL", "TomoroAI/tomoro-colqwen3-embed-4b")
COLQWEN_DIM   = int(os.getenv("COLQWEN_DIM", "320"))  # the model's output dimension
COLQWEN_DEVICE = os.getenv("COLQWEN_DEVICE", "cuda:3")

# ── Qdrant ────────────────────────────────────────────────────────────────────
EMBED_DIM  = int(os.getenv("EMBED_DIM", "4096"))

# ── Local text embedding (RQ1 ablation only) ──
# Qwen3-Embedding-8B, the text-only sibling of the multimodal Qwen3-VL-Embedding-8B (same size, same 4096 dimensions),
# so the ablation isolates the vision-language component. Served by vLLM on LOCAL_TEXT_EMBED_URL; client in
# evaluation/RQ1/collections/local_text_embed.py, with the same instruction (DOMAIN["instruction"]) for both models.
LOCAL_TEXT_EMBED_URL = os.getenv("LOCAL_TEXT_EMBED_URL", "http://localhost:8025")
LOCAL_TEXT_EMBED_DIM = int(os.getenv("LOCAL_TEXT_EMBED_DIM", "4096"))
TEXT_EMBEDDER_MODEL = os.getenv("TEXT_EMBEDDER_MODEL", "Qwen3-Embedding-8B")

# ── Local text reranking (RQ1 ablation only) ──
# Qwen3-Reranker-8B, the text-only sibling of Qwen3-VL-Reranker-8B (same size), isolating the vision-language component
# of reranking. The same vLLM rerank API as the VL reranker, so pointing RERANKER_URL at LOCAL_TEXT_RERANK_URL swaps
# them.
LOCAL_TEXT_RERANK_URL = os.getenv("LOCAL_TEXT_RERANK_URL", "http://localhost:8026")
TEXT_RERANKER_MODEL = os.getenv("TEXT_RERANKER_MODEL", "Qwen3-Reranker-8B")


def reranker_model_for(url: str) -> str:
    """Served model name for a reranker endpoint (the text reranker has its own)."""
    return TEXT_RERANKER_MODEL if url.rstrip("/") == LOCAL_TEXT_RERANK_URL.rstrip("/") else RERANKER_MODEL
# ── Ingestion concurrency (shared by the CLI pipeline and the OpenWebUI ───────
# ── ingestion pipe — also exposed there as overridable Valves) ────────────────
PARSE_WORKERS    = int(os.getenv("PARSE_WORKERS",    "8"))
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "2"))

# ── GPU assignments (informational — actual binding via CUDA_VISIBLE_DEVICES) ─
EMBEDDER_GPU  = int(os.getenv("EMBEDDER_GPU",  "1"))
RERANKER_GPU  = int(os.getenv("RERANKER_GPU",  "2"))
REASONING_GPU = int(os.getenv("REASONING_GPU", "3"))
MINERU_GPU    = int(os.getenv("MINERU_GPU",    "0"))

# ── MinerU ────────────────────────────────────────────────────────────────────
MINERU_API_URL = os.getenv("MINERU_API_URL", "http://localhost:8010")
MINERU_BACKEND = os.getenv("MINERU_BACKEND", "vlm-engine")
MINERU_BACKEND_FOLDER = {
    "pipeline":      "auto",
    "hybrid-engine": "hybrid_auto",
    "vlm-engine":    "vlm",
}
MINERU_OUT_SUBDIR = MINERU_BACKEND_FOLDER[MINERU_BACKEND]
