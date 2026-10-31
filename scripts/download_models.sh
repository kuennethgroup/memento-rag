#!/bin/bash
# scripts/download_models.sh — download all model weights for Memento
#
# Weights are NOT baked into container images. They live on the host and are
# mounted read-only at runtime (see start_*.sh). This script fetches all three
# in one step, parallel to the "build images" step.
#
# Idempotent: hf download skips files already present and resumes partial ones,
# so re-running is safe and cheap.
#
# Usage:
#   bash scripts/download_models.sh            — download all three
#   bash scripts/download_models.sh embedder   — embedder weights only
#   bash scripts/download_models.sh reranker   — reranker weights only
#   bash scripts/download_models.sh reasoning  — reasoning weights only
#
# Requires:  huggingface_hub[cli]  (already in requirements.txt / pyproject.toml)
#
# Disk:  embedder ~16 GB  +  reranker ~16 GB  +  reasoning ~70 GB  ≈ 102 GB total

set -u

MODELS_DIR=${MODELS_DIR:-$HOME/models}

# the `hf` command of the project's own environment (README step 2) when none is on PATH
VENV_BIN="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.venv/bin"
if ! command -v hf &>/dev/null && [ -x "$VENV_BIN/hf" ]; then
    PATH="$VENV_BIN:$PATH"
fi

# repo_id  →  local subdirectory  (the paths MODEL_PATH and RERANKER_MODEL_PATH in config/.env point to)
EMBEDDER_REPO="Qwen/Qwen3-VL-Embedding-8B"
EMBEDDER_DIR="$MODELS_DIR/Qwen3-VL-Embedding-8B"

RERANKER_REPO="Qwen/Qwen3-VL-Reranker-8B"
RERANKER_DIR="$MODELS_DIR/Qwen3-VL-Reranker-8B"

# No default — local reasoning is optional (cloud generation is the default
# path, see config/.env's OPENAI_BASE_URL). If you do run local generation,
# it must be a vision-capable model (image/chart sources are sent as actual
# images, not just captions) — set REASONING_REPO yourself, e.g.:
#   REASONING_REPO="<org>/<vision-capable-model>" bash scripts/download_models.sh reasoning
REASONING_REPO="${REASONING_REPO:-}"
REASONING_DIR="$MODELS_DIR/${REASONING_REPO##*/}"


# ── Helpers ────────────────────────────────────────────────────────────────────
check_hf_cli() {
    if ! command -v hf &>/dev/null && ! command -v huggingface-cli &>/dev/null; then
        echo "[ERROR] 'hf' / 'huggingface-cli' not found."
        echo "        Create the project's environment first (README step 2):"
        echo "        python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt"
        exit 1
    fi
}

# Prefer the modern `hf` command, fall back to legacy `huggingface-cli`
hf_download() {
    local repo=$1
    local dest=$2
    if command -v hf &>/dev/null; then
        hf download "$repo" --local-dir "$dest"
    else
        huggingface-cli download "$repo" --local-dir "$dest"
    fi
}

# Quick integrity hint: warn if any .safetensors looks like an LFS pointer stub
verify_weights() {
    local dest=$1
    local sample
    sample=$(ls "$dest"/*.safetensors 2>/dev/null | head -1)
    if [ -z "$sample" ]; then
        echo "  [WARN] No .safetensors found in $dest — download may be incomplete."
        return
    fi
    if file "$sample" | grep -qi 'ASCII text'; then
        echo "  [WARN] $sample looks like an LFS pointer, not real weights."
        echo "         Re-run this script to complete the download."
    else
        echo "  [OK]   Weights present in $dest"
    fi
}

download_one() {
    local name=$1 repo=$2 dest=$3
    echo ""
    echo "────────────────────────────────────────────────────────────"
    echo "[$name]  $repo"
    echo "         → $dest"
    echo "────────────────────────────────────────────────────────────"
    mkdir -p "$dest"
    hf_download "$repo" "$dest"
    verify_weights "$dest"
}


# ── Dispatch ──────────────────────────────────────────────────────────────────
check_hf_cli
mkdir -p "$MODELS_DIR"

case "${1:-all}" in
    embedder)
        download_one "embedder"  "$EMBEDDER_REPO"  "$EMBEDDER_DIR"
        ;;
    reranker)
        download_one "reranker"  "$RERANKER_REPO"  "$RERANKER_DIR"
        ;;
    reasoning)
        if [ -z "$REASONING_REPO" ]; then
            echo "[ERROR] REASONING_REPO not set — local reasoning has no default model."
            echo "        Set it to a vision-capable model, e.g.:"
            echo "        REASONING_REPO=\"<org>/<vision-capable-model>\" $0 reasoning"
            exit 1
        fi
        download_one "reasoning" "$REASONING_REPO" "$REASONING_DIR"
        ;;
    all)
        echo "Downloading embedder + reranker into $MODELS_DIR  (~32 GB total)"
        df -h "$MODELS_DIR" | tail -1
        download_one "embedder"  "$EMBEDDER_REPO"  "$EMBEDDER_DIR"
        download_one "reranker"  "$RERANKER_REPO"  "$RERANKER_DIR"
        echo ""
        echo "────────────────────────────────────────────────────────────"
        echo "Embedder + reranker downloads complete."
        echo "  Embedder : $EMBEDDER_DIR"
        echo "  Reranker : $RERANKER_DIR"
        if [ -n "$REASONING_REPO" ]; then
            download_one "reasoning" "$REASONING_REPO" "$REASONING_DIR"
            echo "  Reasoning: $REASONING_DIR"
        else
            echo "  Reasoning: skipped (no REASONING_REPO set — cloud generation is the"
            echo "             default; see 'bash $0 reasoning' for local setup)"
        fi
        echo "Next: build images, then start services."
        echo "────────────────────────────────────────────────────────────"
        ;;
    *)
        echo "Usage: $0 [all|embedder|reranker|reasoning]"
        exit 1
        ;;
esac