#!/usr/bin/env bash
# scripts/build_images.sh — build Memento container images in the correct order.
#
# Usage:
#   bash scripts/build_images.sh                # build all core images (no reasoning); also pulls vLLM
#   bash scripts/build_images.sh all            # same as above
#   bash scripts/build_images.sh reasoning      # build local LLM generator image (optional)
#   bash scripts/build_images.sh mineru         # rebuild one image only
#
# Build order (core):
#   1. mineru          — standalone, ~20 min (downloads 3 GB MinerU models during build)
#   2. vllm            — pulls docker.io/vllm/vllm-openai:v0.30.0 (embedder + reranker; no build), ~10 min
#   3. fileserver      — standalone, no GPU deps, seconds
#
# Optional:
#   4. reasoning       — local vLLM generator (only needed if not using external API)
#                        Must be a vision-capable model — size your GPU/disk accordingly

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

log() { echo "[$(date +%H:%M:%S)] $*"; }

build_mineru() {
    log "Building mineru:latest ..."
    podman build -t mineru:latest -f "$PROJECT_ROOT/docker/Dockerfile.mineru" "$PROJECT_ROOT"
    log "mineru:latest done."
}

# embedder + reranker run the official vLLM image (arguments in docker-compose.yml)
pull_vllm() {
    log "Pulling docker.io/vllm/vllm-openai:v0.30.0 ..."
    podman pull docker.io/vllm/vllm-openai:v0.30.0
    log "vllm-openai:v0.30.0 done."
}

build_reasoning() {
    log "Building qwen3-vl-reasoning:latest ..."
    podman build -t qwen3-vl-reasoning:latest -f "$PROJECT_ROOT/docker/Dockerfile.reasoning" "$PROJECT_ROOT"
    log "qwen3-vl-reasoning:latest done."
    log "  To use: uncomment the 'reasoning' service in docker-compose.yml"
    log "  Download weights (vision-capable model required): bash scripts/download_models.sh reasoning"
}

build_fileserver() {
    log "Building memento-fileserver:latest ..."
    podman build -t memento-fileserver:latest -f "$PROJECT_ROOT/docker/Dockerfile.fileserver" "$PROJECT_ROOT"
    log "memento-fileserver:latest done."
}

TARGET="${1:-all}"

case "$TARGET" in
    all)
        build_mineru
        pull_vllm
        build_fileserver
        ;;
    mineru)          build_mineru ;;
    vllm)            pull_vllm ;;
    reasoning)       build_reasoning ;;
    fileserver)      build_fileserver ;;
    *)
        echo "Usage: $0 [all|mineru|vllm|reasoning|fileserver]"
        exit 1
        ;;
esac

echo ""
log "Images:"
podman images | grep -E 'mineru|vllm-openai|qwen3-vl|memento-fileserver' || true
