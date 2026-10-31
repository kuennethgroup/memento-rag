#!/usr/bin/env bash
# Resolve project root relative to this file's location
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATA_DIR="${PROJECT_ROOT}/data"
INPUT_ROOT="${DATA_DIR}/input"
OUTPUT_ROOT="${DATA_DIR}/output"

# Models
EMBEDDER_MODEL_PATH="${HOME}/models/Qwen3-VL-Embedding-8B"
RERANKER_MODEL_PATH="${HOME}/models/Qwen3-VL-Reranker-8B"
REASONING_MODEL_PATH="${HOME}/models/Qwen3-VL-8B-Instruct"

# GPU defaults (override with env vars)
MINERU_GPU="${MINERU_GPU:-1}"
EMBEDDER_GPU="${EMBEDDER_GPU:-1}"
RERANKER_GPU="${RERANKER_GPU:-2}"
REASONING_GPU="${REASONING_GPU:-3}"
