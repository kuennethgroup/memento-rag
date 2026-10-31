#!/usr/bin/env bash
# one_click_setup.sh — one-shot setup for Memento on a new server.
#
# What this does:
#   1. Creates .venv and installs all Python dependencies
#   2. Creates config/.env from the example (if not already present)
#   3. Creates required directories (data/input, data/output, logs/)
#   4. Optionally downloads model weights via huggingface-cli
#   5. Optionally builds all container images
#
# Prerequisites: INSTALL.md step 1 (system packages, the NVIDIA Container Toolkit and its runtime for Podman).
#
# Usage:
#   bash one_click_setup.sh             # guided interactive setup
#   bash one_click_setup.sh --no-models # skip model downloads
#   bash one_click_setup.sh --no-images # skip container image builds

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

SKIP_MODELS=false
SKIP_IMAGES=false
for arg in "$@"; do
    case "$arg" in
        --no-models) SKIP_MODELS=true ;;
        --no-images) SKIP_IMAGES=true ;;
    esac
done

log()  { echo "[install] $*"; }
warn() { echo "[install] WARNING: $*"; }
die()  { echo "[install] ERROR: $*" >&2; exit 1; }
hr()   { echo "────────────────────────────────────────────────────────────"; }

hr
echo "  Memento — installation"
hr
echo ""

# ── 1. Python virtual environment ────────────────────────────────────────────

log "Step 1/5 — Python virtual environment"

command -v python3.12 >/dev/null || die "Python 3.12 is needed (INSTALL.md step 2)"
if [ ! -d .venv ]; then
    python3.12 -m venv .venv
    log "  Created .venv"
else
    log "  .venv already exists — skipping"
fi

source .venv/bin/activate
log "  Activated .venv (Python: $(python3 --version))"

log "  Installing requirements.txt ..."
pip install --quiet --upgrade pip
pip install -r requirements.txt
log "  Dependencies installed."
echo ""

# ── 2. config/.env ────────────────────────────────────────────────────────────

log "Step 2/5 — Configuration"

if [ -f config/.env ]; then
    log "  config/.env already exists — skipping creation"
    log "  Review it against INSTALL.md step 3 (model paths, GPUs, the answer model, passwords)."
else
    cp config/.env.example config/.env

    # Patch MEMENTO_DIR automatically — no manual editing needed for this one
    if grep -q "^MEMENTO_DIR=/absolute/path" config/.env; then
        sed -i "s|^MEMENTO_DIR=/absolute/path/to/memento-rag|MEMENTO_DIR=${PROJECT_ROOT}|" config/.env
        log "  Created config/.env — MEMENTO_DIR set to ${PROJECT_ROOT}"
    fi

    # Patch INPUT_ROOT / OUTPUT_ROOT
    sed -i "s|^INPUT_ROOT=/absolute/path/to/memento-rag/data/input|INPUT_ROOT=${PROJECT_ROOT}/data/input|" config/.env
    sed -i "s|^OUTPUT_ROOT=/absolute/path/to/memento-rag/data/output|OUTPUT_ROOT=${PROJECT_ROOT}/data/output|" config/.env

    # a random signing key for Open WebUI's logins
    sed -i "s|^WEBUI_SECRET_KEY=.*|WEBUI_SECRET_KEY=$(python3 -c 'import secrets; print(secrets.token_hex(32))')|" config/.env

    echo ""
    echo "  config/.env has been created. Fill in these values before starting (INSTALL.md step 3):"
    echo ""
    echo "    DOMAIN_NAME           — add this line: the domain profile, e.g. generic"
    echo "    MODEL_PATH            — absolute path to Qwen3-VL-Embedding-8B weights (step 4 puts them in ~/models)"
    echo "    RERANKER_MODEL_PATH   — absolute path to Qwen3-VL-Reranker-8B weights"
    echo "    REASONING_URL         — your answer model's server, without /v1"
    echo "    REASONING_MODEL_NAME  — the model name it serves"
    echo "    REASONING_API_KEY     — its key (empty for a server without one)"
    echo "    POSTGRES_PASSWORD     — a new password, also inside DATABASE_URL"
    echo "    (WEBUI_SECRET_KEY has been set to a random value.)"
    echo ""
    echo "  Set MINERU_GPU, EMBEDDER_GPU and RERANKER_GPU in config/.env for your GPUs."
    echo "  (Run: nvidia-smi --query-gpu=index,name,memory.free --format=csv)"
    echo ""
    read -rp "  Press Enter when you have edited config/.env, or Ctrl+C to stop here ..."
fi
echo ""

# ── 3. Directories ────────────────────────────────────────────────────────────

log "Step 3/5 — Data directories"
mkdir -p data/input data/output logs
log "  Created: data/input, data/output, logs/"
echo ""

# ── 4. Model weights ─────────────────────────────────────────────────────────

log "Step 4/5 — Model weights"

if [ "$SKIP_MODELS" = true ]; then
    log "  Skipped (--no-models). Download manually:"
    log "    bash scripts/download_models.sh"
else
    echo ""
    echo "  Downloading model weights (embedder ~16 GB + reranker ~17 GB)."
    echo "  This takes ~15 minutes. Run 'bash scripts/download_models.sh' to re-run."
    echo "  Ctrl+C here to skip and download later."
    echo ""
    read -rp "  Proceed with model downloads? [Y/n] " ans
    if [[ "$ans" =~ ^[Nn] ]]; then
        log "  Skipped. Download later with: bash scripts/download_models.sh"
    else
        bash scripts/download_models.sh
    fi
fi
echo ""

# ── 5. Container images ───────────────────────────────────────────────────────

log "Step 5/5 — Container images"

if [ "$SKIP_IMAGES" = true ]; then
    log "  Skipped (--no-images). Build manually:"
    log "    bash scripts/build_images.sh"
else
    echo ""
    echo "  Building 2 container images and pulling vLLM (MinerU takes ~20 min — downloads model weights)."
    echo "  Steps 4 and 5 can run in parallel in separate terminals."
    echo ""
    read -rp "  Proceed with image builds? [Y/n] " ans
    if [[ "$ans" =~ ^[Nn] ]]; then
        log "  Skipped. Build later with: bash scripts/build_images.sh"
    else
        bash scripts/build_images.sh
    fi
fi
echo ""

# ── Done ─────────────────────────────────────────────────────────────────────

hr
echo ""
echo "  Setup complete. Next steps:"
echo ""
echo "  Then follow INSTALL.md steps 6 to 11:"
echo ""
echo "  6. Start the services:"
echo "       bash scripts/services.sh start postgres qdrant mineru embedder reranker files openwebui proxy"
echo "  7. Open http://localhost:8443 (on a remote server: ssh -L 8443:localhost:8443 you@server), register"
echo "     (the first account is the admin), switch on API keys, create one and add it as OPENWEBUI_API_KEY."
echo "  8. .venv/bin/python scripts/setup_openwebui.py trial --domain <your-domain>"
echo "  9. Check the answer model in the Valves, then: bash scripts/services.sh start openwebui"
echo ""
echo "  Health of every service: bash scripts/services.sh status"
echo ""
hr
