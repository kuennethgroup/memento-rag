#!/usr/bin/env bash
# run_ingestion.sh — Parse + embed + ingest PDFs into the vector store
#
# Usage:
#   bash run_ingestion.sh                     # VLM pipeline, all new PDFs (pipelined)
#   bash run_ingestion.sh --ocr               # OCR pipeline, all new PDFs (pipelined)
#   bash run_ingestion.sh --staggered         # parse all → stop MinerU → embed all
#   bash run_ingestion.sh --staggered --ocr   # staggered OCR
#   bash run_ingestion.sh --pdf paper.pdf     # single PDF
#   bash run_ingestion.sh --force             # re-ingest even if already in Qdrant
#   bash run_ingestion.sh --skip-parse        # embed only (parse output exists)
#
# Prerequisites:
#   podman-compose --env-file config/.env up -d
#
# Logs written to logs/ingestion_<timestamp>.log  (VLM)
#              or logs/ingestion_ocr_<timestamp>.log  (OCR)

set -euo pipefail

MEMENTO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$MEMENTO_DIR/logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SERVICES_SH="$MEMENTO_DIR/scripts/services.sh"

log() { echo "[$(date +%H:%M:%S)] [ingestion] $*"; }

# ── Parse flags first (needed to know which service checks to skip) ───────────
OCR=0
STAGGERED=0
PASSTHROUGH_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --ocr)       OCR=1 ;;
        --staggered) STAGGERED=1 ;;
        *)           PASSTHROUGH_ARGS+=("$arg") ;;
    esac
done

# ── Ensure required services are up (starts them if needed, waits for healthy) ─
# Staggered mode manages the embedder itself (stops it before parse, restarts before embed)
if [ $STAGGERED -eq 1 ]; then
    bash "$SERVICES_SH" start qdrant mineru
else
    bash "$SERVICES_SH" start qdrant mineru embedder
fi

export PYTHONPATH="$MEMENTO_DIR${PYTHONPATH:+:$PYTHONPATH}"   # the memento package
if [ $OCR -eq 1 ]; then
    LOG_FILE="$LOG_DIR/ingestion_ocr_${TIMESTAMP}.log"
    # OCR goes through the same ingestion code as VLM (run_pipeline.py's --backend/--collection flags), so the MinerU
    # parsing backend is the only variable that differs between OCR and VLM ingestion.
    OCR_COLLECTION="$(cd "$MEMENTO_DIR" && python3 -c 'from memento.config.settings import OCR_COLLECTION; print(OCR_COLLECTION)')"
    OCR_ARGS=(--backend pipeline --collection "$OCR_COLLECTION")
    log "Pipeline: OCR  ($OCR_COLLECTION, via run_pipeline.py --backend pipeline)"
else
    LOG_FILE="$LOG_DIR/ingestion_${TIMESTAMP}.log"
    OCR_ARGS=()
    log "Pipeline: VLM  (memento-rag)"
fi

# ── Inject --all if neither --all nor --pdf present ───────────────────────────
PIPELINE_ARGS=("${PASSTHROUGH_ARGS[@]}" "${OCR_ARGS[@]}")
if ! printf '%s\n' "${PIPELINE_ARGS[@]}" | grep -qE '^(--all|--pdf)$'; then
    PIPELINE_ARGS=(--all "${PIPELINE_ARGS[@]}")
fi

# ── Activate venv ──────────────────────────────────────────────────────────────
if [ -f "$MEMENTO_DIR/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$MEMENTO_DIR/.venv/bin/activate"
fi

# ── Run pipeline ───────────────────────────────────────────────────────────────
mkdir -p "$LOG_DIR"

if [ $STAGGERED -eq 1 ]; then
    log "Mode: staggered (stop embedder → parse all → stop MinerU → start embedder → embed all)"
    mkdir -p "$LOG_DIR"

    # Stop embedder so MinerU has the full GPU budget for parsing
    log "Stopping embedder to free GPU memory for MinerU..."
    bash "$SERVICES_SH" stop embedder

    log "Stage 1/2: parsing → $LOG_FILE"
    log "Args: ${PIPELINE_ARGS[*]} --parse-only"

    PYTHONUNBUFFERED=1 python -m memento.ingestion.run_pipeline "${PIPELINE_ARGS[@]}" --parse-only \
        2>&1 | tee "$LOG_FILE"

    EXIT_CODE=${PIPESTATUS[0]}
    if [ $EXIT_CODE -ne 0 ]; then
        log "ERROR: parse stage failed. See $LOG_FILE"
        log "Restarting embedder..."
        bash "$SERVICES_SH" start embedder
        exit $EXIT_CODE
    fi

    log "Parse stage complete. Stopping MinerU, restarting embedder..."
    bash "$SERVICES_SH" stop mineru
    bash "$SERVICES_SH" start embedder

    log "Stage 2/2: embedding → $LOG_FILE"
    PYTHONUNBUFFERED=1 python -m memento.ingestion.run_pipeline "${PIPELINE_ARGS[@]}" --skip-parse \
        2>&1 | tee -a "$LOG_FILE"

    EXIT_CODE=${PIPESTATUS[0]}
    if [ $EXIT_CODE -eq 0 ]; then
        log "Staggered ingestion complete. Log saved to $LOG_FILE"
    else
        log "ERROR: embed stage failed. See $LOG_FILE"
        exit $EXIT_CODE
    fi
else
    log "Starting ingestion pipeline → $LOG_FILE"
    log "Args: ${PIPELINE_ARGS[*]}"

    PYTHONUNBUFFERED=1 python -m memento.ingestion.run_pipeline "${PIPELINE_ARGS[@]}" \
        2>&1 | tee "$LOG_FILE"

    EXIT_CODE=${PIPESTATUS[0]}
    if [ $EXIT_CODE -eq 0 ]; then
        log "Ingestion complete. Log saved to $LOG_FILE"
    else
        log "ERROR: pipeline exited with code $EXIT_CODE. See $LOG_FILE"
        exit $EXIT_CODE
    fi
fi
