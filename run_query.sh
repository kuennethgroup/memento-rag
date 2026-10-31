#!/usr/bin/env bash
# run_query.sh: ask Memento Query from the command line: the same pipe as the chat (router, search, answer, summaries,
# research gaps), with the pipe's saved settings and the served library (memento/retrieval/pipe_cli.py)
#
# Usage:
# bash run_query.sh "<question>"
# bash run_query.sh --file queries.txt              # one query per line
# bash run_query.sh "query" --model qwen35          # answer model: an id of config/query_models.yaml
# bash run_query.sh "query" --thorough              # the chat's Thorough check
# bash run_query.sh "query" --json result.json      # save question, answer and sources
#
# Prerequisites: the query services must be running (podman-compose --env-file config/.env up -d).

set -euo pipefail

MEMENTO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$MEMENTO_DIR/logs"
SERVICES_SH="$MEMENTO_DIR/scripts/services.sh"

log() { echo "[$(date +%H:%M:%S)] [query] $*"; }

set -a; source "$MEMENTO_DIR/config/.env"; set +a

# ── Ensure required services are up (starts them if needed, waits for healthy) ─
bash "$SERVICES_SH" start qdrant embedder reranker

# The answer model and every other setting are Memento Query's own (its saved Valves): this runs the same pipe as the
# chat (memento/retrieval/pipe_cli.py), which reports an unreachable model itself.

# ── Activate venv ──────────────────────────────────────────────────────────────
if [ -f "$MEMENTO_DIR/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$MEMENTO_DIR/.venv/bin/activate"
fi

mkdir -p "$LOG_DIR"

# ── File mode: one query per line ─────────────────────────────────────────────
if [ "${1:-}" = "--file" ]; then
    QUERY_FILE="${2:?Usage: $0 --file <queries.txt>}"
    EXTRA_ARGS=("${@:3}")
    [ -f "$QUERY_FILE" ] || { log "ERROR: file not found: $QUERY_FILE"; exit 1; }

    log "Running queries from $QUERY_FILE"
    LINE=0
    while IFS= read -r query || [ -n "$query" ]; do
        # Skip blank lines and comments
        [[ -z "$query" || "$query" == \#* ]] && continue
        LINE=$(( LINE + 1 ))
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "  Query $LINE: $query"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        PYTHONPATH="$MEMENTO_DIR${PYTHONPATH:+:$PYTHONPATH}" python -m memento.retrieval.generate "$query" "${EXTRA_ARGS[@]}"
    done < "$QUERY_FILE"
    log "Done — $LINE queries processed."
    exit 0
fi

# ── Single query mode ─────────────────────────────────────────────────────────
QUERY="${1:?Usage: $0 \"<query>\" [options]  OR  $0 --file queries.txt}"
EXTRA_ARGS=("${@:2}")

PYTHONPATH="$MEMENTO_DIR${PYTHONPATH:+:$PYTHONPATH}" python -m memento.retrieval.generate "$QUERY" "${EXTRA_ARGS[@]}"
