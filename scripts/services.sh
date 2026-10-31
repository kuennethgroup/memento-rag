#!/usr/bin/env bash
# scripts/services.sh — start/stop/status for the full Memento stack: the
# podman-compose backend services AND the OpenWebUI frontend (host tmux
# process). Single entry point for service lifecycle + health/status.
#
# Usage:
#   bash scripts/services.sh start                       # start everything, wait for healthy
#   bash scripts/services.sh start embedder reranker      # start only these, wait for only these
#   bash scripts/services.sh start openwebui               # start just the frontend
#   bash scripts/services.sh start --timeout 300          # custom health-wait threshold (default 600s)
#   bash scripts/services.sh start --no-rollback          # leave a failed start up for debugging
#
#   bash scripts/services.sh stop                         # stop everything (compose down + openwebui)
#   bash scripts/services.sh stop mineru                   # stop only this service (leaves it defined)
#   bash scripts/services.sh stop openwebui                # stop just the frontend
#
#   bash scripts/services.sh status                       # full system status dashboard
#   bash scripts/services.sh status embedder reranker     # just these services' container/health state
#
# Service names: postgres, qdrant, neo4j, embedder, mineru, reranker, files, proxy
# (podman-compose services) and openwebui (host tmux process, started via tmux +
# the venv's open-webui CLI). embedder/reranker are vLLM 0.30.0 containers
# (Qwen3-VL-Embedding-8B on 8021, Qwen3-VL-Reranker-8B on 8022; GPUs EMBEDDER_GPU /
# RERANKER_GPU, memory share EMBEDDER_GPU_MEM / RERANKER_GPU_MEM).
#
# Explicit names only (not part of a bare `start`): reasoning (optional local
# reasoning container, REASONING_GPU) and the RQ2 per-corpus graph stores
# neo4j-ceramics, neo4j-polymer, neo4j-test, neo4j-*-no-llm[-gliner].

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEMENTO_DIR="$(dirname "$SCRIPT_DIR")"
ENV_FILE="$MEMENTO_DIR/config/.env"
LOG_DIR="$MEMENTO_DIR/logs"
COMPOSE=(podman-compose --env-file "$ENV_FILE")

[[ -f "$ENV_FILE" ]] || { echo "ERROR: $ENV_FILE not found."; exit 1; }
# Per-invocation GPU placement (e.g. RERANKER_GPU=0 bash scripts/services.sh start reranker)
# must win over config/.env; every other setting still comes from .env.
_caller_gpu_env=$(export -p | grep -E '^declare -x [A-Z_]+_(GPU|MEM)=' || true)
set -a; source "$ENV_FILE"; set +a
eval "$_caller_gpu_env"

OPENWEBUI_SESSION="openwebui"
GAP_UPDATER_SESSION="gap-updater"   # keeps the research-gap ledger current (scripts/update_gap_ledger.py)
# the second library (scripts/setup_openwebui.py study), SECOND_LIBRARY_ID in config/.env
SECOND_ID="${SECOND_LIBRARY_ID:-polymer}"
GAP_UPDATER_SECOND_SESSION="gap-updater-$SECOND_ID"
OPENWEBUI_PORT="${OPENWEBUI_PORT:-8086}"
# localhost only: users come in through the proxy on PROXY_PORT (docker/Caddyfile, host network, proxies to
# 127.0.0.1:8086); stopping the proxy then closes every outside path
OPENWEBUI_HOST="${OPENWEBUI_HOST:-127.0.0.1}"

# compose service name -> container name (every service with a healthcheck
# in docker-compose.yml)
declare -A CONTAINER_OF=(
    [postgres]=memento-postgres
    [qdrant]=qdrant
    [neo4j]=memento-neo4j
    [neo4j-ceramics]=memento-neo4j-ceramics
    [neo4j-polymer]=memento-neo4j-polymer
    [neo4j-test]=memento-neo4j-test
    [neo4j-ceramics-no-llm]=memento-neo4j-ceramics-no-llm
    [neo4j-polymer-no-llm]=memento-neo4j-polymer-no-llm
    [neo4j-ceramics-no-llm-gliner]=memento-neo4j-ceramics-no-llm-gliner
    [neo4j-polymer-no-llm-gliner]=memento-neo4j-polymer-no-llm-gliner
    [embedder]=embedder
    [mineru]=mineru-api-server
    [reranker]=reranker
    [reasoning]=reasoning
    [files]=memento-files
    [proxy]=memento-proxy
)
PODMAN_SERVICES=(postgres qdrant neo4j embedder mineru reranker files proxy)
# optional podman services: never in the defaults (reasoning's GPU may be a shared card)
OPTIONAL_SERVICES=(reasoning)
# per-corpus graph stores (RQ2): podman services, but never in the defaults
GRAPH_SERVICES=(neo4j-ceramics neo4j-polymer neo4j-test
                neo4j-ceramics-no-llm neo4j-polymer-no-llm neo4j-ceramics-no-llm-gliner neo4j-polymer-no-llm-gliner)
DEFAULT_SERVICES=("${PODMAN_SERVICES[@]}" openwebui)
ALL_SERVICES=("${DEFAULT_SERVICES[@]}" "${OPTIONAL_SERVICES[@]}" "${GRAPH_SERVICES[@]}")

log() { echo "[$(date +%H:%M:%S)] [services] $*"; }

container_health() {
    local name="$1" status health
    status=$(podman inspect "$name" --format '{{.State.Status}}' 2>/dev/null) || { echo "missing"; return; }
    [ "$status" != "running" ] && { echo "$status"; return; }
    health=$(podman inspect "$name" --format '{{.State.Health.Status}}' 2>/dev/null)
    if [ -z "$health" ] || [ "$health" = "<nil>" ]; then
        echo "healthy"  # no healthcheck defined — running is sufficient
    else
        echo "$health"
    fi
}

usage() {
    awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "${BASH_SOURCE[0]}"
    exit 1
}

validate_services() {
    for s in "$@"; do
        local ok=0
        for known in "${ALL_SERVICES[@]}"; do
            [ "$s" = "$known" ] && ok=1 && break
        done
        if [ $ok -eq 0 ]; then
            echo "Unknown service: $s (expected one of: ${ALL_SERVICES[*]})" >&2
            exit 1
        fi
    done
}

# Splits "$@" into the global array/flag PODMAN_REQ and WANT_OPENWEBUI.
split_services() {
    PODMAN_REQ=()
    WANT_OPENWEBUI=0
    for s in "$@"; do
        if [ "$s" = "openwebui" ]; then
            WANT_OPENWEBUI=1
        else
            PODMAN_REQ+=("$s")
        fi
    done
}

# ── OpenWebUI (host tmux process) ───────────────────────────────────────────
rotate_openwebui_logs() {
    mkdir -p "$LOG_DIR"
    local today
    today=$(date +%Y-%m-%d)
    for f in "$LOG_DIR"/openwebui-*.log; do
        [[ -f "$f" ]] || continue
        [[ "$f" == *"$today"* ]] && continue
        gzip -f "$f" && log "openwebui: compressed $(basename "$f")"
    done
}

patch_openwebui_logo() {
    local LOGO="${WEBUI_LOGO:-}"
    [ -z "$LOGO" ] && return 0   # no logo configured — skip silently
    local PKG="$MEMENTO_DIR/.venv/lib/python3.12/site-packages/open_webui"
    local STATIC="$PKG/static"
    local FRONTEND="$PKG/frontend/static"
    # the site root serves its own favicon.png (frontend/favicon.png)
    cp "$LOGO" "$PKG/frontend/favicon.png" 2>/dev/null || true
    for dir in "$STATIC" "$FRONTEND"; do
        [ -d "$dir" ] || continue
        for f in logo.png favicon.png favicon-96x96.png apple-touch-icon.png \
                 favicon-dark.png splash.png splash-dark.png; do
            cp "$LOGO" "$dir/$f" 2>/dev/null || true
        done
        # the venv's Python: it has Pillow (the system python3 may not, and favicon.ico / favicon.svg would stay Open
        # WebUI's own)
        "$MEMENTO_DIR/.venv/bin/python" -c "
import base64
from pathlib import Path
from PIL import Image
img = Image.open('$LOGO').convert('RGBA')
img.save('$dir/favicon.ico', format='ICO', sizes=[(16,16),(32,32),(48,48),(64,64)])
# index.html lists an SVG icon ahead of the PNG/ICO ones, and modern browsers
# prefer it when present — overwriting it keeps the OI default from winning.
b64 = base64.b64encode(Path('$LOGO').read_bytes()).decode()
svg = ('<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"500\" height=\"500\">'
       f'<image width=\"500\" height=\"500\" href=\"data:image/png;base64,{b64}\"/></svg>')
Path('$dir/favicon.svg').write_text(svg)
" 2>/dev/null || true
    done
}

# The " (Open WebUI)" suffix OpenWebUI appends to the app name (open_webui/env.py) is off by default: this is a Master's
# thesis deployment with far fewer than 50 users in any 30 days, which clause 4 (i) of the Open WebUI licence allows
# (openwebui/README.md, "Licence and branding"). OPENWEBUI_NAME_SUFFIX=on in config/.env restores it (then restart
# OpenWebUI). The credit "Built with Open WebUI" stays in the corner either way (openwebui/loader.js).
patch_openwebui_name_suffix() {
    local ENV_PY="$MEMENTO_DIR/.venv/lib/python3.12/site-packages/open_webui/env.py"
    [ -f "$ENV_PY" ] || return 0
    local ON="    WEBUI_NAME += ' (Open WebUI)'"
    local OFF="    pass  # Memento: ' (Open WebUI)' name suffix off, see openwebui/README.md (licence clause 4 (i))"
    if [ "${OPENWEBUI_NAME_SUFFIX:-off}" = "on" ]; then
        grep -qF "$OFF" "$ENV_PY" && sed -i "s|^${OFF//|/\|}\$|${ON//|/\|}|" "$ENV_PY"
    else
        grep -qxF "$ON" "$ENV_PY" && sed -i "s|^${ON//|/\|}\$|${OFF//|/\|}|" "$ENV_PY"
    fi
    return 0
}

# Memento's CSS and script additions (openwebui/custom.css, openwebui/loader.js) into OpenWebUI's custom slots
# (/static/custom.css, /static/loader.js). OpenWebUI
# rebuilds static/ from frontend/static/ at start, so both copies are written, like the logo above.
apply_openwebui_css() {
    local PKG="$MEMENTO_DIR/.venv/lib/python3.12/site-packages/open_webui"
    # custom.css = OpenWebUI's custom stylesheet slot, loader.js = its custom script slot (both empty by default)
    for f in custom.css loader.js; do
        [ -f "$MEMENTO_DIR/openwebui/$f" ] || continue
        for dir in "$PKG/static" "$PKG/frontend/static"; do
            [ -d "$dir" ] && cp "$MEMENTO_DIR/openwebui/$f" "$dir/$f" 2>/dev/null || true
        done
    done
    # guide clips for the start-screen labels: openwebui/static/memento/ → /static/memento/
    if [ -d "$MEMENTO_DIR/openwebui/static/memento" ]; then
        for dir in "$PKG/static" "$PKG/frontend/static"; do
            [ -d "$dir" ] && cp -r "$MEMENTO_DIR/openwebui/static/memento" "$dir/" 2>/dev/null || true
        done
    fi
}

start_openwebui() {
    rotate_openwebui_logs
    local LOG_FILE="$LOG_DIR/openwebui-$(date +%Y-%m-%d).log"

    if tmux has-session -t "$OPENWEBUI_SESSION" 2>/dev/null; then
        log "openwebui: killing existing tmux session..."
        tmux kill-session -t "$OPENWEBUI_SESSION"
        sleep 1
    fi

    patch_openwebui_logo
    patch_openwebui_name_suffix
    apply_openwebui_css

    log "openwebui: starting on $OPENWEBUI_HOST:$OPENWEBUI_PORT, logging to logs/openwebui-$(date +%Y-%m-%d).log ..."
    # The session clears the environment a running tmux server would hand it and reads config/.env itself
    # (scripts/openwebui_session.sh): a changed setting or key applies at this start, and no secret is on the command line.
    tmux new-session -d -s "$OPENWEBUI_SESSION" \
        "bash '$MEMENTO_DIR/scripts/openwebui_session.sh' '$MEMENTO_DIR' '$OPENWEBUI_HOST' '$OPENWEBUI_PORT' '$LOG_FILE'; \
         echo '[EXITED] Press enter to close'; read"

    local retries=60
    until curl -sf "http://localhost:$OPENWEBUI_PORT/health" >/dev/null 2>&1; do
        sleep 2
        retries=$(( retries - 1 ))
        [ $retries -le 0 ] && { log "openwebui: ERROR — timed out. Check: tmux attach -t $OPENWEBUI_SESSION"; return 1; }
    done
    log "openwebui: healthy ✓  (http://localhost:$OPENWEBUI_PORT)"
    sync_openwebui_models "$LOG_FILE"
    start_gap_updater
}

# The research-gap ledger's updater: every minute, documents added to or removed from the library are processed (their
# open questions, the checks against them). Started and stopped with Open WebUI; config/.env sourced (it needs the
# database and the Semantic Scholar key); GAP_LEDGER_UPDATES=false makes it exit at once. A second updater exits by
# itself (its lock file).
start_gap_updater() {
    tmux has-session -t "$GAP_UPDATER_SESSION" 2>/dev/null && tmux kill-session -t "$GAP_UPDATER_SESSION"
    tmux new-session -d -s "$GAP_UPDATER_SESSION" \
        "cd '$MEMENTO_DIR' && set -a && . config/.env && set +a && \
         '$MEMENTO_DIR/.venv/bin/python' scripts/update_gap_ledger.py >> '$LOG_DIR/gap_updater.log' 2>&1"
    log "gap updater: started (logs/gap_updater.log)"
    # the second copy: only once it is enabled (installed, configured, active)
    tmux has-session -t "$GAP_UPDATER_SECOND_SESSION" 2>/dev/null && tmux kill-session -t "$GAP_UPDATER_SECOND_SESSION"
    if (cd "$MEMENTO_DIR" && set -a && . config/.env && set +a && \
            "$MEMENTO_DIR/.venv/bin/python" scripts/copy_state_cli.py --function-id "memento_query_$SECOND_ID" \
            >> "$LOG_DIR/gap_updater_$SECOND_ID.log" 2>&1); then
        tmux new-session -d -s "$GAP_UPDATER_SECOND_SESSION" \
            "cd '$MEMENTO_DIR' && set -a && . config/.env && set +a && \
             '$MEMENTO_DIR/.venv/bin/python' scripts/update_gap_ledger.py --function-id 'memento_query_$SECOND_ID' \
             >> '$LOG_DIR/gap_updater_$SECOND_ID.log' 2>&1"
        log "gap updater (second library): started (logs/gap_updater_$SECOND_ID.log)"
    else
        log "gap updater (second library): not started, the second library is not enabled"
    fi
}

stop_gap_updater() {
    if tmux has-session -t "$GAP_UPDATER_SESSION" 2>/dev/null; then
        tmux kill-session -t "$GAP_UPDATER_SESSION"
        log "gap updater: stopped."
    fi
    if tmux has-session -t "$GAP_UPDATER_SECOND_SESSION" 2>/dev/null; then
        tmux kill-session -t "$GAP_UPDATER_SECOND_SESSION"
        log "gap updater (second library): stopped."
    fi
}

# The picker entries (names, descriptions, group access, default model and order) live in Open WebUI's database; they
# are written from config/query_models.yaml at every start, so editing the config and restarting is enough. A failure
# only warns: Open WebUI keeps running with the entries it already has.
sync_openwebui_models() {
    local log_file="$1"
    if "$MEMENTO_DIR/.venv/bin/python" "$MEMENTO_DIR/scripts/setup_openwebui.py" \
            --url "http://localhost:$OPENWEBUI_PORT" sync-models >> "$log_file" 2>&1; then
        log "openwebui: models synced from config/query_models.yaml ✓"
    else
        log "openwebui: WARNING: model sync failed (details in logs/$(basename "$log_file")); the picker keeps its old entries"
    fi
}

stop_openwebui() {
    stop_gap_updater
    if tmux has-session -t "$OPENWEBUI_SESSION" 2>/dev/null; then
        tmux kill-session -t "$OPENWEBUI_SESSION"
        log "openwebui: stopped."
    else
        log "openwebui: not running."
    fi
}

status_openwebui() {
    if tmux has-session -t "$OPENWEBUI_SESSION" 2>/dev/null; then
        if curl -sf "http://localhost:$OPENWEBUI_PORT/health" >/dev/null 2>&1; then
            printf "  %-20s running (healthy)\n" "openwebui"
        else
            printf "  %-20s running (not responding on :%s)\n" "openwebui" "$OPENWEBUI_PORT"
        fi
    else
        printf "  %-20s absent\n" "openwebui"
    fi
}

# ── start ─────────────────────────────────────────────────────────────────────
cmd_start() {
    local timeout=600 rollback=1 services=()
    while [ $# -gt 0 ]; do
        case "$1" in
            --timeout)     timeout="$2"; shift 2 ;;
            --timeout=*)   timeout="${1#*=}"; shift ;;
            --no-rollback) rollback=0; shift ;;
            -*) echo "Unknown flag: $1" >&2; exit 1 ;;
            *) services+=("$1"); shift ;;
        esac
    done

    local all_requested=0
    if [ ${#services[@]} -eq 0 ]; then
        services=("${DEFAULT_SERVICES[@]}")
        all_requested=1
    else
        validate_services "${services[@]}"
    fi

    split_services "${services[@]}"

    if [ ${#PODMAN_REQ[@]} -gt 0 ]; then
        # Skip `up -d` for services already healthy — podman-compose's `up -d`
        # on an existing running container logs a noisy (harmless) "name already
        # in use" error before falling back to `podman start`.
        local to_start=() containers=()
        for s in "${PODMAN_REQ[@]}"; do
            local c="${CONTAINER_OF[$s]}"
            containers+=("$c")
            local h
            h=$(container_health "$c")
            [ "$h" != "healthy" ] && to_start+=("$s")
        done

        if [ ${#to_start[@]} -gt 0 ]; then
            log "Starting: ${to_start[*]}"
            "${COMPOSE[@]}" up -d "${to_start[@]}"
        else
            log "Already healthy: ${PODMAN_REQ[*]}"
        fi

        log "Waiting up to ${timeout}s for: ${containers[*]}"
        local start_ts=$(date +%s)
        local pending=("${containers[@]}")
        local failed=""

        while [ ${#pending[@]} -gt 0 ]; do
            local still_pending=()
            for name in "${pending[@]}"; do
                local health
                health=$(container_health "$name")
                case "$health" in
                    healthy)   log "$name: healthy ✓" ;;
                    unhealthy) failed="$name (unhealthy)" ;;
                    missing)   failed="$name (container not found)" ;;
                    *)         still_pending+=("$name") ;;
                esac
            done
            [ -n "$failed" ] && break

            pending=("${still_pending[@]}")
            [ ${#pending[@]} -eq 0 ] && break

            local elapsed=$(( $(date +%s) - start_ts ))
            if [ $elapsed -ge $timeout ]; then
                failed="${pending[*]} (timed out after ${timeout}s)"
                break
            fi
            sleep 5
        done

        if [ -n "$failed" ]; then
            log "ERROR: $failed"
            log "Check logs with: podman logs <container>"
            if [ $rollback -eq 1 ]; then
                if [ $all_requested -eq 1 ]; then
                    log "Bringing the whole stack back down (use --no-rollback to leave it up for debugging)..."
                    "${COMPOSE[@]}" down
                else
                    log "Stopping the service(s) just started (use --no-rollback to leave them up for debugging)..."
                    "${COMPOSE[@]}" stop "${PODMAN_REQ[@]}"
                fi
            fi
            exit 1
        fi
        log "Podman services healthy ✓"
    fi

    if [ $WANT_OPENWEBUI -eq 1 ]; then
        start_openwebui || exit 1
    fi

    log "All requested services healthy ✓"
}

# ── stop ──────────────────────────────────────────────────────────────────────
cmd_stop() {
    local services=("$@")
    if [ ${#services[@]} -eq 0 ]; then
        log "Stopping + removing the whole stack (down)..."
        "${COMPOSE[@]}" down
        stop_openwebui
        return
    fi

    validate_services "${services[@]}"
    split_services "${services[@]}"

    if [ ${#PODMAN_REQ[@]} -gt 0 ]; then
        log "Stopping: ${PODMAN_REQ[*]}"
        "${COMPOSE[@]}" stop "${PODMAN_REQ[@]}"
    fi
    if [ $WANT_OPENWEBUI -eq 1 ]; then
        stop_openwebui
    fi
}

# ── status ────────────────────────────────────────────────────────────────────
check_http() {
    local name="$1" url="$2" optional="${3:-}"
    if curl -sf --max-time 3 "$url" >/dev/null 2>&1; then
        printf "  ✓ %-16s %s\n" "$name" "$url"
    elif [ -n "$optional" ]; then
        printf "  ○ %-16s %s  (optional — not running)\n" "$name" "$url"
    else
        printf "  ✗ %-16s %s  (not reachable)\n" "$name" "$url"
    fi
}

container_status() {
    local name="$1"
    if podman container exists "$name" 2>/dev/null; then
        local state health
        state=$(podman inspect "$name" --format '{{.State.Status}}' 2>/dev/null)
        health=$(podman inspect "$name" --format '{{.State.Health.Status}}' 2>/dev/null)
        if [ -n "$health" ] && [ "$health" != "<nil>" ]; then
            echo "$state ($health)"
        else
            echo "$state"
        fi
    else
        echo "absent"
    fi
}

check_postgres() {
    if podman exec memento-postgres pg_isready -q 2>/dev/null; then
        printf "  ✓ %-16s %s\n" "Postgres" "localhost:5433"
    else
        printf "  ✗ %-16s %s  (not reachable)\n" "Postgres" "localhost:5433"
    fi
}

print_dashboard() {
    echo ""
    echo "══════════════════════════════════════════════"
    echo "  Memento — system status"
    echo "══════════════════════════════════════════════"
    echo ""

    echo "  Core services (podman-compose):"
    check_postgres
    check_http "Qdrant"    "http://localhost:6333/healthz"
    check_http "Neo4j"     "http://localhost:7474"
    check_http "MinerU"    "http://localhost:8010/health"
    check_http "Embedder"  "http://localhost:8021/health"
    check_http "Reranker"  "http://localhost:8022/health"
    check_http "Reasoning" "http://localhost:8023/health" "optional"
    check_http "Files"     "http://localhost:8024/health"
    echo ""

    echo "  Frontend:"
    check_http "OpenWebUI" "http://localhost:$OPENWEBUI_PORT/health"
    check_http "Remote proxy" "http://localhost:${PROXY_PORT:-8443}/health"
    echo ""

    echo "  Container state:"
    for s in "${PODMAN_SERVICES[@]}" "${OPTIONAL_SERVICES[@]}" "${GRAPH_SERVICES[@]}"; do
        local c="${CONTAINER_OF[$s]}"
        printf "  %-24s %s\n" "$c" "$(container_status "$c")"
    done
    echo ""

    echo "  GPU memory:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader 2>/dev/null \
        | awk -F', ' '{printf "  GPU %-2s  %-28s %s / %s\n", $1, $2, $3, $4}' \
        || echo "  (nvidia-smi not available)"
    echo ""

    echo "  Logs  →  logs/"
    ls "$LOG_DIR"/*.log 2>/dev/null | while read -r f; do
        printf "    %-34s %s\n" "$(basename "$f")" "$(du -sh "$f" 2>/dev/null | cut -f1)"
    done

    echo ""
    echo "══════════════════════════════════════════════"
    echo ""
}

cmd_status() {
    local services=("$@")
    if [ ${#services[@]} -eq 0 ]; then
        print_dashboard
        return
    fi

    validate_services "${services[@]}"
    for s in "${services[@]}"; do
        if [ "$s" = "openwebui" ]; then
            status_openwebui
            continue
        fi
        local c="${CONTAINER_OF[$s]}"
        if podman container exists "$c" 2>/dev/null; then
            local state health
            state=$(podman inspect "$c" --format '{{.State.Status}}' 2>/dev/null)
            health=$(podman inspect "$c" --format '{{.State.Health.Status}}' 2>/dev/null)
            if [ -n "$health" ] && [ "$health" != "<nil>" ]; then
                printf "  %-20s %s (%s)\n" "$c" "$state" "$health"
            else
                printf "  %-20s %s\n" "$c" "$state"
            fi
        else
            printf "  %-20s absent\n" "$c"
        fi
    done
}

# ── dispatch ──────────────────────────────────────────────────────────────────
ACTION="${1:-}"
[ $# -gt 0 ] && shift || true

case "$ACTION" in
    start)  cmd_start "$@" ;;
    stop)   cmd_stop "$@" ;;
    status) cmd_status "$@" ;;
    *)      usage ;;
esac
