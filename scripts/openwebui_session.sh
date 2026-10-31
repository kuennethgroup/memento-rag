#!/usr/bin/env bash
# scripts/openwebui_session.sh: Open WebUI as scripts/services.sh starts it in its tmux session. A tmux server keeps the
# environment it was first started with, so the session clears it and reads config/.env afresh: a changed setting or
# key takes effect at the next start, and no secret value appears on a command line. OPENAI_API_KEY is left out because
# Open WebUI would make it the key of its own default connection; Memento's functions read config/.env themselves.
#   bash scripts/openwebui_session.sh <memento_dir> <host> <port> <log_file>
set -uo pipefail
if [[ "${MEMENTO_OPENWEBUI_CLEAN:-}" != 1 ]]; then
    exec env -i MEMENTO_OPENWEBUI_CLEAN=1 HOME="$HOME" PATH="$PATH" LANG="${LANG:-C.UTF-8}" TERM="${TERM:-xterm}" \
        bash "$0" "$@"
fi
unset MEMENTO_OPENWEBUI_CLEAN
root="$1" host="$2" port="$3" log="$4"
cd "$root" || exit 1
set -a
. config/.env
set +a
unset OPENAI_API_KEY
export WEBUI_NAME="${WEBUI_NAME:-Memento}" DATA_DIR="$HOME/.open-webui"
"$root/.venv/bin/open-webui" serve --host "$host" --port "$port" >> "$log" 2>&1
