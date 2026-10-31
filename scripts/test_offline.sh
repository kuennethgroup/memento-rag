#!/usr/bin/env bash
# Run the suite against disposable state without contacting configured model services.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${MEMENTO_TEST_PYTHON:-${repo_root}/.venv/bin/python}"
if [[ ! -x "$python_bin" ]]; then
    echo "Set MEMENTO_TEST_PYTHON or install requirements/dev.txt in .venv." >&2
    exit 1
fi
test_state="$(mktemp -d "${TMPDIR:-/tmp}/memento-offline.XXXXXXXX")"
trap 'rm -rf -- "$test_state"' EXIT
cd "$repo_root"
# Existing config/.env cannot override these values (dotenv uses override=False).
# Proxies also catch historical HTTP model addresses embedded in experiment code.
# This is test isolation, not a network sandbox for arbitrary untrusted tests.
export MEMENTO_DIR="$repo_root" DOMAIN_NAME=polymer-engineering
# the default library of the two-library setup the tests model (DOMAIN_NAME stays the recordings' domain)
export MEMENTO_DEFAULT_COLLECTION=memento-comp-mat-rag
export DATA_DIR="$test_state/data" INPUT_ROOT="$test_state/input" OUTPUT_ROOT="$test_state/output"
export TIKTOKEN_CACHE_DIR="${TIKTOKEN_CACHE_DIR:-${TMPDIR:-/tmp}/data-gym-cache}"
export TMPDIR="$test_state" PYTHONDONTWRITEBYTECODE=1
export DATABASE_URL="sqlite:///$test_state/tests.db" NEO4J_URI=bolt://127.0.0.1:9
export WEBUI_SECRET_KEY=isolated-offline-test-key
export OPENAI_API_KEY=offline-test CLOUD_API_KEY=offline-test
export OPENAI_BASE_URL=http://127.0.0.1:9/v1
export EMBEDDER_URL=http://127.0.0.1:9 RERANKER_URL=http://127.0.0.1:9
export MINERU_API_URL=http://127.0.0.1:9 QDRANT_URL=http://127.0.0.1:9
export HTTP_PROXY=http://127.0.0.1:9 HTTPS_PROXY=http://127.0.0.1:9 ALL_PROXY=http://127.0.0.1:9
export http_proxy="$HTTP_PROXY" https_proxy="$HTTPS_PROXY" all_proxy="$ALL_PROXY"
export NO_PROXY= no_proxy=
# A throwaway Qdrant for the integration tests (MEMENTO_TEST_QDRANT_URL) passes the proxy, by its host:port only
if [[ -n "${MEMENTO_TEST_QDRANT_URL:-}" ]]; then
    qdrant_hostport="${MEMENTO_TEST_QDRANT_URL#*://}"
    export NO_PROXY="${qdrant_hostport%%/*}" no_proxy="${qdrant_hostport%%/*}"
fi
mkdir -p "$DATA_DIR" "$INPUT_ROOT" "$OUTPUT_ROOT"
if [[ $# -eq 0 ]]; then set -- tests/; fi
"$python_bin" -m pytest -q -p no:cacheprovider "$@"
