#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTEST_TMP="$REPO_ROOT/tmp/pytest_tmp"

mkdir -p "$PYTEST_TMP"

exec env TMPDIR="$PYTEST_TMP" "$REPO_ROOT/.venv/bin/python" -m pytest "$@"
