#!/usr/bin/env bash
# HTML-only local preview. Reuse the latest cached notebook outputs even when
# code changes; never generate datasets or execute notebook cells. Results may
# be stale. Use docs_rebuild_fast.sh to refresh outputs, or docs_rebuild.sh for
# a clean build. Missing caches or changed code-cell counts fail the preview.
set -euo pipefail
PYTHON="uv run --extra docs python"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # hybrax/docs/
SRC="$ROOT/source"; OUT="$ROOT/html"; SCRATCH="$ROOT/_scratch"
JUPYTER_EXECUTE="$ROOT/jupyter_execute"
mkdir -p "$SCRATCH"

LOG="$SCRATCH/build_fastfast.log"
echo "Preview only: saved outputs may be stale; notebook code will not run."
echo "Build log: $LOG  (tail -f \"$LOG\" in another terminal to watch progress)"
BUILD_START=$(date +%s)
rm -f "$OUT"/_static/mystnb.*.css
# Separate doctrees keep a preview from marking code changes as executed in
# subsequent normal builds. The output directory matches the other scripts.
$PYTHON "$ROOT/_fastfast.py" -b html -j 4 \
    -d "$SCRATCH/doctrees_fastfast" "$SRC" "$OUT" 2>&1 | tee "$LOG"

if compgen -G "$JUPYTER_EXECUTE"/*.png > /dev/null; then
    mkdir -p "$JUPYTER_EXECUTE/figures"
    mv "$JUPYTER_EXECUTE"/*.png "$JUPYTER_EXECUTE/figures/"
fi

# Same warning gate as the other builds.
if grep -E "WARNING|ERROR" "$LOG" | grep -v "source/autoapi/" | grep -v "\[IPKernelApp\]" > "$SCRATCH/our_warnings_fastfast.log"; then
    echo
    echo "docs preview failed: warnings in hand-written pages" >&2
    cat "$SCRATCH/our_warnings_fastfast.log" >&2
    exit 1
fi

echo "Built (preview, saved outputs): $OUT/index.html  ($(( $(date +%s) - BUILD_START ))s)"
