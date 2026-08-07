#!/usr/bin/env bash
set -euo pipefail
PYTHON="/home/mgotsmy/anaconda3/envs/bench13/bin/python"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # bp-docs/
SRC="$ROOT/source"; OUT="$ROOT/html"; SCRATCH="$ROOT/_scratch"
mkdir -p "$SCRATCH"
rm -rf "$OUT" "$SCRATCH/doctrees"   # full clean build → no stale html orphans in the committed artifact
rm -rf "$SRC/narrative"             # legacy: bp-docs no longer copies the packages' agent docs

# Regenerate the demo datasets the tutorials/gallery execute against.
"$PYTHON" "$SRC/_data/generate.py"

LOG="$SCRATCH/build.log"
"$PYTHON" -m sphinx -b html -d "$SCRATCH/doctrees" "$SRC" "$OUT" 2>&1 | tee "$LOG"

# Warning gate. We cannot use -W: autoapi renders the packages' docstrings into RST
# and emits warnings we cannot fix from here (and at least one of them is logged
# without a `type`, so suppress_warnings cannot reach it). Instead, fail on any
# warning that points at a page *we* wrote — dead cross-references included.
if grep -E "WARNING|ERROR" "$LOG" | grep -v "source/autoapi/" > "$SCRATCH/our_warnings.log"; then
    echo
    echo "docs build failed: warnings in hand-written pages" >&2
    cat "$SCRATCH/our_warnings.log" >&2
    exit 1
fi

echo "Built: $OUT/index.html"
