#!/usr/bin/env bash
set -euo pipefail
PYTHON="/home/mgotsmy/anaconda3/envs/bench13/bin/python"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # bp-docs/
SRC="$ROOT/source"; OUT="$ROOT/html"; SCRATCH="$ROOT/_scratch"
mkdir -p "$SCRATCH"
rm -rf "$OUT" "$SCRATCH/doctrees"   # full clean build → no stale html orphans in the committed artifact
rm -rf "$SRC/narrative"
mkdir -p "$SRC/narrative/bp-train" "$SRC/narrative/bp-format"
cp "$ROOT/../bp-train/documentation/"[0-9]*.md  "$SRC/narrative/bp-train/"  2>/dev/null || true
cp "$ROOT/../bp-format/documentation/"[0-9]*.md "$SRC/narrative/bp-format/" 2>/dev/null || true
# the per-package design rationales are merged into source/design_rationale.md
rm -f "$SRC/narrative/bp-train/01_design_rationale.md" "$SRC/narrative/bp-format/01_design_rationale.md"
"$PYTHON" -m sphinx -b html -d "$SCRATCH/doctrees" "$SRC" "$OUT"
echo "Built: $OUT/index.html"
