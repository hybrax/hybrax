#!/usr/bin/env bash
set -euo pipefail
PYTHON="/home/mgotsmy/anaconda3/envs/bench13/bin/python"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # bp-docs/
SRC="$ROOT/source"; OUT="$ROOT/html"; SCRATCH="$ROOT/_scratch"
JUPYTER_EXECUTE="$ROOT/jupyter_execute"
mkdir -p "$SCRATCH"
rm -rf "$OUT" "$SCRATCH/doctrees" "$JUPYTER_EXECUTE"
# myst-nb's notebook execution cache is keyed on cell source only, not on the demo
# data files those cells load, so a data-generator change alone does not invalidate
# it — clear it every time or "full clean build" silently serves stale plots/output.
rm -rf "$SCRATCH/jupyter_cache"
rm -rf "$SRC/narrative"             # legacy: bp-docs no longer copies hybrax's agent docs

# Regenerate the demo datasets the tutorials/gallery execute against.
"$PYTHON" "$SRC/_data/generate.py"

LOG="$SCRATCH/build.log"
echo "Build log: $LOG  (tail -f \"$LOG\" in another terminal to watch progress)"
BUILD_START=$(date +%s)
# -j 4: each page's setup cell isolates its own WORK dir and subprocess, so
# parallel pages don't share state, only CPU/memory. 4 matches the cap that
# kept hybrax's own pytest -n from OOM-killing this WSL box; raise with
# caution, not by default.
"$PYTHON" -m sphinx -b html -j 4 -d "$SCRATCH/doctrees" "$SRC" "$OUT" 2>&1 | tee "$LOG"

# myst-nb writes every executed cell's image output flat into jupyter_execute/,
# hash-named for content-addressed dedup (see myst_nb.core.render.render_image) —
# real, not cosmetic: it's what stops a re-run from silently overwriting an image
# a still-live page points at. By the time sphinx-build above returns, Sphinx has
# already copied every image it needs into html/_images/, so moving these into
# their own subdirectory now touches nothing the built site depends on — it only
# keeps them from cluttering jupyter_execute/<page>/*.ipynb for anyone browsing
# the checked-out gallery/tutorial notebooks.
if compgen -G "$JUPYTER_EXECUTE"/*.png > /dev/null; then
    mkdir -p "$JUPYTER_EXECUTE/figures"
    mv "$JUPYTER_EXECUTE"/*.png "$JUPYTER_EXECUTE/figures/"
fi

# Warning gate. We cannot use -W: autoapi renders hybrax's docstrings into RST
# and emits warnings we cannot fix from here (and at least one of them is logged
# without a `type`, so suppress_warnings cannot reach it). Instead, fail on any
# warning that points at a page *we* wrote — dead cross-references included.
if grep -E "WARNING|ERROR" "$LOG" | grep -v "source/autoapi/" > "$SCRATCH/our_warnings.log"; then
    echo
    echo "docs build failed: warnings in hand-written pages" >&2
    cat "$SCRATCH/our_warnings.log" >&2
    exit 1
fi

echo "Built: $OUT/index.html  ($(( $(date +%s) - BUILD_START ))s)"
