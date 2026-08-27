#!/usr/bin/env bash
set -euo pipefail
PYTHON="/home/mgotsmy/code/bpbench/hybrax/.venv/bin/python"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # hybrax/docs/
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
$PYTHON "$SRC/_data/generate.py"

# Keep hybrax/examples/*'s frozen demo-data snapshots in sync with the live
# generator above. These files stay checked into git (examples/ works
# standalone, without docs tooling), but every full rebuild refreshes them
# from generate.py's current output, so a demo-data change never goes stale
# unnoticed: if a rebuild changes something, it shows up as a normal
# git diff in examples/ for review.
OUT_DATA="$SRC/_data/out"
EXAMPLES="$ROOT/../examples"
sync_example() {
    local dataset="$1" example="$2"
    shift 2
    for f in "$@"; do
        [ -f "$OUT_DATA/$dataset/$f" ] && cp "$OUT_DATA/$dataset/$f" "$EXAMPLES/$example/$f"
    done
}
sync_example demo_batch tutorial_02_look_at_it data.json
sync_example demo_batch tutorial_03_train data.json
sync_example demo_batch tutorial_04_your_first_custom_py data.json
sync_example demo_batch tutorial_05_predict data.json ground_truth.json
cp "$OUT_DATA/demo_batch/raw/offline.csv" "$EXAMPLES/tutorial_01_your_first_dataset/offline.csv"
sync_example demo_batch gallery_dense_loss data.json
sync_example demo_batch gallery_mechanistic_rates data.json ground_truth.json
sync_example demo_batch gallery_freezing data.json
sync_example demo_batch gallery_gaussian_process data.json
sync_example demo_batch gallery_kan data.json
sync_example demo_batch gallery_stateful data.json
sync_example demo_batch gallery_loo data.json
sync_example demo_fedbatch gallery_fed_batch data.json
sync_example demo_continuous_overflow gallery_continuous_overflow data.json ground_truth.json
sync_example demo_fedbatch gallery_augmentation data.json
sync_example demo_products gallery_knowledge_transfer data.json
sync_example demo_ecoli_fba gallery_fba_hyb data.json
sync_example demo_ecoli_blend gallery_pls_dfba data.json
sync_example demo_optfed gallery_optfed data.json ground_truth.json
sync_example demo_glutamine_decay gallery_glutamine_decay data.json ground_truth.json
sync_example demo_modeled_pv gallery_modeled_pv data.json ground_truth.json
sync_example demo_spline_jump gallery_pseudobatch_splines data.json

LOG="$SCRATCH/build.log"
echo "Build log: $LOG  (tail -f \"$LOG\" in another terminal to watch progress)"
BUILD_START=$(date +%s)
# -j 4: each page's setup cell isolates its own WORK dir and subprocess, so
# parallel pages don't share state, only CPU/memory. 4 matches the cap that
# kept hybrax's own pytest -n from OOM-killing this WSL box; raise with
# caution, not by default.
$PYTHON -m sphinx -b html -j 4 -d "$SCRATCH/doctrees" "$SRC" "$OUT" 2>&1 | tee "$LOG"

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
if grep -E "WARNING|ERROR" "$LOG" | grep -v "source/autoapi/" | grep -v "\[IPKernelApp\]" > "$SCRATCH/our_warnings.log"; then
    echo
    echo "docs build failed: warnings in hand-written pages" >&2
    cat "$SCRATCH/our_warnings.log" >&2
    exit 1
fi

echo "Built: $OUT/index.html  ($(( $(date +%s) - BUILD_START ))s)"
