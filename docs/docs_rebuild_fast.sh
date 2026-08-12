#!/usr/bin/env bash
# Fast, incremental companion to docs_rebuild.sh, for local iteration only.
#
# docs_rebuild.sh wipes html/, doctrees/, jupyter_execute/ and the myst-nb
# execution cache on every run, on purpose: myst-nb's cache is keyed on
# {code-cell} source text only, not on the demo data files those cells load,
# so a generate.py change alone does not invalidate it -- an unwiped cache can
# silently keep serving a page's stale numbers/plots after its underlying data
# changed. That correctness guarantee is why the full script exists and stays
# untouched.
#
# This script skips those wipes. Sphinx's own doctree cache and myst-nb's
# execution cache then only rebuild pages whose .md source (or its {code-cell}
# content) actually changed since the last run, which is the entire point:
# editing one gallery page re-executes that one page, not all thirty. It is
# for fast local iteration, not for verifying a change is done.
#
# The one thing to remember: if you edit source/_data/generate.py itself
# (change what a dataset contains) without also touching the .md source of
# every page built on that dataset, this script will not detect the staleness
# -- run the full docs_rebuild.sh instead, or after, in that case.
#
# Always finish with a clean docs_rebuild.sh run before calling anything done.
set -euo pipefail
PYTHON="/home/mgotsmy/anaconda3/envs/bench13/bin/python"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # bp-docs/
SRC="$ROOT/source"; OUT="$ROOT/html"; SCRATCH="$ROOT/_scratch"
JUPYTER_EXECUTE="$ROOT/jupyter_execute"
mkdir -p "$SCRATCH"

# Cheap and deterministic: always safe to rerun, keeps datasets fresh without
# needing a cache wipe to do it.
"$PYTHON" "$SRC/_data/generate.py"

LOG="$SCRATCH/build.log"
"$PYTHON" -m sphinx -b html -d "$SCRATCH/doctrees" "$SRC" "$OUT" 2>&1 | tee "$LOG"

if compgen -G "$JUPYTER_EXECUTE"/*.png > /dev/null; then
    mkdir -p "$JUPYTER_EXECUTE/figures"
    mv "$JUPYTER_EXECUTE"/*.png "$JUPYTER_EXECUTE/figures/"
fi

# Same warning gate as docs_rebuild.sh: speed is not an excuse to lower the bar.
if grep -E "WARNING|ERROR" "$LOG" | grep -v "source/autoapi/" > "$SCRATCH/our_warnings.log"; then
    echo
    echo "docs build failed: warnings in hand-written pages" >&2
    cat "$SCRATCH/our_warnings.log" >&2
    exit 1
fi

echo "Built (incremental): $OUT/index.html"
