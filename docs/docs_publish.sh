#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$ROOT/docs_rebuild.sh"
cd "$ROOT"
git add -A   # source + scripts + rebuilt html/ (generated dirs are gitignored)
git commit -m "docs: rebuild $(date -Is)" || { echo "nothing to publish"; exit 0; }
git push -u origin HEAD
echo "Published. Colleagues: git pull && open html/index.html"
