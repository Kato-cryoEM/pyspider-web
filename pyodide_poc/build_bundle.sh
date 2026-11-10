#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="pyodide_poc/bundle.zip"
rm -f "$OUT"

# Copy pyspider core alongside backend/ in a temp stage
STAGE=$(mktemp -d)
trap "rm -rf $STAGE" EXIT

# backend/ (keep bundled_data — needed by several challenges + the β-gal
# volume loader). Exclude the vite build output `_static/` which isn't
# used by the Pyodide build.
rsync -a --exclude '__pycache__' --exclude '*.pyc' \
  --exclude '_static' \
  backend "$STAGE/"

# pyspider core — prefer the sibling copy inside the repo (so CI /
# GitHub Actions work), fall back to the dev-machine absolute path.
if [ -d pyspider ]; then
  PYSPIDER_SRC="pyspider"
elif [ -d /Applications/spider/pyspider/pyspider ]; then
  PYSPIDER_SRC="/Applications/spider/pyspider/pyspider"
else
  echo "ERROR: cannot find pyspider/ core package." >&2
  exit 1
fi
rsync -a --exclude '__pycache__' --exclude '*.pyc' \
  "$PYSPIDER_SRC" "$STAGE/"

(cd "$STAGE" && zip -qr "$OLDPWD/$OUT" backend pyspider)
echo "Built $OUT"
ls -lh "$OUT"
