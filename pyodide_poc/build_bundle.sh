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

# pyspider core
rsync -a --exclude '__pycache__' --exclude '*.pyc' \
  /Applications/spider/pyspider/pyspider "$STAGE/"

(cd "$STAGE" && zip -qr "$OLDPWD/$OUT" backend pyspider)
echo "Built $OUT"
ls -lh "$OUT"
