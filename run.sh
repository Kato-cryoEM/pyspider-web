#!/usr/bin/env bash
# One-shot launcher: FastAPI serves both the pre-built React UI and the API
# on http://127.0.0.1:8000. No Node.js required.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

PORT="${PORT:-8000}"

if [ ! -d frontend/dist ]; then
  cat >&2 <<EOF
[error] frontend/dist not found.

This distribution normally ships with a pre-built UI. If you removed it or
you're working from source, rebuild with:

    cd frontend && npm install && npm run build

(needs Node.js 18+)
EOF
  exit 1
fi

echo "==> starting pyspider-web on http://127.0.0.1:$PORT"
echo "==> open http://localhost:$PORT/  (Ctrl-C to stop)"
PYTHONPATH="$HERE" exec python3 -m uvicorn backend.main:app \
  --host 127.0.0.1 --port "$PORT"
