#!/usr/bin/env bash
# Start the view server, replacing any instance already on the port.
set -euo pipefail
cd "$(dirname "$0")"
PORT="${PORT:-8842}"
PY="${OMEZARR_VIEW_PYTHON:-/data/nmechtel/bioengine/.dev/omezarr-probe/.venv/bin/python}"
if old=$(lsof -t -i:"$PORT" 2>/dev/null); then kill $old 2>/dev/null || true; sleep 2; fi
PYTHONPATH=. PORT="$PORT" nohup "$PY" -m omezarr_view.server > server.log 2>&1 &
echo "started pid $! on port $PORT"
