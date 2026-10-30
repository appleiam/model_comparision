#!/usr/bin/env bash
# Starts the FastAPI backend (port 8000) and the Vite frontend (port 5173) together. Ctrl+C stops both.
set -euo pipefail
cd "$(dirname "$0")"

PY=backend/.venv/bin/python
if [ ! -x "$PY" ]; then
  echo "[setup] creating backend/.venv ..."
  python3 -m venv backend/.venv
  "$PY" -m pip install --upgrade pip
  "$PY" -m pip install -r backend/requirements.txt
fi
if [ ! -d frontend/node_modules ]; then
  echo "[setup] npm install ..."
  (cd frontend && npm install)
fi

(cd backend && exec .venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000) &
BACK=$!
(cd frontend && exec npm run dev) &
FRONT=$!
trap 'kill $BACK $FRONT 2>/dev/null' EXIT INT TERM

echo "Backend:  http://127.0.0.1:8000/system"
echo "Frontend: http://localhost:5173"
wait
