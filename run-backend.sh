#!/usr/bin/env bash
# Run FastAPI backend the same way as systemd.
set -e
cd "$(dirname "$0")/backend"

# Create venv + install deps if needed (first-time bootstrap)
if [ ! -x ".venv/bin/python" ]; then
  python3 -m venv .venv
  .venv/bin/pip install --upgrade pip setuptools wheel
  .venv/bin/pip install -r requirements.txt
fi

# Load environment variables if backend/.env exists
if [ -f ".env" ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi

export TOKENIZERS_PARALLELISM=false

echo "Starting backend on 127.0.0.1:8000 ..."
exec .venv/bin/uvicorn main:app --host 127.0.0.1 --port 8000
