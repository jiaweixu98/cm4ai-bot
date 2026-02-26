#!/usr/bin/env bash
# Run the FastAPI backend locally using uv
set -e
cd "$(dirname "$0")/backend"

# Create venv + install deps if needed
if [ ! -d ".venv" ]; then
  uv venv --python 3.10
  uv pip install -r requirements.txt
fi

export PYTORCH_ENABLE_MPS_FALLBACK=1
export TOKENIZERS_PARALLELISM=false

echo "Starting backend (without auto-reload to prevent PyTorch MPS segfaults on macOS)..."
uv run uvicorn main:app --host 0.0.0.0 --port 8000
