#!/usr/bin/env bash
# Run the CM4AI Streamlit app locally (CPU-only, no MPS)
# PYTORCH_ENABLE_MPS_FALLBACK=1  -- prevents segfault on Apple Silicon by falling back to CPU
# TOKENIZERS_PARALLELISM=false   -- avoids tokenizer warning in forked processes
# --server.fileWatcherType none  -- avoids hot-reload fork conflicts with PyTorch
PYTORCH_ENABLE_MPS_FALLBACK=1 \
TOKENIZERS_PARALLELISM=false \
.venv/bin/streamlit run pages/CM4AI_Search.py \
  --server.fileWatcherType none \
  "$@"
