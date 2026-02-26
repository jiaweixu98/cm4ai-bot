# ---------- Stage 1: build deps ----------
FROM python:3.10-slim AS builder

WORKDIR /app

# System deps for faiss-cpu, numpy, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------- Stage 2: runtime ----------
FROM python:3.10-slim

WORKDIR /app

# Runtime deps only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY . .

# Env defaults
ENV PYTORCH_ENABLE_MPS_FALLBACK=1 \
    TOKENIZERS_PARALLELISM=false \
    PORT=8000

EXPOSE 8000

# Railway injects PORT; use it
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT}
