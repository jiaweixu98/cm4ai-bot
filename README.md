# CM4AI Bot

CM4AI Bot is the matrix recommender application for Bridge2AI.

This repo contains:

- `frontend/`: Next.js UI
- `backend/`: FastAPI service

The companion repo [`bridge2aikg`](https://github.com/jiaweixu98/bridge2aikg) owns the graph application. In the integrated deployment, `bridge2aikg` is mounted at `/` and `cm4ai-bot` is mounted under `/matrix`.

## Before You Start

- This repository does **not** include the large runtime data files required by the backend.
- Download the shared runtime data bundles from [Google Drive](https://drive.google.com/drive/folders/1Nw6ysd0YUuWfkJIpBBtgfCBFZPpaw1vx?usp=sharing).
- By default, the backend reads runtime data from `data/`.
- For this repo, extract `cm4ai-bot-data.tar.gz` into `data/`.
- If you do not set `LOCAL_DATA_DIR`, a normal local run expects these files under `data/`:
  - `updated_author_nodes_with_papers.json`
  - `author_n_publications.json`
  - `author_knowledge_graph_2024.json`
  - `author_ids.pkl`
  - `faiss_index.bin`
  - `author_embeddings.pkl` or `author_embeddings.npy`
- If you develop locally with `bridge2aikg`, note that `bridge2aikg` uses a **mock ORCID session** during `npm run dev`.
- Real ORCID sign-in, callback handling, token exchange, and identity-linked feedback or admin behavior should be tested only in a deployed production-like environment.

## How This Repo Fits With `bridge2aikg`

- `bridge2aikg` handles the graph UI and graph-side APIs.
- `cm4ai-bot` handles the recommender UI and recommendation backend.
- When developing locally, it is simplest to run the two repos as separate apps on separate ports.

## Local Setup

Requirements:

- Python 3
- `python3-venv`
- Node.js 20+
- npm

### Backend setup

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
cp .env.example .env
```

Required in `backend/.env`:

- `OPENAI_API_KEY=...`

Optional:

- `CACHE_DIR=...`
- `LOCAL_DATA_DIR=...`

### Frontend setup

```bash
cd frontend
npm install
cp .env.local.example .env.local
```

## Local Development

Run the backend:

```bash
cd backend
source .venv/bin/activate
uvicorn main:app --reload --port 8000
```

Run the frontend in a second terminal:

```bash
cd frontend
npm install
cp .env.local.example .env.local
npm run dev -- --port 3000
```

Typical local URLs:

- Frontend: `http://localhost:3000`
- Backend: `http://localhost:8000`

The default `frontend/.env.local.example` points:

- `NEXT_PUBLIC_API_URL` at `http://localhost:8000`
- `BRIDGE_REPORT_API_URL` at `http://localhost:5173/api/report-error`

If you are exercising report-feedback flows, run `bridge2aikg` locally too, or change `BRIDGE_REPORT_API_URL` to another reachable graph app instance.

## Local Development With `bridge2aikg`

If your change spans both repos, also run the graph app:

```bash
cd ../bridge2aikg
npm install
cp .env.example .env
npm run dev
```

Typical full local setup:

- `bridge2aikg` on `http://localhost:5173`
- `cm4ai-bot` frontend on `http://localhost:3000`
- `cm4ai-bot` backend on `http://localhost:8000`

## Authentication Modes During Development

When `bridge2aikg` runs in development mode:

- sign-in uses a mock ORCID session
- real ORCID OAuth is not exercised on localhost
- this is expected, because the ORCID client cannot use localhost callback URLs

In production or production-like environments:

- `bridge2aikg` must use real ORCID OAuth
- ORCID callback, token exchange, and identity-linked flows should be verified there

## What To Test Where

Use this split consistently:

- Test on localhost:
  - matrix UI changes
  - API integration with the local backend
  - graph-to-matrix local integration
  - feedback flows against the local graph app
- Test in a deployed production-like environment:
  - real ORCID sign-in
  - callback and token exchange
  - feedback or admin behavior tied to a real ORCID identity

## Collaboration Workflow

Recommended team workflow:

1. Develop and test locally.
2. Push changes to GitHub.
3. Let the OCI operator pull the new commit onto the deployment machine.
4. Validate there before restarting production services.

OCI-specific operations are intentionally documented outside this README so the main repo docs stay collaborator-focused.

## Data and Persistence Notes

- Runtime cache is under `tmp/matrix_cache/`
- This folder is intentionally not tracked in git because it can grow large

## Correct Pipeline Integration

Data generation example:

- `PIPELINE_ROOT=/path/to/correct_pipline_code`
- `DATA_ROOT=/path/to/data-root`

```bash
export PIPELINE_ROOT=/path/to/correct_pipline_code
export DATA_ROOT=/path/to/data-root
```

Runtime data path used by backend:

- `$DATA_ROOT/cm4ai-bot/data`

Canonical pipeline build:

```bash
cd "$PIPELINE_ROOT"
python run_pipeline.py --embedding-model specter2 --layout-method umap
```

Sync command:

```bash
python "$PIPELINE_ROOT/08_sync_to_apps.py"
```

Contract gate command:

```bash
python "$PIPELINE_ROOT/09_smoke_test_apps.py"
```

Release path:

1. `run_pipeline.py`
2. `08_sync_to_apps.py`
3. `09_smoke_test_apps.py`

Cache-first rebuild example:

```bash
cd "$PIPELINE_ROOT"
python 05_build_embeddings.py --embedding-model specter2 --author-agg-mode paper_weighted --devices cuda:1,cuda:2,cuda:3 --batch-size 32 --max-papers-per-author 12 --layout-method umap
python 06_export_bridge2aikg.py
python 07_export_cm4ai_bot.py
python 08_sync_to_apps.py
python 09_smoke_test_apps.py
```

Rollback:

```bash
python "$PIPELINE_ROOT/08_sync_to_apps.py" --snapshot-id <snapshot_id> --no-snapshot
```

Recommended backend env example:

```env
OPENAI_API_KEY=your_real_key
CACHE_DIR=/path/to/cm4ai-bot/tmp/matrix_cache
LOCAL_DATA_DIR=/path/to/cm4ai-bot/data
```

Feedback reports are forwarded to the graph app through `BRIDGE_REPORT_API_URL`, which defaults to the Bridge2AI graph service in local integrated development.

