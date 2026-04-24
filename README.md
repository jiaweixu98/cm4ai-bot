# CM4AI Bot

CM4AI Bot is the matrix recommender application for Bridge2AI.

This repo contains:

- `frontend/`: Next.js UI
- `backend/`: FastAPI service

The companion repo [`bridge2aikg`](https://github.com/jiaweixu98/bridge2aikg) owns the graph application. In the integrated deployment, `bridge2aikg` is mounted at `/` and `cm4ai-bot` is mounted under `/matrix`.

## What lives here

- The published recommendation runtime backed by static data files and a FAISS index.
- The matrix frontend (Next.js) and backend (FastAPI).
- **Bridge integration** with three coupled touch points:
  - Feedback forwarding to the graph app.
  - `POST /api/author-preview` used by bridge to place a newly created catalog author on the atlas (SPECTER2 embedding + FAISS nearest authors).
  - Read-through fallback to the bridge author catalog and collaborator overrides when an author is not in the static export.
- **Saved MATRIX chat sessions** persisted per Bridge-authenticated user in Postgres, restored when the user reopens MATRIX for the same focal author.
- **Stop control** on chat, search, and rerank so a user can abort a run that is in progress.

## Prerequisites

- Python 3
- `python3-venv`
- Node.js 20+
- npm
- PostgreSQL shared with `bridge2aikg`. This service only writes the `matrix_chat_sessions` table; all other tables belong to bridge and are created by bridge itself.
- The shared runtime data bundle for `cm4ai-bot`.

This repository does **not** include the large backend runtime files. By default, the backend reads them from `data/`.

Required runtime files:

- `data/updated_author_nodes_with_papers.json`
- `data/author_n_publications.json`
- `data/author_knowledge_graph_2024.json`
- `data/author_ids.pkl`
- `data/faiss_index.bin`
- `data/author_embeddings.pkl` or `data/author_embeddings.npy`

## Quick start

### Backend

```bash
cd /data/jiawei_data/cm4ai-bot/backend
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
cp .env.example .env
PORT=8000 python main.py
```

### Frontend

```bash
cd /data/jiawei_data/cm4ai-bot/frontend
npm install
cp .env.local.example .env.local
npm run dev -- -H 127.0.0.1 -p 3000
```

Local URLs:

- Frontend: `http://127.0.0.1:3000`
- Backend: `http://127.0.0.1:8000`

## Environment

### `backend/.env`

Copy `backend/.env.example` to `backend/.env`.

Most important settings:

- `OPENAI_API_KEY` — required for chat / query generation / reranking.
- `LOCAL_DATA_DIR` — where the static runtime files live.
- `CACHE_DIR` — where the backend stores local caches.
- `BRIDGE_REPORT_API_URL` — bridge feedback/report endpoint.
- `BRIDGE_CATALOG_API_URL` — bridge catalog API base, usually `http://127.0.0.1:5173/api/catalog`.
- `BRIDGE_COLLABORATORS_API_URL` — bridge collaborator API base, usually `http://127.0.0.1:5173/api/collaborators`.
- `BRIDGE_INTERNAL_API_TOKEN` — shared secret with bridge. Used for two things:
  1. Validates inbound requests from bridge (`x-bridge-api-token`) on `/api/author-preview`.
  2. HMAC secret that verifies MATRIX session tokens (`mx_user_token`) issued by `bridge2aikg`'s `/api/matrix/session-auth`.
  Must exactly match the value configured on the bridge side.
- `DATABASE_URL` — Postgres for persisting MATRIX chat sessions (`matrix_chat_sessions` table). Usually the same `bridge2ai` database used by `bridge2aikg`.

### `frontend/.env.local`

Copy `frontend/.env.local.example` to `frontend/.env.local`.

Most important settings:

- `NEXT_PUBLIC_API_URL=http://127.0.0.1:8000`
- `BRIDGE_REPORT_API_URL=http://127.0.0.1:5173/api/report-error`

## Bridge handoff behavior

When bridge opens MATRIX (via the atlas detail panel or the quick-action), it forwards two query parameters:

- `aid` — the focal author id.
- `mx_user_token` — an HMAC-signed session token issued by bridge's `/api/matrix/session-auth`. Carries the Bridge-authenticated ORCID and a 6-hour expiry.

On first load the frontend:

1. Reads `mx_user_token` from the URL, stashes it in `sessionStorage`, and removes it from the URL bar so it is not accidentally copied/shared.
2. Calls `GET /api/chat-sessions?aid=...` with `x-matrix-user-token` set from that token.
3. Auto-loads the most recent session for that `(aid, owner_orcid)` pair, if any. A fresh session is created the first time the user sends a message.

When MATRIX is opened without a token (directly, or by an unauthenticated visitor), the session sidebar shows a friendly explainer and everything still works in-tab — chats just are not persisted.

## Matrix chat sessions API

Every endpoint requires the `x-matrix-user-token` header; otherwise it returns 401. The token is validated by HMAC against `BRIDGE_INTERNAL_API_TOKEN` (shared with bridge) and must not be expired.

- `GET /api/chat-sessions?aid=...` — list the most recent sessions for the current ORCID and focal author.
- `GET /api/chat-sessions/{session_id}` — full session (messages + state snapshot).
- `POST /api/chat-sessions` — create a new session. Body: `{ aid, focal_author_name, messages, state }`.
- `PUT /api/chat-sessions/{session_id}` — upsert messages + state for a session. Returns 403 if the session is owned by a different ORCID.

Session rows are scoped by `owner_orcid`, so a user can never read or overwrite someone else's sessions even by guessing a UUID.

## Author preview for bridge onboarding

- `POST /api/author-preview` — body: `{ full_name, affiliation, papers: [{ title, journal?, year?, doi?, url? }], top_k? }`.
- When the paper list is non-empty, the backend batch-encodes paper titles with SPECTER2 and uses the mean vector as the author embedding. Otherwise it falls back to a compact "Name / Affiliation / Papers" text representation.
- The embedding is run against the loaded FAISS index and the top nearest static authors are returned with de-duplicated author ids, a normalized similarity score, and name/affiliation metadata.
- When `BRIDGE_INTERNAL_API_TOKEN` is configured, requests must include a matching `x-bridge-api-token` header. Open for local dev otherwise.

## Bridge catalog fallback

The static FAISS path is still the primary path for already-published authors. It is intentionally **not** rebuilt.

When an `aid` shows up that is not in the static export:

- `_get_catalog_author` fetches `GET /api/catalog/authors/{aid}` from bridge and caches the record for 60s.
- `GET /api/author/{aid}` returns a normalized author record instead of 404.
- `_retrieve_candidates` also excludes the focal author's already-known collaborators (resolved via `GET /api/collaborators/{aid}` on bridge), so the matrix recommender never pitches someone you already work with, even when they're a catalog author that the static graph doesn't reach within its 2-hop window.

Together with `/api/author-preview`, this means newly added authors flow through bridge → cm4ai end-to-end without rebuilding the FAISS bundle.

## Local integrated development with `bridge2aikg`

If you are testing bridge-to-matrix handoff, saved sessions, author onboarding, or the new author catalog fallback, run all three services:

1. `cm4ai-bot` backend on `8000`
2. `cm4ai-bot` frontend on `3000`
3. `bridge2aikg` on `5173`

Recommended startup order:

### 1. CM4AI backend

```bash
cd /data/jiawei_data/cm4ai-bot/backend
source .venv/bin/activate
PORT=8000 python main.py
```

### 2. CM4AI frontend

```bash
cd /data/jiawei_data/cm4ai-bot/frontend
npm run dev -- -H 127.0.0.1 -p 3000
```

### 3. bridge2aikg

```bash
cd /data/jiawei_data/bridge2aikg
npm install
cp .env.example .env
npm run dev -- --host 127.0.0.1 --port 5173
```

Expected URLs:

- Bridge graph: `http://127.0.0.1:5173`
- Matrix frontend: `http://127.0.0.1:3000`
- Matrix backend: `http://127.0.0.1:8000`

## Testing locally

### Basic matrix test

1. Start backend and frontend.
2. Open `http://127.0.0.1:3000`.
3. Confirm the app loads and the backend responds.
4. Pick a known static author id and run a search flow.

### Full bridge-to-matrix test

1. Start all three services.
2. Open `http://127.0.0.1:5173`.
3. Click into an author and open "Find Teaming Opportunities".
4. Confirm the matrix modal opens and the backend returns author data.

### Saved sessions

1. From bridge, open an author into MATRIX (so `mx_user_token` is forwarded).
2. Send a few chat messages; watch the saved-sessions sidebar populate on the left.
3. Click "New session" — confirm a fresh chat appears and the old one is still reachable.
4. Reload the tab and confirm the most recent session is auto-restored.

### Stop control

1. Start a search that would normally take a few seconds.
2. Click **Stop** next to Send before it finishes.
3. Confirm the run stops and an "Stopped the current run." message appears, and the UI returns to IDLE (or DONE if candidates were already loaded).

### New author fallback

1. Start all three services plus Postgres for bridge.
2. In `bridge2aikg`, create a new author from `/admin/drafts/...` and promote it.
3. Open that author in MATRIX from the atlas.
4. Confirm:
   - The matrix handoff no longer 404s.
   - Author details in matrix populate from the bridge catalog fallback.
   - Any admin-curated collaborators are excluded from the recommender output.

## What local dev does **not** validate

Localhost is good for:

- matrix UI changes
- local API integration (catalog fallback, preview, collaborator exclusion)
- saved sessions and stop control
- bridge-to-matrix handoff
- feedback forwarding to a local bridge app

You still need a production-like environment for:

- real ORCID OAuth in `bridge2aikg`
- bridge flows tied to non-mock user identity

## Runtime data and persistence

- Static matrix runtime data lives under `data/` unless `LOCAL_DATA_DIR` overrides it.
- Runtime caches live under `tmp/matrix_cache/` unless `CACHE_DIR` overrides it.
- Large runtime data and caches are intentionally not committed to git.
- Postgres is only used for the `matrix_chat_sessions` table on this side.

This repo has two author data paths:

1. **Static published author data** from local files and FAISS.
2. **Bridge catalog fallback** for newly added authors not yet present in the static export.

## Relationship to the pipeline

The expensive full author export and FAISS refresh still come from the external pipeline.

Use the pipeline when you want to regenerate the canonical static matrix data. For day-to-day app development, local author-onboarding tests, and chat-session work, you do **not** need to rerun the full pipeline.

## Reproducing on another machine

For a teammate to reproduce this setup cleanly, share:

- the Git commit SHA for this repo
- the matching Git commit SHA for `bridge2aikg`
- the runtime data bundle version
- a Postgres instance shared with `bridge2aikg`
- a matching `BRIDGE_INTERNAL_API_TOKEN` on both sides (the saved-sessions HMAC secret must agree)
- the `.env.example` / `.env.local.example`-based local setup described above

If those pieces are in place, the local integrated flow should be reproducible without extra unpublished scripts.
