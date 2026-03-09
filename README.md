# CM4AI Bot (OCI Deployment)

This project runs as:

- Frontend: Next.js (`frontend/`) on port `3000`
- Backend: FastAPI (`backend/`) on port `8000`
- Reverse proxy: Nginx (`bridge2ai.labs-app.com`)

The app is designed to be maintained directly on an OCI server, with GitHub used as source backup and change history.

## 1) Current Architecture

- User visits `https://bridge2ai.labs-app.com`
- Nginx routes:
  - `/api/*` -> `127.0.0.1:8000` (FastAPI)
  - `/*` -> `127.0.0.1:3000` (Next.js)
- FastAPI loads local cache data from:
  - `tmp/matrix_cache/`

## 2) Prerequisites on Server

- Ubuntu/Linux
- `python3`, `python3-venv`, `pip`
- `node` + `npm` (Node 22 LTS recommended)
- `nginx`

## 3) Backend Setup

```bash
cd /home/ubuntu/cm4ai-bot/backend
python3 -m venv .venv
.venv/bin/pip install --upgrade pip setuptools wheel
.venv/bin/pip install -r requirements.txt
cp .env.example .env
```

Required in `backend/.env`:

- `OPENAI_API_KEY=...`

Optional:

- `CACHE_DIR=...` (default: `/home/ubuntu/cm4ai-bot/tmp/matrix_cache`)
- `LOCAL_DATA_DIR=...` (default: `/home/ubuntu/cm4ai-bot/data`)

## 4) Frontend Setup

```bash
cd /home/ubuntu/cm4ai-bot/frontend
npm ci
npm run build
```

## 5) Run (Manual)

Manual scripts are aligned with production startup:

```bash
cd /home/ubuntu/cm4ai-bot
./run-backend.sh
./run-frontend.sh
```

## 6) Run (Production, Recommended)

Use `systemd` services:

- `cm4ai-backend.service`
- `cm4ai-frontend.service`

Useful commands:

```bash
systemctl status cm4ai-backend cm4ai-frontend
sudo systemctl restart cm4ai-backend cm4ai-frontend
journalctl -u cm4ai-backend -f
journalctl -u cm4ai-frontend -f
```

## 7) Nginx

Nginx config file:

- `/etc/nginx/sites-available/default`

Validate + reload after changes:

```bash
sudo nginx -t
sudo systemctl reload nginx
```

## 8) GitHub Backup Workflow

Recommended daily workflow on OCI:

```bash
cd /home/ubuntu/cm4ai-bot
git status
git add -A
git commit -m "your message"
git push origin main
```

To check if local changes are already pushed:

```bash
git status -sb
git branch -vv
```

## 9) Upgrade / Deploy Checklist

1. Pull latest code:
   - `git pull --rebase origin main`
2. Backend deps:
   - `backend/.venv/bin/pip install -r backend/requirements.txt`
3. Frontend build:
   - `cd frontend && npm ci && npm run build`
4. Restart services:
   - `sudo systemctl restart cm4ai-backend cm4ai-frontend`
5. Verify:
   - `curl http://127.0.0.1:8000/api/health`
   - `curl -I http://127.0.0.1:3000`
   - `curl -k https://bridge2ai.labs-app.com/api/health`

## 10) Data and Persistence Notes

- Runtime cache is under `tmp/matrix_cache/`
- This folder is intentionally not tracked in git (large files)
- For disaster recovery, back up:
  - repo (`/home/ubuntu/cm4ai-bot`)
  - cache (`/home/ubuntu/cm4ai-bot/tmp/matrix_cache`)
  - nginx config (`/etc/nginx/sites-available/default`)
  - systemd unit files (`/etc/systemd/system/cm4ai-*.service`)

