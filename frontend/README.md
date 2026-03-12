# CM4AI Frontend (Next.js)

This is the frontend for `cm4ai-bot` and is deployed behind `/matrix` on the same host as `bridge2aikg`.

## Local Development

```bash
cd /data/jiawei_data/cm4ai-bot/frontend
npm ci
NEXT_PUBLIC_API_URL=http://127.0.0.1:8001 npm run dev -- -H 127.0.0.1 -p 3000
```

Open:

- `http://127.0.0.1:3000`

## Build

```bash
cd /data/jiawei_data/cm4ai-bot/frontend
npm ci
npm run build
```

## Runtime Dependency

Frontend API requests depend on backend data files synced from the canonical pipeline:

1. `python /data/jiawei_data/correct_pipline_code/run_pipeline.py --embedding-model specter2 --layout-method umap`
2. `python /data/jiawei_data/correct_pipline_code/08_sync_to_apps.py`
3. `python /data/jiawei_data/correct_pipline_code/09_smoke_test_apps.py`

Paper embedding cache reuse is handled by pipeline step 05 and does not require frontend changes.

`09_smoke_test_apps.py` may print warning-only lines (`dataset_count=0`, filtered invalid IDs) while still passing.
