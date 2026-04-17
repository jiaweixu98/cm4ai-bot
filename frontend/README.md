# CM4AI Frontend (Next.js)

This package is the frontend for [`cm4ai-bot`](../README.md). The companion repo [`bridge2aikg`](https://github.com/jiaweixu98/bridge2aikg) owns the graph application that links into the matrix flow.

## Local Development

Create a local env file:

```bash
cp .env.local.example .env.local
```

Start the frontend:

```bash
npm install
npm run dev -- --port 3000
```

Default local URL:

- `http://localhost:3000`

By default, `frontend/.env.local.example` points:

- `NEXT_PUBLIC_API_URL` to `http://localhost:8000`
- `BRIDGE_REPORT_API_URL` to `http://localhost:5173/api/report-error`

So the backend should be running on `8000`, and the graph app should be running on `5173` if you want to test report-feedback flows end to end.

## Local Development With Backend

Run the backend in another terminal:

```bash
cd ../backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn main:app --reload --port 8000
```

If you are testing the integrated Bridge2AI flow, also run `bridge2aikg` separately on its default dev port (`5173`).

In local development, `bridge2aikg` uses a mock ORCID session instead of real ORCID OAuth. Real ORCID login must be verified in a deployed environment.

## Build

```bash
npm install
npm run build
```

## Runtime Dependency

Frontend API requests depend on backend data files synced from the canonical pipeline.

Example:

- `PIPELINE_ROOT=/path/to/correct_pipline_code`

1. `python "$PIPELINE_ROOT/run_pipeline.py" --embedding-model specter2 --layout-method umap`
2. `python "$PIPELINE_ROOT/08_sync_to_apps.py"`
3. `python "$PIPELINE_ROOT/09_smoke_test_apps.py"`

Paper embedding cache reuse is handled by pipeline step 05 and does not require frontend changes.

`09_smoke_test_apps.py` may print warning-only lines (`dataset_count=0`, filtered invalid IDs) while still passing.
