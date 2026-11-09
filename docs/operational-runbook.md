# Operational Runbook

## Fresh setup

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -e ".[dev]"
cd dashboard
pnpm install
```

Use `source .venv/bin/activate` and `.venv/bin/python` on macOS/Linux.

## Start development services

```powershell
cd dashboard
pnpm dev
```

This starts Vite on `127.0.0.1:5173`, proxies `/api` and `/runs` to the Python
server, and starts `python -m game_performance.server` on port 8000.

## Build a static dashboard

Run the Python export against a completed model report, then build:

```powershell
python -m game_performance.dashboard
cd dashboard
pnpm build
```

The result is under `artifacts/dashboard/`. The build cleans only the
`artifacts/dashboard/assets` subtree before compiling.

## Run notebooks in order

1. `01_data_exploration.ipynb` — source configuration, cleaning, quality, and
   `artifacts/processed/clean_game_daily.pkl`.
2. `02_feature_engineering.ipynb` — snapshots and
   `artifacts/features/historical_snapshots.joblib`.
3. `03_model_experiments.ipynb` — production training and
   `artifacts/models/training_results.joblib`.
4. `04_training_and_forecasting.ipynb` — scoring and report exports.

Set `GAME_DATA_FILE` or edit the notebook source-file cell before running.

## Verification checklist

- `python -m pytest` passes.
- `python -m compileall src` passes.
- `pnpm build` passes from `dashboard/`.
- `artifacts/dashboard/dashboard_data.json` exists after a successful run.
- `GET /api/dashboard-data` returns HTTP 200.
- A sample upload returns worksheet names.
- A run reaches `completed` and both report download URLs return files.
- Report sheets contain the expected business/technical outputs.

## Support triage

1. Capture the browser message, upload name, worksheet, job id, and timestamp.
2. Inspect the Python API terminal for the detailed exception.
3. Query `GET /api/runs/{job_id}` to distinguish queued, running, and failed.
4. Check input headers, date coverage, unique launch dates, and row counts.
5. Confirm the expected artifact path exists under `artifacts/uploads/{run_id}`.
6. Only retry after identifying whether the failure is bad input, insufficient
   history, an environment issue, or a code/model issue.

## Safe cleanup

Generated artifacts and uploaded workbooks may be commercially sensitive. Do
not delete them blindly. Follow the owner’s retention policy and preserve the
run directory when it is needed for audit or support evidence.

## Port and process checks

If the dashboard cannot reach the API, check that port 8000 is available and
that the Vite child process started the same Python environment that installed
the package. Override the Python executable with `PYTHON` in Vite if needed.
