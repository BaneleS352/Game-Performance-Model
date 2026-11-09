# Local API Reference

The API is implemented by `src/game_performance/server.py` and is intended for
the local Vite dashboard. It binds to `127.0.0.1:8000` by default.

## `GET /api/dashboard-data`

Returns the latest `artifacts/dashboard/dashboard_data.json`.

- `200`: dashboard data object.
- `404`: no successful export has published dashboard data yet.

## `POST /api/uploads`

Uploads a workbook.

Headers:

- `X-File-Name`: URL-encoded original file name.
- `Content-Length`: required and no larger than 65 MB.

Body: raw file bytes.

Response `201`:

```json
{
  "upload_id": "20260925T065305Z_a28c0b8e",
  "sheets": ["Sheet1", "Sheet2"]
}
```

The server stores the file as `data/uploads/{upload_id}_{safe_name}`.

## `POST /api/uploads/{upload_id}/run`

Starts a background production run.

Body:

```json
{ "sheet_name": "Sheet1" }
```

Response `202` contains a job object with `job_id`, `status`, `progress`,
`message`, `upload_id`, `upload_name`, and `sheet_name`.

## `GET /api/runs/{job_id}`

Returns the current in-memory job state.

Completed responses include a `result` object with `run_id`, summary, artifact
paths, and download URLs. Failed responses include `error` and `message`.

## `GET /runs/{run_id}/business-report.xlsx`

Downloads the run’s concise stakeholder workbook.

## `GET /runs/{run_id}/model-analysis.xlsx`

Downloads the run’s complete technical workbook.

## Static content

All other GET requests are delegated to `SimpleHTTPRequestHandler` serving the
dashboard directory. Vite’s development server serves the frontend and proxies
API/report requests to this backend.

## Operational notes

There is no authentication, CORS policy, rate limiter, persistent job store,
or cross-process coordination. Do not expose the default server to an
untrusted network without adding those controls.
