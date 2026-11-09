# `server.py`

## Purpose

Provides the local threaded HTTP server used by the dashboard for uploads,
worksheet discovery, background production jobs, status polling, dashboard data,
and report downloads.

## Inputs and outputs

`create_server(host, port)` builds a `ThreadingHTTPServer`; `main()` parses
`--host`/`--port`. `DashboardRequestHandler` implements the API documented in
[`docs/api-reference.md`](../../docs/api-reference.md). `UploadState` keeps
uploads and job dictionaries with a lock.

## Processing and errors

Upload bodies are bounded at 65 MB and require consistent Content-Length
headers. Expected `ValueError`/JSON/file errors map to 400/404; unexpected
exceptions map to 500 and are logged. Jobs run in daemon threads and transition
through queued/running/completed/failed.

## Dependencies and related modules

Uses only Python stdlib plus `config.py` and `upload_pipeline.py`. Vite starts
it as `python -m game_performance.server`.
