# `dashboard/src/` Folder Guide

## Folder purpose

This folder contains the React application and its CSS. It has no direct model
logic; it consumes dashboard JSON produced by the Python package and calls the
local HTTP API for uploads, job status, and report links.

## File inventory

| File | Purpose | Dependencies | Description |
|---|---|---|---|
| `main.jsx` | Application entry and views | React 19, React DOM, Fetch API | Dashboard, upload panel, workbook previews, ledger modes, formatting, and polling. |
| `styles.css` | Presentation layer | CSS | Layout, colors, tables, responsive behavior, and reduced-motion handling. |

## Key processes

`main.jsx` loads the initial data, chooses the dashboard/sheet/upload view,
filters and sorts forecast rows, and renders virtualised ledger rows. It formats
money as ZAR, percentages as locale percentages, and dates/types using metadata
from `dashboard_data.json`.

## Data dependencies

The expected top-level JSON fields are `summary`, `systems`, `columns`,
`forecasts`, `top_stakes`, `top_clients`, `review_candidates`, and
`workbook_sheets`. Required upload headers are repeated in the UI for user
guidance.

## Related modules

- Python producer: [`game_performance/dashboard.py`](../../src/game_performance/dashboard.md)
- API: [`game_performance/server.py`](../../src/game_performance/server.md)
