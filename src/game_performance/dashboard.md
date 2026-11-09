# `dashboard.py`

## Purpose

Transforms the model-analysis workbook into the JSON/JavaScript data contract
consumed by React.

## Key functions

- `build_dashboard_data(workbook_path)`: loads Forecasts and Holdout Comparison,
  keeps the latest checkpoint per game, calculates summary/system/top/review
  datasets, and previews workbook sheets.
- `export_dashboard(workbook_path, output_dir)`: writes `dashboard_data.json`
  and `dashboard_data.js`.

## Processing and outputs

Values are JSON-safe (timestamps become ISO strings, NumPy scalars become native
values, infinities become null). Default preview rows and top/review limits are
250. Systems are sorted by predicted stakes; forecast metadata supplies labels,
types, and default visibility.

## Errors and related modules

Missing workbook/sheets or malformed columns propagate pandas errors. Uses
pandas, NumPy, JSON, and feeds `dashboard/src/main.jsx` and the upload pipeline.
