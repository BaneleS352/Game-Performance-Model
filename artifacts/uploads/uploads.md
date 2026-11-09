# `artifacts/uploads/` Folder Guide

## Folder purpose

Run-scoped audit directories created by uploaded-workbook scoring.

## Folder inventory

Each `<run_id>/` contains some or all of:

| Pattern | Purpose |
|---|---|
| `clean_game_daily.pkl` | Run-specific cleaned input. |
| `historical_snapshots.joblib` | Run-specific feature snapshots. |
| `training_results.joblib` | Run-specific model/evaluation bundle. |
| `current_predictions.pkl` | Run-specific forecasts. |
| `scoring_log.pkl` | Deduplication/lineage log. |
| `Business Report.xlsx` | Concise business output. |
| `Model Analysis.xlsx` | Complete technical output. |
| `dashboard/` | Staged dashboard JSON/JS before publication. |
| `<run_id>.md` | Optional local run guide if a run is retained as a manually curated audit package. |

Run ids are UTC timestamp plus a short random suffix. These directories are
immutable audit snapshots after the pipeline returns, except for manual
retention/cleanup operations.
