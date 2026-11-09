# `src/game_performance/` Folder Guide

## Folder purpose

Reusable Python implementation of the game-performance forecasting platform.
The package is the source of truth for ingestion, feature engineering, model
training, scoring, reporting, dashboard serialization, upload orchestration,
and the local HTTP API.

## File inventory

| File | Purpose | Dependencies | Description |
|---|---|---|---|
| `__init__.py` | Package identity | None | Exposes package version. |
| `config.py` | Settings | stdlib | Paths, thresholds, checkpoints, and defaults. |
| `data.py` | Ingestion/cleaning | pandas, NumPy | Schema validation and canonical daily data. |
| `features.py` | Feature engineering | pandas, NumPy | Point-in-time snapshots and feature matrices. |
| `validation.py` | Temporal validation | pandas, NumPy, scikit-learn | Splits, targets, folds, metrics. |
| `impact.py` | Impact detection | pandas, NumPy | Inferred spike/drop periods and forecast flags. |
| `modeling.py` | Model training | CatBoost, scikit-learn, pandas, NumPy | Tuning, fitting, calibration, uncertainty, evaluation. |
| `scoring.py` | Forecast scoring | pandas, NumPy | Model inference, derived outcomes, review, recommendations. |
| `reporting.py` | Report generation | pandas, NumPy, openpyxl | Holdout comparisons and Excel workbooks. |
| `dashboard.py` | Dashboard data | pandas, NumPy | JSON/JS serialization and previews. |
| `upload_pipeline.py` | Production orchestration | Package modules, pandas | Runs the complete uploaded-workbook flow and publishes artifacts. |
| `server.py` | HTTP API | Python stdlib, package modules | Upload, background jobs, status, data, and report endpoints. |
| `io.py` | Persistence | joblib, pandas | Pickle and joblib helpers. |
| `*.md` | Module guides | Markdown | Dedicated documentation for each module. |

## Execution order

`server.py` → `upload_pipeline.py` → `data.py` → `impact.py` and `features.py`
→ `modeling.py` → `scoring.py` → `reporting.py` → `dashboard.py`.
`config.py` and `io.py` support the entire flow; `validation.py` supports
training and reporting.

## Related services and structures

The package has no database or external service. Its primary structures are
Pandas DataFrames, `(checkpoint, horizon)` snapshot mappings, model bundles,
run dictionaries, and dashboard JSON. See [Data flow](../../docs/data-flow.md)
and [Module reference](../../docs/module-reference.md).
