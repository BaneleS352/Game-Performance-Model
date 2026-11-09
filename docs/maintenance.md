# Maintenance Guide

## Change boundaries

- `data.py` owns source schema, cleaning, and canonical lifecycle columns.
- `features.py` owns snapshot construction and model feature definitions.
- `validation.py` owns temporal partitions, business targets, and regression
  metrics.
- `modeling.py` owns CatBoost training, tuning, calibration, uncertainty, and
  model bundles.
- `scoring.py` owns current scoring, derived outputs, review flags, and
  recommendations.
- `reporting.py` owns holdout calculations and workbook schemas.
- `dashboard.py` owns report-to-dashboard JSON transformation.
- `upload_pipeline.py` owns orchestration, run staging, and publication.
- `server.py` owns HTTP transport and background job state.
- `dashboard/src/main.jsx` owns browser navigation, filters, rendering, and
  upload polling.

Keep changes within the narrowest owner module and update the corresponding
tests and docs.

## Compatibility contracts

1. Preserve the seven required source headers unless a deliberate migration is
   documented.
2. Preserve artifact keys consumed by `scoring.py`, `reporting.py`, and the
   dashboard.
3. Treat workbook sheet names and public column names as user-facing API.
4. Keep future targets out of current scoring features.
5. Preserve chronological splitting and development-only threshold fitting.
6. Maintain atomic publication semantics for live artifacts.
7. Update the dashboard column metadata when adding/removing forecast fields.

## Adding a new metric

1. Add/validate the source field in `data.py`.
2. Add snapshot aggregates and/or ratios in `features.py`.
3. Decide whether the metric is a driver, outcome, classification goal, or
   reporting-only field.
4. Update model target sets, non-negative handling, interval/confidence logic,
   and scoring output.
5. Add holdout/report columns and dashboard metadata.
6. Add unit tests for cleaning, feature leakage, scoring, and export.
7. Update [data and reporting](data-and-reporting.md) and the module docs.

## Adding a new checkpoint or horizon

Update `Settings.candidate_checkpoints`, `forecast_horizon`, and any notebook
or reporting assumptions. Verify that the source has sufficient history and
that run-time training budget is acceptable. Re-run model-family tests and
confirm the selected/available pairs in the output.

## Model changes

Keep quick/standard/production budgets explicit. Record training mode,
selection mode, checkpoint, horizon, training window, target transforms,
calibration method, and interval method in the model bundle/evaluation output.
Do not compare holdout metrics from a run that changed its split logic without
recording the change.

## Security and privacy

The current app is local and unauthenticated. Before network exposure, add
authentication, authorization, request-origin controls, rate limiting, durable
job storage, and secure report access. Review whether pickle/joblib artifacts
are trusted before loading; these formats can execute code when unpickled.

## Documentation synchronization checklist

- Update the relevant module/script/notebook document.
- Update data/reporting and API docs for schema or endpoint changes.
- Update folder inventories for new files.
- Update README links if the entry point changes.
- Run tests/build and record any intentionally changed behavior.
