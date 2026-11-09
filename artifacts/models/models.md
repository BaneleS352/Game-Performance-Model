# `artifacts/models/` Folder Guide

## Folder purpose

Stores the latest serialized training results and model bundles.

## File inventory

| File | Purpose |
|---|---|
| `training_results.joblib` | Model bundles, evaluations, holdout predictions, selection, calibration, and metadata. |
| `models.md` | Folder guide. |

Produced by notebook 03 or `run_full_production_pipeline`; consumed by scoring
and the score-only upload path.
