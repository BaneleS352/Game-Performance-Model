# `03_model_experiments.ipynb`

## Purpose and location

Run rolling temporal validation, tune representative model configurations, fit
all production checkpoint/horizon families, and persist their evaluations.

## Inputs and outputs

Input: `artifacts/features/historical_snapshots.joblib`. Output:
`artifacts/models/training_results.joblib`, containing model bundles, regression
and classification evaluation, holdout predictions, calibration, and skipped
configuration details.

## Dependencies and process

Calls `train_all_model_families` in production mode, with regression and
classification enabled and no family cap. It prints available configurations,
trained families, evaluation shapes, and skipped configurations.

## Error handling

Insufficient rows or unique dates can prevent valid temporal splits and cause
training errors. Inspect `skipped_configurations` before treating a missing
checkpoint/horizon as a model bug. Run before notebook 04.
