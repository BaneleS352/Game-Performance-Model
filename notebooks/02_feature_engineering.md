# `02_feature_engineering.ipynb`

## Purpose and location

Construct leakage-safe historical lifecycle snapshots from the canonical daily
pickle.

## Inputs and outputs

Input: `artifacts/processed/clean_game_daily.pkl`. Output:
`artifacts/features/historical_snapshots.joblib`, a mapping keyed by
`(checkpoint, horizon)`.

## Dependencies and process

Loads `Settings.horizon_strategy`, calls `build_all_historical_snapshots`,
prints snapshot sizes, categorical features, and model feature count, then
persists the mapping. Future columns exist only in historical snapshots used for
training.

## Errors and related notebook

Missing/invalid cleaned data or insufficient history can produce empty snapshots
or downstream training failures. Run after notebook 01 and before notebook 03.
