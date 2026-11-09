# `features.py`

## Purpose

Builds point-in-time lifecycle snapshots and feature matrices without future
data leakage.

## Inputs and outputs

`build_lifecycle_snapshot(data, obs_window, horizon, require_future, as_of_date)`
groups cleaned data by `Game_ID`, aggregates the observation window, and adds
future targets only when required. `prepare_features` returns categorical,
numeric, and missing-indicator columns expected by CatBoost. `build_all_historical_snapshots`
returns a `(checkpoint, horizon) -> DataFrame` mapping.

## Data processing

Features include calendar fields, totals, means, volatility, slopes, growth,
recent/early ratios, weekly aggregates, commercial ratios, log values, and size
bands. Current scoring uses snapshots with `require_future=False`.

## Errors and dependencies

Games lacking the requested observation window/future horizon are skipped. Uses
pandas and NumPy; feeds `modeling.py`, `scoring.py`, and notebooks 02/04.
