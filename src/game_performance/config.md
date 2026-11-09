# `config.py`

## Purpose

Centralises immutable default settings and derives artifact directories from the
project root.

## Inputs and outputs

`GAME_DATA_FILE` overrides `Settings.source_file`. `Settings` exposes paths for
processed data, features, models, and reports, plus model fractions,
checkpoints, horizon, thresholds, calibration, and training defaults.

## Key functions

- `Settings.horizon_strategy()` maps each checkpoint to the configured horizon.
- `Settings.ensure_directories()` creates output directories.
- `SETTINGS` is the module-level default instance.

## Errors and related modules

Relative source paths are resolved against the project root. Invalid settings
are generally detected by downstream validation/modeling. Used by notebooks,
`upload_pipeline.py`, `server.py`, and `modeling.py`.
