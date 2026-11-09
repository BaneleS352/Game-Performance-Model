# `notebooks/` Folder Guide

## Folder purpose

Thin orchestration notebooks for exploration, feature generation, model
experiments, and forecast/report export. Production logic belongs in the
reusable `game_performance` package.

## File inventory

| File | Purpose | Dependencies | Description |
|---|---|---|---|
| `01_data_exploration.ipynb` | Clean and inspect input | `data.py`, `config.py`, `io.py` | Writes canonical daily data. |
| `02_feature_engineering.ipynb` | Build snapshots | `features.py`, `io.py` | Writes historical snapshot joblib. |
| `03_model_experiments.ipynb` | Train/evaluate models | `modeling.py`, `io.py` | Writes model results/bundles. |
| `04_training_and_forecasting.ipynb` | Score and export | `scoring.py`, `reporting.py`, `io.py` | Writes predictions and Excel report. |
| `notebooks.md` | Folder guide | Markdown | Execution order and notebook ownership. |

## Execution order and data dependencies

Run notebooks in numeric order. Each consumes the persisted artifact from the
previous stage. Set `GAME_DATA_FILE` or update the source cell before notebook
01; do not assume the historical absolute path is available on another machine.

## Related documentation

- [Platform workflow](../docs/platform-overview.md)
- [Operational runbook](../docs/operational-runbook.md)
