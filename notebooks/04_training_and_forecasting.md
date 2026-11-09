# `04_training_and_forecasting.ipynb`

## Purpose and location

Load the cleaned data/model bundle, generate current lifecycle forecasts, build
the preferred 14-day-to-7-day holdout comparison when available, and export
stakeholder outputs.

## Inputs and outputs

Inputs: `artifacts/processed/clean_game_daily.pkl` and
`artifacts/models/training_results.joblib`. Outputs:
`artifacts/reports/current_predictions.pkl` and
`artifacts/reports/model_business_report.xlsx`.

## Dependencies and process

Uses `run_daily_lifecycle_scoring`, `build_holdout_comparison`, and `export_excel`.
The preferred `(14, 7)` pair is used if available; otherwise the last available
pair is selected with a warning.

## Error handling

An empty holdout raises a clear error asking for notebook 03 first. If the
preferred pair is absent, inspect skipped configurations and verify that the
source contains enough history.
