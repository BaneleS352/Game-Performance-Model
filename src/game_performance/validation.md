# `validation.py`

## Purpose

Provides chronological splits, rolling folds, business high-performer labels,
training-window restriction, and regression metrics.

## Key functions

- `split_training_validation_holdout`: row-targeted boundaries that keep launch
  dates intact, defaulting to 70/15/15.
- `split_development_and_holdout`: separates the final chronological test dates.
- `rolling_date_splits`: builds expanding temporal folds.
- `create_high_performance_target`: fits 80th-percentile goal thresholds on
  development only and applies them to holdout.
- `regression_metrics`: returns MAE, RMSE, R2, and WAPE.

## Inputs, outputs, errors

Inputs are snapshot DataFrames with `First_Observed_Date` and target columns;
outputs are split DataFrames, metadata, labels, thresholds, or metrics. Invalid
fractions, missing dates, too few unique dates, or unmet row minima raise
`ValueError`. Uses pandas, NumPy, and scikit-learn.
