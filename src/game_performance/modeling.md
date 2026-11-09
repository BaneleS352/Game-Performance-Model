# `modeling.py`

## Purpose

Implements CatBoost model-family training, temporal tuning, regression/classification
evaluation, probability calibration, confidence, prediction intervals, and
model-bundle assembly.

## Inputs and outputs

Primary entry points are `train_all_model_families`,
`train_selected_model_configuration`, and `select_best_configuration`. Inputs
are historical snapshot mappings and `Settings`; outputs include model bundles,
regression/classification evaluation, holdout predictions, calibration,
uncertainty, tuning, selection, and skipped configurations.

## Processing logic

Training modes are `quick`, `standard`, and `production`. Regressors use
commercial drivers (`clients`, bets/client, stake/bet, GGR margin), derive
outcomes, log-transform non-negative targets, blend against baselines, and fit
residual/conformal intervals. Classifiers target client/stakes high performers,
may use isotonic probability calibration, and select validation thresholds with
a production floor of 0.80.

## Errors and dependencies

Insufficient rows/classes/folds or unavailable configurations are recorded or
raise `ValueError`. Uses CatBoost, scikit-learn, pandas, NumPy, and internal
features/validation/config modules. Feeds `scoring.py` and reporting.
