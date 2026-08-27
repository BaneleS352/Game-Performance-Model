from __future__ import annotations

import time
from typing import Any, Iterable, Literal

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from .config import Settings
from .features import REGRESSION_TARGETS, categorical_features, prepare_features
from .validation import (
    create_high_performance_target,
    regression_metrics,
    rolling_date_splits,
    split_development_and_holdout,
)

TrainingMode = Literal["quick", "standard", "production"]

NON_NEGATIVE_TARGETS = {"target_clients", "target_stakes"}

REGRESSOR_PARAMETER_CANDIDATES = [
    {"_target_transform": "raw", "depth": 5, "learning_rate": 0.05, "l2_leaf_reg": 5, "random_strength": 1.0, "bagging_temperature": 0.5},
    {"_target_transform": "raw", "depth": 6, "learning_rate": 0.04, "l2_leaf_reg": 6, "random_strength": 1.0, "bagging_temperature": 1.0},
    {"_target_transform": "raw", "depth": 7, "learning_rate": 0.03, "l2_leaf_reg": 8, "random_strength": 0.5, "bagging_temperature": 1.0},
    {"_target_transform": "log", "depth": 5, "learning_rate": 0.05, "l2_leaf_reg": 5, "random_strength": 1.0, "bagging_temperature": 0.5},
    {"_target_transform": "log", "depth": 6, "learning_rate": 0.04, "l2_leaf_reg": 6, "random_strength": 1.0, "bagging_temperature": 1.0},
    {"_target_transform": "log", "depth": 7, "learning_rate": 0.03, "l2_leaf_reg": 8, "random_strength": 0.5, "bagging_temperature": 1.0},
]

CLASSIFIER_PARAMETER_CANDIDATES = [
    {"depth": 5, "learning_rate": 0.05, "l2_leaf_reg": 5, "random_strength": 1.0, "bagging_temperature": 0.5},
    {"depth": 6, "learning_rate": 0.04, "l2_leaf_reg": 7, "random_strength": 1.0, "bagging_temperature": 1.0},
]

TRAINING_BUDGETS: dict[TrainingMode, dict[str, int]] = {
    # Use this while debugging notebooks. It should finish much faster on a laptop.
    "quick": {
        "regressor_iterations": 250,
        "classifier_iterations": 200,
        "early_stopping_rounds": 25,
        "max_tuning_candidates": 1,
        "max_families": 2,
    },
    # Balanced default for local model development.
    "standard": {
        "regressor_iterations": 500,
        "classifier_iterations": 400,
        "early_stopping_rounds": 50,
        "max_tuning_candidates": 2,
        "max_families": 0,
    },
    # Full search budget. Use for the final training run, not for notebook debugging.
    "production": {
        "regressor_iterations": 900,
        "classifier_iterations": 700,
        "early_stopping_rounds": 100,
        "max_tuning_candidates": 0,
        "max_families": 0,
    },
}


def _setting(settings: Settings, name: str, default: Any) -> Any:
    """Read an optional Settings attribute without requiring a config.py migration."""
    return getattr(settings, name, default)


def _training_mode(settings: Settings, override: TrainingMode | None = None) -> TrainingMode:
    mode = override or _setting(settings, "training_mode", "quick")
    if mode not in TRAINING_BUDGETS:
        raise ValueError(f"training_mode must be one of {sorted(TRAINING_BUDGETS)}; got {mode!r}.")
    return mode  # type: ignore[return-value]


def _budget(settings: Settings, mode: TrainingMode | None = None) -> dict[str, int]:
    return TRAINING_BUDGETS[_training_mode(settings, mode)].copy()


def _thread_count(settings: Settings) -> int:
    # -1 lets CatBoost use all available cores. Override with settings.catboost_thread_count if needed.
    return int(_setting(settings, "catboost_thread_count", -1))


def _fit_verbose(settings: Settings) -> bool | int:
    return _setting(settings, "catboost_verbose", False)


def _limit_candidates(candidates: list[dict[str, Any]], settings: Settings, mode: TrainingMode | None) -> list[dict[str, Any]]:
    max_candidates = int(_setting(settings, "max_tuning_candidates", _budget(settings, mode)["max_tuning_candidates"]))
    if max_candidates <= 0:
        return candidates
    return candidates[:max_candidates]


def _safe_concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    frames = [frame for frame in frames if isinstance(frame, pd.DataFrame) and not frame.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _safe_roc_auc(y_true: pd.Series, probability: np.ndarray) -> float:
    return float(roc_auc_score(y_true, probability)) if y_true.nunique(dropna=True) == 2 else np.nan


def _score_regression_target(target: str, y_pred: np.ndarray) -> np.ndarray:
    if target in NON_NEGATIVE_TARGETS:
        return np.maximum(y_pred, 0)
    return y_pred


def _target_transform(params: dict[str, Any] | None, target: str) -> str:
    transform = (params or {}).get("_target_transform", "raw")
    return "log" if transform == "log" and target in NON_NEGATIVE_TARGETS else "raw"


def _model_params(params: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in params.items() if not key.startswith("_")}


def _target_for_fit(target: str, y: Iterable[float], transform: str = "raw") -> np.ndarray:
    values = np.asarray(y, dtype="float64")
    if transform == "log" and target in NON_NEGATIVE_TARGETS:
        return np.log1p(np.maximum(values, 0))
    return values


def _prediction_from_fit(target: str, y_pred: Iterable[float], transform: str = "raw") -> np.ndarray:
    values = np.asarray(y_pred, dtype="float64")
    if transform == "log" and target in NON_NEGATIVE_TARGETS:
        values = np.expm1(values)
    return _score_regression_target(target, values)


def _confidence_from_wape(wape: float) -> float:
    if pd.isna(wape):
        return np.nan
    return float(np.clip(1 - wape, 0, 1))


def _absolute_percentage_error(actual: Iterable[float], predicted: Iterable[float]) -> pd.Series:
    actual_series = pd.Series(actual, dtype="float64").reset_index(drop=True)
    predicted_series = pd.Series(predicted, dtype="float64").reset_index(drop=True)
    denominator = actual_series.abs().replace(0, np.nan)
    error = (predicted_series - actual_series).abs() / denominator
    return error.mask(denominator.isna(), predicted_series.ne(actual_series).astype(float))


def build_confidence_table(actual: Iterable[float], predicted: Iterable[float], *, bins: int = 5) -> dict[str, Any]:
    actual_series = pd.Series(actual, dtype="float64").reset_index(drop=True)
    predicted_series = pd.Series(predicted, dtype="float64").reset_index(drop=True)
    error = _absolute_percentage_error(actual_series, predicted_series)
    valid = pd.DataFrame({"actual": actual_series, "predicted": predicted_series, "error": error}).dropna()
    fallback = _confidence_from_wape(regression_metrics(actual_series, predicted_series)["WAPE"])
    if valid.empty:
        return {"fallback": fallback, "bins": []}

    try:
        groups = pd.qcut(valid["predicted"], q=min(bins, len(valid)), duplicates="drop")
    except ValueError:
        groups = None

    rows: list[dict[str, float]] = []
    if groups is not None:
        for _, group in valid.groupby(groups, observed=True):
            if group.empty:
                continue
            rows.append(
                {
                    "min_pred": float(group["predicted"].min()),
                    "max_pred": float(group["predicted"].max()),
                    "confidence": float(np.clip(1 - group["error"].median(), 0, 1)),
                }
            )

    if not rows:
        rows.append(
            {
                "min_pred": float(valid["predicted"].min()),
                "max_pred": float(valid["predicted"].max()),
                "confidence": fallback,
            }
        )

    return {"fallback": fallback, "bins": rows}


def confidence_from_prediction(
    predicted: Iterable[float],
    table: dict[str, Any] | None,
    observation_coverage: Iterable[float] | None = None,
) -> np.ndarray:
    predicted_series = pd.Series(predicted, dtype="float64").reset_index(drop=True)
    fallback = np.nan if not table else table.get("fallback", np.nan)
    result = pd.Series(fallback, index=predicted_series.index, dtype="float64")
    for row in (table or {}).get("bins", []):
        lower = row.get("min_pred", -np.inf)
        upper = row.get("max_pred", np.inf)
        mask = predicted_series.between(lower, upper, inclusive="both")
        result.loc[mask] = row.get("confidence", fallback)
    if observation_coverage is not None:
        coverage = pd.Series(observation_coverage, dtype="float64").reset_index(drop=True).clip(lower=0, upper=1)
        result = result * coverage.fillna(1)
    return result.clip(lower=0, upper=1).to_numpy()


def _baseline_prediction(frame: pd.DataFrame, target: str) -> np.ndarray:
    short = target.replace("target_", "")
    week_2_column = f"{short}_week_2_sum"
    week_1_column = f"{short}_week_1_sum"
    if week_2_column in frame.columns and week_1_column in frame.columns:
        week_1 = pd.to_numeric(frame[week_1_column], errors="coerce").fillna(0)
        week_2 = pd.to_numeric(frame[week_2_column], errors="coerce").fillna(0)
        growth = ((week_2 - week_1) / week_1.abs().replace(0, np.nan)).clip(lower=-0.5, upper=1.0).fillna(0)
        return np.maximum((week_2 * (1 + growth)).to_numpy(dtype="float64"), 0)
    if week_2_column in frame.columns:
        return np.maximum(pd.to_numeric(frame[week_2_column], errors="coerce").fillna(0).to_numpy(), 0)
    return np.zeros(len(frame), dtype="float64")


def _best_blend_weight(actual: Iterable[float], model_prediction: np.ndarray, baseline_prediction: np.ndarray) -> float:
    best_weight = 1.0
    best_score = np.inf
    for weight in np.linspace(0, 1, 11):
        blended = weight * model_prediction + (1 - weight) * baseline_prediction
        metrics = regression_metrics(actual, blended)
        score = metrics["WAPE"] if pd.notna(metrics["WAPE"]) else metrics["MAE"]
        if pd.notna(score) and score < best_score:
            best_score = float(score)
            best_weight = float(weight)
    return best_weight


def _build_regressor(params: dict[str, Any], settings: Settings, mode: TrainingMode | None = None) -> CatBoostRegressor:
    budget = _budget(settings, mode)
    transform = params.get("_target_transform", "raw")
    return CatBoostRegressor(
        loss_function="RMSE" if transform == "log" else "MAE",
        eval_metric="RMSE" if transform == "log" else "MAE",
        iterations=int(_setting(settings, "regressor_iterations", budget["regressor_iterations"])),
        random_seed=settings.random_seed,
        verbose=_fit_verbose(settings),
        allow_writing_files=False,
        od_type="Iter",
        thread_count=_thread_count(settings),
        **_model_params(params),
    )


def _build_classifier(params: dict[str, Any], settings: Settings, mode: TrainingMode | None = None) -> CatBoostClassifier:
    budget = _budget(settings, mode)
    return CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=int(_setting(settings, "classifier_iterations", budget["classifier_iterations"])),
        random_seed=settings.random_seed,
        verbose=_fit_verbose(settings),
        allow_writing_files=False,
        auto_class_weights="Balanced",
        od_type="Iter",
        thread_count=_thread_count(settings),
        **params,
    )


def _fit_regressor(
    model: CatBoostRegressor,
    train: pd.DataFrame,
    validation: pd.DataFrame | None,
    target: str,
    categorical: list[str],
    settings: Settings,
    mode: TrainingMode | None = None,
    transform: str = "raw",
) -> CatBoostRegressor:
    budget = _budget(settings, mode)
    fit_kwargs: dict[str, Any] = {"cat_features": categorical}
    if validation is not None and not validation.empty:
        fit_kwargs["eval_set"] = (
            prepare_features(validation, categorical),
            _target_for_fit(target, validation[target], transform),
        )
        fit_kwargs["early_stopping_rounds"] = int(
            _setting(settings, "early_stopping_rounds", budget["early_stopping_rounds"])
        )
        fit_kwargs["use_best_model"] = True

    model.fit(prepare_features(train, categorical), _target_for_fit(target, train[target], transform), **fit_kwargs)
    return model


def _fit_classifier(
    model: CatBoostClassifier,
    train: pd.DataFrame,
    validation: pd.DataFrame | None,
    categorical: list[str],
    settings: Settings,
    mode: TrainingMode | None = None,
) -> CatBoostClassifier:
    budget = _budget(settings, mode)
    fit_kwargs: dict[str, Any] = {"cat_features": categorical}
    if validation is not None and not validation.empty and validation["is_high_performer"].nunique() >= 2:
        fit_kwargs["eval_set"] = (prepare_features(validation, categorical), validation["is_high_performer"])
        fit_kwargs["early_stopping_rounds"] = int(
            _setting(settings, "early_stopping_rounds", budget["early_stopping_rounds"])
        )
        fit_kwargs["use_best_model"] = True

    model.fit(prepare_features(train, categorical), train["is_high_performer"], **fit_kwargs)
    return model


def tune_regressor(
    development: pd.DataFrame,
    target: str,
    checkpoint: int,
    horizon: int,
    categorical: list[str],
    settings: Settings,
    mode: TrainingMode | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    records: list[dict[str, Any]] = []
    best_params: dict[str, Any] | None = None
    best_score = np.inf

    folds = rolling_date_splits(
        development,
        n_splits=settings.rolling_validation_folds,
        min_training_rows=settings.min_training_rows,
        min_validation_rows=settings.min_validation_rows,
    )

    for candidate_id, params in enumerate(_limit_candidates(REGRESSOR_PARAMETER_CANDIDATES, settings, mode), 1):
        scores: list[float] = []
        for fold_number, (train, validation) in enumerate(folds, 1):
            train = train.dropna(subset=[target]).copy()
            validation = validation.dropna(subset=[target]).copy()
            if train.empty or validation.empty:
                continue

            model = _build_regressor(params, settings, mode)
            transform = _target_transform(params, target)
            _fit_regressor(model, train, validation, target, categorical, settings, mode, transform)
            prediction = _prediction_from_fit(
                target,
                model.predict(prepare_features(validation, categorical)),
                transform,
            )
            metrics = regression_metrics(validation[target], prediction)
            selection_score = metrics["WAPE"] if pd.notna(metrics["WAPE"]) else metrics["MAE"]
            scores.append(float(selection_score))
            records.append(
                {
                    "Checkpoint": checkpoint,
                    "Horizon": horizon,
                    "Target": target,
                    "Candidate": candidate_id,
                    "Fold": fold_number,
                    "Best_Iteration": model.get_best_iteration(),
                    **params,
                    **metrics,
                }
            )

        if scores and float(np.mean(scores)) < best_score:
            best_score = float(np.mean(scores))
            best_params = params.copy()

    if best_params is None:
        raise ValueError(f"Unable to tune {target} for day {checkpoint}, horizon {horizon}.")

    return best_params, pd.DataFrame(records)


def tune_classifier(
    development: pd.DataFrame,
    checkpoint: int,
    horizon: int,
    categorical: list[str],
    settings: Settings,
    mode: TrainingMode | None = None,
) -> tuple[dict[str, Any] | None, pd.DataFrame]:
    records: list[dict[str, Any]] = []
    best_params: dict[str, Any] | None = None
    best_auc = -np.inf

    if development["is_high_performer"].nunique(dropna=True) < 2:
        return None, pd.DataFrame()

    folds = rolling_date_splits(
        development,
        n_splits=settings.rolling_validation_folds,
        min_training_rows=settings.min_training_rows,
        min_validation_rows=settings.min_validation_rows,
    )

    for candidate_id, params in enumerate(_limit_candidates(CLASSIFIER_PARAMETER_CANDIDATES, settings, mode), 1):
        aucs: list[float] = []
        for fold_number, (train, validation) in enumerate(folds, 1):
            if train["is_high_performer"].nunique() < 2 or validation["is_high_performer"].nunique() < 2:
                continue

            model = _build_classifier(params, settings, mode)
            _fit_classifier(model, train, validation, categorical, settings, mode)
            probability = model.predict_proba(prepare_features(validation, categorical))[:, 1]
            auc = _safe_roc_auc(validation["is_high_performer"], probability)
            if pd.notna(auc):
                aucs.append(float(auc))
            records.append(
                {
                    "Checkpoint": checkpoint,
                    "Horizon": horizon,
                    "Target": "is_high_performer",
                    "Candidate": candidate_id,
                    "Fold": fold_number,
                    "Best_Iteration": model.get_best_iteration(),
                    **params,
                    "ROC_AUC": auc,
                }
            )

        if aucs and float(np.mean(aucs)) > best_auc:
            best_auc = float(np.mean(aucs))
            best_params = params.copy()

    return best_params, pd.DataFrame(records)


def choose_representative_snapshot(snapshots: dict[tuple[int, int], pd.DataFrame], settings: Settings) -> tuple[tuple[int, int], pd.DataFrame]:
    preferred_key = (settings.representative_tuning_checkpoint, settings.representative_tuning_horizon)
    minimum = settings.min_training_rows + settings.min_validation_rows
    preferred = snapshots.get(preferred_key, pd.DataFrame())
    if isinstance(preferred, pd.DataFrame) and len(preferred) >= minimum:
        return preferred_key, preferred

    eligible = [(key, frame) for key, frame in snapshots.items() if isinstance(frame, pd.DataFrame) and len(frame) >= minimum]
    if not eligible:
        raise ValueError("No lifecycle snapshot has enough rows for representative tuning.")

    eligible.sort(
        key=lambda item: (
            abs(item[0][0] - settings.representative_tuning_checkpoint),
            abs(item[0][1] - settings.representative_tuning_horizon),
            -len(item[1]),
        )
    )
    return eligible[0]


def tune_shared_hyperparameters(
    snapshots: dict[tuple[int, int], pd.DataFrame],
    settings: Settings,
    *,
    run_regression: bool = True,
    run_classification: bool = False,
    targets: Iterable[str] | None = None,
    mode: TrainingMode | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, int, int]:
    (checkpoint, horizon), snapshot = choose_representative_snapshot(snapshots, settings)
    development, holdout, _ = split_development_and_holdout(snapshot, settings.final_test_fraction)
    if run_classification:
        development, _, _ = create_high_performance_target(development, holdout, settings.high_performer_percentile)
    categorical = categorical_features(snapshot)

    selected_targets = list(targets or REGRESSION_TARGETS)
    shared: dict[str, Any] = {"regressors": {}, "classifier": None}
    frames: list[pd.DataFrame] = []

    if run_regression:
        for target in selected_targets:
            params, fold_results = tune_regressor(development, target, checkpoint, horizon, categorical, settings, mode)
            shared["regressors"][target] = params
            if not fold_results.empty:
                frames.append(fold_results)

    if run_classification:
        params, fold_results = tune_classifier(development, checkpoint, horizon, categorical, settings, mode)
        shared["classifier"] = params
        if not fold_results.empty:
            frames.append(fold_results)

    return shared, _safe_concat(frames), checkpoint, horizon


def _base_prediction_frame(holdout: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Game_ID",
        "TournamentName",
        "SystemName",
        "First_Observed_Date",
        "Obs_Window",
        "Horizon",
    ]
    return holdout[[column for column in columns if column in holdout.columns]].copy()


def train_model_family(
    snapshot: pd.DataFrame,
    checkpoint: int,
    horizon: int,
    shared_parameters: dict[str, Any],
    tuning_checkpoint: int,
    tuning_horizon: int,
    settings: Settings,
    *,
    run_regression: bool = True,
    run_classification: bool = False,
    targets: Iterable[str] | None = None,
    mode: TrainingMode | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any], pd.DataFrame]:
    development, holdout, holdout_start = split_development_and_holdout(snapshot, settings.final_test_fraction)
    threshold = np.nan
    if run_classification:
        development, holdout, threshold = create_high_performance_target(
            development,
            holdout,
            settings.high_performer_percentile,
        )
    categorical = categorical_features(snapshot)
    selected_targets = list(targets or REGRESSION_TARGETS)

    bundle: dict[str, Any] = {
        "checkpoint": checkpoint,
        "horizon": horizon,
        "categorical_features": categorical,
        "regressors": {},
        "classifier": None,
        "best_parameters": {},
        "confidence_scores": {},
        "high_score_threshold": threshold,
        "holdout_start_date": holdout_start,
        "tuned_on_checkpoint": tuning_checkpoint,
        "tuned_on_horizon": tuning_horizon,
        "training_mode": _training_mode(settings, mode),
    }
    evaluations: list[dict[str, Any]] = []
    predictions = _base_prediction_frame(holdout)

    if run_regression:
        for target in selected_targets:
            params = shared_parameters.get("regressors", {}).get(target)
            if params is None:
                continue

            development_target = development.dropna(subset=[target]).copy()
            holdout_target = holdout.dropna(subset=[target]).copy()
            if development_target.empty or holdout_target.empty:
                continue

            train, validation, _ = split_development_and_holdout(development_target, 0.15)
            if train.empty:
                continue
            if validation.empty:
                validation = None

            model = _build_regressor(params, settings, mode)
            transform = _target_transform(params, target)
            _fit_regressor(model, train, validation, target, categorical, settings, mode, transform)
            model_prediction = _prediction_from_fit(
                target,
                model.predict(prepare_features(holdout_target, categorical)),
                transform,
            )
            baseline = _baseline_prediction(holdout_target, target)
            baseline_metrics = regression_metrics(holdout_target[target], baseline)
            blend_weight = 1.0

            if validation is not None and not validation.empty:
                validation_prediction = _prediction_from_fit(
                    target,
                    model.predict(prepare_features(validation, categorical)),
                    transform,
                )
                validation_baseline = _baseline_prediction(validation, target)
                blend_weight = _best_blend_weight(validation[target], validation_prediction, validation_baseline)
                validation_prediction = blend_weight * validation_prediction + (1 - blend_weight) * validation_baseline
                confidence_table = build_confidence_table(validation[target], validation_prediction)
            else:
                confidence_table = {"fallback": np.nan, "bins": []}

            prediction = blend_weight * model_prediction + (1 - blend_weight) * baseline
            metrics = regression_metrics(holdout_target[target], prediction)
            if pd.isna(confidence_table.get("fallback")):
                confidence_table["fallback"] = _confidence_from_wape(metrics["WAPE"])
            confidence = confidence_from_prediction(
                prediction,
                confidence_table,
                holdout_target.get("observation_coverage"),
            )

            evaluations.append(
                {
                    "Checkpoint": checkpoint,
                    "Horizon": horizon,
                    "Target": target,
                    "Holdout_Start": holdout_start,
                    "Best_Iteration": model.get_best_iteration(),
                    **params,
                    **metrics,
                    "Model_Blend_Weight": blend_weight,
                    "Baseline_MAE": baseline_metrics["MAE"],
                    "Baseline_RMSE": baseline_metrics["RMSE"],
                    "Baseline_R2": baseline_metrics["R2"],
                    "Baseline_WAPE": baseline_metrics["WAPE"],
                    "WAPE_Improvement": baseline_metrics["WAPE"] - metrics["WAPE"],
                }
            )
            bundle["regressors"][target] = model
            bundle["best_parameters"][target] = params
            bundle.setdefault("target_transforms", {})[target] = transform
            bundle.setdefault("blend_weights", {})[target] = blend_weight
            bundle.setdefault("confidence_tables", {})[target] = confidence_table
            bundle["confidence_scores"][target] = float(np.nanmean(confidence))
            short_name = target.replace("target_", "")
            predictions.loc[holdout_target.index, f"actual_{short_name}"] = holdout_target[target]
            predictions.loc[holdout_target.index, f"baseline_{short_name}_week_4"] = baseline
            predictions.loc[holdout_target.index, f"predicted_{short_name}_week_4"] = prediction
            predictions.loc[holdout_target.index, f"confidence_{short_name}"] = confidence
            week_3_column = f"actual_{short_name}_week_3"
            week_4_column = f"actual_{short_name}_week_4"
            if week_3_column in holdout_target.columns:
                predictions.loc[holdout_target.index, week_3_column] = holdout_target[week_3_column]
            if week_4_column in holdout_target.columns:
                predictions.loc[holdout_target.index, week_4_column] = holdout_target[week_4_column]
            for week in (1, 2):
                week_column = f"{target}_week_{week}"
                if week_column in holdout_target.columns:
                    predictions.loc[holdout_target.index, f"actual_{short_name}_week_{week}"] = holdout_target[
                        week_column
                    ]

    class_eval: dict[str, Any] = {}
    params = shared_parameters.get("classifier")
    if run_classification and params is not None and development["is_high_performer"].nunique() >= 2:
        train, validation, _ = split_development_and_holdout(development, 0.15)
        classifier = _build_classifier(params, settings, mode)

        if train["is_high_performer"].nunique() >= 2:
            validation_for_fit = validation if validation["is_high_performer"].nunique() >= 2 else None
            _fit_classifier(classifier, train, validation_for_fit, categorical, settings, mode)
        else:
            _fit_classifier(classifier, development, None, categorical, settings, mode)

        probability = classifier.predict_proba(prepare_features(holdout, categorical))[:, 1]
        predicted = (probability >= float(_setting(settings, "classification_threshold", 0.5))).astype(int)
        y_true = holdout["is_high_performer"]
        class_eval = {
            "Checkpoint": checkpoint,
            "Horizon": horizon,
            "Holdout_Start": holdout_start,
            "Best_Iteration": classifier.get_best_iteration(),
            "Accuracy": accuracy_score(y_true, predicted),
            "Precision": precision_score(y_true, predicted, zero_division=0),
            "Recall": recall_score(y_true, predicted, zero_division=0),
            "F1": f1_score(y_true, predicted, zero_division=0),
            "ROC_AUC": _safe_roc_auc(y_true, probability),
            **params,
        }
        bundle["classifier"] = classifier
        bundle["best_parameters"]["is_high_performer"] = params
        predictions["probability_high_performance"] = probability
        predictions["predicted_high_performer"] = predicted

    return bundle, pd.DataFrame(evaluations), class_eval, predictions


def train_all_model_families(
    snapshots: dict[tuple[int, int], pd.DataFrame],
    settings: Settings,
    *,
    run_regression: bool = True,
    run_classification: bool = False,
    targets: Iterable[str] | None = None,
    mode: TrainingMode | None = None,
    max_families: int | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Tune shared parameters once, then train one model family per lifecycle snapshot.

    Parameters
    ----------
    snapshots:
        Mapping of ``(checkpoint_day, horizon_days)`` to feature/target snapshots.
    settings:
        Project settings object.
    run_regression / run_classification:
        Switch model families on/off while debugging. This is useful when CatBoost
        classifiers are the bottleneck.
    targets:
        Optional subset of regression targets. Example: ["target_ggr"].
    mode:
        ``"quick"``, ``"standard"`` or ``"production"``. Defaults to
        ``settings.training_mode`` if present, otherwise ``"quick"``.
    max_families:
        Optional cap on the number of checkpoint/horizon families trained.
        Defaults to the mode budget. Use 0 or None for no cap.
    verbose:
        Print progress and timing information.
    """
    active_mode = _training_mode(settings, mode)
    budget = _budget(settings, active_mode)
    if max_families is None:
        max_families = int(_setting(settings, "max_model_families", budget["max_families"]))

    started_at = time.time()
    shared, tuning_df, tuned_checkpoint, tuned_horizon = tune_shared_hyperparameters(
        snapshots,
        settings,
        run_regression=run_regression,
        run_classification=run_classification,
        targets=targets,
        mode=active_mode,
    )

    bundles: dict[tuple[int, int], dict[str, Any]] = {}
    regression_frames: list[pd.DataFrame] = []
    classification_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    skipped: list[dict[str, Any]] = []

    minimum = settings.min_training_rows + settings.min_validation_rows
    trained_count = 0

    for checkpoint_horizon, snapshot in snapshots.items():
        checkpoint, horizon = checkpoint_horizon
        if len(snapshot) < minimum:
            skipped.append({"Checkpoint": checkpoint, "Horizon": horizon, "Reason": f"Only {len(snapshot)} eligible rows"})
            continue
        if max_families and trained_count >= max_families:
            skipped.append({"Checkpoint": checkpoint, "Horizon": horizon, "Reason": f"Skipped by max_families={max_families}"})
            continue

        family_started_at = time.time()
        if verbose:
            print(f"Training family day {checkpoint} -> horizon {horizon} [{active_mode}] ...")

        try:
            bundle, regression_eval, classification_eval, predictions = train_model_family(
                snapshot,
                checkpoint,
                horizon,
                shared,
                tuned_checkpoint,
                tuned_horizon,
                settings,
                run_regression=run_regression,
                run_classification=run_classification,
                targets=targets,
                mode=active_mode,
            )
            bundles[checkpoint_horizon] = bundle
            trained_count += 1
            if not regression_eval.empty:
                regression_frames.append(regression_eval)
            if classification_eval:
                classification_rows.append(classification_eval)
            if not predictions.empty:
                prediction_frames.append(predictions)
        except ValueError as exc:
            skipped.append({"Checkpoint": checkpoint, "Horizon": horizon, "Reason": str(exc)})

        if verbose:
            elapsed = time.time() - family_started_at
            print(f"Finished day {checkpoint} -> horizon {horizon} in {elapsed:,.1f}s")

    return {
        "model_bundles": bundles,
        "regression_evaluation": _safe_concat(regression_frames),
        "classification_evaluation": pd.DataFrame(classification_rows),
        "hyperparameter_tuning": tuning_df,
        "holdout_predictions": _safe_concat(prediction_frames),
        "skipped_configurations": pd.DataFrame(skipped),
        "training_metadata": {
            "training_mode": active_mode,
            "run_regression": run_regression,
            "run_classification": run_classification,
            "targets": list(targets or REGRESSION_TARGETS),
            "tuned_on_checkpoint": tuned_checkpoint,
            "tuned_on_horizon": tuned_horizon,
            "trained_families": trained_count,
            "elapsed_seconds": time.time() - started_at,
        },
    }
