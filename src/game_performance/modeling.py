from __future__ import annotations

import time
from typing import Any, Iterable, Literal

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import Settings
from .features import (
    COMMERCIAL_DRIVER_TARGETS,
    OUTCOME_TARGETS,
    REGRESSION_TARGETS,
    categorical_features,
    prepare_features,
)
from .validation import (
    create_high_performance_target,
    regression_metrics,
    restrict_training_window,
    rolling_date_splits,
    split_training_validation_holdout,
)

TrainingMode = Literal["quick", "standard", "production"]

NON_NEGATIVE_TARGETS = {
    "target_clients",
    "target_bets_per_client",
    "target_stake_per_bet",
    "target_betcount",
    "target_stakes",
}
ProbabilityCalibrator = dict[str, Any]
CLASSIFICATION_TARGETS = {
    "client": "is_high_client_performer",
    "stakes": "is_high_stakes_performer",
}

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
        "max_tuning_candidates": 3,
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


def _safe_log_loss(y_true: pd.Series | np.ndarray, probability: np.ndarray) -> float:
    if len(y_true) == 0:
        return np.nan
    return float(log_loss(y_true, np.clip(probability, 1e-6, 1 - 1e-6), labels=[0, 1]))


def _expected_calibration_error(
    y_true: pd.Series | np.ndarray,
    probability: np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    if len(y_true) == 0:
        return np.nan

    frame = pd.DataFrame(
        {
            "actual": np.asarray(y_true, dtype=float),
            "probability": np.clip(np.asarray(probability, dtype=float), 0, 1),
        }
    ).dropna()
    if frame.empty:
        return np.nan

    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(frame["probability"], bins[1:-1], right=False)
    total = len(frame)
    error = 0.0

    for bin_id in range(n_bins):
        bucket = frame[bin_ids == bin_id]
        if bucket.empty:
            continue
        error += len(bucket) / total * abs(bucket["probability"].mean() - bucket["actual"].mean())

    return float(error)


def probability_calibration_bins(
    y_true: pd.Series | np.ndarray,
    probability: np.ndarray,
    *,
    raw_probability: np.ndarray | None = None,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Summarise how predicted probabilities line up with actual hit rates."""
    frame = pd.DataFrame(
        {
            "actual": np.asarray(y_true, dtype=float),
            "probability": np.clip(np.asarray(probability, dtype=float), 0, 1),
        }
    )
    if raw_probability is not None:
        frame["raw_probability"] = np.clip(np.asarray(raw_probability, dtype=float), 0, 1)
    frame = frame.dropna(subset=["actual", "probability"])

    bins = np.linspace(0, 1, n_bins + 1)
    rows: list[dict[str, Any]] = []
    if frame.empty:
        frame["_bin"] = pd.Series(dtype=int)
    else:
        frame["_bin"] = np.digitize(frame["probability"], bins[1:-1], right=False)

    for bin_id in range(n_bins):
        lower = float(bins[bin_id])
        upper = float(bins[bin_id + 1])
        bucket = frame[frame["_bin"] == bin_id]
        mean_probability = bucket["probability"].mean() if not bucket.empty else np.nan
        actual_rate = bucket["actual"].mean() if not bucket.empty else np.nan
        row = {
            "Probability_Bin": f"{lower:.1f}-{upper:.1f}",
            "Bin_Lower": lower,
            "Bin_Upper": upper,
            "Count": int(len(bucket)),
            "Mean_Calibrated_Probability": float(mean_probability) if pd.notna(mean_probability) else np.nan,
            "Actual_High_Performer_Rate": float(actual_rate) if pd.notna(actual_rate) else np.nan,
            "Calibration_Error": (
                float(abs(mean_probability - actual_rate))
                if pd.notna(mean_probability) and pd.notna(actual_rate)
                else np.nan
            ),
        }
        if raw_probability is not None:
            raw_mean = bucket["raw_probability"].mean() if not bucket.empty else np.nan
            row["Mean_Raw_Probability"] = float(raw_mean) if pd.notna(raw_mean) else np.nan
        rows.append(row)

    return pd.DataFrame(rows)


def _score_regression_target(target: str, y_pred: np.ndarray) -> np.ndarray:
    if target in NON_NEGATIVE_TARGETS:
        return np.maximum(y_pred, 0)
    return y_pred


def _is_log_transform(transform: str | None) -> bool:
    return transform in {"log", "log1p"}


def regression_target_transform_name(target: str) -> str:
    if target in NON_NEGATIVE_TARGETS:
        return "log"
    return "raw"


def transform_regression_target(target: str, values: pd.Series | np.ndarray) -> np.ndarray:
    return _target_for_fit(target, values, regression_target_transform_name(target))


def inverse_transform_regression_target(
    target: str,
    values: pd.Series | np.ndarray,
    *,
    transform: str | None = None,
) -> np.ndarray:
    return _prediction_from_fit(target, values, transform or regression_target_transform_name(target))


def score_regression_prediction(
    target: str,
    raw_prediction: pd.Series | np.ndarray,
    *,
    transform: str | None = None,
) -> np.ndarray:
    return inverse_transform_regression_target(target, raw_prediction, transform=transform)


def _target_transform(params: dict[str, Any] | None, target: str) -> str:
    transform = (params or {}).get("_target_transform", regression_target_transform_name(target))
    return "log" if _is_log_transform(transform) and target in NON_NEGATIVE_TARGETS else "raw"


def _model_params(params: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in params.items() if not key.startswith("_")}


def _target_for_fit(target: str, y: Iterable[float], transform: str = "raw") -> np.ndarray:
    values = np.asarray(y, dtype="float64")
    if _is_log_transform(transform) and target in NON_NEGATIVE_TARGETS:
        return np.log1p(np.maximum(values, 0))
    return values


def _prediction_from_fit(target: str, y_pred: Iterable[float], transform: str = "raw") -> np.ndarray:
    values = np.asarray(y_pred, dtype="float64")
    if _is_log_transform(transform) and target in NON_NEGATIVE_TARGETS:
        values = np.expm1(values)
    return _score_regression_target(target, values)


def _confidence_from_wape(wape: float) -> float:
    if pd.isna(wape):
        return np.nan
    return float(np.clip(1 - wape, 0, 1))


def build_prediction_interval_table(
    actual: Iterable[float],
    predicted: Iterable[float],
    *,
    lower_quantile: float = 0.10,
    upper_quantile: float = 0.90,
    bins: int = 5,
    min_bin_rows: int = 25,
) -> dict[str, Any]:
    actual_series = pd.to_numeric(pd.Series(actual), errors="coerce").astype("float64").reset_index(drop=True)
    predicted_series = pd.to_numeric(pd.Series(predicted), errors="coerce").astype("float64").reset_index(drop=True)
    valid = pd.DataFrame({"actual": actual_series, "predicted": predicted_series}).replace([np.inf, -np.inf], np.nan).dropna()
    if valid.empty:
        return {"lower_residual": 0.0, "upper_residual": 0.0, "bins": []}

    residual = valid["actual"] - valid["predicted"]
    table: dict[str, Any] = {
        "lower_residual": float(residual.quantile(lower_quantile)),
        "upper_residual": float(residual.quantile(upper_quantile)),
        "bins": [],
    }

    number_of_bins = min(bins, len(valid) // max(1, min_bin_rows), valid["predicted"].nunique())
    if number_of_bins < 2:
        return table

    try:
        groups = pd.qcut(valid["predicted"], q=int(number_of_bins), duplicates="drop")
    except ValueError:
        return table

    for _, group in valid.groupby(groups, observed=True):
        group_residual = group["actual"] - group["predicted"]
        table["bins"].append(
            {
                "min_pred": float(group["predicted"].min()),
                "max_pred": float(group["predicted"].max()),
                "lower_residual": float(group_residual.quantile(lower_quantile)),
                "upper_residual": float(group_residual.quantile(upper_quantile)),
                "rows": int(len(group)),
            }
        )
    return table


def build_conformal_interval_table(
    actual: Iterable[float],
    predicted: Iterable[float],
    *,
    coverage: float = 0.80,
) -> dict[str, Any]:
    """Build a split-conformal interval from out-of-fold absolute residuals."""
    if not 0 < coverage < 1:
        raise ValueError("coverage must be strictly between 0 and 1.")

    valid = pd.DataFrame(
        {
            "actual": pd.to_numeric(pd.Series(actual), errors="coerce"),
            "predicted": pd.to_numeric(pd.Series(predicted), errors="coerce"),
        }
    ).replace([np.inf, -np.inf], np.nan).dropna()
    if valid.empty:
        return {"method": "split_conformal", "coverage": coverage, "absolute_residual": 0.0, "rows": 0}

    residual = (valid["actual"] - valid["predicted"]).abs().to_numpy(dtype="float64")
    quantile = min(1.0, float(np.ceil((len(residual) + 1) * coverage) / len(residual)))
    return {
        "method": "split_conformal",
        "coverage": coverage,
        "absolute_residual": float(np.quantile(residual, quantile, method="higher")),
        "rows": int(len(residual)),
    }


def prediction_interval_from_table(
    predicted: Iterable[float],
    table: dict[str, float] | None,
    *,
    non_negative: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    prediction = np.asarray(predicted, dtype="float64")
    table = table or {}
    if table.get("method") == "split_conformal":
        residual = float(table.get("absolute_residual", 0.0))
        low, high = prediction - residual, prediction + residual
        if non_negative:
            low, high = np.maximum(low, 0), np.maximum(high, 0)
        return low, high

    lower_residual = np.full(len(prediction), float(table.get("lower_residual", 0.0)))
    upper_residual = np.full(len(prediction), float(table.get("upper_residual", 0.0)))
    bands = table.get("bins", [])
    assigned = np.zeros(len(prediction), dtype=bool)

    for band in bands:
        minimum = float(band["min_pred"])
        maximum = float(band["max_pred"])
        mask = (prediction >= minimum) & (prediction <= maximum)
        lower_residual[mask] = float(band["lower_residual"])
        upper_residual[mask] = float(band["upper_residual"])
        assigned |= mask

    if bands and (~assigned).any():
        centers = np.asarray([(float(band["min_pred"]) + float(band["max_pred"])) / 2 for band in bands])
        nearest = np.abs(prediction[~assigned, None] - centers).argmin(axis=1)
        unassigned = np.flatnonzero(~assigned)
        lower_residual[unassigned] = [float(bands[index]["lower_residual"]) for index in nearest]
        upper_residual[unassigned] = [float(bands[index]["upper_residual"]) for index in nearest]

    low = prediction + lower_residual
    high = prediction + upper_residual
    low = np.minimum(low, prediction)
    high = np.maximum(high, prediction)
    if non_negative:
        low, high = np.maximum(low, 0), np.maximum(high, 0)
    return low, high


def _absolute_percentage_error(actual: Iterable[float], predicted: Iterable[float]) -> pd.Series:
    actual_series = pd.Series(actual, dtype="float64").reset_index(drop=True)
    predicted_series = pd.Series(predicted, dtype="float64").reset_index(drop=True)
    denominator = actual_series.abs().replace(0, np.nan)
    error = (predicted_series - actual_series).abs() / denominator
    return error.mask(denominator.isna(), predicted_series.ne(actual_series).astype(float))

def build_confidence_table(
    actual: Iterable[float],
    predicted: Iterable[float],
    *,
    bins: int = 10,
    min_bin_rows: int = 10,
) -> dict[str, Any]:
    """
    Build prediction-specific confidence bands from validation data.

    Confidence means:
        1.00 = historically very accurate predictions in this range
        0.80 = median historical error was approximately 20%
        0.50 = median historical error was approximately 50%
        0.00 = historically very unreliable

    The table is learned ONLY from validation data. It does not use
    future actual values when scoring new games.
    """

    actual_series = pd.to_numeric(
        pd.Series(actual).reset_index(drop=True),
        errors="coerce",
    ).astype("float64")

    predicted_series = pd.to_numeric(
        pd.Series(predicted).reset_index(drop=True),
        errors="coerce",
    ).astype("float64")

    error = _absolute_percentage_error(
        actual_series,
        predicted_series,
    )

    valid = pd.DataFrame(
        {
            "actual": actual_series,
            "predicted": predicted_series,
            "error": error,
        }
    ).replace([np.inf, -np.inf], np.nan)

    valid = valid.dropna(
        subset=["actual", "predicted", "error"]
    )

    # Global fallback confidence.
    metrics = regression_metrics(
        actual_series.dropna(),
        predicted_series.loc[actual_series.notna()],
    )

    fallback = _confidence_from_wape(
        metrics["WAPE"]
    )

    if valid.empty:
        return {
            "fallback": fallback,
            "bins": [],
        }

    # ---------------------------------------------------------
    # Determine how many bins are sensible.
    #
    # We do not want tiny bins containing only one or two games.
    # ---------------------------------------------------------
    max_bins_from_rows = max(
        1,
        len(valid) // max(1, min_bin_rows),
    )

    number_of_bins = min(
        bins,
        max_bins_from_rows,
        valid["predicted"].nunique(),
    )

    number_of_bins = max(
        1,
        int(number_of_bins),
    )

    try:
        groups = pd.qcut(
            valid["predicted"],
            q=number_of_bins,
            duplicates="drop",
        )

    except ValueError:
        groups = None

    rows: list[dict[str, float]] = []

    if groups is not None:

        grouped = valid.groupby(
            groups,
            observed=True,
        )

        for _, group in grouped:

            if group.empty:
                continue

            median_error = float(
                group["error"].median()
            )

            confidence = float(
                np.clip(
                    1 - median_error,
                    0,
                    1,
                )
            )

            rows.append(
                {
                    "min_pred": float(
                        group["predicted"].min()
                    ),
                    "max_pred": float(
                        group["predicted"].max()
                    ),
                    "confidence": confidence,
                    "rows": int(len(group)),
                    "median_error": median_error,
                }
            )

    # ---------------------------------------------------------
    # Fallback if qcut cannot create useful groups.
    # ---------------------------------------------------------
    if not rows:

        median_error = float(
            valid["error"].median()
        )

        rows.append(
            {
                "min_pred": float(
                    valid["predicted"].min()
                ),
                "max_pred": float(
                    valid["predicted"].max()
                ),
                "confidence": float(
                    np.clip(
                        1 - median_error,
                        0,
                        1,
                    )
                ),
                "rows": int(len(valid)),
                "median_error": median_error,
            }
        )

    # Ensure bins are sorted.
    rows = sorted(
        rows,
        key=lambda row: row["min_pred"],
    )

    return {
        "fallback": fallback,
        "bins": rows,
    }


def confidence_from_prediction(
    predicted: Iterable[float],
    table: dict[str, Any] | None,
    observation_coverage: Iterable[float] | None = None,
) -> np.ndarray:
    """
    Assign confidence to each individual prediction.

    The confidence comes from the validation prediction range that the
    forecast falls into.

    Observation coverage is then applied as a reliability adjustment.

    Example:
        learned confidence = 0.80
        observation coverage = 0.75

        final confidence = 0.60
    """

    predicted_series = pd.to_numeric(
        pd.Series(predicted).reset_index(drop=True),
        errors="coerce",
    ).astype("float64")

    fallback = (
        np.nan
        if not table
        else table.get("fallback", np.nan)
    )

    result = pd.Series(
        fallback,
        index=predicted_series.index,
        dtype="float64",
    )

    bins = (
        []
        if not table
        else table.get("bins", [])
    )

    # ---------------------------------------------------------
    # Assign confidence from prediction bands.
    # ---------------------------------------------------------
    for row in bins:

        lower = float(
            row.get(
                "min_pred",
                -np.inf,
            )
        )

        upper = float(
            row.get(
                "max_pred",
                np.inf,
            )
        )

        confidence = row.get(
            "confidence",
            fallback,
        )

        mask = predicted_series.between(
            lower,
            upper,
            inclusive="both",
        )

        result.loc[mask] = confidence

    # ---------------------------------------------------------
    # Predictions outside the validation range.
    #
    # Use the nearest confidence band instead of immediately
    # falling back to global WAPE.
    # ---------------------------------------------------------
    if bins:

        ordered_bins = sorted(
            bins,
            key=lambda row: row["min_pred"],
        )

        first_bin = ordered_bins[0]
        last_bin = ordered_bins[-1]

        below = (
            predicted_series
            < float(first_bin["min_pred"])
        )

        above = (
            predicted_series
            > float(last_bin["max_pred"])
        )

        result.loc[below] = first_bin[
            "confidence"
        ]

        result.loc[above] = last_bin[
            "confidence"
        ]

    # ---------------------------------------------------------
    # Observation coverage adjustment.
    # ---------------------------------------------------------
    if observation_coverage is not None:

        coverage = pd.to_numeric(
            pd.Series(
                observation_coverage
            ).reset_index(drop=True),
            errors="coerce",
        ).astype("float64")

        coverage = coverage.clip(
            lower=0,
            upper=1,
        )

        # Missing coverage should not destroy an otherwise valid
        # confidence estimate.
        coverage = coverage.fillna(1.0)

        result = result * coverage

    return (
        result
        .clip(lower=0, upper=1)
        .to_numpy(dtype="float64")
    )

def _baseline_prediction(frame: pd.DataFrame, target: str) -> np.ndarray:
    def constrain(values: np.ndarray) -> np.ndarray:
        return np.maximum(values, 0) if target in NON_NEGATIVE_TARGETS else values

    driver_feature = {
        "target_bets_per_client": "bets_per_client",
        "target_stake_per_bet": "stake_per_bet",
        "target_ggr_margin": "ggr_margin",
    }.get(target)
    if driver_feature is not None:
        values = pd.to_numeric(frame.get(driver_feature, pd.Series(np.nan, index=frame.index)), errors="coerce")
        return constrain(values.fillna(0).to_numpy(dtype="float64"))

    short = target.replace("target_", "")
    recent_values = frame[f"{short}_last_7_mean"] if f"{short}_last_7_mean" in frame.columns else None
    recent_mean = pd.to_numeric(recent_values, errors="coerce") if recent_values is not None else None
    if recent_mean is not None and recent_mean.notna().any():
        return constrain(recent_mean.fillna(0).to_numpy(dtype="float64") * 7)

    observed_values = frame[f"obs_{short}_sum"] if f"obs_{short}_sum" in frame.columns else None
    observed = pd.to_numeric(observed_values, errors="coerce") if observed_values is not None else None
    window_values = frame["Obs_Window"] if "Obs_Window" in frame.columns else None
    window = pd.to_numeric(window_values, errors="coerce").replace(0, np.nan) if window_values is not None else None
    if observed is not None and window is not None and observed.notna().any():
        return constrain((observed.fillna(0) / window * 7).fillna(0).to_numpy(dtype="float64"))

    week_3_column = f"{short}_week_3_sum"
    week_2_column = f"{short}_week_2_sum"
    week_1_column = f"{short}_week_1_sum"
    if week_3_column in frame.columns and week_2_column in frame.columns:
        week_2 = pd.to_numeric(frame[week_2_column], errors="coerce").fillna(0)
        week_3 = pd.to_numeric(frame[week_3_column], errors="coerce")
        has_week_3 = week_3.notna()
        growth = ((week_3 - week_2) / week_2.abs().replace(0, np.nan)).clip(lower=-0.5, upper=1.0).fillna(0)
        prediction = np.maximum((week_3.fillna(0) * (1 + growth)).to_numpy(dtype="float64"), 0)
        if has_week_3.any():
            return constrain(prediction)
    if week_2_column in frame.columns and week_1_column in frame.columns:
        week_1 = pd.to_numeric(frame[week_1_column], errors="coerce").fillna(0)
        week_2 = pd.to_numeric(frame[week_2_column], errors="coerce").fillna(0)
        growth = ((week_2 - week_1) / week_1.abs().replace(0, np.nan)).clip(lower=-0.5, upper=1.0).fillna(0)
        return constrain((week_2 * (1 + growth)).to_numpy(dtype="float64"))
    if week_2_column in frame.columns:
        return constrain(pd.to_numeric(frame[week_2_column], errors="coerce").fillna(0).to_numpy())
    return np.zeros(len(frame), dtype="float64")


def derive_commercial_outputs(driver_values: dict[str, Iterable[float]]) -> dict[str, np.ndarray]:
    """Recompose compatible commercial outcomes from the four driver forecasts."""
    clients = np.maximum(np.asarray(driver_values["target_clients"], dtype="float64"), 0)
    bets_per_client = np.maximum(np.asarray(driver_values["target_bets_per_client"], dtype="float64"), 0)
    stake_per_bet = np.maximum(np.asarray(driver_values["target_stake_per_bet"], dtype="float64"), 0)
    ggr_margin = np.asarray(driver_values["target_ggr_margin"], dtype="float64")
    betcount = clients * bets_per_client
    stakes = betcount * stake_per_bet
    return {
        "target_clients": clients,
        "target_betcount": betcount,
        "target_stakes": stakes,
        "target_ggr": stakes * ggr_margin,
    }


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


def apply_probability_calibration(
    raw_probability: pd.Series | np.ndarray,
    calibrator: ProbabilityCalibrator | None = None,
) -> np.ndarray:
    probability = np.clip(np.asarray(raw_probability, dtype=float), 0, 1)
    if not calibrator:
        return probability

    model = calibrator.get("model") if isinstance(calibrator, dict) else calibrator
    if model is None:
        return probability

    calibrated = model.predict(probability)
    return np.clip(np.asarray(calibrated, dtype=float), 0, 1)


def _fit_probability_calibrator(
    y_true: pd.Series,
    raw_probability: np.ndarray,
    settings: Settings,
) -> tuple[ProbabilityCalibrator, np.ndarray]:
    raw_probability = np.clip(np.asarray(raw_probability, dtype=float), 0, 1)
    min_rows = int(_setting(settings, "min_calibration_rows", 50))
    enough_rows = len(y_true) >= min_rows
    enough_classes = y_true.nunique(dropna=True) == 2
    enough_probability_values = np.unique(raw_probability[~np.isnan(raw_probability)]).size >= 2

    if enough_rows and enough_classes and enough_probability_values:
        calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
        calibrator.fit(raw_probability, y_true)
        bundle: ProbabilityCalibrator = {"method": "isotonic", "model": calibrator}
        return bundle, apply_probability_calibration(raw_probability, bundle)

    bundle = {"method": "identity", "model": None}
    return bundle, raw_probability


def select_classification_threshold(
    y_true: pd.Series | np.ndarray,
    probability: pd.Series | np.ndarray,
    *,
    default: float = 0.5,
) -> float:
    y_true = np.asarray(y_true, dtype=int)
    probability = np.clip(np.asarray(probability, dtype=float), 0, 1)
    if len(y_true) == 0:
        return float(default)

    thresholds = np.unique(np.concatenate(([0.0, default, 1.0], probability)))
    best_threshold = float(default)
    best_f1 = -np.inf
    best_distance = np.inf

    for threshold in thresholds:
        predicted = (probability >= threshold).astype(int)
        score = f1_score(y_true, predicted, zero_division=0)
        distance = abs(float(threshold) - default)
        if score > best_f1 or (score == best_f1 and distance < best_distance):
            best_f1 = float(score)
            best_threshold = float(threshold)
            best_distance = distance

    return best_threshold


def _classification_diagnostics(
    y_true: pd.Series,
    probability: np.ndarray,
    predicted: np.ndarray,
    *,
    n_bins: int,
) -> dict[str, float]:
    return {
        "Accuracy": float(accuracy_score(y_true, predicted)),
        "Precision": float(precision_score(y_true, predicted, zero_division=0)),
        "Recall": float(recall_score(y_true, predicted, zero_division=0)),
        "F1": float(f1_score(y_true, predicted, zero_division=0)),
        "ROC_AUC": _safe_roc_auc(y_true, probability),
        "Brier_Score": float(brier_score_loss(y_true, probability)) if len(y_true) else np.nan,
        "Log_Loss": _safe_log_loss(y_true, probability),
        "Expected_Calibration_Error": _expected_calibration_error(y_true, probability, n_bins=n_bins),
    }


def _build_regressor(
    params: dict[str, Any],
    settings: Settings,
    mode: TrainingMode | None = None,
    *,
    iterations: int | None = None,
) -> CatBoostRegressor:
    budget = _budget(settings, mode)
    transform = params.get("_target_transform", "raw")
    return CatBoostRegressor(
        loss_function="RMSE" if _is_log_transform(transform) else "MAE",
        eval_metric="RMSE" if _is_log_transform(transform) else "MAE",
        iterations=iterations or int(_setting(settings, "regressor_iterations", budget["regressor_iterations"])),
        random_seed=settings.random_seed,
        verbose=_fit_verbose(settings),
        allow_writing_files=False,
        od_type="Iter",
        thread_count=_thread_count(settings),
        **_model_params(params),
    )


def _build_classifier(
    params: dict[str, Any],
    settings: Settings,
    mode: TrainingMode | None = None,
    *,
    iterations: int | None = None,
) -> CatBoostClassifier:
    budget = _budget(settings, mode)
    return CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=iterations or int(_setting(settings, "classifier_iterations", budget["classifier_iterations"])),
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
    target_column: str = "is_high_performer",
) -> CatBoostClassifier:
    budget = _budget(settings, mode)
    fit_kwargs: dict[str, Any] = {"cat_features": categorical}
    if validation is not None and not validation.empty and validation[target_column].nunique() >= 2:
        fit_kwargs["eval_set"] = (prepare_features(validation, categorical), validation[target_column])
        fit_kwargs["early_stopping_rounds"] = int(
            _setting(settings, "early_stopping_rounds", budget["early_stopping_rounds"])
        )
        fit_kwargs["use_best_model"] = True

    model.fit(prepare_features(train, categorical), train[target_column], **fit_kwargs)
    return model


def _fitted_iterations(model: Any, fallback: int) -> int:
    best = model.get_best_iteration()
    trees = int(getattr(model, "tree_count_", 0) or 0)
    if best is not None and best >= 0 and (not trees or trees <= int(best) + 1):
        return int(best) + 1
    return trees or fallback


def _adaptive_rolling_date_splits(
    development: pd.DataFrame,
    settings: Settings,
    training_window_days: int | None = None,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    dates = pd.to_datetime(development["First_Observed_Date"]).dropna().unique()
    max_splits = max(1, len(dates) - 2)
    preferred_splits = min(int(settings.rolling_validation_folds), max_splits)

    for n_splits in range(preferred_splits, 0, -1):
        try:
            return rolling_date_splits(
                development,
                n_splits=n_splits,
                min_training_rows=settings.min_training_rows,
                min_validation_rows=settings.min_validation_rows,
                training_window_days=training_window_days,
            )
        except ValueError:
            continue

    raise ValueError("No rolling fold met minimum row requirements.")


def tune_regressor(
    development: pd.DataFrame,
    target: str,
    checkpoint: int,
    horizon: int,
    categorical: list[str],
    settings: Settings,
    mode: TrainingMode | None = None,
    training_window_days: int | None = None,
    validation: pd.DataFrame | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """
    Tune a CatBoost regressor using the supplied temporal validation split.

    Parameters
    ----------
    development:
        Development/training dataset.
    target:
        Regression target column.
    checkpoint:
        Lifecycle observation checkpoint.
    horizon:
        Forecast horizon.
    categorical:
        Categorical feature columns.
    settings:
        Project Settings object.
    mode:
        Training mode: quick, standard, or production.
    training_window_days:
        Optional rolling training-window restriction.
    validation:
        Optional fixed validation data. When omitted, rolling date folds are
        retained for callers that explicitly request backtesting.
    """
    records: list[dict[str, Any]] = []
    best_params: dict[str, Any] | None = None
    best_score = np.inf

    if validation is None:
        folds = _adaptive_rolling_date_splits(development, settings, training_window_days)
    else:
        training = restrict_training_window(
            development,
            training_window_days,
            reference_date=pd.to_datetime(validation["First_Observed_Date"]).min(),
        )
        if len(training) < settings.min_training_rows or len(validation) < settings.min_validation_rows:
            raise ValueError("The temporal training or validation split did not meet minimum row requirements.")
        folds = [(training, validation)]

    for candidate_id, params in enumerate(
        _limit_candidates(
            REGRESSOR_PARAMETER_CANDIDATES,
            settings,
            mode,
        ),
        1,
    ):
        scores: list[float] = []

        for fold_number, (train, validation) in enumerate(folds, 1):
            train = train.dropna(subset=[target]).copy()
            validation = validation.dropna(subset=[target]).copy()

            if train.empty or validation.empty:
                continue

            model = _build_regressor(
                params,
                settings,
                mode,
            )

            transform = _target_transform(
                params,
                target,
            )

            _fit_regressor(
                model,
                train,
                validation,
                target,
                categorical,
                settings,
                mode,
                transform,
            )

            prediction = _prediction_from_fit(
                target,
                model.predict(
                    prepare_features(
                        validation,
                        categorical,
                    )
                ),
                transform,
            )

            metrics = regression_metrics(
                validation[target],
                prediction,
            )

            selection_score = (
                metrics["WAPE"]
                if pd.notna(metrics["WAPE"])
                else metrics["MAE"]
            )

            scores.append(
                float(selection_score)
            )

            records.append(
                {
                    "Checkpoint": checkpoint,
                    "Horizon": horizon,
                    "Target": target,
                    "Target_Transform": transform,
                    "Training_Window_Days": training_window_days,
                    "Candidate": candidate_id,
                    "Fold": fold_number,
                    "Best_Iteration": model.get_best_iteration(),
                    **params,
                    **metrics,
                }
            )

        if scores:
            mean_score = float(np.mean(scores))

            if mean_score < best_score:
                best_score = mean_score
                best_params = params.copy()

    if best_params is None:
        raise ValueError(
            f"Unable to tune {target} for "
            f"day {checkpoint}, horizon {horizon}."
        )

    return best_params, pd.DataFrame(records)

def tune_classifier(
    development: pd.DataFrame,
    checkpoint: int,
    horizon: int,
    categorical: list[str],
    settings: Settings,
    mode: TrainingMode | None = None,
    target_column: str = "is_high_performer",
    training_window_days: int | None = None,
    validation: pd.DataFrame | None = None,
) -> tuple[dict[str, Any] | None, pd.DataFrame]:
    records: list[dict[str, Any]] = []
    best_params: dict[str, Any] | None = None
    best_auc = -np.inf

    if development[target_column].nunique(dropna=True) < 2:
        return None, pd.DataFrame()

    if validation is None:
        folds = _adaptive_rolling_date_splits(development, settings, training_window_days)
    else:
        training = restrict_training_window(
            development,
            training_window_days,
            reference_date=pd.to_datetime(validation["First_Observed_Date"]).min(),
        )
        if len(training) < settings.min_training_rows or len(validation) < settings.min_validation_rows:
            raise ValueError("The temporal training or validation split did not meet minimum row requirements.")
        folds = [(training, validation)]

    for candidate_id, params in enumerate(_limit_candidates(CLASSIFIER_PARAMETER_CANDIDATES, settings, mode), 1):
        aucs: list[float] = []
        for fold_number, (train, validation) in enumerate(folds, 1):
            if train[target_column].nunique() < 2 or validation[target_column].nunique() < 2:
                continue

            model = _build_classifier(params, settings, mode)
            _fit_classifier(model, train, validation, categorical, settings, mode, target_column)
            probability = model.predict_proba(prepare_features(validation, categorical))[:, 1]
            auc = _safe_roc_auc(validation[target_column], probability)
            if pd.notna(auc):
                aucs.append(float(auc))
            records.append(
                {
                    "Checkpoint": checkpoint,
                    "Horizon": horizon,
                    "Target": target_column,
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
    run_classification: bool = True,
    targets: Iterable[str] | None = None,
    mode: TrainingMode | None = None,
    training_window_days: int | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, int, int]:
    (checkpoint, horizon), snapshot = choose_representative_snapshot(snapshots, settings)
    development, validation, _, _ = split_training_validation_holdout(
        snapshot,
        training_fraction=settings.training_fraction,
        validation_fraction=settings.validation_fraction,
        holdout_fraction=settings.final_test_fraction,
        min_training_rows=settings.min_training_rows,
        min_validation_rows=settings.min_validation_rows,
    )
    if run_classification:
        development, validation, _ = create_high_performance_target(
            development,
            validation,
            settings.high_performer_percentile,
        )
    categorical = categorical_features(snapshot)

    selected_targets = list(targets or REGRESSION_TARGETS)
    shared: dict[str, Any] = {"regressors": {}, "classifiers": {}}
    frames: list[pd.DataFrame] = []

    if run_regression:
        for target in selected_targets:
            params, fold_results = tune_regressor(
                development,
                target,
                checkpoint,
                horizon,
                categorical,
                settings,
                mode,
                training_window_days,
                validation,
            )
            shared["regressors"][target] = params
            if not fold_results.empty:
                frames.append(fold_results)

    if run_classification:
        for goal, target_column in CLASSIFICATION_TARGETS.items():
            params, fold_results = tune_classifier(
                development,
                checkpoint,
                horizon,
                categorical,
                settings,
                mode,
                target_column,
                training_window_days,
                validation,
            )
            shared["classifiers"][goal] = params
            if not fold_results.empty:
                frames.append(fold_results)

    return shared, _safe_concat(frames), checkpoint, horizon


def _base_prediction_frame(holdout: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Game_ID",
        "TournamentName",
        "SystemName",
        "Provider",
        "First_Observed_Date",
        "Obs_Window",
        "Horizon",
        "future_performance_score",
        "is_high_client_performer",
        "is_high_stakes_performer",
        "is_balanced_high_performer",
        "is_high_performer",
    ]
    return holdout[[column for column in columns if column in holdout.columns]].copy()


def _rolling_oof_regression_predictions(
    frame: pd.DataFrame,
    target: str,
    params: dict[str, Any],
    categorical: list[str],
    settings: Settings,
    mode: TrainingMode | None,
    training_window_days: int | None,
) -> tuple[pd.DataFrame, list[int]]:
    """Return strictly forward out-of-fold predictions for one driver."""
    records: list[pd.DataFrame] = []
    iterations: list[int] = []
    transform = _target_transform(params, target)
    fallback_iterations = int(_setting(settings, "regressor_iterations", _budget(settings, mode)["regressor_iterations"]))

    for train, validation in _adaptive_rolling_date_splits(frame, settings, training_window_days):
        train = train.dropna(subset=[target]).copy()
        validation = validation.dropna(subset=[target]).copy()
        if train.empty or validation.empty:
            continue
        model = _build_regressor(params, settings, mode)
        _fit_regressor(model, train, validation, target, categorical, settings, mode, transform)
        model_prediction = _prediction_from_fit(
            target,
            model.predict(prepare_features(validation, categorical)),
            transform,
        )
        records.append(
            pd.DataFrame(
                {
                    "actual": validation[target].to_numpy(dtype="float64"),
                    "model_prediction": model_prediction,
                    "baseline_prediction": _baseline_prediction(validation, target),
                },
                index=validation.index,
            )
        )
        iterations.append(_fitted_iterations(model, fallback_iterations))

    return (pd.concat(records).sort_index() if records else pd.DataFrame()), iterations


def _uncertainty_validation(
    predictions: pd.DataFrame,
    *,
    checkpoint: int,
    horizon: int,
    coverage: float,
) -> pd.DataFrame:
    """Validate holdout interval coverage by age, provider/system, and value band."""
    if predictions.empty:
        return pd.DataFrame()

    provider = predictions.get("Provider", predictions.get("SystemName", pd.Series("Unknown", index=predictions.index)))
    provider = provider.fillna("Unknown").astype("string")
    volume = pd.Series("All", index=predictions.index, dtype="string")
    stakes = pd.to_numeric(
        predictions.get("predicted_stakes", pd.Series(np.nan, index=predictions.index)), errors="coerce"
    )
    if stakes.notna().sum() >= 3 and stakes.nunique(dropna=True) >= 3:
        try:
            volume = pd.qcut(stakes, 3, labels=("Low", "Medium", "High"), duplicates="drop").astype("string")
        except ValueError:
            pass

    segments = {
        "Age / Checkpoint": pd.Series(str(checkpoint), index=predictions.index),
        "Provider": provider,
        "Volume": volume.fillna("Unknown"),
    }
    rows: list[dict[str, Any]] = []
    for target in OUTCOME_TARGETS:
        short = target.removeprefix("target_")
        actual = pd.to_numeric(
            predictions.get(f"actual_{short}", pd.Series(np.nan, index=predictions.index)), errors="coerce"
        )
        low = pd.to_numeric(
            predictions.get(f"predicted_{short}_low", pd.Series(np.nan, index=predictions.index)), errors="coerce"
        )
        high = pd.to_numeric(
            predictions.get(f"predicted_{short}_high", pd.Series(np.nan, index=predictions.index)), errors="coerce"
        )
        valid = actual.notna() & low.notna() & high.notna()
        if not valid.any():
            continue
        for segment_type, values in segments.items():
            for segment, index in values[valid].groupby(values[valid], dropna=False).groups.items():
                mask = valid.copy()
                mask.loc[:] = False
                mask.loc[index] = True
                rows.append(
                    {
                        "Checkpoint": checkpoint,
                        "Horizon": horizon,
                        "Target": target,
                        "Segment_Type": segment_type,
                        "Segment": str(segment),
                        "Rows": int(mask.sum()),
                        "Nominal_Coverage": coverage,
                        "Observed_Coverage": float(actual[mask].between(low[mask], high[mask], inclusive="both").mean()),
                        "Average_Range_Width": float((high[mask] - low[mask]).mean()),
                    }
                )
    return pd.DataFrame(rows)


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
    run_classification: bool = True,
    targets: Iterable[str] | None = None,
    mode: TrainingMode | None = None,
    training_window_days: int | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, list[dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    development, validation, holdout, split_metadata = split_training_validation_holdout(
        snapshot,
        training_fraction=settings.training_fraction,
        validation_fraction=settings.validation_fraction,
        holdout_fraction=settings.final_test_fraction,
        min_training_rows=settings.min_training_rows,
        min_validation_rows=settings.min_validation_rows,
    )
    holdout_start = split_metadata["holdout_start"]
    pre_holdout = pd.concat([development, validation]).sort_values("First_Observed_Date")
    pre_holdout = restrict_training_window(
        pre_holdout,
        training_window_days,
        reference_date=pd.Timestamp(holdout_start),
    )
    development = restrict_training_window(
        development,
        training_window_days,
        reference_date=split_metadata["validation_start"],
    )
    threshold = np.nan
    if run_classification:
        development, validation, threshold = create_high_performance_target(
            development,
            validation,
            settings.high_performer_percentile,
        )
        _, holdout, _ = create_high_performance_target(development, holdout, settings.high_performer_percentile)
    categorical = categorical_features(snapshot)
    selected_targets = list(targets or REGRESSION_TARGETS)
    if run_regression and set(selected_targets) != set(COMMERCIAL_DRIVER_TARGETS):
        raise ValueError("Commercial forecasting requires all four commercial driver targets.")

    bundle: dict[str, Any] = {
        "checkpoint": checkpoint,
        "horizon": horizon,
        "categorical_features": categorical,
        "regressors": {},
        "forecast_strategy": "commercial_drivers_v1",
        "commercial_driver_targets": list(COMMERCIAL_DRIVER_TARGETS),
        "target_transforms": {},
        "regression_target_transforms": {},
        "classifier": None,
        "classifiers": {},
        "probability_calibrator": {"method": "identity", "model": None},
        "probability_calibrators": {},
        "classification_threshold": float(_setting(settings, "classification_threshold", 0.80)),
        "classification_thresholds": {},
        "best_parameters": {},
        "confidence_scores": {},
        "prediction_interval_tables": {},
        "review_confidence_threshold": float(_setting(settings, "review_confidence_threshold", 0.50)),
        "review_relative_width_threshold": float(_setting(settings, "review_relative_width_threshold", 1.00)),
        "high_score_threshold": threshold,
        "holdout_start_date": holdout_start,
        "tuned_on_checkpoint": tuning_checkpoint,
        "tuned_on_horizon": tuning_horizon,
        "training_mode": _training_mode(settings, mode),
        "training_window_days": training_window_days,
        "temporal_split": split_metadata,
    }
    evaluations: list[dict[str, Any]] = []
    predictions = _base_prediction_frame(holdout)

    if run_regression:
        driver_predictions: dict[str, np.ndarray] = {}
        driver_baselines: dict[str, np.ndarray] = {}
        oof_predictions: dict[str, pd.DataFrame] = {}
        fitted_iterations: dict[str, int] = {}

        for target in selected_targets:
            params = shared_parameters.get("regressors", {}).get(target)
            if params is None:
                raise ValueError(f"No tuned parameters were available for commercial driver {target}.")
            train_target = pre_holdout.dropna(subset=[target]).copy()
            if train_target.empty:
                raise ValueError(f"No pre-holdout rows are available for commercial driver {target}.")

            oof, fold_iterations = _rolling_oof_regression_predictions(
                pre_holdout,
                target,
                params,
                categorical,
                settings,
                mode,
                training_window_days,
            )
            if oof.empty:
                raise ValueError(f"No rolling out-of-fold predictions were available for commercial driver {target}.")

            blend_weight = _best_blend_weight(oof["actual"], oof["model_prediction"].to_numpy(), oof["baseline_prediction"].to_numpy())
            oof["prediction"] = blend_weight * oof["model_prediction"] + (1 - blend_weight) * oof["baseline_prediction"]
            fallback_iterations = int(_setting(settings, "regressor_iterations", _budget(settings, mode)["regressor_iterations"]))
            selected_iterations = int(np.median(fold_iterations)) if fold_iterations else fallback_iterations
            transform = _target_transform(params, target)
            model = _build_regressor(params, settings, mode, iterations=selected_iterations)
            _fit_regressor(model, train_target, None, target, categorical, settings, mode, transform)

            model_prediction = _prediction_from_fit(target, model.predict(prepare_features(holdout, categorical)), transform)
            baseline = _baseline_prediction(holdout, target)
            driver_predictions[target] = blend_weight * model_prediction + (1 - blend_weight) * baseline
            driver_baselines[target] = baseline
            oof_predictions[target] = oof
            fitted_iterations[target] = selected_iterations
            bundle["regressors"][target] = model
            bundle["best_parameters"][target] = params
            bundle["target_transforms"][target] = transform
            bundle["regression_target_transforms"][target] = transform
            bundle.setdefault("blend_weights", {})[target] = blend_weight

        oof_driver_predictions: dict[str, np.ndarray] = {}
        oof_driver_baselines: dict[str, np.ndarray] = {}
        for target in COMMERCIAL_DRIVER_TARGETS:
            prediction = pd.Series(np.nan, index=pre_holdout.index, dtype="float64")
            baseline = pd.Series(np.nan, index=pre_holdout.index, dtype="float64")
            prediction.loc[oof_predictions[target].index] = oof_predictions[target]["prediction"]
            baseline.loc[oof_predictions[target].index] = oof_predictions[target]["baseline_prediction"]
            oof_driver_predictions[target] = prediction.to_numpy()
            oof_driver_baselines[target] = baseline.to_numpy()

        outcome_predictions = derive_commercial_outputs(driver_predictions)
        outcome_baselines = derive_commercial_outputs(driver_baselines)
        oof_outcomes = derive_commercial_outputs(oof_driver_predictions)
        oof_baselines = derive_commercial_outputs(oof_driver_baselines)
        coverage = float(_setting(settings, "prediction_interval_coverage", 0.80))

        for target in OUTCOME_TARGETS:
            short_name = target.removeprefix("target_")
            oof_actual = pd.to_numeric(pre_holdout[target], errors="coerce").to_numpy(dtype="float64")
            oof_prediction = oof_outcomes[target]
            oof_valid = ~np.isnan(oof_actual) & ~np.isnan(oof_prediction)
            if not oof_valid.any():
                raise ValueError(f"No valid out-of-fold outcome predictions were available for {target}.")
            confidence_table = build_confidence_table(oof_actual[oof_valid], oof_prediction[oof_valid])
            oof_metrics = regression_metrics(oof_actual[oof_valid], oof_prediction[oof_valid])
            metric_confidence = _confidence_from_wape(oof_metrics["WAPE"])
            confidence_table["fallback"] = metric_confidence
            interval_table = build_conformal_interval_table(
                oof_actual[oof_valid], oof_prediction[oof_valid], coverage=coverage
            )

            actual = pd.to_numeric(holdout[target], errors="coerce").to_numpy(dtype="float64")
            prediction = outcome_predictions[target]
            baseline = outcome_baselines[target]
            prediction_low, prediction_high = prediction_interval_from_table(
                prediction,
                interval_table,
                non_negative=target in NON_NEGATIVE_TARGETS,
            )
            valid = ~np.isnan(actual) & ~np.isnan(prediction)
            metrics = regression_metrics(actual[valid], prediction[valid]) if valid.any() else {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan, "WAPE": np.nan}
            baseline_metrics = regression_metrics(actual[valid], baseline[valid]) if valid.any() else metrics
            range_valid = valid & ~np.isnan(prediction_low) & ~np.isnan(prediction_high)
            range_coverage = float(
                np.mean((actual[range_valid] >= prediction_low[range_valid]) & (actual[range_valid] <= prediction_high[range_valid]))
            ) if range_valid.any() else np.nan
            confidence = confidence_from_prediction(prediction, confidence_table, holdout.get("observation_coverage"))

            evaluations.append(
                {
                    "Checkpoint": checkpoint,
                    "Horizon": horizon,
                    "Target": target,
                    "Target_Transform": "derived",
                    "Holdout_Start": holdout_start,
                    "Commercial_Drivers": ", ".join(COMMERCIAL_DRIVER_TARGETS),
                    "OOF_Rows": int(oof_valid.sum()),
                    **metrics,
                    "Baseline_MAE": baseline_metrics["MAE"],
                    "Baseline_RMSE": baseline_metrics["RMSE"],
                    "Baseline_R2": baseline_metrics["R2"],
                    "Baseline_WAPE": baseline_metrics["WAPE"],
                    "WAPE_Improvement": baseline_metrics["WAPE"] - metrics["WAPE"],
                    "Range_Coverage": range_coverage,
                    "Average_Range_Width": float(np.nanmean(prediction_high - prediction_low)),
                    "Interval_Method": interval_table["method"],
                    "Interval_Nominal_Coverage": coverage,
                }
            )
            bundle.setdefault("confidence_tables", {})[target] = confidence_table
            bundle["prediction_interval_tables"][target] = interval_table
            bundle["confidence_scores"][target] = float(metric_confidence)
            predictions[f"actual_{short_name}"] = actual
            predictions[f"baseline_{short_name}"] = baseline
            predictions[f"predicted_{short_name}_low"] = prediction_low
            predictions[f"predicted_{short_name}"] = prediction
            predictions[f"predicted_{short_name}_high"] = prediction_high
            predictions[f"confidence_{short_name}"] = confidence

        bundle["driver_fitted_iterations"] = fitted_iterations
        bundle["uncertainty_validation"] = _uncertainty_validation(
            predictions,
            checkpoint=checkpoint,
            horizon=horizon,
            coverage=coverage,
        )

    class_eval: list[dict[str, Any]] = []
    calibration_frames: list[pd.DataFrame] = []
    if run_classification:
        default_threshold = float(_setting(settings, "classification_threshold", 0.80))
        n_bins = int(_setting(settings, "calibration_bins", 10))

        for goal, target_column in CLASSIFICATION_TARGETS.items():
            params = shared_parameters.get("classifiers", {}).get(goal)
            if params is None or development[target_column].nunique() < 2:
                continue

            probe_classifier = _build_classifier(params, settings, mode)
            if development[target_column].nunique() >= 2:
                validation_for_fit = validation if validation[target_column].nunique() >= 2 else None
                _fit_classifier(probe_classifier, development, validation_for_fit, categorical, settings, mode, target_column)
            else:
                _fit_classifier(probe_classifier, development, None, categorical, settings, mode, target_column)

            classification_threshold = default_threshold
            calibrator: ProbabilityCalibrator = {"method": "identity", "model": None}
            validation_eval: dict[str, Any] = {}

            if validation is not None and not validation.empty and validation[target_column].nunique() >= 2:
                raw_validation_probability = probe_classifier.predict_proba(prepare_features(validation, categorical))[:, 1]
                calibrator, validation_probability = _fit_probability_calibrator(
                    validation[target_column],
                    raw_validation_probability,
                    settings,
                )
                classification_threshold = select_classification_threshold(
                    validation[target_column],
                    validation_probability,
                    default=default_threshold,
                )
                classification_threshold = max(default_threshold, classification_threshold)
                validation_predicted = (validation_probability >= classification_threshold).astype(int)
                validation_eval = {
                    f"Validation_{key}": value
                    for key, value in _classification_diagnostics(
                        validation[target_column],
                        validation_probability,
                        validation_predicted,
                        n_bins=n_bins,
                    ).items()
                }

            selected_iterations = _fitted_iterations(
                probe_classifier,
                int(_setting(settings, "classifier_iterations", _budget(settings, mode)["classifier_iterations"])),
            )
            classifier = _build_classifier(params, settings, mode, iterations=selected_iterations)
            _fit_classifier(classifier, development, None, categorical, settings, mode, target_column)
            raw_probability = classifier.predict_proba(prepare_features(holdout, categorical))[:, 1]
            probability = apply_probability_calibration(raw_probability, calibrator)
            predicted = (probability >= classification_threshold).astype(int)
            y_true = holdout[target_column]
            class_eval.append(
                {
                    "Checkpoint": checkpoint,
                    "Horizon": horizon,
                    "Target": target_column,
                    "Holdout_Start": holdout_start,
                    "Training_Window_Days": training_window_days,
                    "Best_Iteration": selected_iterations,
                    "Calibrator_Method": calibrator["method"],
                    "Classification_Threshold": classification_threshold,
                    **_classification_diagnostics(y_true, probability, predicted, n_bins=n_bins),
                    "Raw_ROC_AUC": _safe_roc_auc(y_true, raw_probability),
                    "Raw_Brier_Score": float(brier_score_loss(y_true, raw_probability)) if len(y_true) else np.nan,
                    "Raw_Log_Loss": _safe_log_loss(y_true, raw_probability),
                    "Raw_Expected_Calibration_Error": _expected_calibration_error(
                        y_true,
                        raw_probability,
                        n_bins=n_bins,
                    ),
                    **validation_eval,
                    **params,
                }
            )

            calibration_bins = probability_calibration_bins(
                y_true,
                probability,
                raw_probability=raw_probability,
                n_bins=n_bins,
            )
            calibration_bins.insert(0, "Dataset", "Holdout")
            calibration_bins.insert(0, "Target", target_column)
            calibration_bins.insert(0, "Horizon", horizon)
            calibration_bins.insert(0, "Checkpoint", checkpoint)
            calibration_frames.append(calibration_bins)

            bundle["classifiers"][goal] = classifier
            bundle["probability_calibrators"][goal] = calibrator
            bundle["classification_thresholds"][goal] = classification_threshold
            bundle["best_parameters"][target_column] = params
            predictions[f"raw_probability_high_{goal}_performer"] = raw_probability
            predictions[f"probability_high_{goal}_performer"] = probability
            predictions[f"classification_threshold_high_{goal}_performer"] = classification_threshold
            predictions[f"predicted_high_{goal}_performer"] = predicted

        if {"probability_high_client_performer", "probability_high_stakes_performer"}.issubset(predictions.columns):
            predictions["probability_high_performance"] = predictions[
                ["probability_high_client_performer", "probability_high_stakes_performer"]
            ].min(axis=1)
            predictions["raw_probability_high_performance"] = predictions[
                ["raw_probability_high_client_performer", "raw_probability_high_stakes_performer"]
            ].min(axis=1)
            predictions["classification_threshold"] = default_threshold
            predictions["predicted_high_performer"] = (
                predictions["predicted_high_client_performer"].eq(1)
                & predictions["predicted_high_stakes_performer"].eq(1)
            ).astype(int)

    return bundle, pd.DataFrame(evaluations), class_eval, predictions, _safe_concat(calibration_frames)


def select_best_configuration(
    snapshots: dict[tuple[int, int], pd.DataFrame],
    settings: Settings,
    *,
    targets: Iterable[str] | None = None,
    mode: TrainingMode | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Select checkpoint and training window using the fixed temporal validation split."""
    selected_targets = list(targets or REGRESSION_TARGETS)
    records: list[dict[str, Any]] = []
    selection_mode = mode or _setting(settings, "selection_mode", "quick")

    for (checkpoint, horizon), snapshot in sorted(snapshots.items()):
        minimum = settings.min_training_rows + settings.min_validation_rows
        if len(snapshot) < minimum:
            continue
        development, validation, _, _ = split_training_validation_holdout(
            snapshot,
            training_fraction=settings.training_fraction,
            validation_fraction=settings.validation_fraction,
            holdout_fraction=settings.final_test_fraction,
            min_training_rows=settings.min_training_rows,
            min_validation_rows=settings.min_validation_rows,
        )
        categorical = categorical_features(snapshot)

        for training_window_days in settings.training_window_days:
            target_scores: dict[str, float] = {}
            fold_counts: list[int] = []
            for target in selected_targets:
                try:
                    params, tuning = tune_regressor(
                        development,
                        target,
                        checkpoint,
                        horizon,
                        categorical,
                        settings,
                        selection_mode,
                        training_window_days,
                        validation,
                    )
                except ValueError:
                    continue
                if tuning.empty:
                    continue

                candidate_scores = tuning.groupby("Candidate", as_index=False)["WAPE"].mean().dropna()
                if candidate_scores.empty:
                    continue
                best_candidate = candidate_scores.sort_values("WAPE").iloc[0]
                target_scores[target] = float(best_candidate["WAPE"])
                fold_counts.append(int((tuning["Candidate"] == best_candidate["Candidate"]).sum()))

            if len(target_scores) != len(selected_targets):
                continue
            records.append(
                {
                    "Checkpoint": checkpoint,
                    "Horizon": horizon,
                    "Training_Window_Days": training_window_days,
                    "WAPE_Clients": target_scores.get("target_clients", np.nan),
                    "WAPE_Bets_Per_Client": target_scores.get("target_bets_per_client", np.nan),
                    "WAPE_Stake_Per_Bet": target_scores.get("target_stake_per_bet", np.nan),
                    "WAPE_GGR_Margin": target_scores.get("target_ggr_margin", np.nan),
                    "Mean_WAPE": float(np.mean(list(target_scores.values()))),
                    "Validation_Folds": min(fold_counts) if fold_counts else 0,
                    "Selection_Mode": selection_mode,
                }
            )

    evaluation = pd.DataFrame(records).sort_values(
        ["Mean_WAPE", "WAPE_Clients", "WAPE_Bets_Per_Client", "Checkpoint"],
        kind="stable",
    ).reset_index(drop=True)
    if evaluation.empty:
        raise ValueError("No checkpoint and training-window configuration met validation requirements.")

    best = evaluation.iloc[0].to_dict()
    return best, evaluation


def train_selected_model_configuration(
    snapshots: dict[tuple[int, int], pd.DataFrame],
    settings: Settings,
    selection: dict[str, Any],
    *,
    run_regression: bool = True,
    run_classification: bool = True,
    targets: Iterable[str] | None = None,
    mode: TrainingMode | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Tune and fit exactly the configuration selected by development backtests."""
    checkpoint = int(selection["Checkpoint"])
    horizon = int(selection["Horizon"])
    key = (checkpoint, horizon)
    snapshot = snapshots.get(key)
    if snapshot is None or snapshot.empty:
        raise ValueError(f"Selected snapshot {key} is unavailable.")

    training_window = selection.get("Training_Window_Days")
    if pd.isna(training_window):
        training_window = None
    elif training_window is not None:
        training_window = int(training_window)

    shared, tuning, tuned_checkpoint, tuned_horizon = tune_shared_hyperparameters(
        {key: snapshot},
        settings,
        run_regression=run_regression,
        run_classification=run_classification,
        targets=targets,
        mode=mode,
        training_window_days=training_window,
    )
    if verbose:
        print(f"Training selected family day {checkpoint} -> horizon {horizon} ...")
    bundle, regression, classification, predictions, calibration = train_model_family(
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
        mode=mode,
        training_window_days=training_window,
    )
    bundle["selected_configuration"] = selection
    return {
        "model_bundles": {key: bundle},
        "regression_evaluation": regression,
        "uncertainty_validation": bundle.get("uncertainty_validation", pd.DataFrame()),
        "classification_evaluation": pd.DataFrame(classification),
        "classification_calibration": calibration,
        "hyperparameter_tuning": tuning,
        "holdout_predictions": predictions,
        "skipped_configurations": pd.DataFrame(),
        "selection_evaluation": selection,
        "training_metadata": {
            "training_mode": _training_mode(settings, mode),
            "selection_mode": _setting(settings, "selection_mode", "quick"),
            "trained_families": 1,
            "selected_checkpoint": checkpoint,
            "selected_horizon": horizon,
            "selected_training_window_days": training_window,
        },
    }


def train_all_model_families(
    snapshots: dict[tuple[int, int], pd.DataFrame],
    settings: Settings,
    *,
    run_regression: bool = True,
    run_classification: bool = True,
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
        Optional subset of regression targets. Example: ["target_clients"].
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
    bundles: dict[tuple[int, int], dict[str, Any]] = {}
    regression_frames: list[pd.DataFrame] = []
    classification_rows: list[dict[str, Any]] = []
    calibration_frames: list[pd.DataFrame] = []
    uncertainty_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    tuning_frames: list[pd.DataFrame] = []
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
            shared, tuning_df, tuned_checkpoint, tuned_horizon = tune_shared_hyperparameters(
                {checkpoint_horizon: snapshot},
                settings,
                run_regression=run_regression,
                run_classification=run_classification,
                targets=targets,
                mode=active_mode,
            )
            bundle, regression_eval, classification_eval, predictions, calibration_bins = train_model_family(
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
            if not tuning_df.empty:
                tuning_frames.append(tuning_df)
            if not regression_eval.empty:
                regression_frames.append(regression_eval)
            if classification_eval:
                classification_rows.extend(classification_eval)
            if not calibration_bins.empty:
                calibration_frames.append(calibration_bins)
            uncertainty = bundle.get("uncertainty_validation")
            if isinstance(uncertainty, pd.DataFrame) and not uncertainty.empty:
                uncertainty_frames.append(uncertainty)
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
        "uncertainty_validation": _safe_concat(uncertainty_frames),
        "classification_evaluation": pd.DataFrame(classification_rows),
        "classification_calibration": _safe_concat(calibration_frames),
        "hyperparameter_tuning": _safe_concat(tuning_frames),
        "holdout_predictions": _safe_concat(prediction_frames),
        "skipped_configurations": pd.DataFrame(skipped),
        "training_metadata": {
            "training_mode": active_mode,
            "run_regression": run_regression,
            "run_classification": run_classification,
            "targets": list(targets or REGRESSION_TARGETS),
            "tuning_scope": "per_model_family",
            "trained_families": trained_count,
            "elapsed_seconds": time.time() - started_at,
        },
    }
