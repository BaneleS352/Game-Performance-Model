from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def split_development_and_holdout(snapshot: pd.DataFrame, test_fraction: float = 0.15):
    data = snapshot.dropna(subset=["First_Observed_Date"]).copy()
    data["First_Observed_Date"] = pd.to_datetime(data["First_Observed_Date"])
    data = data.sort_values("First_Observed_Date").reset_index(drop=True)
    dates = np.sort(data["First_Observed_Date"].unique())
    if len(dates) < 8:
        raise ValueError("Not enough unique launch dates for temporal validation and holdout testing.")
    test_date_count = max(1, int(np.ceil(len(dates) * test_fraction)))
    test_start = pd.Timestamp(dates[-test_date_count])
    return (
        data[data["First_Observed_Date"] < test_start].copy(),
        data[data["First_Observed_Date"] >= test_start].copy(),
        test_start,
    )


def rolling_date_splits(
    development: pd.DataFrame,
    *,
    n_splits: int,
    min_training_rows: int,
    min_validation_rows: int,
):
    dates = np.sort(pd.to_datetime(development["First_Observed_Date"]).dropna().unique())
    if len(dates) < n_splits + 2:
        raise ValueError("Not enough unique dates for requested rolling folds.")
    blocks = [b for b in np.array_split(dates, n_splits + 1) if len(b)]
    folds = []
    for i in range(1, len(blocks)):
        train_dates = np.concatenate(blocks[:i])
        validation_dates = blocks[i]
        train = development[development["First_Observed_Date"].isin(train_dates)].copy()
        validation = development[development["First_Observed_Date"].isin(validation_dates)].copy()
        if len(train) >= min_training_rows and len(validation) >= min_validation_rows:
            folds.append((train, validation))
    if not folds:
        raise ValueError("No rolling fold met minimum row requirements.")
    return folds


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denominator = np.abs(y_true).sum()
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "WAPE": float(np.abs(y_true - y_pred).sum() / denominator) if denominator else np.nan,
    }


def create_high_performance_target(
    development: pd.DataFrame,
    holdout: pd.DataFrame,
    percentile: float = 0.80,
):
    """Fit score references on development only, then apply them to holdout."""
    weights = {
        "target_ggr": 0.40,
        "target_stakes": 0.25,
        "target_clients": 0.15,
        "target_betcount": 0.10,
        "target_ggr_pct": 0.10,
    }
    references = {metric: np.sort(development[metric].dropna().to_numpy()) for metric in weights}
    medians = {metric: development[metric].median() for metric in weights}

    def empirical_percentile(values, reference):
        if len(reference) == 0:
            return np.zeros(len(values))
        return np.searchsorted(reference, np.asarray(values), side="right") / len(reference)

    def score(frame):
        result = frame.copy()
        composite = np.zeros(len(result), dtype=float)
        for metric, weight in weights.items():
            # Important: holdout missing values use development medians, not holdout medians.
            values = pd.to_numeric(result[metric], errors="coerce").fillna(medians[metric])
            composite += weight * empirical_percentile(values, references[metric])
        result["future_performance_score"] = composite
        return result

    development_scored = score(development)
    holdout_scored = score(holdout)
    threshold = float(development_scored["future_performance_score"].quantile(percentile))
    development_scored["is_high_performer"] = (development_scored["future_performance_score"] >= threshold).astype(int)
    holdout_scored["is_high_performer"] = (holdout_scored["future_performance_score"] >= threshold).astype(int)
    return development_scored, holdout_scored, threshold
