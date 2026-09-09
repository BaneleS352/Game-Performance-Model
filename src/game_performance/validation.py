from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BUSINESS_GOAL_TARGETS = {
    "clients": ("target_clients", "is_high_client_performer"),
    "stakes": ("target_stakes", "is_high_stakes_performer"),
}


def split_training_validation_holdout(
    snapshot: pd.DataFrame,
    *,
    training_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    holdout_fraction: float = 0.15,
    date_column: str = "First_Observed_Date",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Split chronological data by unique dates into train, validation, and holdout sets."""
    fractions = np.asarray((training_fraction, validation_fraction, holdout_fraction), dtype=float)
    if np.any(fractions <= 0) or not np.isclose(fractions.sum(), 1.0):
        raise ValueError("Training, validation, and holdout fractions must be positive and sum to 1.")
    if date_column not in snapshot:
        raise KeyError(f"Missing temporal split column: {date_column}")

    data = snapshot.dropna(subset=[date_column]).copy()
    data[date_column] = pd.to_datetime(data[date_column])
    data = data.sort_values(date_column).reset_index(drop=True)
    dates = np.sort(data[date_column].unique())
    if len(dates) < 3:
        raise ValueError("At least three unique dates are required for temporal training, validation, and holdout splits.")

    counts = np.floor(len(dates) * fractions).astype(int)
    for index in np.argsort(-(len(dates) * fractions - counts), kind="stable")[: len(dates) - counts.sum()]:
        counts[index] += 1
    for index in np.flatnonzero(counts == 0):
        donor = int(np.argmax(counts))
        counts[donor] -= 1
        counts[index] += 1

    training_dates, validation_dates, holdout_dates = np.split(dates, np.cumsum(counts)[:-1])
    training = data[data[date_column].isin(training_dates)].copy()
    validation = data[data[date_column].isin(validation_dates)].copy()
    holdout = data[data[date_column].isin(holdout_dates)].copy()
    metadata = {
        "date_column": date_column,
        "total_days": int(len(dates)),
        "training_days": int(len(training_dates)),
        "validation_days": int(len(validation_dates)),
        "holdout_days": int(len(holdout_dates)),
        "training_start": pd.Timestamp(training_dates[0]),
        "validation_start": pd.Timestamp(validation_dates[0]),
        "holdout_start": pd.Timestamp(holdout_dates[0]),
    }
    return training, validation, holdout, metadata


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


def restrict_training_window(
    development: pd.DataFrame,
    window_days: int | None,
    *,
    reference_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Keep all history or only the requested recent launch-date window."""
    if window_days is None or development.empty:
        return development.copy()
    dates = pd.to_datetime(development["First_Observed_Date"], errors="coerce")
    reference = pd.Timestamp(reference_date) if reference_date is not None else dates.max()
    cutoff = reference - pd.Timedelta(days=int(window_days) - 1)
    return development.loc[dates.ge(cutoff)].copy()


def rolling_date_splits(
    development: pd.DataFrame,
    *,
    n_splits: int,
    min_training_rows: int,
    min_validation_rows: int,
    training_window_days: int | None = None,
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
        if training_window_days is not None and not validation.empty:
            train = restrict_training_window(
                train,
                training_window_days,
                reference_date=pd.Timestamp(validation_dates[0]),
            )
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
    """Fit business-goal thresholds on development only, then apply them to holdout."""
    references = {
        metric: np.sort(development[metric].dropna().to_numpy())
        for metric, _ in BUSINESS_GOAL_TARGETS.values()
    }
    medians = {
        metric: development[metric].median()
        for metric, _ in BUSINESS_GOAL_TARGETS.values()
    }
    thresholds = {
        flag: float(development[metric].quantile(percentile))
        for metric, flag in BUSINESS_GOAL_TARGETS.values()
    }

    def empirical_percentile(values, reference):
        if len(reference) == 0:
            return np.zeros(len(values))
        return np.searchsorted(reference, np.asarray(values), side="right") / len(reference)

    def score(frame):
        result = frame.copy()
        composite = np.zeros(len(result), dtype=float)
        for metric, flag in BUSINESS_GOAL_TARGETS.values():
            # Important: holdout missing values use development medians, not holdout medians.
            values = pd.to_numeric(result[metric], errors="coerce").fillna(medians[metric])
            composite += 0.50 * empirical_percentile(values, references[metric])
            result[flag] = values.ge(thresholds[flag]).astype(int)
        result["is_balanced_high_performer"] = (
            result["is_high_client_performer"].eq(1)
            & result["is_high_stakes_performer"].eq(1)
        ).astype(int)
        result["future_performance_score"] = composite
        result["is_high_performer"] = result["is_balanced_high_performer"]
        return result

    development_scored = score(development)
    holdout_scored = score(holdout)
    return development_scored, holdout_scored, thresholds
