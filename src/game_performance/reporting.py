from __future__ import annotations

import numpy as np
import pandas as pd

from .features import OBSERVATION_WINDOW_DAYS, PREDICTION_WEEK_END
from .validation import create_high_performance_target, split_development_and_holdout


def _period_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    first = pd.to_datetime(result["First_Observed_Date"], errors="coerce")
    checkpoint = pd.to_numeric(result.get("Obs_Window", OBSERVATION_WINDOW_DAYS), errors="coerce")
    horizon = pd.to_numeric(result.get("Horizon", PREDICTION_WEEK_END - OBSERVATION_WINDOW_DAYS), errors="coerce")
    result["Observation_Start"] = first
    result["Observation_End"] = first + pd.to_timedelta(checkpoint - 1, unit="D")
    result["Prediction_Start"] = first + pd.to_timedelta(checkpoint, unit="D")
    result["Prediction_End"] = result["Prediction_Start"] + pd.to_timedelta(horizon - 1, unit="D")
    return result


def _prediction_probability(actual: pd.Series, predicted: pd.Series) -> pd.Series:
    denominator = actual.abs().replace(0, np.nan)
    probability = 1 - (predicted - actual).abs() / denominator
    probability = probability.clip(lower=0, upper=1)
    return probability.mask(denominator.isna(), predicted.eq(actual).astype(float))


def _growth_percentage(week_1: pd.Series, week_2: pd.Series) -> pd.Series:
    denominator = week_1.abs().replace(0, np.nan)
    return ((week_2 - week_1) / denominator).mask(denominator.isna())


def _is_off_by(actual: pd.Series, predicted: pd.Series, threshold: float) -> pd.Series:
    denominator = actual.abs().replace(0, np.nan)
    error = (predicted - actual).abs() / denominator
    return error.gt(threshold).fillna(False).astype("int8")


def add_actual_week_columns(holdout_predictions: pd.DataFrame, daily_data: pd.DataFrame) -> pd.DataFrame:
    """Add first- and second-week actual clients/stakes from the daily source data."""
    if holdout_predictions.empty:
        return holdout_predictions.copy()

    required = {"Game_ID", "Days_Since_First_Observed", "Clients", "Stakes"}
    missing = required.difference(daily_data.columns)
    if missing:
        raise KeyError(f"Missing required daily data columns: {sorted(missing)}")

    result = holdout_predictions.copy()
    pairs = result[["Game_ID", "Obs_Window"]].drop_duplicates().copy()
    lookup = daily_data[list(required)].merge(pairs, on="Game_ID", how="inner")
    lookup["Days_Since_First_Observed"] = pd.to_numeric(
        lookup["Days_Since_First_Observed"],
        errors="coerce",
    )
    lookup["Obs_Window"] = pd.to_numeric(lookup["Obs_Window"], errors="coerce")
    lookup["future_day"] = lookup["Days_Since_First_Observed"] - lookup["Obs_Window"]

    for metric, source in (("clients", "Clients"), ("stakes", "Stakes")):
        week_1 = (
            lookup[lookup["future_day"].between(1, 7)]
            .groupby(["Game_ID", "Obs_Window"], as_index=False)[source]
            .sum(min_count=1)
            .rename(columns={source: f"actual_{metric}_week_1"})
        )
        week_2 = (
            lookup[lookup["future_day"].between(8, 14)]
            .groupby(["Game_ID", "Obs_Window"], as_index=False)[source]
            .sum(min_count=1)
            .rename(columns={source: f"actual_{metric}_week_2"})
        )
        result = result.merge(week_1, on=["Game_ID", "Obs_Window"], how="left")
        result = result.merge(week_2, on=["Game_ID", "Obs_Window"], how="left")

    return result


def build_holdout_comparison(
    holdout_predictions: pd.DataFrame,
    checkpoint: int | None = None,
    horizon: int | None = None,
) -> pd.DataFrame:
    """Create a business-friendly holdout comparison report."""

    test = holdout_predictions.copy()
    if checkpoint is not None:
        test = test[test["Obs_Window"] == checkpoint]
    if horizon is not None:
        test = test[test["Horizon"] == horizon]

    if test.empty:
        return test

    test = _period_columns(test)

    metrics = [
        "clients",
        "stakes",
    ]

    # ------------------------------------------------------------------
    # Error calculations
    # ------------------------------------------------------------------
    for metric in metrics:

        actual = f"actual_{metric}_week_4"
        predicted = f"predicted_{metric}_week_4"

        if {actual, predicted}.issubset(test.columns):

            variance = test[predicted] - test[actual]

            test[f"variance_{metric}"] = variance
            test[f"probability_{metric}"] = _prediction_probability(test[actual], test[predicted])
            week_3 = f"actual_{metric}_week_3"
            week_4 = f"actual_{metric}_week_4"
            if {week_3, week_4}.issubset(test.columns):
                test[f"week_3_to_week_4_{metric}_growth_pct"] = _growth_percentage(test[week_3], test[week_4])

    off_10 = []
    off_20 = []
    for metric in metrics:
        actual = f"actual_{metric}_week_4"
        predicted = f"predicted_{metric}_week_4"
        if {actual, predicted}.issubset(test.columns):
            off_10.append(_is_off_by(test[actual], test[predicted], 0.10))
            off_20.append(_is_off_by(test[actual], test[predicted], 0.20))
    if off_10:
        test["off_by_10_pct"] = pd.concat(off_10, axis=1).max(axis=1)
        test["off_by_20_pct"] = pd.concat(off_20, axis=1).max(axis=1)

    # ------------------------------------------------------------------
    # Business-friendly column ordering
    # ------------------------------------------------------------------
    ordered_columns = [

        # ==============================================================
        # Game information
        # ==============================================================
        "Game_ID",
        "TournamentName",
        "SystemName",
        "First_Observed_Date",
        "Observation_Start",
        "Observation_End",
        "Prediction_Start",
        "Prediction_End",
        "Obs_Window",
        "Horizon",
        "off_by_10_pct",
        "off_by_20_pct",

    ]

    # ==============================================================
    # Group metrics together
    # ==============================================================
    for metric in metrics:

        ordered_columns.extend(
            [
                f"actual_{metric}_week_3",
                f"actual_{metric}_week_4",
                f"week_3_to_week_4_{metric}_growth_pct",
                f"baseline_{metric}_week_4",
                f"predicted_{metric}_week_4",
                f"confidence_{metric}",
                f"variance_{metric}",
                f"probability_{metric}",
            ]
        )

    # ------------------------------------------------------------------
    # Keep only columns that exist
    # ------------------------------------------------------------------
    ordered_columns = [
        col
        for col in ordered_columns
        if col in test.columns
    ]

    return test[ordered_columns]


def build_training_view(
    snapshot: pd.DataFrame,
    *,
    checkpoint=14,
    horizon=7,
    test_fraction=0.15,
    percentile=0.80,
):
    development, holdout, holdout_start = split_development_and_holdout(snapshot, test_fraction)
    development, _, threshold = create_high_performance_target(development, holdout, percentile)
    development = _period_columns(development, checkpoint, horizon)
    return development, holdout_start, threshold


def _make_excel_safe(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of frame with Excel-compatible datetime columns.

    Excel cannot store timezone-aware datetime values. This helper removes
    timezone information while preserving the displayed local timestamp.
    """
    safe = frame.copy()

    for column in safe.columns:
        series = safe[column]

        if pd.api.types.is_datetime64tz_dtype(series):
            safe[column] = series.dt.tz_localize(None)
            continue

        if series.dtype == "object":
            safe[column] = series.map(
                lambda value: value.replace(tzinfo=None)
                if hasattr(value, "tzinfo") and value.tzinfo is not None
                else value
            )

    return safe


def _high_performers(comparison: pd.DataFrame) -> pd.DataFrame:
    if "predicted_high_performer" not in comparison.columns:
        return comparison.iloc[0:0].copy()
    return comparison[comparison["predicted_high_performer"] == 1].copy()


def _forecast_export_columns(forecasts: pd.DataFrame) -> pd.DataFrame:
    allowed = [
        "Game_ID",
        "TournamentName",
        "SystemName",
        "First_Observed_Date",
        "Lifecycle_Stage",
        "Obs_Window",
        "Horizon",
        "active_observation_days",
        "observation_coverage",
        "predicted_clients_week_4",
        "confidence_clients",
        "predicted_stakes_week_4",
        "confidence_stakes",
        "Checkpoint",
        "As_Of_Date",
        "scored_at",
    ]
    return forecasts[[column for column in allowed if column in forecasts.columns]].copy()


def _model_accuracy(regression: pd.DataFrame) -> pd.DataFrame:
    if regression.empty or "Target" not in regression.columns:
        return regression.iloc[0:0].copy()
    keep_targets = {"target_clients", "target_stakes"}
    return regression[regression["Target"].isin(keep_targets)].copy()


def export_excel(
    path,
    *,
    comparison: pd.DataFrame,
    forecasts: pd.DataFrame,
    regression: pd.DataFrame,
    classification: pd.DataFrame,
):
    """Export business/model outputs to an Excel workbook."""
    path.parent.mkdir(parents=True, exist_ok=True)

    sheets = [
        ("Holdout Comparison", comparison),
        ("Forecasts", _forecast_export_columns(forecasts)),
        ("Model Accuracy", _model_accuracy(regression)),
    ]

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, frame in sheets:
            if not isinstance(frame, pd.DataFrame):
                continue

            excel_frame = _make_excel_safe(frame)

            excel_frame.to_excel(
                writer,
                sheet_name=name[:31],
                index=False,
            )

            ws = writer.sheets[name[:31]]

            # Freeze header
            ws.freeze_panes = "A2"

            # Auto-size columns
            for column_cells in ws.columns:
                length = max(
                    len(str(cell.value))
                    if cell.value is not None else 0
                    for cell in column_cells
                )

                ws.column_dimensions[
                    column_cells[0].column_letter
                ].width = min(max(length + 3, 12), 35)

    return path
