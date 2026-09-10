from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .features import FORECAST_HORIZON_DAYS, OBSERVATION_WINDOW_DAYS
from .validation import create_high_performance_target, split_development_and_holdout


FORECAST_CAPABILITY_COLUMNS = [
    "History_Days",
    "Forecast_Days",
    "Games_Evaluated",
    "Input_Metrics",
    "Derived_Variables",
    "Clients_Average_Off_Percent",
    "Stakes_Average_Off_Percent",
    "BetCount_Average_Off_Percent",
    "GGR_Average_Off_Percent",
    "Average_Off_Percent",
    "Prediction_Range_Hit_Percent",
    "Summary",
]
FORECAST_METRICS = {
    "clients": "Clients",
    "stakes": "Stakes",
    "betcount": "BetCount",
    "ggr": "GGR",
}
DERIVED_VARIABLES = (
    "totals, averages, variability, first/last values, trends, growth, recent 3/7-day averages, "
    "recent shares/ratios, peaks, weekly changes, per-client/per-bet ratios and GGR margin"
)


def _period_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    first = pd.to_datetime(result["First_Observed_Date"], errors="coerce")
    checkpoint = pd.to_numeric(result.get("Obs_Window", OBSERVATION_WINDOW_DAYS), errors="coerce")
    horizon = pd.to_numeric(result.get("Horizon", FORECAST_HORIZON_DAYS), errors="coerce")
    result["Observation_Start"] = first
    result["Observation_End"] = first + pd.to_timedelta(checkpoint - 1, unit="D")
    result["Prediction_Start"] = first + pd.to_timedelta(checkpoint, unit="D")
    result["Prediction_End"] = result["Prediction_Start"] + pd.to_timedelta(horizon - 1, unit="D")
    return result


def _prediction_probability(actual: pd.Series, predicted: pd.Series) -> pd.Series:
    """
    Calculate prediction accuracy as a probability between 0 and 1.

    1.0 = exact prediction
    0.9 = prediction is 10% off
    0.8 = prediction is 20% off
    0.0 = prediction is 100%+ off

    Missing actual/predicted values remain NaN.
    If actual is zero:
        - predicted zero -> 1.0
        - predicted non-zero -> 0.0
    """

    # Force both inputs to numeric.
    # pd.NA, invalid strings, etc. become np.nan.
    actual = pd.to_numeric(actual, errors="coerce").astype("float64")
    predicted = pd.to_numeric(predicted, errors="coerce").astype("float64")

    # Start with missing values
    probability = pd.Series(np.nan, index=actual.index, dtype="float64")

    # Rows where both values exist
    valid = actual.notna() & predicted.notna()

    # Normal case: actual is not zero
    nonzero_actual = valid & actual.ne(0)

    probability.loc[nonzero_actual] = (
        1
        - (
            predicted.loc[nonzero_actual]
            - actual.loc[nonzero_actual]
        ).abs()
        / actual.loc[nonzero_actual].abs()
    ).clip(lower=0, upper=1)

    # Special case: actual = 0
    zero_actual = valid & actual.eq(0)

    probability.loc[zero_actual] = (
        predicted.loc[zero_actual]
        .eq(0)
        .astype("float64")
    )

    return probability


def _is_off_by(actual: pd.Series, predicted: pd.Series, threshold: float) -> pd.Series:
    actual = pd.to_numeric(actual, errors="coerce")
    predicted = pd.to_numeric(predicted, errors="coerce")
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

    metrics = ["clients", "stakes", "betcount", "ggr"]

    # ------------------------------------------------------------------
    # Error calculations
    # ------------------------------------------------------------------
    for metric in metrics:
        actual = f"actual_{metric}"
        predicted = f"predicted_{metric}"
        low = f"predicted_{metric}_low"
        high = f"predicted_{metric}_high"

        if {actual, predicted}.issubset(test.columns):
            variance = test[predicted] - test[actual]

            test[f"variance_{metric}"] = variance
            test[f"probability_{metric}"] = _prediction_probability(test[actual], test[predicted])
            if {low, high}.issubset(test.columns):
                range_valid = test[actual].notna() & test[low].notna() & test[high].notna()
                test[f"{metric}_actual_in_range"] = test[actual].between(
                    test[low], test[high], inclusive="both"
                ).where(range_valid).astype("Int8")
                test[f"{metric}_range_width"] = test[high] - test[low]

    off_10 = []
    off_20 = []
    for metric in metrics:
        actual = f"actual_{metric}"
        predicted = f"predicted_{metric}"
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
        "future_performance_score",
        "is_high_client_performer",
        "is_high_stakes_performer",
        "is_balanced_high_performer",
        "is_high_performer",
        "raw_probability_high_client_performer",
        "probability_high_client_performer",
        "classification_threshold_high_client_performer",
        "predicted_high_client_performer",
        "raw_probability_high_stakes_performer",
        "probability_high_stakes_performer",
        "classification_threshold_high_stakes_performer",
        "predicted_high_stakes_performer",
        "raw_probability_high_performance",
        "probability_high_performance",
        "classification_threshold",
        "predicted_high_performer",
    ]

    for metric in metrics:
        ordered_columns.extend(
            [
                f"actual_{metric}",
                f"baseline_{metric}",
                f"predicted_{metric}_low",
                f"predicted_{metric}",
                f"predicted_{metric}_high",
                f"confidence_{metric}",
                f"variance_{metric}",
                f"probability_{metric}",
                f"{metric}_actual_in_range",
                f"{metric}_range_width",
            ]
        )

    ordered_columns = [
        col
        for col in ordered_columns
        if col in test.columns
    ]

    remaining_columns = [
        col
        for col in test.columns
        if col not in ordered_columns
    ]

    return test[ordered_columns + remaining_columns]


def build_forecast_capability_report(comparison: pd.DataFrame) -> pd.DataFrame:
    """Summarise X history days -> Y forecast days -> Z games -> W average error."""
    if comparison.empty or not {"Obs_Window", "Horizon"}.issubset(comparison.columns):
        return pd.DataFrame(columns=FORECAST_CAPABILITY_COLUMNS)

    frame = comparison.copy()
    frame["Obs_Window"] = pd.to_numeric(frame["Obs_Window"], errors="coerce")
    frame["Horizon"] = pd.to_numeric(frame["Horizon"], errors="coerce")
    frame = frame.dropna(subset=["Obs_Window", "Horizon"])
    rows = []

    for (history_days, forecast_days), group in frame.groupby(["Obs_Window", "Horizon"], sort=True):
        average_off: dict[str, float] = {}
        range_hits: list[float] = []

        for metric in FORECAST_METRICS:
            actual_column = f"actual_{metric}"
            predicted_column = f"predicted_{metric}"
            if actual_column not in group or predicted_column not in group:
                average_off[metric] = np.nan
                continue
            actual = pd.to_numeric(group[actual_column], errors="coerce")
            predicted = pd.to_numeric(group[predicted_column], errors="coerce")

            valid = actual.notna() & predicted.notna()
            denominator = actual.loc[valid].abs().sum()
            average_off[metric] = (
                float((predicted.loc[valid] - actual.loc[valid]).abs().sum() / denominator)
                if valid.any() and denominator
                else np.nan
            )

            low_column = f"predicted_{metric}_low"
            high_column = f"predicted_{metric}_high"
            if low_column in group and high_column in group:
                low = pd.to_numeric(group[low_column], errors="coerce")
                high = pd.to_numeric(group[high_column], errors="coerce")
                range_valid = actual.notna() & low.notna() & high.notna()
                range_hits.extend(actual.loc[range_valid].between(low.loc[range_valid], high.loc[range_valid]).astype(float))

        measured = [value for value in average_off.values() if pd.notna(value)]
        mean_off = float(np.mean(measured)) if measured else np.nan
        range_coverage = float(np.mean(range_hits)) if range_hits else np.nan
        games = int(group["Game_ID"].nunique()) if "Game_ID" in group else int(len(group))
        history_days = int(history_days)
        forecast_days = int(forecast_days)
        error_text = f"{mean_off:.1%}" if pd.notna(mean_off) else "unavailable"
        coverage_text = (
            f"; {range_coverage:.1%} of actual metric values fell inside the prediction ranges"
            if pd.notna(range_coverage)
            else ""
        )

        rows.append(
            {
                "History_Days": history_days,
                "Forecast_Days": forecast_days,
                "Games_Evaluated": games,
                "Input_Metrics": ", ".join(FORECAST_METRICS.values()),
                "Derived_Variables": DERIVED_VARIABLES,
                **{
                    f"{label}_Average_Off_Percent": value * 100 if pd.notna(value) else np.nan
                    for metric, label in FORECAST_METRICS.items()
                    for value in [average_off[metric]]
                },
                "Average_Off_Percent": mean_off * 100 if pd.notna(mean_off) else np.nan,
                "Prediction_Range_Hit_Percent": range_coverage * 100 if pd.notna(range_coverage) else np.nan,
                "Summary": (
                    f"Using {history_days} days of daily Clients, Stakes, BetCount and GGR history, "
                    f"the model forecasts the next {forecast_days} days for {games} holdout games "
                    f"with {error_text} average error (WAPE){coverage_text}."
                ),
            }
        )

    return pd.DataFrame(rows, columns=FORECAST_CAPABILITY_COLUMNS)


def build_calibration_report(comparison: pd.DataFrame, *, bins: int = 10) -> pd.DataFrame:
    """Bucket calibrated high-performer probability against observed hit rates."""
    required = {"is_high_performer", "probability_high_performance"}
    if comparison.empty or not required.issubset(comparison.columns):
        return pd.DataFrame()

    frame = pd.DataFrame(
        {
            "actual": pd.to_numeric(comparison["is_high_performer"], errors="coerce"),
            "probability": pd.to_numeric(comparison["probability_high_performance"], errors="coerce").clip(0, 1),
        }
    )
    if "raw_probability_high_performance" in comparison.columns:
        frame["raw_probability"] = pd.to_numeric(
            comparison["raw_probability_high_performance"],
            errors="coerce",
        ).clip(0, 1)
    frame = frame.dropna(subset=["actual", "probability"])
    if frame.empty:
        return pd.DataFrame()

    edges = np.linspace(0, 1, bins + 1)
    frame["_bin"] = np.digitize(frame["probability"], edges[1:-1], right=False)
    rows: list[dict[str, float | int | str]] = []

    for bin_id in range(bins):
        lower = float(edges[bin_id])
        upper = float(edges[bin_id + 1])
        bucket = frame[frame["_bin"] == bin_id]
        mean_probability = bucket["probability"].mean() if not bucket.empty else np.nan
        actual_rate = bucket["actual"].mean() if not bucket.empty else np.nan
        row: dict[str, float | int | str] = {
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
        if "raw_probability" in frame.columns:
            raw_mean = bucket["raw_probability"].mean() if not bucket.empty else np.nan
            row["Mean_Raw_Probability"] = float(raw_mean) if pd.notna(raw_mean) else np.nan
        rows.append(row)

    return pd.DataFrame(rows)


def build_training_view(
    snapshot: pd.DataFrame,
    *,
    checkpoint=OBSERVATION_WINDOW_DAYS,
    horizon=FORECAST_HORIZON_DAYS,
    test_fraction=0.15,
    percentile=0.80,
):
    development, holdout, holdout_start = split_development_and_holdout(snapshot, test_fraction)
    development, _, threshold = create_high_performance_target(development, holdout, percentile)
    development = _period_columns(development)
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


BUSINESS_REPORT_EXCLUDED_COLUMNS = {
    "TournamentName",
    "SystemName",
    "First_Observed_Date",
    "Observation_Start",
    "Observation_End",
    "Prediction_Start",
    "Prediction_End",
    "future_performance_score",
    "is_high_client_performer",
    "is_high_stakes_performer",
    "is_balanced_high_performer",
    "raw_probability_high_client_performer",
    "probability_high_client_performer",
    "classification_threshold_high_client_performer",
    "predicted_high_client_performer",
    "raw_probability_high_stakes_performer",
    "probability_high_stakes_performer",
    "classification_threshold_high_stakes_performer",
    "predicted_high_stakes_performer",
    "raw_probability_high_performance",
    "probability_high_performance",
    "classification_threshold",
}


def _business_report_view(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(columns=BUSINESS_REPORT_EXCLUDED_COLUMNS.intersection(frame.columns)).copy()


def _forecast_export_columns(forecasts: pd.DataFrame) -> pd.DataFrame:
    allowed = [
        "Game_ID",
        "TournamentName",
        "SystemName",
        "First_Observed_Date",
        "Lifecycle_Stage",
        "Prediction_Start",
        "Prediction_End",
        "Obs_Window",
        "Horizon",
        "active_observation_days",
        "observation_coverage",
        "actual_clients",
        "baseline_clients",
        "predicted_clients_low",
        "predicted_clients",
        "predicted_clients_high",
        "confidence_clients",
        "actual_betcount",
        "baseline_betcount",
        "predicted_betcount_low",
        "predicted_betcount",
        "predicted_betcount_high",
        "confidence_betcount",
        "actual_ggr",
        "baseline_ggr",
        "predicted_ggr_low",
        "predicted_ggr",
        "predicted_ggr_high",
        "confidence_ggr",
        "actual_stakes",
        "baseline_stakes",
        "predicted_stakes_low",
        "predicted_stakes",
        "predicted_stakes_high",
        "confidence_stakes",
        "requires_review",
        "review_reason",
        "review_priority",
        "probability_high_client_performer",
        "predicted_high_client_performer",
        "probability_high_stakes_performer",
        "predicted_high_stakes_performer",
        "raw_probability_high_performance",
        "probability_high_performance",
        "classification_threshold",
        "predicted_high_performer",
        "Checkpoint",
        "As_Of_Date",
        "scored_at",
    ]
    return forecasts[[column for column in allowed if column in forecasts.columns]].copy()


def _model_accuracy(regression: pd.DataFrame) -> pd.DataFrame:
    if regression.empty or "Target" not in regression.columns:
        return regression.iloc[0:0].copy()
    keep_targets = {"target_clients", "target_stakes", "target_betcount", "target_ggr"}
    return regression[regression["Target"].isin(keep_targets)].copy()


def _score_calculation_sheet() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Metric": "High client performer",
                "Target field": "is_high_client_performer",
                "Weight": np.nan,
                "Calculation note": "Forecast-window clients at or above the 80th percentile from the development dataset.",
            },
            {
                "Metric": "High stakes performer",
                "Target field": "is_high_stakes_performer",
                "Weight": np.nan,
                "Calculation note": "Forecast-window stakes at or above the 80th percentile from the development dataset.",
            },
            {
                "Metric": "Balanced high performer",
                "Target field": "is_balanced_high_performer",
                "Weight": np.nan,
                "Calculation note": "Marked when both client and stakes high-performer flags are true.",
            },
        ]
    )


def _write_excel(path: Path, sheets: list[tuple[str, pd.DataFrame]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, frame in sheets:
            if not isinstance(frame, pd.DataFrame):
                continue

            excel_frame = _make_excel_safe(frame)
            excel_frame.to_excel(writer, sheet_name=name[:31], index=False)
            ws = writer.sheets[name[:31]]
            ws.freeze_panes = "A2"

            if name == "future_performance_score_calculator":
                ws["F1"] = "Formula"
                ws["F2"] = (
                    "High client/stakes performers are games at or above each "
                    "development 80th percentile threshold."
                )
                ws["F4"] = "High performer flag"
                ws["F5"] = "Balanced high performer is marked when both client and stakes flags are true."

            for column_cells in ws.columns:
                length = max(
                    len(str(cell.value)) if cell.value is not None else 0
                    for cell in column_cells
                )
                ws.column_dimensions[column_cells[0].column_letter].width = min(max(length + 3, 12), 35)
    return path


def export_business_report(
    path,
    *,
    comparison: pd.DataFrame,
    forecasts: pd.DataFrame,
) -> Path:
    """Export the concise workbook used by business stakeholders."""
    return _write_excel(
        Path(path),
        [
            ("Forecast Capability", build_forecast_capability_report(comparison)),
            ("Holdout Comparison", _business_report_view(comparison)),
            ("High Performers", _business_report_view(_high_performers(forecasts))),
            ("Forecasting", _business_report_view(_forecast_export_columns(forecasts))),
        ],
    )


def export_excel(
    path,
    *,
    comparison: pd.DataFrame,
    forecasts: pd.DataFrame,
    regression: pd.DataFrame,
    classification: pd.DataFrame,
    calibration: pd.DataFrame | None = None,
    configuration_leaderboard: pd.DataFrame | None = None,
    uncertainty_validation: pd.DataFrame | None = None,
):
    """Export the complete Model Analysis workbook."""
    path = Path(path)
    calibration = build_calibration_report(comparison) if calibration is None else calibration

    sheets = [
        ("Forecast Capability", build_forecast_capability_report(comparison)),
        ("Holdout Comparison", comparison),
        ("High Performers", _high_performers(comparison)),
        ("Forecasts", _forecast_export_columns(forecasts)),
        ("Model Accuracy", _model_accuracy(regression)),
    ]
    sheets.extend(
        (name, frame)
        for name, frame in (
            ("Classification Metrics", classification),
            ("Calibration", calibration),
            ("Uncertainty Validation", uncertainty_validation),
        )
        if isinstance(frame, pd.DataFrame) and not frame.empty
    )
    if isinstance(configuration_leaderboard, pd.DataFrame) and not configuration_leaderboard.empty:
        sheets.append(("Configuration Selection", configuration_leaderboard))
    sheets.append(("future_performance_score_calculator", _score_calculation_sheet()))

    return _write_excel(path, sheets)
