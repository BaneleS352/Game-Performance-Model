from __future__ import annotations

import numpy as np
import pandas as pd

from .validation import create_high_performance_target, split_development_and_holdout


def _period_columns(frame: pd.DataFrame, checkpoint: int, horizon: int) -> pd.DataFrame:
    result = frame.copy()
    first = pd.to_datetime(result["First_Observed_Date"], errors="coerce")
    result["Observation_Start"] = first
    result["Observation_End"] = first + pd.to_timedelta(checkpoint - 1, unit="D")
    result["Prediction_Start"] = first + pd.to_timedelta(checkpoint, unit="D")
    result["Prediction_End"] = result["Prediction_Start"] + pd.to_timedelta(horizon - 1, unit="D")
    return result


def _prediction_probability(actual: pd.Series, predicted: pd.Series) -> pd.Series:
    denominator = actual.abs().replace(0, np.nan)
    probability = 1 - (predicted - actual).abs() / denominator
    probability = probability.clip(lower=0, upper=1)
    return probability.mask(denominator.isna(), np.where(predicted.eq(actual), 1.0, 0.0))


def build_holdout_comparison(
    holdout_predictions: pd.DataFrame,
    checkpoint: int = 14,
    horizon: int = 7,
) -> pd.DataFrame:
    """Create a business-friendly holdout comparison report."""

    test = holdout_predictions[
        (holdout_predictions["Obs_Window"] == checkpoint)
        & (holdout_predictions["Horizon"] == horizon)
    ].copy()

    if test.empty:
        return test

    test = _period_columns(test, checkpoint, horizon)

    metrics = [
        "clients",
        "betcount",
        "stakes",
        "ggr",
        "ggr_pct",
    ]

    # ------------------------------------------------------------------
    # Error calculations
    # ------------------------------------------------------------------
    for metric in metrics:

        actual = f"actual_{metric}"
        predicted = f"predicted_{metric}"

        if {actual, predicted}.issubset(test.columns):

            variance = test[predicted] - test[actual]

            test[f"variance_{metric}"] = variance
            test[f"probability_{metric}"] = _prediction_probability(test[actual], test[predicted])

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

        # ==============================================================
        # Classification
        # ==============================================================
        "future_performance_score",
        "is_high_performer",
        "probability_high_performance",
        "predicted_high_performer",
    ]

    # ==============================================================
    # Group metrics together
    # ==============================================================
    for metric in metrics:

        ordered_columns.extend(
            [
                f"actual_{metric}",
                f"predicted_{metric}",
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

    # Preserve any remaining columns
    remaining_columns = [
        col
        for col in test.columns
        if col not in ordered_columns
    ]

    return test[
        ordered_columns + remaining_columns
    ]


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
    return forecasts.drop(columns=["recommendation", "Best_Iteration"], errors="ignore")


def _score_calculation_sheet() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Metric": "Future GGR",
                "Target field": "target_ggr",
                "Weight": 0.40,
                "Calculation note": "Empirical percentile within the development dataset.",
            },
            {
                "Metric": "Future Stakes",
                "Target field": "target_stakes",
                "Weight": 0.25,
                "Calculation note": "Empirical percentile within the development dataset.",
            },
            {
                "Metric": "Future Clients",
                "Target field": "target_clients",
                "Weight": 0.15,
                "Calculation note": "Empirical percentile within the development dataset.",
            },
            {
                "Metric": "Future Bet Count",
                "Target field": "target_betcount",
                "Weight": 0.10,
                "Calculation note": "Empirical percentile within the development dataset.",
            },
            {
                "Metric": "Future GGR %",
                "Target field": "target_ggr_pct",
                "Weight": 0.10,
                "Calculation note": "Empirical percentile within the development dataset.",
            },
        ]
    )


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
        ("High Performers", _high_performers(comparison)),
        ("Forecasts", _forecast_export_columns(forecasts)),
        ("future_performance_score_calculator", _score_calculation_sheet()),
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

            if name == "future_performance_score_calculator":
                ws["F1"] = "Formula"
                ws["F2"] = (
                    "future_performance_score = 40% * percentile(Future GGR) "
                    "+ 25% * percentile(Future Stakes) "
                    "+ 15% * percentile(Future Clients) "
                    "+ 10% * percentile(Future Bet Count) "
                    "+ 10% * percentile(Future GGR %)"
                )
                ws["F4"] = "High performer flag"
                ws["F5"] = (
                    "Marked high performer when future_performance_score is at or above "
                    "the 80th percentile threshold from the development scored dataset."
                )
                ws["C2"].number_format = "0%"
                ws["C3"].number_format = "0%"
                ws["C4"].number_format = "0%"
                ws["C5"].number_format = "0%"
                ws["C6"].number_format = "0%"

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
