from pathlib import Path

import pandas as pd
from openpyxl import load_workbook

from game_performance.reporting import build_calibration_report, build_holdout_comparison, export_excel


def test_holdout_comparison_uses_week4_probability_and_off_flags():
    holdout = pd.DataFrame(
        {
            "Game_ID": [1],
            "First_Observed_Date": ["2026-01-01"],
            "Obs_Window": [14],
            "Horizon": [14],
            "actual_clients_week_3": [80.0],
            "actual_clients_week_4": [100.0],
            "predicted_clients_week_4_low": [70.0],
            "predicted_clients_week_4": [85.0],
            "predicted_clients_week_4_high": [105.0],
            "confidence_clients": [0.75],
        }
    )

    comparison = build_holdout_comparison(holdout)

    expected = [
        "actual_clients_week_3",
        "actual_clients_week_4",
        "week_3_to_week_4_clients_growth_pct",
        "predicted_clients_week_4_low",
        "predicted_clients_week_4",
        "predicted_clients_week_4_high",
        "confidence_clients",
        "variance_clients",
        "probability_clients",
        "clients_actual_in_range",
        "clients_range_width",
    ]
    start = comparison.columns.get_loc("actual_clients_week_3")

    assert list(comparison.columns[start : start + len(expected)]) == expected
    assert comparison.loc[0, "week_3_to_week_4_clients_growth_pct"] == 0.25
    assert comparison.loc[0, "variance_clients"] == -15.0
    assert comparison.loc[0, "probability_clients"] == 0.85
    assert comparison.loc[0, "clients_actual_in_range"] == 1
    assert comparison.loc[0, "clients_range_width"] == 35.0
    assert comparison.loc[0, "off_by_10_pct"] == 1
    assert comparison.loc[0, "off_by_20_pct"] == 0
    assert "accuracy_clients" not in comparison.columns
    assert "absolute_error_clients" not in comparison.columns
    assert "variance_pct_clients" not in comparison.columns
    assert "High_Performer_Match" not in comparison.columns


def test_holdout_comparison_keeps_missing_range_actuals_missing():
    holdout = pd.DataFrame(
        {
            "Game_ID": [1],
            "First_Observed_Date": ["2026-01-01"],
            "Obs_Window": [14],
            "Horizon": [14],
            "actual_clients_week_4": [None],
            "predicted_clients_week_4_low": [70.0],
            "predicted_clients_week_4": [85.0],
            "predicted_clients_week_4_high": [105.0],
        }
    )

    comparison = build_holdout_comparison(holdout)

    assert pd.isna(comparison.loc[0, "clients_actual_in_range"])


def test_calibration_report_buckets_confidence_against_actual_rate():
    comparison = pd.DataFrame(
        {
            "is_high_performer": [0, 1, 1],
            "probability_high_performance": [0.05, 0.72, 0.76],
            "raw_probability_high_performance": [0.15, 0.90, 0.95],
        }
    )

    report = build_calibration_report(comparison, bins=2)

    assert list(report["Probability_Bin"]) == ["0.0-0.5", "0.5-1.0"]
    assert list(report["Count"]) == [1, 2]
    assert report.loc[1, "Actual_High_Performer_Rate"] == 1.0
    assert "Mean_Raw_Probability" in report.columns


def test_export_excel_keeps_clients_stakes_metrics_and_calibration_sheets():
    comparison = pd.DataFrame(
        {
            "Game_ID": [1, 2],
            "predicted_high_performer": [0, 1],
            "actual_clients": [10, 20],
            "predicted_clients": [11, 19],
            "actual_stakes": [100, 200],
            "predicted_stakes": [110, 190],
        }
    )

    path = Path("artifacts/test_report.xlsx")
    workbook = None
    try:
        export_excel(
            path,
            comparison=comparison,
            forecasts=pd.DataFrame(
                {
                    "Game_ID": [1],
                    "recommendation": ["Promote"],
                    "Best_Iteration": [42],
                    "predicted_clients_week_4_low": [8],
                    "predicted_clients_week_4": [10],
                    "predicted_clients_week_4_high": [12],
                    "confidence_clients": [0.8],
                    "predicted_stakes_week_4_low": [90],
                    "predicted_stakes_week_4": [100],
                    "predicted_stakes_week_4_high": [110],
                    "confidence_stakes": [0.7],
                    "probability_high_client_performer": [0.8],
                    "predicted_high_client_performer": [1],
                    "probability_high_stakes_performer": [0.7],
                    "predicted_high_stakes_performer": [1],
                    "predicted_ggr": [5],
                }
            ),
            regression=pd.DataFrame(
                {
                    "Target": ["target_clients", "target_ggr", "target_stakes"],
                    "MAE": [1.0, 2.0, 3.0],
                }
            ),
            classification=pd.DataFrame(),
        )

        workbook = load_workbook(path, read_only=True)

        assert workbook.sheetnames == [
            "Holdout Comparison",
            "High Performers",
            "Forecasts",
            "Model Accuracy",
            "future_performance_score_calcul",
        ]

        forecast_headers = next(workbook["Forecasts"].iter_rows(values_only=True))
        assert forecast_headers == (
            "Game_ID",
            "predicted_clients_week_4_low",
            "predicted_clients_week_4",
            "predicted_clients_week_4_high",
            "confidence_clients",
            "predicted_stakes_week_4_low",
            "predicted_stakes_week_4",
            "predicted_stakes_week_4_high",
            "confidence_stakes",
            "probability_high_client_performer",
            "predicted_high_client_performer",
            "probability_high_stakes_performer",
            "predicted_high_stakes_performer",
        )

        accuracy_rows = list(workbook["Model Accuracy"].iter_rows(values_only=True))
        assert accuracy_rows[0] == ("Target", "MAE")
        assert accuracy_rows[1] == ("target_clients", 1.0)
        assert accuracy_rows[2] == ("target_stakes", 3.0)

        score_rows = list(workbook["future_performance_score_calcul"].iter_rows(values_only=True))
        assert score_rows[0][:3] == ("Metric", "Target field", "Weight")
        assert score_rows[1][:2] == ("High client performer", "is_high_client_performer")
        assert score_rows[2][:2] == ("High stakes performer", "is_high_stakes_performer")
        assert score_rows[3][:2] == ("Balanced high performer", "is_balanced_high_performer")
    finally:
        if workbook is not None:
            workbook.close()
        path.unlink(missing_ok=True)
