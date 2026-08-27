from pathlib import Path

import pandas as pd
from openpyxl import load_workbook

from game_performance.reporting import build_holdout_comparison, export_excel


def test_holdout_comparison_uses_week4_confidence_and_off_flags():
    holdout = pd.DataFrame(
        {
            "Game_ID": [1],
            "First_Observed_Date": ["2026-01-01"],
            "Obs_Window": [14],
            "Horizon": [14],
            "actual_clients_week_3": [80.0],
            "actual_clients_week_4": [100.0],
            "predicted_clients_week_4": [85.0],
            "confidence_clients": [0.75],
        }
    )

    comparison = build_holdout_comparison(holdout)

    expected = [
        "actual_clients_week_3",
        "actual_clients_week_4",
        "week_3_to_week_4_clients_growth_pct",
        "predicted_clients_week_4",
        "confidence_clients",
        "variance_clients",
        "probability_clients",
    ]
    start = comparison.columns.get_loc("actual_clients_week_3")

    assert list(comparison.columns[start : start + len(expected)]) == expected
    assert comparison.loc[0, "week_3_to_week_4_clients_growth_pct"] == 0.25
    assert comparison.loc[0, "variance_clients"] == -15.0
    assert comparison.loc[0, "probability_clients"] == 0.85
    assert comparison.loc[0, "off_by_10_pct"] == 1
    assert comparison.loc[0, "off_by_20_pct"] == 0
    assert "absolute_error_clients" not in comparison.columns
    assert "variance_pct_clients" not in comparison.columns
    assert "High_Performer_Match" not in comparison.columns


def test_export_excel_keeps_stakes_and_clients_sheets_only():
    comparison = pd.DataFrame(
        {
            "Game_ID": [1, 2],
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
                    "predicted_clients_week_4": [10],
                    "confidence_clients": [0.8],
                    "predicted_stakes_week_4": [100],
                    "confidence_stakes": [0.7],
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
            "Forecasts",
            "Model Accuracy",
        ]

        forecast_headers = next(workbook["Forecasts"].iter_rows(values_only=True))
        assert forecast_headers == (
            "Game_ID",
            "predicted_clients_week_4",
            "confidence_clients",
            "predicted_stakes_week_4",
            "confidence_stakes",
        )

        accuracy_rows = list(workbook["Model Accuracy"].iter_rows(values_only=True))
        assert accuracy_rows[0] == ("Target", "MAE")
        assert accuracy_rows[1] == ("target_clients", 1.0)
        assert accuracy_rows[2] == ("target_stakes", 3.0)
    finally:
        if workbook is not None:
            workbook.close()
        path.unlink(missing_ok=True)
