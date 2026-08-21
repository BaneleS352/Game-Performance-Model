from pathlib import Path

import pandas as pd
from openpyxl import load_workbook

from game_performance.reporting import build_holdout_comparison, export_excel


def test_holdout_comparison_uses_probability_metric_layout():
    holdout = pd.DataFrame(
        {
            "Game_ID": [1],
            "First_Observed_Date": ["2026-01-01"],
            "Obs_Window": [14],
            "Horizon": [7],
            "future_performance_score": [0.81],
            "actual_clients": [100.0],
            "predicted_clients": [90.0],
        }
    )

    comparison = build_holdout_comparison(holdout)

    expected = [
        "actual_clients",
        "predicted_clients",
        "variance_clients",
        "probability_clients",
    ]
    start = comparison.columns.get_loc("actual_clients")

    assert list(comparison.columns[start : start + len(expected)]) == expected
    assert comparison.loc[0, "variance_clients"] == -10.0
    assert comparison.loc[0, "probability_clients"] == 0.9
    assert "absolute_error_clients" not in comparison.columns
    assert "variance_pct_clients" not in comparison.columns
    assert "High_Performer_Match" not in comparison.columns


def test_export_excel_adds_high_performers_sheet():
    comparison = pd.DataFrame(
        {
            "Game_ID": [1, 2],
            "future_performance_score": [0.90, 0.70],
            "predicted_high_performer": [0, 1],
            "actual_clients": [10, 20],
            "predicted_clients": [11, 19],
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
                    "predicted_clients": [10],
                }
            ),
            regression=pd.DataFrame(),
            classification=pd.DataFrame(),
        )

        workbook = load_workbook(path, read_only=True)
        sheet = workbook["High Performers"]
        rows = list(sheet.iter_rows(values_only=True))

        assert workbook.sheetnames == [
            "Holdout Comparison",
            "High Performers",
            "Forecasts",
            "future_performance_score_calcul",
        ]
        assert rows[0] == tuple(comparison.columns)
        assert rows[1] == (2, 0.70, 1, 20, 19)
        assert len(rows) == 2

        forecast_headers = next(workbook["Forecasts"].iter_rows(values_only=True))
        assert forecast_headers == ("Game_ID", "predicted_clients")

        score_rows = list(workbook["future_performance_score_calcul"].iter_rows(values_only=True))
        assert score_rows[0][:3] == ("Metric", "Target field", "Weight")
        assert score_rows[1][:3] == ("Future GGR", "target_ggr", 0.40)
        assert score_rows[5][:3] == ("Future GGR %", "target_ggr_pct", 0.10)
    finally:
        if workbook is not None:
            workbook.close()
        path.unlink(missing_ok=True)
