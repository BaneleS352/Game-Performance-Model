from pathlib import Path

import json
import shutil

import pandas as pd

from game_performance.dashboard import build_dashboard_data, export_dashboard


def _sample_workbook(path: Path) -> None:
    forecasts = pd.DataFrame(
        {
            "Game_ID": ["SYS::Game A", "SYS::Game B"],
            "TournamentName": ["Game A", "Game B"],
            "SystemName": ["SYS", "SYS"],
            "First_Observed_Date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "predicted_clients_week_4": [100, 50],
            "confidence_clients": [0.8, 0.6],
            "predicted_stakes_week_4": [1000, 500],
            "confidence_stakes": [0.7, 0.4],
            "probability_high_performance": [0.9, 0.1],
            "active_observation_days": [14, 14],
            "observation_coverage": [1.0, 1.0],
            "As_Of_Date": pd.to_datetime(["2026-02-01", "2026-02-01"]),
            "predicted_ggr": [999, 999],
        }
    )
    holdout = pd.DataFrame(
        {
            "Game_ID": ["SYS::Game C"],
            "off_by_10_pct": [1],
            "off_by_20_pct": [0],
            "probability_clients": [0.9],
            "probability_stakes": [0.85],
        }
    )
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        holdout.to_excel(writer, sheet_name="Holdout Comparison", index=False)
        forecasts.to_excel(writer, sheet_name="Forecasts", index=False)


def test_dashboard_data_summarises_workbook():
    workspace = Path("artifacts/test_dashboard")
    workbook = workspace / "report.xlsx"
    try:
        workspace.mkdir(parents=True, exist_ok=True)
        _sample_workbook(workbook)

        data = build_dashboard_data(workbook)

        assert data["summary"]["games_scored"] == 2
        assert data["summary"]["holdout_games"] == 1
        assert data["summary"]["total_predicted_stakes_week_4"] == 1500
        assert data["summary"]["avg_probability_clients"] == 0.9
        assert data["systems"][0]["SystemName"] == "SYS"
        assert data["forecasts"][0]["confidence_clients"] == 0.8
        assert data["forecasts"][1]["confidence_stakes"] == 0.4
        assert data["forecasts"][0]["probability_high_performance"] == 0.9
        assert data["forecasts"][0]["predicted_ggr"] == 999
        assert [sheet["name"] for sheet in data["workbook_sheets"]] == ["Holdout Comparison", "Forecasts"]
        assert data["workbook_sheets"][1]["totalRows"] == 2
        assert data["workbook_sheets"][1]["rows"][0]["TournamentName"] == "Game A"
        assert [column["key"] for column in data["columns"]["forecasts"]] == list(
            pd.read_excel(workbook, sheet_name="Forecasts", nrows=0).columns
        )
        assert any(
            column["key"] == "predicted_ggr" and column["type"] == "money"
            for column in data["columns"]["forecasts"]
        )
        assert any(
            column["key"] == "predicted_stakes_week_4" and column["defaultVisible"]
            for column in data["columns"]["forecasts"]
        )
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_export_dashboard_writes_json_and_local_js():
    workspace = Path("artifacts/test_dashboard")
    workbook = workspace / "report.xlsx"
    output = workspace / "dashboard"
    try:
        workspace.mkdir(parents=True, exist_ok=True)
        _sample_workbook(workbook)

        json_path = export_dashboard(workbook, output)

        assert json_path.exists()
        assert (output / "dashboard_data.js").exists()
        assert json.loads(json_path.read_text(encoding="utf-8"))["summary"]["games_scored"] == 2
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_dashboard_uses_latest_checkpoint_for_each_game():
    workspace = Path("artifacts/test_dashboard")
    workbook = workspace / "report.xlsx"
    try:
        workspace.mkdir(parents=True, exist_ok=True)
        forecasts = pd.DataFrame(
            {
                "Game_ID": ["SYS::Game A", "SYS::Game A"],
                "TournamentName": ["Game A", "Game A"],
                "SystemName": ["SYS", "SYS"],
                "Checkpoint": [14, 21],
                "predicted_clients_week_4": [100, 120],
                "confidence_clients": [0.6, 0.8],
                "predicted_stakes_week_4": [1000, 1500],
                "confidence_stakes": [0.5, 0.7],
            }
        )
        with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
            pd.DataFrame({"Game_ID": ["SYS::Game A"]}).to_excel(
                writer, sheet_name="Holdout Comparison", index=False
            )
            forecasts.to_excel(writer, sheet_name="Forecasts", index=False)

        data = build_dashboard_data(workbook)

        assert data["summary"]["games_scored"] == 1
        assert data["forecasts"][0]["predicted_stakes_week_4"] == 1500
        assert data["forecasts"][0]["confidence_clients"] == 0.8
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_dashboard_uses_latest_checkpoint_holdout_metrics():
    workspace = Path("artifacts/test_dashboard")
    workbook = workspace / "report.xlsx"
    try:
        workspace.mkdir(parents=True, exist_ok=True)
        forecasts = pd.DataFrame(
            {
                "Game_ID": ["SYS::Game A"],
                "SystemName": ["SYS"],
                "predicted_clients_week_4": [100],
                "confidence_clients": [0.8],
                "predicted_stakes_week_4": [1000],
                "confidence_stakes": [0.7],
            }
        )
        holdout = pd.DataFrame(
            {
                "Game_ID": ["SYS::Game A", "SYS::Game A"],
                "Obs_Window": [14, 21],
                "off_by_10_pct": [1, 0],
                "off_by_20_pct": [1, 0],
                "probability_clients": [0.2, 0.8],
                "probability_stakes": [0.3, 0.9],
            }
        )
        with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
            holdout.to_excel(writer, sheet_name="Holdout Comparison", index=False)
            forecasts.to_excel(writer, sheet_name="Forecasts", index=False)

        data = build_dashboard_data(workbook)

        assert data["summary"]["holdout_games"] == 1
        assert data["summary"]["holdout_off_by_10_rate"] == 0
        assert data["summary"]["avg_probability_stakes"] == 0.9
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
